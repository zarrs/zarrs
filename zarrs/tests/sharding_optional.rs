#![allow(missing_docs)]

//! Tests for the `sharding_indexed` codec with optional data types.

use std::fmt::Debug;
use std::num::NonZeroU64;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::ShardingCodecBuilder;
use zarrs::array::{ArrayBuilder, ArraySubset, DataType, ElementOwned, FillValue, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, CodecOptions,
    CodecSpecificOptions, UnboundArrayToBytesCodecTraits,
};

const SHAPE: [u64; 2] = [8, 6];
const SHARD_SHAPE: [u64; 2] = [4, 6];
const SUBCHUNK_SHAPE: [u64; 2] = [2, 2];

/// Elements at `indices` of an array with shape [`SHAPE`].
fn select<T: Clone>(elements: &[T], indices: &[[u64; 2]]) -> Vec<T> {
    indices
        .iter()
        .map(|[i, j]| elements[usize::try_from(i * SHAPE[1] + j).unwrap()].clone())
        .collect()
}

/// The innermost data type of (possibly nested) optional data types, and the nesting depth.
fn optional_innermost(data_type: &DataType) -> (&DataType, usize) {
    let mut data_type = data_type;
    let mut depth = 0;
    while let Some(inner) = data_type.optional_inner() {
        data_type = inner;
        depth += 1;
    }
    (data_type, depth)
}

/// Build a decode target from views of the data and validity masks (outer to inner).
fn nested_target<'a>(
    data_view: &'a mut ArrayBytesFixedDisjointView<'a>,
    mask_views: &'a mut [ArrayBytesFixedDisjointView<'a>],
) -> ArrayBytesDecodeIntoTarget<'a> {
    if let Some((mask_view, mask_views)) = mask_views.split_first_mut() {
        ArrayBytesDecodeIntoTarget::Optional(
            Box::new(nested_target(data_view, mask_views)),
            mask_view,
        )
    } else {
        ArrayBytesDecodeIntoTarget::Fixed(data_view)
    }
}

/// Decode `$subset` of fixed length (optional) `$data_type` data with `$decode_into` into a new target.
macro_rules! decode_into_new {
    ($data_type:expr, $subset:expr, |$target:ident| $decode_into:expr) => {{
        let (inner_data_type, depth) = optional_innermost($data_type);
        let data_type_size = inner_data_type.fixed_size().unwrap();
        let shape = $subset.shape().to_vec();
        let num_elements = usize::try_from($subset.num_elements())?;
        let mut data = vec![0u8; num_elements * data_type_size];
        let mut masks = vec![vec![0u8; num_elements]; depth];
        {
            let subset = ArraySubset::new_with_shape(shape.clone());
            let mut data_view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut data),
                    data_type_size,
                    &shape,
                    subset.clone(),
                )?
            };
            let mut mask_views = masks
                .iter_mut()
                .map(|mask| unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(mask),
                        1,
                        &shape,
                        subset.clone(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let $target = nested_target(&mut data_view, &mut mask_views);
            $decode_into?;
        }
        masks
            .into_iter()
            .rev()
            .fold(ArrayBytes::new_flen(data), ArrayBytes::with_optional_mask)
    }};
}

fn sharding_optional_round_trip<T: ElementOwned + Clone + PartialEq + Debug>(
    data_type: DataType,
    elements: &[T],
) -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(
        SHAPE.to_vec(),
        SHARD_SHAPE.to_vec(),
        data_type.clone(),
        FillValue::new_optional_null(),
    );
    builder.subchunk_shape(SUBCHUNK_SHAPE.to_vec());
    let array = builder.build(store, "/")?;
    array.store_array_subset(
        &array.subset_all(),
        T::to_array_bytes(&data_type, elements)?,
    )?;

    // Retrieve the entire array (spanning multiple shards)
    assert_eq!(
        array.retrieve_array_subset::<Vec<T>>(&array.subset_all())?,
        elements
    );

    // Retrieve a subset spanning multiple shards and partial subchunks
    let subset = ArraySubset::new_with_ranges(&[3..7, 1..4]);
    let indices: Vec<[u64; 2]> = (3..7).flat_map(|i| (1..4).map(move |j| [i, j])).collect();
    assert_eq!(
        array.retrieve_array_subset::<Vec<T>>(&subset)?,
        select(elements, &indices)
    );
    Ok(())
}

/// Encode a shard, then decode it entirely and partially with a generic indexer spanning multiple subchunks.
async fn sharding_optional_codec<T: ElementOwned + Clone + PartialEq + Debug>(
    data_type: DataType,
    elements: &[T],
) -> Result<(), Box<dyn std::error::Error>> {
    let shard_shape = SHARD_SHAPE.map(|size| NonZeroU64::new(size).unwrap());
    let shard_elements = &elements[..usize::try_from(SHARD_SHAPE.iter().product::<u64>())?];
    let codec = ShardingCodecBuilder::new(
        SUBCHUNK_SHAPE
            .map(|size| NonZeroU64::new(size).unwrap())
            .to_vec(),
        &data_type,
    )
    .build_arc()
    .with_context(
        data_type.clone(),
        FillValue::new_optional_null(),
        &CodecSpecificOptions::default(),
    )?;
    let options = CodecOptions::default();
    let encoded = codec
        .encode(
            T::to_array_bytes(&data_type, shard_elements)?,
            &shard_shape,
            &options,
        )?
        .into_vec();

    // Decode
    let decoded = codec.decode(encoded.clone().into(), &shard_shape, &options)?;
    assert_eq!(T::from_array_bytes(&data_type, decoded)?, shard_elements);

    // Partial decode with a generic indexer
    let indexer = vec![vec![0, 0], vec![3, 5], vec![2, 3], vec![0, 1], vec![3, 0]];
    let expected = select(
        shard_elements,
        &indexer.iter().map(|i| [i[0], i[1]]).collect::<Vec<_>>(),
    );
    let partial_decoder =
        codec
            .clone()
            .partial_decoder(Arc::new(encoded.clone()), &shard_shape, &options)?;
    let decoded = partial_decoder.partial_decode(&indexer, &options)?;
    assert_eq!(T::from_array_bytes(&data_type, decoded)?, expected);

    // Partial decode with an array subset spanning multiple (partial) subchunks
    let subset = ArraySubset::new_with_ranges(&[1..4, 1..5]);
    let expected_subset = select(
        shard_elements,
        &(1..4)
            .flat_map(|i| (1..5).map(move |j| [i, j]))
            .collect::<Vec<_>>(),
    );
    let fixed = optional_innermost(&data_type).0.is_fixed();
    let decoded = partial_decoder.partial_decode(&subset, &options)?;
    assert_eq!(T::from_array_bytes(&data_type, decoded)?, expected_subset);
    if fixed {
        let decoded = decode_into_new!(&data_type, subset, |target| partial_decoder
            .partial_decode_into(&subset, target, &options));
        assert_eq!(T::from_array_bytes(&data_type, decoded)?, expected_subset);
    }

    #[cfg(feature = "async")]
    {
        let partial_decoder = codec
            .clone()
            .async_partial_decoder(Arc::new(encoded), &shard_shape, &options)
            .await?;
        let decoded = partial_decoder.partial_decode(&indexer, &options).await?;
        assert_eq!(T::from_array_bytes(&data_type, decoded)?, expected);
        let decoded = partial_decoder.partial_decode(&subset, &options).await?;
        assert_eq!(T::from_array_bytes(&data_type, decoded)?, expected_subset);
        if fixed {
            let decoded = decode_into_new!(&data_type, subset, |target| partial_decoder
                .partial_decode_into(&subset, target, &options)
                .await);
            assert_eq!(T::from_array_bytes(&data_type, decoded)?, expected_subset);
        }
    }
    Ok(())
}

async fn sharding_optional<T: ElementOwned + Clone + PartialEq + Debug>(
    data_type: DataType,
    element: impl Fn(u64) -> T,
) -> Result<(), Box<dyn std::error::Error>> {
    let elements: Vec<T> = (0..SHAPE.iter().product()).map(element).collect();
    sharding_optional_round_trip(data_type.clone(), &elements)?;
    sharding_optional_codec(data_type, &elements).await
}

#[tokio::test]
async fn sharding_optional_uint8() -> Result<(), Box<dyn std::error::Error>> {
    sharding_optional(data_type::uint8().to_optional(), |i| {
        (i % 3 != 0).then(|| u8::try_from(i).unwrap())
    })
    .await
}

#[tokio::test]
async fn sharding_optional_optional_uint16() -> Result<(), Box<dyn std::error::Error>> {
    sharding_optional(
        data_type::uint16().to_optional().to_optional(),
        |i| match i % 3 {
            0 => None,
            1 => Some(None),
            _ => Some(Some(u16::try_from(i).unwrap())),
        },
    )
    .await
}

#[tokio::test]
async fn sharding_optional_string() -> Result<(), Box<dyn std::error::Error>> {
    sharding_optional(data_type::string().to_optional(), |i| {
        (i % 4 != 0).then(|| format!("element {i}"))
    })
    .await
}

#[tokio::test]
async fn sharding_optional_missing_subchunks() -> Result<(), Box<dyn std::error::Error>> {
    // The subchunks of the first two rows are entirely the (null) fill value, so they are not stored
    let row = |i: u64| i / SHAPE[1];
    sharding_optional(data_type::uint8().to_optional(), |i| {
        (row(i) >= 2 && i % 3 != 0).then(|| u8::try_from(i).unwrap())
    })
    .await?;
    sharding_optional(
        data_type::uint16().to_optional().to_optional(),
        |i| match (row(i), i % 3) {
            (0 | 1, _) | (_, 0) => None,
            (_, 1) => Some(None),
            _ => Some(Some(u16::try_from(i).unwrap())),
        },
    )
    .await
}

#![allow(missing_docs)]

//! Decoding into targets whose shape differs from the decoded shape, which only needs the number of elements to match.

use std::num::NonZeroU64;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::chunk_cache::ChunkCacheEncodedLruChunkLimit;
use zarrs::array::codec::ShardingCodecBuilder;
use zarrs::array::{
    ArrayBuilder, ArrayCached, ArraySubset, ComputeResources, DataType, FillValue, data_type,
};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, CodecOptions,
    CodecSpecificOptions, Resources, UnboundArrayToBytesCodecTraits,
};

const ARRAY_SHAPE: [u64; 2] = [8, 8];
const SHARD_SHAPE: [u64; 2] = [4, 8];
const SUBCHUNK_SHAPE: [u64; 2] = [2, 2];
const SENTINEL: u8 = 0xa5;

/// The shape of an output array and the subset of it covered by a target of `num_elements` elements.
fn layouts(num_elements: u64) -> [(Vec<u64>, ArraySubset); 2] {
    [
        // 1D, offset from the start
        (
            vec![num_elements + 4],
            ArraySubset::new_with_ranges(&[2..num_elements + 2]),
        ),
        // 2D with a different shape to the decoded shape, offset from the start
        (
            vec![3, num_elements / 2 + 2],
            ArraySubset::new_with_ranges(&[1..3, 1..num_elements / 2 + 1]),
        ),
    ]
}

/// Output buffers for a target covering `subset` of an output array with `shape`.
struct LayoutBuffers {
    data: Vec<u8>,
    masks: Vec<Vec<u8>>,
    shape: Vec<u64>,
    subset: ArraySubset,
}

impl LayoutBuffers {
    fn new(data_type: &DataType, (shape, subset): &(Vec<u64>, ArraySubset)) -> Self {
        let depth = usize::from(data_type.is_optional());
        let num_elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
        Self {
            data: vec![SENTINEL; num_elements],
            masks: vec![vec![SENTINEL; num_elements]; depth],
            shape: shape.clone(),
            subset: subset.clone(),
        }
    }

    fn target(&mut self) -> ArrayBytesDecodeIntoTarget<'_> {
        // SAFETY: each view is the only view of its buffer.
        let data_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut self.data),
                1,
                &self.shape,
                self.subset.clone(),
            )
        }
        .unwrap();
        let mask_views: Vec<_> = self
            .masks
            .iter_mut()
            .map(|mask| {
                // SAFETY: each view is the only view of its buffer.
                unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(mask),
                        1,
                        &self.shape,
                        self.subset.clone(),
                    )
                }
                .unwrap()
            })
            .collect();
        ArrayBytesDecodeIntoTarget::from(data_view)
            .with_optional_masks(mask_views)
            .unwrap()
    }

    /// Return the elements of the target in C order, after checking that elements outside of the target are untouched.
    fn into_array_bytes(self) -> ArrayBytes<'static> {
        let linearised: Vec<usize> = self
            .subset
            .linearised_indices(&self.shape)
            .unwrap()
            .into_iter()
            .map(|index| usize::try_from(index).unwrap())
            .collect();
        let extract = |buffer: &[u8]| -> Vec<u8> {
            for (index, &byte) in buffer.iter().enumerate() {
                if !linearised.contains(&index) {
                    assert_eq!(
                        byte, SENTINEL,
                        "element {index} outside the target was written"
                    );
                }
            }
            linearised.iter().map(|&index| buffer[index]).collect()
        };
        let masks: Vec<_> = self.masks.iter().map(|mask| extract(mask)).collect();
        ArrayBytes::new_flen(extract(&self.data)).with_optional_masks(masks)
    }
}

fn data_types() -> [DataType; 2] {
    [data_type::uint8(), data_type::uint8().to_optional()]
}

fn fill_value(data_type: &DataType) -> FillValue {
    if data_type.is_optional() {
        FillValue::new_optional_null()
    } else {
        FillValue::from(0u8)
    }
}

/// Array bytes of `num_elements` elements, with every third element missing for optional data.
fn elements(data_type: &DataType, num_elements: u64) -> ArrayBytes<'static> {
    let data: Vec<u8> = (0..num_elements)
        .map(|i| u8::try_from(i).unwrap())
        .collect();
    if data_type.is_optional() {
        let mask = (0..num_elements)
            .map(|i| u8::from(i % 3 != 0))
            .collect::<Vec<_>>();
        ArrayBytes::new_flen(data).with_optional_mask(mask)
    } else {
        ArrayBytes::new_flen(data)
    }
}

#[tokio::test]
async fn sharding_decode_into_target_shape() -> Result<(), Box<dyn std::error::Error>> {
    let shard_shape = SHARD_SHAPE.map(|size| NonZeroU64::new(size).unwrap());
    let options = CodecOptions::default();
    let subset = ArraySubset::new_with_ranges(&[1..3, 0..8]);
    for data_type in data_types() {
        let codec = ShardingCodecBuilder::new(
            SUBCHUNK_SHAPE
                .map(|size| NonZeroU64::new(size).unwrap())
                .to_vec(),
            &data_type,
        )
        .build_arc()
        .with_context(
            data_type.clone(),
            Some(fill_value(&data_type)),
            &CodecSpecificOptions::default(),
        )?;
        let encoded = codec
            .encode(
                elements(&data_type, 32),
                &shard_shape,
                &options,
                &ComputeResources::default(),
            )?
            .into_vec();
        let decoded = codec.decode(
            encoded.clone().into(),
            &shard_shape,
            &options,
            &ComputeResources::default(),
        )?;
        let partial_decoder = codec.clone().partial_decoder(
            Arc::new(encoded.clone()),
            &shard_shape,
            &options,
            &Resources::default(),
        )?;
        let decoded_subset =
            partial_decoder.partial_decode(&subset, &options, &Resources::default())?;
        #[cfg(feature = "async")]
        let async_partial_decoder = codec
            .clone()
            .async_partial_decoder(
                Arc::new(encoded.clone()),
                &shard_shape,
                &options,
                &Resources::default(),
            )
            .await?;

        for layout in layouts(32) {
            let mut buffers = LayoutBuffers::new(&data_type, &layout);
            codec.decode_into(
                encoded.clone().into(),
                &shard_shape,
                &mut buffers.target(),
                &options,
                &ComputeResources::default(),
            )?;
            assert_eq!(
                buffers.into_array_bytes(),
                decoded,
                "decode_into {}",
                layout.1
            );
        }
        for layout in layouts(16) {
            let mut buffers = LayoutBuffers::new(&data_type, &layout);
            partial_decoder.partial_decode_into(
                &subset,
                &mut buffers.target(),
                &options,
                &Resources::default(),
            )?;
            assert_eq!(
                buffers.into_array_bytes(),
                decoded_subset,
                "partial_decode_into {}",
                layout.1
            );

            #[cfg(feature = "async")]
            {
                let mut buffers = LayoutBuffers::new(&data_type, &layout);
                async_partial_decoder
                    .partial_decode_into(
                        &subset,
                        &mut buffers.target(),
                        &options,
                        &Resources::default(),
                    )
                    .await?;
                assert_eq!(
                    buffers.into_array_bytes(),
                    decoded_subset,
                    "async partial_decode_into {}",
                    layout.1
                );
            }
        }
    }
    Ok(())
}

/// Check the `_into` retrieval methods of `$array` (named by `$methods`, followed by `.await` if async) against retrieval into new array bytes.
macro_rules! check_retrieve_into {
    ($data_type:expr, $array:expr, [$chunk:ident, $chunk_into:ident, $partial:ident, $partial_into:ident, $subset:ident, $subset_into:ident] $($await:tt)*) => {{
        let data_type: &DataType = $data_type;
        let resources = Resources::default();

        // A chunk
        let expected = $array.$chunk::<ArrayBytes>(&[1, 0], &resources)$($await)*?;
        for layout in layouts(32) {
            let mut buffers = LayoutBuffers::new(data_type, &layout);
            $array.$chunk_into(&[1, 0], &mut buffers.target(), &resources)$($await)*?;
            assert_eq!(buffers.into_array_bytes(), expected, "{} {}", stringify!($chunk_into), layout.1);
        }

        // A partial chunk spanning partial subchunks
        let chunk_subset = ArraySubset::new_with_ranges(&[1..3, 0..8]);
        let expected = $array.$partial::<ArrayBytes>(&[1, 0], &chunk_subset, &resources)$($await)*?;
        for layout in layouts(16) {
            let mut buffers = LayoutBuffers::new(data_type, &layout);
            $array.$partial_into(&[1, 0], &chunk_subset, &mut buffers.target(), &resources)$($await)*?;
            assert_eq!(buffers.into_array_bytes(), expected, "{} {}", stringify!($partial_into), layout.1);
        }

        // An array subset spanning multiple chunks
        let array_subset = ArraySubset::new_with_ranges(&[3..5, 0..8]);
        let expected = $array.$subset::<ArrayBytes>(&array_subset, &resources)$($await)*?;
        for layout in layouts(16) {
            let mut buffers = LayoutBuffers::new(data_type, &layout);
            $array.$subset_into(&array_subset, &mut buffers.target(), &resources)$($await)*?;
            assert_eq!(buffers.into_array_bytes(), expected, "{} {}", stringify!($subset_into), layout.1);
        }
    }};
}

fn array_builder(data_type: &DataType) -> ArrayBuilder {
    let mut builder = ArrayBuilder::new(
        ARRAY_SHAPE.to_vec(),
        SHARD_SHAPE.to_vec(),
        data_type.clone(),
        fill_value(data_type),
    );
    builder.subchunk_shape(SUBCHUNK_SHAPE.to_vec());
    builder
}

#[test]
fn array_retrieve_into_target_shape() -> Result<(), Box<dyn std::error::Error>> {
    for data_type in data_types() {
        let array =
            Arc::new(array_builder(&data_type).build(Arc::new(MemoryStore::default()), "/")?);
        array.store_array_subset(
            &array.subset_all(),
            elements(&data_type, 64),
            &Resources::default(),
        )?;
        check_retrieve_into!(
            &data_type,
            array,
            [
                retrieve_chunk,
                retrieve_chunk_into,
                retrieve_partial_chunk,
                retrieve_partial_chunk_into,
                retrieve_array_subset,
                retrieve_array_subset_into
            ]
        );
        let cached = ArrayCached::new(array.clone(), ChunkCacheEncodedLruChunkLimit::new(4));
        check_retrieve_into!(
            &data_type,
            cached,
            [
                retrieve_chunk,
                retrieve_chunk_into,
                retrieve_partial_chunk,
                retrieve_partial_chunk_into,
                retrieve_array_subset,
                retrieve_array_subset_into
            ]
        );
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_array_retrieve_into_target_shape() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::array::chunk_cache::AsyncChunkCacheEncodedLruChunkLimit;
    use zarrs_storage::store::AsyncMemoryStore;

    for data_type in data_types() {
        let array =
            Arc::new(array_builder(&data_type).build(Arc::new(AsyncMemoryStore::new()), "/")?);
        array
            .async_store_array_subset(
                &array.subset_all(),
                elements(&data_type, 64),
                &Resources::default(),
            )
            .await?;
        check_retrieve_into!(
            &data_type,
            array,
            [
                async_retrieve_chunk,
                async_retrieve_chunk_into,
                async_retrieve_partial_chunk,
                async_retrieve_partial_chunk_into,
                async_retrieve_array_subset,
                async_retrieve_array_subset_into
            ]
            .await
        );
        let cached = ArrayCached::new(array.clone(), AsyncChunkCacheEncodedLruChunkLimit::new(4));
        check_retrieve_into!(
            &data_type,
            cached,
            [
                async_retrieve_chunk,
                async_retrieve_chunk_into,
                async_retrieve_partial_chunk,
                async_retrieve_partial_chunk_into,
                async_retrieve_array_subset,
                async_retrieve_array_subset_into
            ]
            .await
        );
    }
    Ok(())
}

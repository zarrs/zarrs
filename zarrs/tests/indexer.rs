//! Tests for generic indexers.
#![cfg(feature = "async")]

use std::{num::NonZeroU64, sync::Arc};

use itertools::Itertools;
use zarrs::{
    array::{
        codec::{
            ArrayToBytesCodecTraits, BytesCodec, CodecOptions, ShardingCodecBuilder, SqueezeCodec,
            TransposeCodec, TransposeOrder, VlenCodec,
        },
        ChunkRepresentation, CodecChain, DataType, ElementOwned,
    },
    array_subset::ArraySubset,
    indexer::{IncompatibleIndexerError, Indexer},
};
use zarrs_metadata::ChunkShape;

fn indexer_basic<T: Indexer>(indexer: T, indices: Vec<u64>, contiguous_indices: Vec<(u64, u64)>) {
    assert_eq!(
        indexer
            .iter_contiguous_linearised_indices(&[4, 4])
            .unwrap()
            .collect_vec(),
        contiguous_indices
    );
    assert_eq!(
        indexer
            .iter_linearised_indices(&[4, 4])
            .unwrap()
            .collect_vec(),
        indices
    );
    assert!(matches!(
        indexer.iter_linearised_indices(&[4, 4, 4]),
        Err(IncompatibleIndexerError::IncompatibleDimensionality(_))
    )); // incompatible dimensionality
    assert!(matches!(
        indexer.iter_contiguous_linearised_indices(&[4, 4, 4]),
        Err(IncompatibleIndexerError::IncompatibleDimensionality(_))
    )); // incompatible dimensionality
    assert!(matches!(
        indexer.iter_linearised_indices(&[3, 3]),
        Err(IncompatibleIndexerError::OutOfBounds(_, _))
    )); // OOB
    assert!(matches!(
        indexer.iter_contiguous_linearised_indices(&[3, 3]),
        Err(IncompatibleIndexerError::OutOfBounds(_, _))
    )); // OOB
    let subset = ArraySubset::new_with_shape(vec![4, 4]);
    assert!(matches!(
        subset.extract_elements(&vec![0u8, 4 * 4], &[5, 5]),
        Err(IncompatibleIndexerError::IncompatibleLength(_, _))
    ));
}

#[test]
fn indexer_array_subset1() {
    let indexer = ArraySubset::new_with_ranges(&[1..4, 2..4]);
    indexer_basic(
        indexer,
        vec![6, 7, 10, 11, 14, 15],
        vec![(6, 2), (10, 2), (14, 2)],
    );
}

#[test]
fn indexer_array_subset1_ref() {
    let indexer = ArraySubset::new_with_ranges(&[1..4, 2..4]);
    indexer_basic(
        &indexer,
        vec![6, 7, 10, 11, 14, 15],
        vec![(6, 2), (10, 2), (14, 2)],
    );
}

#[test]
fn indexer_array_subset2() {
    let indexer = ArraySubset::new_with_ranges(&[0..1, 0..4]);
    indexer_basic(&indexer, vec![0, 1, 2, 3], vec![(0, 4)]);
}

#[test]
fn indexer_array_subsets_list() {
    let indexer = [
        ArraySubset::new_with_ranges(&[1..4, 2..4]),
        ArraySubset::new_with_ranges(&[0..1, 0..4]),
    ];
    indexer_basic(
        indexer,
        vec![6, 7, 10, 11, 14, 15, 0, 1, 2, 3],
        vec![(6, 2), (10, 2), (14, 2), (0, 4)],
    );
}

#[test]
fn indexer_array_subsets_slice() {
    let indexer = [
        ArraySubset::new_with_ranges(&[1..4, 2..4]),
        ArraySubset::new_with_ranges(&[0..1, 0..4]),
    ];
    indexer_basic(
        &indexer,
        vec![6, 7, 10, 11, 14, 15, 0, 1, 2, 3],
        vec![(6, 2), (10, 2), (14, 2), (0, 4)],
    );
}

#[test]
fn indexer_array_subsets_vec() {
    let indexer = vec![
        ArraySubset::new_with_ranges(&[1..4, 2..4]),
        ArraySubset::new_with_ranges(&[0..1, 0..4]),
    ];
    indexer_basic(
        indexer,
        vec![6, 7, 10, 11, 14, 15, 0, 1, 2, 3],
        vec![(6, 2), (10, 2), (14, 2), (0, 4)],
    );
}

#[async_generic::async_generic]
fn indexer_array_subsets_impl<T: ElementOwned>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    shape: &ChunkShape,
    indexer: &dyn Indexer,
    data_type: DataType,
    bytes: &[T],
) -> Vec<T> {
    let decoded_representation =
        ChunkRepresentation::new(shape.to_vec(), data_type.clone(), 0u32).unwrap();
    let encoded_chunk = Arc::new(
        codec
            .encode(
                T::into_array_bytes(&data_type, bytes).unwrap(),
                &decoded_representation,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_owned(),
    );

    let partial_decoder = if _async {
        codec
            .async_partial_decoder(
                encoded_chunk,
                &decoded_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap()
    } else {
        codec
            .partial_decoder(
                encoded_chunk,
                &decoded_representation,
                &CodecOptions::default(),
            )
            .unwrap()
    };

    T::from_array_bytes(
        &data_type,
        if _async {
            partial_decoder
                .partial_decode(indexer, &CodecOptions::default())
                .await
        } else {
            partial_decoder.partial_decode(indexer, &CodecOptions::default())
        }
        .unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn async_indexer_array_subsets_fixed() {
    let shape: ChunkShape = vec![
        NonZeroU64::new(1).unwrap(),
        NonZeroU64::new(4).unwrap(),
        NonZeroU64::new(4).unwrap(),
    ]
    .into();
    let indexer = [
        ArraySubset::new_with_ranges(&[0..1, 1..4, 2..4]),
        ArraySubset::new_with_ranges(&[0..1, 0..1, 0..4]),
    ];
    let elements: Vec<f32> = (0..shape.num_elements_usize())
        .map(|i| i as f32)
        .collect_vec();
    let expected = vec![6.0, 7.0, 10.0, 11.0, 14.0, 15.0, 0.0, 1.0, 2.0, 3.0];

    let codecs: Vec<Arc<dyn ArrayToBytesCodecTraits>> = vec![
        Arc::new(BytesCodec::little()),
        Arc::new(CodecChain::new(
            vec![
                Arc::new(SqueezeCodec::new()),
                #[cfg(feature = "transpose")]
                Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
            ],
            Arc::new(BytesCodec::little()),
            vec![],
        )),
        Arc::new(CodecChain::new(
            vec![
                Arc::new(SqueezeCodec::new()),
                #[cfg(feature = "transpose")]
                Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
            ],
            ShardingCodecBuilder::new(
                vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(2).unwrap()].into(),
            )
            .build_arc(),
            vec![],
        )),
    ];

    for codec in codecs {
        assert_eq!(
            indexer_array_subsets_impl(
                codec.clone(),
                &shape,
                &indexer,
                DataType::Float32,
                &elements
            ),
            expected
        );
        #[cfg(feature = "async")]
        assert_eq!(
            indexer_array_subsets_impl_async(
                codec.clone(),
                &shape,
                &indexer,
                DataType::Float32,
                &elements
            )
            .await,
            expected
        );
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_indexer_array_subsets_variable() {
    let shape: ChunkShape = vec![
        NonZeroU64::new(1).unwrap(),
        NonZeroU64::new(4).unwrap(),
        NonZeroU64::new(4).unwrap(),
    ]
    .into();
    let indexer = [
        ArraySubset::new_with_ranges(&[0..1, 1..4, 2..4]),
        ArraySubset::new_with_ranges(&[0..1, 0..1, 0..4]),
    ];
    let elements: Vec<String> = (0usize..shape.num_elements_usize())
        .map(|i| {
            std::iter::repeat_n(char::from_digit((i + 10) as u32, 26).unwrap(), i + 1)
                .collect::<String>()
        })
        .collect_vec();
    println!("{elements:#?}");
    // let expected = vec![6.0, 7.0, 10.0, 11.0, 14.0, 15.0, 0.0, 1.0, 2.0, 3.0];
    let expected = vec![
        "ggggggg",
        "hhhhhhhh",
        "kkkkkkkkkkk",
        "llllllllllll",
        "ooooooooooooooo",
        "pppppppppppppppp",
        "a",
        "bb",
        "ccc",
        "dddd",
    ];

    let codecs: Vec<Arc<dyn ArrayToBytesCodecTraits>> = vec![
        Arc::new(VlenCodec::default()),
        Arc::new(CodecChain::new(
            vec![
                Arc::new(SqueezeCodec::new()),
                #[cfg(feature = "transpose")]
                Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
            ],
            Arc::new(VlenCodec::default()),
            vec![],
        )),
        Arc::new(CodecChain::new(
            vec![
                Arc::new(SqueezeCodec::new()),
                #[cfg(feature = "transpose")]
                Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
            ],
            ShardingCodecBuilder::new(
                vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(2).unwrap()].into(),
            )
            .array_to_bytes_codec(Arc::new(VlenCodec::default()))
            .build_arc(),
            vec![],
        )),
    ];

    for codec in codecs {
        assert_eq!(
            indexer_array_subsets_impl(
                codec.clone(),
                &shape,
                &indexer,
                DataType::String,
                &elements
            ),
            expected
        );
        assert_eq!(
            indexer_array_subsets_impl_async(codec, &shape, &indexer, DataType::String, &elements)
                .await,
            expected,
        );
    }
}

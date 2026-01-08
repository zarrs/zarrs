//! Tests for generic indexers.
#![cfg(feature = "async")]

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use itertools::Itertools;
use zarrs::array::codec::{
    ArrayToBytesCodecTraits, BytesCodec, BytesPartialDecoderTraits, BytesPartialEncoderTraits,
    CodecOptions, ShardingCodecBuilder, SqueezeCodec, VlenCodec,
};
#[cfg(feature = "transpose")]
use zarrs::array::codec::{TransposeCodec, TransposeOrder};
use zarrs::array::{
    ArrayIndices, ArrayIndicesTinyVec, ChunkShape, ChunkShapeTraits, CodecChain, DataType,
    ElementOwned, data_type,
};
use zarrs::array_subset::ArraySubset;
use zarrs::indexer::{IncompatibleIndexerError, Indexer};
use zarrs_data_type::FillValue;

fn indexer_basic<T: Indexer>(
    indexer: T,
    dimensionality: usize,
    output_shape: Vec<u64>,
    indices: Vec<ArrayIndices>,
    linearised_indices: Vec<u64>,
    contiguous_indices: Vec<(u64, u64)>,
) {
    assert_eq!(indexer.dimensionality(), dimensionality);
    assert_eq!(indexer.len(), indices.len() as u64);
    assert_eq!(indexer.is_empty(), indices.is_empty());
    assert_eq!(indexer.output_shape(), output_shape);
    assert_eq!(
        indexer
            .iter_contiguous_linearised_indices(&[4, 4])
            .unwrap()
            .collect_vec(),
        contiguous_indices
    );
    assert_eq!(
        indexer.iter_indices().collect_vec(),
        indices
            .into_iter()
            .map(ArrayIndicesTinyVec::Heap)
            .collect_vec()
    );
    assert_eq!(
        indexer
            .iter_linearised_indices(&[4, 4])
            .unwrap()
            .collect_vec(),
        linearised_indices
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
        subset.extract_elements(&[0u8, 4 * 4], &[5, 5]),
        Err(IncompatibleIndexerError::IncompatibleLength(_, _))
    ));
}

#[test]
fn indexer_indices_list() {
    let indexer = [vec![0, 0], vec![0, 1], vec![0, 3], vec![1, 1]];
    indexer_basic(
        indexer,
        2,
        vec![4],
        vec![vec![0, 0], vec![0, 1], vec![0, 3], vec![1, 1]],
        vec![0, 1, 3, 5],
        vec![(0, 1), (1, 1), (3, 1), (5, 1)], // TODO: Fusion of contiguous indices
    );
}

#[test]
fn indexer_indices_vec() {
    let indexer = vec![vec![0, 0], vec![0, 1], vec![0, 3], vec![1, 1]];
    indexer_basic(
        indexer,
        2,
        vec![4],
        vec![vec![0, 0], vec![0, 1], vec![0, 3], vec![1, 1]],
        vec![0, 1, 3, 5],
        vec![(0, 1), (1, 1), (3, 1), (5, 1)], // TODO: Fusion of contiguous indices
    );
}

#[test]
fn indexer_indices_slice() {
    let indexer = vec![vec![0, 0], vec![0, 1], vec![0, 3], vec![1, 1]];
    indexer_basic(
        &indexer,
        2,
        vec![4],
        vec![vec![0, 0], vec![0, 1], vec![0, 3], vec![1, 1]],
        vec![0, 1, 3, 5],
        vec![(0, 1), (1, 1), (3, 1), (5, 1)], // TODO: Fusion of contiguous indices
    );
}

#[test]
fn indexer_array_subset1() {
    let indexer = ArraySubset::new_with_ranges(&[1..4, 2..4]);
    indexer_basic(
        indexer,
        2,
        vec![3, 2],
        vec![
            vec![1, 2],
            vec![1, 3],
            vec![2, 2],
            vec![2, 3],
            vec![3, 2],
            vec![3, 3],
        ],
        vec![6, 7, 10, 11, 14, 15],
        vec![(6, 2), (10, 2), (14, 2)],
    );
}

#[test]
fn indexer_array_subset1_ref() {
    let indexer = ArraySubset::new_with_ranges(&[1..4, 2..4]);
    indexer_basic(
        &indexer,
        2,
        vec![3, 2],
        vec![
            vec![1, 2],
            vec![1, 3],
            vec![2, 2],
            vec![2, 3],
            vec![3, 2],
            vec![3, 3],
        ],
        vec![6, 7, 10, 11, 14, 15],
        vec![(6, 2), (10, 2), (14, 2)],
    );
}

#[test]
fn indexer_array_subset2() {
    let indexer = ArraySubset::new_with_ranges(&[0..1, 0..4]);
    indexer_basic(
        &indexer,
        2,
        vec![1, 4],
        vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![0, 3]],
        vec![0, 1, 2, 3],
        vec![(0, 4)],
    );
}

#[test]
fn indexer_array_subsets_list() {
    let indexer = [
        ArraySubset::new_with_ranges(&[1..4, 2..4]),
        ArraySubset::new_with_ranges(&[0..1, 0..4]),
    ];
    indexer_basic(
        indexer,
        2,
        vec![10],
        vec![
            vec![1, 2],
            vec![1, 3],
            vec![2, 2],
            vec![2, 3],
            vec![3, 2],
            vec![3, 3],
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
        ],
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
        2,
        vec![10],
        vec![
            vec![1, 2],
            vec![1, 3],
            vec![2, 2],
            vec![2, 3],
            vec![3, 2],
            vec![3, 3],
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
        ],
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
        2,
        vec![10],
        vec![
            vec![1, 2],
            vec![1, 3],
            vec![2, 2],
            vec![2, 3],
            vec![3, 2],
            vec![3, 3],
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
        ],
        vec![6, 7, 10, 11, 14, 15, 0, 1, 2, 3],
        vec![(6, 2), (10, 2), (14, 2), (0, 4)],
    );
}

#[async_generic::async_generic]
fn indexer_partial_decode_impl<T: ElementOwned>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    data_type: DataType,
    bytes: &[T],
) -> Vec<T> {
    let fill_value = FillValue::from(0u32);
    let encoded_chunk = Arc::new(
        codec
            .encode(
                T::to_array_bytes(&data_type, bytes).unwrap(),
                shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_owned(),
    );

    let partial_decoder = if _async {
        codec
            .clone()
            .async_partial_decoder(
                encoded_chunk.clone(),
                shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .await
            .unwrap()
    } else {
        codec
            .clone()
            .partial_decoder(
                encoded_chunk.clone(),
                shape,
                &data_type,
                &fill_value,
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

// #[async_generic::async_generic]
fn indexer_partial_encode_impl<T: ElementOwned>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    elements_partial_encode: &[T],
    data_type: DataType,
    bytes: &[T],
) -> Vec<T> {
    let fill_value = FillValue::from(0u32);
    let encoded_chunk = Arc::new(
        codec
            .encode(
                T::to_array_bytes(&data_type, bytes).unwrap(),
                shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_owned(),
    );

    // TODO: Async partial encoder
    let output = Arc::new(Mutex::new(Some(encoded_chunk.to_vec())));
    let partial_encoder = codec
        .clone()
        .partial_encoder(
            output.clone(),
            shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )
        .unwrap();
    assert_eq!(
        partial_encoder.supports_partial_encode(),
        codec.partial_encoder_capability().partial_encode && output.supports_partial_encode()
    );
    assert_eq!(
        partial_encoder.supports_partial_decode(),
        codec.partial_decoder_capability().partial_decode && output.supports_partial_decode()
    );

    partial_encoder
        .partial_encode(
            indexer,
            &T::to_array_bytes(&data_type, elements_partial_encode).unwrap(),
            &CodecOptions::default(),
        )
        .unwrap();

    let output = output.lock().unwrap().clone().unwrap();
    T::from_array_bytes(
        &data_type,
        codec
            .decode(
                output.into(),
                shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
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
    ];
    let indexer = [
        ArraySubset::new_with_ranges(&[0..1, 1..4, 2..4]),
        ArraySubset::new_with_ranges(&[0..1, 0..1, 0..4]),
    ];
    let elements: Vec<f32> = (0..shape.num_elements_usize())
        .map(|i| i as f32)
        .collect_vec();
    let expected = vec![6.0, 7.0, 10.0, 11.0, 14.0, 15.0, 0.0, 1.0, 2.0, 3.0];

    let elements_partial_encode = vec![
        60.0, 70.0, 100.0, 110.0, 140.0, 150.0, 0.0, 10.0, 20.0, 30.0,
    ];
    let expected_partial_encode = vec![
        0.0, 10.0, 20.0, 30.0, //
        4.0, 5.0, 60.0, 70.0, //
        8.0, 9.0, 100.0, 110.0, //
        12.0, 13.0, 140.0, 150.0, //
    ];

    let codecs: Vec<(Arc<dyn ArrayToBytesCodecTraits>, bool)> = vec![
        (Arc::new(BytesCodec::little()), true),
        (
            Arc::new(CodecChain::new(
                vec![
                    Arc::new(SqueezeCodec::new()),
                    #[cfg(feature = "transpose")]
                    Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
                ],
                Arc::new(BytesCodec::little()),
                vec![],
            )),
            true,
        ),
        (
            Arc::new(CodecChain::new(
                vec![
                    Arc::new(SqueezeCodec::new()),
                    #[cfg(feature = "transpose")]
                    Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
                ],
                ShardingCodecBuilder::new(
                    vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(2).unwrap()],
                    &data_type::float32(),
                )
                .build_arc(),
                vec![],
            )),
            false, // FIXME: Add generic indexing support to sharding indexed partial encoder
        ),
    ];

    for (codec, test_partial_encoding) in codecs {
        assert_eq!(
            indexer_partial_decode_impl(
                codec.clone(),
                &shape,
                &indexer,
                data_type::float32(),
                &elements
            ),
            expected
        );
        #[cfg(feature = "async")]
        assert_eq!(
            indexer_partial_decode_impl_async(
                codec.clone(),
                &shape,
                &indexer,
                data_type::float32(),
                &elements
            )
            .await,
            expected
        );
        if test_partial_encoding {
            assert_eq!(
                indexer_partial_encode_impl(
                    codec.clone(),
                    &shape,
                    &indexer,
                    &elements_partial_encode,
                    data_type::float32(),
                    &elements,
                ),
                expected_partial_encode
            );
        }
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_indexer_array_subsets_variable() {
    let shape: ChunkShape = vec![
        NonZeroU64::new(1).unwrap(),
        NonZeroU64::new(4).unwrap(),
        NonZeroU64::new(4).unwrap(),
    ];
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

    let elements_partial_encode = [
        "60.0", "70.0", "100.0", "110.0", "140.0", "150.0", "0.0", "10.0", "20.0", "30.0",
    ]
    .into_iter()
    .map(std::string::ToString::to_string)
    .collect::<Vec<_>>();
    let expected_partial_encode = vec![
        "0.0",
        "10.0",
        "20.0",
        "30.0", //
        "eeeee",
        "ffffff",
        "60.0",
        "70.0", //
        "iiiiiiiii",
        "jjjjjjjjjj",
        "100.0",
        "110.0", //
        "mmmmmmmmmmmmm",
        "nnnnnnnnnnnnnn",
        "140.0",
        "150.0", //
    ];
    let codecs: Vec<(Arc<dyn ArrayToBytesCodecTraits>, bool)> = vec![
        (Arc::new(VlenCodec::default()), true),
        (
            Arc::new(CodecChain::new(
                vec![
                    Arc::new(SqueezeCodec::new()),
                    #[cfg(feature = "transpose")]
                    Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
                ],
                Arc::new(VlenCodec::default()),
                vec![],
            )),
            true,
        ),
        (
            Arc::new(CodecChain::new(
                vec![
                    Arc::new(SqueezeCodec::new()),
                    #[cfg(feature = "transpose")]
                    Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
                ],
                ShardingCodecBuilder::new(
                    vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(2).unwrap()],
                    &data_type::string(),
                )
                .array_to_bytes_codec(Arc::new(VlenCodec::default()))
                .build_arc(),
                vec![],
            )),
            false, // FIXME: Add generic indexing support to sharding indexed partial encoder
        ),
    ];

    for (codec, test_partial_encoding) in codecs {
        assert_eq!(
            indexer_partial_decode_impl(
                codec.clone(),
                &shape,
                &indexer,
                data_type::string(),
                &elements
            ),
            expected
        );
        assert_eq!(
            indexer_partial_decode_impl_async(
                codec.clone(),
                &shape,
                &indexer,
                data_type::string(),
                &elements
            )
            .await,
            expected,
        );
        if test_partial_encoding {
            assert_eq!(
                indexer_partial_encode_impl(
                    codec.clone(),
                    &shape,
                    &indexer,
                    &elements_partial_encode,
                    data_type::string(),
                    &elements
                ),
                expected_partial_encode
            );
        }
    }
}

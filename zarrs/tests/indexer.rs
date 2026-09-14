//! Tests for generic indexers.
#![cfg(feature = "async")]

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use itertools::Itertools;
use zarrs::array::codec::{BytesCodec, ShardingCodecBuilder, SqueezeCodec, VlenCodec};
#[cfg(feature = "transpose")]
use zarrs::array::codec::{TransposeCodec, TransposeOrder};
use zarrs::array::{
    ArrayIndices, ArrayIndicesTinyVec, ArraySubset, ChunkShape, ChunkShapeTraits, CodecChain,
    DataType, ElementOwned, Indexer, IndexerError, data_type,
};
use zarrs_codec::{
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecOptions,
    UnboundArrayToBytesCodecTraits,
};
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
        Err(IndexerError::IncompatibleDimensionality(_))
    )); // incompatible dimensionality
    assert!(matches!(
        indexer.iter_contiguous_linearised_indices(&[4, 4, 4]),
        Err(IndexerError::IncompatibleDimensionality(_))
    )); // incompatible dimensionality
    assert!(matches!(
        indexer.iter_linearised_indices(&[3, 3]),
        Err(IndexerError::OutOfBounds(_, _))
    )); // OOB
    assert!(matches!(
        indexer.iter_contiguous_linearised_indices(&[3, 3]),
        Err(IndexerError::OutOfBounds(_, _))
    )); // OOB
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
    codec: Arc<dyn UnboundArrayToBytesCodecTraits>,
    shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    data_type: DataType,
    bytes: &[T],
) -> Vec<T> {
    let fill_value = FillValue::from(0u32);
    let bound_codec = codec.with_context(data_type.clone(), fill_value).unwrap();
    let encoded_chunk = Arc::new(
        bound_codec
            .encode(
                T::to_array_bytes(&data_type, bytes).unwrap(),
                shape,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_vec(),
    );

    let partial_decoder = if _async {
        bound_codec
            .clone()
            .async_partial_decoder(encoded_chunk.clone(), shape, &CodecOptions::default())
            .await
            .unwrap()
    } else {
        bound_codec
            .clone()
            .partial_decoder(encoded_chunk, shape, &CodecOptions::default())
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
    codec: Arc<dyn UnboundArrayToBytesCodecTraits>,
    shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    elements_partial_encode: &[T],
    data_type: DataType,
    bytes: &[T],
) -> Vec<T> {
    let fill_value = FillValue::from(0u32);
    let bound_codec = codec.with_context(data_type.clone(), fill_value).unwrap();
    let encoded_chunk = Arc::new(
        bound_codec
            .encode(
                T::to_array_bytes(&data_type, bytes).unwrap(),
                shape,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_vec(),
    );

    // TODO: Async partial encoder
    let output = Arc::new(Mutex::new(Some(encoded_chunk.to_vec())));
    let partial_encoder = bound_codec
        .clone()
        .partial_encoder(output.clone(), shape, &CodecOptions::default())
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
        bound_codec
            .decode(output.into(), shape, &CodecOptions::default())
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

    let codecs: Vec<(Arc<dyn UnboundArrayToBytesCodecTraits>, bool)> = vec![
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
    let codecs: Vec<(Arc<dyn UnboundArrayToBytesCodecTraits>, bool)> = vec![
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

/// `Array` read/write of a chunk with a generic (non-subset) indexer.
#[test]
fn array_chunk_subset_generic_indexer() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::array::ArrayBuilder;
    use zarrs::storage::store::MemoryStore;

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    array.store_chunk(&[0, 0], &(0u16..16).collect::<Vec<_>>())?;

    // Chunk-relative list of indices, in a deliberately non-monotonic order.
    let indexer: Vec<ArrayIndices> = vec![vec![0, 1], vec![3, 3], vec![1, 0], vec![2, 2]];

    // Read
    let elements: Vec<u16> = array.retrieve_chunk_subset(&[0, 0], &indexer)?;
    assert_eq!(elements, vec![1, 15, 4, 10]);

    // ... and the output is flattened, matching `Indexer::output_shape`
    assert_eq!(indexer.output_shape(), vec![4]);

    // Write
    array.store_chunk_subset(&[0, 0], &indexer, &[100u16, 101, 102, 103])?;
    let chunk: Vec<u16> = array.retrieve_chunk(&[0, 0])?;
    assert_eq!(
        chunk,
        vec![0, 100, 2, 3, 102, 5, 6, 7, 8, 9, 103, 11, 12, 13, 14, 101]
    );

    // An out-of-bounds indexer is rejected by the codec layer
    let oob: Vec<ArrayIndices> = vec![vec![0, 4]];
    assert!(
        array
            .retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &oob)
            .is_err()
    );

    // An array subset still works and is not flattened
    let subset = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let elements: Vec<u16> = array.retrieve_chunk_subset(&[0, 0], &subset)?;
    assert_eq!(elements, vec![0, 100, 102, 5]);

    Ok(())
}

/// `erase_chunks` with a scattered list of chunk indices.
#[test]
fn array_erase_chunks_generic_indexer() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::array::ArrayBuilder;
    use zarrs::storage::store::MemoryStore;

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    for chunk in [[0u64, 0], [0, 1], [1, 0], [1, 1]] {
        array.store_chunk(&chunk, &[1u16, 1, 1, 1])?;
    }

    // Erase the two chunks on the anti-diagonal, which is not an array subset.
    let chunks: Vec<ArrayIndices> = vec![vec![0, 1], vec![1, 0]];
    array.erase_chunks(&chunks)?;

    assert!(array.retrieve_encoded_chunk(&[0, 0])?.is_some());
    assert!(array.retrieve_encoded_chunk(&[0, 1])?.is_none());
    assert!(array.retrieve_encoded_chunk(&[1, 0])?.is_none());
    assert!(array.retrieve_encoded_chunk(&[1, 1])?.is_some());

    // `retrieve_encoded_chunks` follows `iter_indices` order
    let encoded = array.retrieve_encoded_chunks(&chunks)?;
    assert_eq!(encoded.len(), 2);
    assert!(encoded.iter().all(Option::is_none));

    // Chunk indices out-of-bounds of the chunk grid or with an incompatible dimensionality are rejected
    for bad in [vec![vec![9, 9]], vec![vec![0]]] {
        assert!(array.erase_chunks(&bad).is_err());
        assert!(array.retrieve_encoded_chunks(&bad).is_err());
    }

    Ok(())
}

/// Partial encoding a sharded chunk with a generic indexer is not yet supported and must
/// report an error rather than silently writing the wrong bytes.
#[test]
fn array_store_chunk_subset_sharded_generic_indexer_unsupported()
-> Result<(), Box<dyn std::error::Error>> {
    use zarrs::array::ArrayBuilder;
    use zarrs::storage::store::MemoryStore;

    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16);
    builder.array_to_bytes_codec(
        ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint16())
            .build_arc(),
    );
    let array = builder
        .build(store, "/array")?
        .with_codec_options(CodecOptions::default().with_experimental_partial_encoding(true));
    array.store_chunk(&[0, 0], &(0u16..16).collect::<Vec<_>>())?;

    let indexer: Vec<ArrayIndices> = vec![vec![0, 1], vec![3, 3]];
    assert!(
        array
            .store_chunk_subset(&[0, 0], &indexer, &[100u16, 101])
            .is_err()
    );

    // The chunk is untouched
    let chunk: Vec<u16> = array.retrieve_chunk(&[0, 0])?;
    assert_eq!(chunk, (0u16..16).collect::<Vec<_>>());

    Ok(())
}

/// A generic indexer is validated against the chunk shape even if the cached chunk is absent.
#[test]
fn array_cached_chunk_subset_generic_indexer_absent_chunk() -> Result<(), Box<dyn std::error::Error>>
{
    use zarrs::array::chunk_cache::ChunkCacheDecodedLruChunkLimit;
    use zarrs::array::{ArrayBuilder, ArrayCached};
    use zarrs::storage::store::MemoryStore;

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    let cached = ArrayCached::new(array.into(), ChunkCacheDecodedLruChunkLimit::new(4));

    // Chunk [0, 0] is absent, so the fill value is returned
    let indexer: Vec<ArrayIndices> = vec![vec![0, 1], vec![1, 1]];
    let elements: Vec<u16> = cached.retrieve_chunk_subset(&[0, 0], &indexer)?;
    assert_eq!(elements, vec![0, 0]);

    // ... but an out-of-bounds or incompatible indexer is still rejected
    let oob: Vec<ArrayIndices> = vec![vec![0, 2]];
    assert!(
        cached
            .retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &oob)
            .is_err()
    );
    let bad_dimensionality: Vec<ArrayIndices> = vec![vec![0]];
    assert!(
        cached
            .retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &bad_dimensionality)
            .is_err()
    );
    assert!(
        cached
            .retrieve_chunk_subset::<Vec<u16>>(&[9, 9], &indexer)
            .is_err()
    );

    Ok(())
}

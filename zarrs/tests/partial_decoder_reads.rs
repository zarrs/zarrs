#![allow(missing_docs)]

//! The reads of array partial decoders, which read what every request needs once when they are created.

use std::sync::Arc;

use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArraySubset, Resources, data_type};
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::CodecOptions;

type MetricsStore = PerformanceMetricsStorageAdapter<MemoryStore>;

/// An array of two 64 x 64 `u16` chunks (the second missing) on a store that counts reads.
fn array(
    sharded: bool,
    data_type: zarrs::array::DataType,
) -> (Arc<MetricsStore>, Array<MetricsStore>) {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::new(),
    )));
    let mut builder = ArrayBuilder::new(vec![128, 64], vec![64, 64], data_type, 0u16);
    builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
    if sharded {
        builder.subchunk_shape(vec![16, 16]);
    }
    let array = builder.build(store.clone(), "/").unwrap();
    (store, array)
}

#[test]
fn sharded_partial_decoder_reads_index_once() {
    let (store, array) = array(true, data_type::uint16());
    let data: Vec<u16> = (0..64 * 64).map(|i| i as u16).collect();
    array
        .store_chunk(&[0, 0], data.as_slice(), &Resources::default())
        .unwrap();
    let options = CodecOptions::default();
    store.reset();
    let decoder = array
        .partial_decoder(&[0, 0], &Resources::default())
        .unwrap();
    assert_eq!(store.reads(), 1, "the index is read");
    assert!(decoder.exists().unwrap());

    // One subchunk, then a region of four subchunks
    for (ranges, reads) in [([20..30, 40..48], 1), ([8..24, 8..24], 4)] {
        store.reset();
        let subset = ArraySubset::new_with_ranges(&ranges);
        let bytes = decoder
            .partial_decode(&subset, &options, &Resources::default())
            .unwrap()
            .into_owned();
        assert_eq!(store.reads(), reads, "{ranges:?}");
        let expected: Vec<u16> = array
            .retrieve_array_subset(&subset, &Resources::default())
            .unwrap();
        assert_eq!(
            bytes.into_fixed().unwrap().as_ref(),
            bytemuck::cast_slice::<u16, u8>(&expected)
        );
    }

    // A missing shard is known to be missing once its index is read
    store.reset();
    let missing = array
        .partial_decoder(&[1, 0], &Resources::default())
        .unwrap();
    assert!(!missing.exists().unwrap());
    let bytes = missing
        .partial_decode(
            &ArraySubset::new_with_ranges(&[0..4, 0..4]),
            &options,
            &Resources::default(),
        )
        .unwrap()
        .into_owned();
    assert_eq!(bytes.into_fixed().unwrap().as_ref(), &[0u8; 32]);
    assert_eq!(store.reads(), 1, "missing shard");
}

#[test]
fn compressed_partial_decoder_reads_chunk_once() {
    let (store, array) = array(false, data_type::uint16());
    let data: Vec<u16> = (0..64 * 64).map(|i| i as u16).collect();
    array
        .store_chunk(&[0, 0], data.as_slice(), &Resources::default())
        .unwrap();
    store.reset();
    let decoder = array
        .partial_decoder(&[0, 0], &Resources::default())
        .unwrap();
    assert_eq!(store.reads(), 1, "the chunk is read");
    assert!(decoder.size_held() > 0);
    let subset = ArraySubset::new_with_ranges(&[1..3, 2..4]);
    let bytes = decoder
        .partial_decode(&subset, &CodecOptions::default(), &Resources::default())
        .unwrap()
        .into_owned();
    assert_eq!(store.reads(), 1, "decoded from memory");
    assert_eq!(
        bytes.into_fixed().unwrap().as_ref(),
        bytemuck::cast_slice::<u16, u8>(&[66, 67, 130, 131])
    );
}

#[test]
fn variable_length_partial_decoder_decodes_chunk_once() {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::new(),
    )));
    let array = ArrayBuilder::new(vec![4], vec![4], data_type::string(), "")
        .build(store.clone(), "/")
        .unwrap();
    array
        .store_chunk(&[0], vec!["a", "bb", "ccc", "dddd"], &Resources::default())
        .unwrap();
    store.reset();
    let decoder = array.partial_decoder(&[0], &Resources::default()).unwrap();
    assert_eq!(store.reads(), 1, "the chunk is read");
    let bytes = decoder
        .partial_decode(
            &ArraySubset::new_with_ranges(&[1..3]),
            &CodecOptions::default(),
            &Resources::default(),
        )
        .unwrap();
    assert_eq!(store.reads(), 1, "decoded from memory");
    let (bytes, offsets) = bytes.into_variable().unwrap().into_parts();
    assert_eq!(bytes.as_ref(), b"bbccc");
    assert_eq!(
        offsets,
        zarrs::array::ArrayBytesOffsets::try_from(&[0u32, 2, 5]).unwrap()
    );
}

#![allow(missing_docs)]
#![cfg(feature = "sharding")]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{BytesToBytesCodecTraits, CodecOptions};
use zarrs::array::{ArrayBuilder, ArraySubset, data_type};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::ReadableStorageTraits;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;

fn array_partial_encode_sharding(
    sharding_index_location: ShardingIndexLocation,
    inner_bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let opt = CodecOptions::default().with_experimental_partial_encoding(true);

    let store = std::sync::Arc::new(MemoryStore::default());
    // let log_writer = Arc::new(std::sync::Mutex::new(
    //     // std::io::BufWriter::new(
    //     std::io::stdout(),
    //     //    )
    // ));
    // let store = Arc::new(
    //     zarrs_storage::storage_adapter::usage_log::UsageLogStorageAdapter::new(
    //         store.clone(),
    //         log_writer.clone(),
    //         || chrono::Utc::now().format("[%T%.3f] ").to_string(),
    //     ),
    // );
    let store_perf = Arc::new(PerformanceMetricsStorageAdapter::new(store.clone()));

    let array_path = "/";
    let mut builder = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        data_type::uint16(),
        0u16,
    );
    builder
        .array_to_bytes_codec(Arc::new(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(1).unwrap(); 2], &data_type::uint16())
                .index_bytes_to_bytes_codecs(vec![])
                .index_location(sharding_index_location)
                .bytes_to_bytes_codecs(inner_bytes_to_bytes_codecs.clone())
                .build(),
        ))
        .bytes_to_bytes_codecs(vec![]);
    // .storage_transformers(vec![].into())

    let array = builder.build(store_perf.clone(), array_path).unwrap();

    let get_bytes_0_0 = || {
        let key = array.chunk_key_encoding().encode(&[0, 0]);
        store.get(&key)
    };

    let expected_writes_per_shard = match sharding_index_location {
        ShardingIndexLocation::Start => 2, // Separate write for inner chunks and index
        ShardingIndexLocation::End => 1,   // Combined write for inner chunks and index
    };

    let chunks_per_shard = 2 * 2;
    let shard_index_size = size_of::<u64>() * 2 * chunks_per_shard;
    assert!(get_bytes_0_0()?.is_none());
    assert_eq!(store_perf.reads(), 0);
    assert_eq!(store_perf.bytes_read(), 0);

    // [1, 0]
    // [0, 0]
    array.store_array_subset_opt(&ArraySubset::new_with_ranges(&[0..1, 0..1]), &[1u16], &opt)?;
    assert_eq!(store_perf.reads(), 1); // index
    assert_eq!(store_perf.writes(), expected_writes_per_shard);
    assert_eq!(store_perf.bytes_read(), 0);
    if inner_bytes_to_bytes_codecs.is_empty() {
        assert_eq!(
            get_bytes_0_0()?.unwrap().len(),
            shard_index_size + size_of::<u16>()
        );
    }
    store_perf.reset();

    // [0, 0]
    // [0, 0]
    array.store_array_subset_opt(&ArraySubset::new_with_ranges(&[0..1, 0..1]), &[0u16], &opt)?;
    assert_eq!(store_perf.reads(), 1); // index
    assert_eq!(store_perf.writes(), 0);
    if inner_bytes_to_bytes_codecs.is_empty() {
        assert_eq!(store_perf.bytes_read(), shard_index_size);
    }
    assert!(get_bytes_0_0()?.is_none());
    store_perf.reset();

    // [1, 2]
    // [0, 0]
    array.store_array_subset_opt(
        &ArraySubset::new_with_ranges(&[0..1, 0..2]),
        &[1u16, 2],
        &opt,
    )?;
    assert_eq!(store_perf.reads(), 1); // index
    assert_eq!(store_perf.writes(), expected_writes_per_shard);
    if inner_bytes_to_bytes_codecs.is_empty() {
        assert_eq!(
            get_bytes_0_0()?.unwrap().len(),
            shard_index_size + size_of::<u16>() * 2
        );
    }
    assert_eq!(array.retrieve_chunk::<Vec<u16>>(&[0, 0])?, vec![1, 2, 0, 0]);
    store_perf.reset();

    // Check that the shard is entirely rewritten when possible, rather than appended
    // [3, 4]
    // [0, 0]
    array.store_array_subset_opt(
        &ArraySubset::new_with_ranges(&[0..1, 0..2]),
        &[3u16, 4],
        &opt,
    )?;
    assert_eq!(store_perf.reads(), 1); // index + 1x inner chunk
    assert_eq!(store_perf.writes(), expected_writes_per_shard);
    if inner_bytes_to_bytes_codecs.is_empty() {
        assert_eq!(store_perf.bytes_read(), shard_index_size);
    }
    if inner_bytes_to_bytes_codecs.is_empty() {
        assert_eq!(
            get_bytes_0_0()?.unwrap().len(),
            shard_index_size + size_of::<u16>() * 2
        );
    }
    assert_eq!(array.retrieve_chunk::<Vec<u16>>(&[0, 0])?, vec![3, 4, 0, 0]);
    store_perf.reset();

    // [99, 4]
    // [5, 0]
    array.store_array_subset_opt(
        &ArraySubset::new_with_ranges(&[0..2, 0..1]),
        &[99u16, 5],
        &opt,
    )?;
    assert_eq!(store_perf.reads(), 1); // index
    assert_eq!(store_perf.writes(), expected_writes_per_shard);
    if inner_bytes_to_bytes_codecs.is_empty() {
        assert_eq!(
            get_bytes_0_0()?.unwrap().len(),
            shard_index_size + size_of::<u16>() * 4 // 1 stale inner chunk + 3 inner chunks
        );
    }
    assert_eq!(
        array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
        vec![99, 4, 5, 0]
    );
    store_perf.reset();

    // [99, 4]
    // [5, 100]
    store_perf.reset();
    array.store_array_subset_opt(
        &ArraySubset::new_with_ranges(&[1..2, 1..2]),
        &[100u16],
        &opt,
    )?;
    assert_eq!(store_perf.reads(), 1); // index
    assert_eq!(store_perf.writes(), expected_writes_per_shard);
    if inner_bytes_to_bytes_codecs.is_empty() {
        assert_eq!(
            get_bytes_0_0()?.unwrap().len(),
            shard_index_size + size_of::<u16>() * 5 // 1 stale inner chunk + 4 inner chunks
        );
    }
    store_perf.reset();

    assert_eq!(
        array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
        vec![99, 4, 5, 100]
    );

    Ok(())
}

#[test]
fn array_partial_encode_sharding_index_start() {
    array_partial_encode_sharding(ShardingIndexLocation::Start, vec![]).unwrap();
}

#[test]
fn array_partial_encode_sharding_index_end() {
    array_partial_encode_sharding(ShardingIndexLocation::End, vec![]).unwrap();
}

#[test]
fn array_partial_encode_sharding_index_compressed() {
    #[cfg(feature = "blosc")]
    use zarrs::metadata_ext::codec::blosc::{
        BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
    };
    #[cfg(feature = "bz2")]
    use zarrs::metadata_ext::codec::bz2::Bz2CompressionLevel;

    for index_location in &[ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        array_partial_encode_sharding(
            *index_location,
            vec![
                #[cfg(feature = "gzip")]
                Arc::new(zarrs::array::codec::GzipCodec::new(5).unwrap()),
                #[cfg(feature = "zstd")]
                Arc::new(zarrs::array::codec::ZstdCodec::new(
                    5.try_into().unwrap(),
                    true,
                )),
                #[cfg(feature = "bz2")]
                Arc::new(zarrs::array::codec::Bz2Codec::new(
                    Bz2CompressionLevel::try_from(5u8).unwrap(),
                )),
                #[cfg(feature = "blosc")]
                Arc::new(
                    zarrs::array::codec::BloscCodec::new(
                        BloscCompressor::BloscLZ,
                        BloscCompressionLevel::try_from(5u8).unwrap(),
                        None,
                        BloscShuffleMode::NoShuffle,
                        None,
                    )
                    .unwrap(),
                ),
                #[cfg(feature = "crc32c")]
                Arc::new(zarrs::array::codec::Crc32cCodec::new()),
                #[cfg(feature = "adler32")]
                Arc::new(zarrs::array::codec::Adler32Codec::default()),
            ],
        )
        .unwrap();
    }
}

fn array_partial_encode_sharding_compact(
    sharding_index_location: ShardingIndexLocation,
    inner_bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test the compact_chunk() API by creating a shard with gaps
    // Strategy: Write a large compressible chunk, then overwrite with random data
    // This will create gaps in the shard that need compaction

    let opt = CodecOptions::default().with_experimental_partial_encoding(true);

    let store = std::sync::Arc::new(MemoryStore::default());
    let store_perf = Arc::new(PerformanceMetricsStorageAdapter::new(store.clone()));

    let array_path = "/";
    let mut builder = ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 4], // regular chunk shape (shard)
        data_type::uint16(),
        0u16,
    );
    builder
        .array_to_bytes_codec(Arc::new(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint16()) // 2x2 inner chunks
                .index_bytes_to_bytes_codecs(vec![])
                .index_location(sharding_index_location)
                .bytes_to_bytes_codecs(inner_bytes_to_bytes_codecs.clone())
                .build(),
        ))
        .bytes_to_bytes_codecs(vec![]);

    let array = builder.build(store_perf.clone(), array_path).unwrap();

    let get_bytes_0_0 = || {
        let key = array.chunk_key_encoding().encode(&[0, 0]);
        store.get(&key)
    };

    // Step 1: Write a large compressible pattern (all same values)
    // This fills multiple inner chunks with highly compressible data
    let compressible_data = vec![42u16; 16]; // Fill all 16 elements of the shard
    array.store_chunk_opt(&[0, 0], &compressible_data, &opt)?;

    let size_after_first_write = get_bytes_0_0()?.unwrap().len();

    // Step 2: Overwrite with different data in some inner chunks
    // This creates gaps as the old compressed data is marked stale
    // Write to inner chunk [0,0] (elements [0..2, 0..2] of the shard)
    let random_data1 = vec![100u16, 101, 102, 103];
    array.store_array_subset_opt(
        &ArraySubset::new_with_ranges(&[0..2, 0..2]),
        &random_data1,
        &opt,
    )?;

    // Write to inner chunk [1,0] (elements [2..4, 0..2] of the shard)
    let random_data2 = vec![200u16, 201, 202, 203];
    array.store_array_subset_opt(
        &ArraySubset::new_with_ranges(&[2..4, 0..2]),
        &random_data2,
        &opt,
    )?;

    let size_after_overwrites = get_bytes_0_0()?.unwrap().len();

    // After the overwrites, the shard should be larger (has gaps/stale data)
    assert!(
        size_after_overwrites >= size_after_first_write,
        "Shard should not shrink after partial overwrites (may have gaps)"
    );

    // Step 3: Compact the chunk
    store_perf.reset();
    let compaction_occurred = array.compact_chunk(&[0, 0], &opt)?;

    // Verify that compaction occurred
    assert!(
        compaction_occurred,
        "Compaction should occur when the shard has gaps"
    );

    // Verify I/O metrics
    assert_eq!(store_perf.reads(), 1); // Read the shard
    assert_eq!(store_perf.writes(), 1); // Write compacted shard (single write for already-encoded bytes)

    let size_after_compaction = get_bytes_0_0()?.unwrap().len();

    // Verify that the shard is smaller after compaction
    assert!(
        size_after_compaction < size_after_overwrites,
        "Compacted shard ({size_after_compaction}) should be smaller than pre-compaction size ({size_after_overwrites})"
    );

    // Verify data integrity after compaction
    // The 4x4 array in row-major order with 2x2 inner chunks:
    let expected_data = vec![
        // Row 0 (elements 0-3): inner chunks [0,0] cols 0-1, then [0,1] cols 2-3
        100, 101, 42, 42,
        // Row 1 (elements 4-7): inner chunks [0,0] cols 0-1, then [0,1] cols 2-3
        102, 103, 42, 42,
        // Row 2 (elements 8-11): inner chunks [1,0] cols 0-1, then [1,1] cols 2-3
        200, 201, 42, 42,
        // Row 3 (elements 12-15): inner chunks [1,0] cols 0-1, then [1,1] cols 2-3
        202, 203, 42, 42,
    ];
    assert_eq!(
        array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
        expected_data,
        "Data should be unchanged after compaction"
    );

    // Step 4: Verify idempotency - compacting an already-compact shard should return false
    store_perf.reset();
    let compaction_occurred_2 = array.compact_chunk(&[0, 0], &opt)?;

    assert!(
        !compaction_occurred_2,
        "Compaction should not occur on an already-compact shard"
    );

    // Verify I/O metrics - should read but not write
    assert_eq!(store_perf.reads(), 1);
    assert_eq!(store_perf.writes(), 0);

    Ok(())
}

#[test]
fn array_partial_encode_sharding_compact_index_start() {
    array_partial_encode_sharding_compact(ShardingIndexLocation::Start, vec![]).unwrap();
}

#[test]
fn array_partial_encode_sharding_compact_index_end() {
    array_partial_encode_sharding_compact(ShardingIndexLocation::End, vec![]).unwrap();
}

#[test]
fn array_partial_encode_sharding_compact_index_compressed() {
    #[cfg(feature = "blosc")]
    use zarrs::metadata_ext::codec::blosc::{
        BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
    };
    #[cfg(feature = "bz2")]
    use zarrs::metadata_ext::codec::bz2::Bz2CompressionLevel;

    for index_location in &[ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let result = array_partial_encode_sharding_compact(
            *index_location,
            vec![
                #[cfg(feature = "gzip")]
                Arc::new(zarrs::array::codec::GzipCodec::new(5).unwrap()),
                #[cfg(feature = "zstd")]
                Arc::new(zarrs::array::codec::ZstdCodec::new(
                    5.try_into().unwrap(),
                    true,
                )),
                #[cfg(feature = "bz2")]
                Arc::new(zarrs::array::codec::Bz2Codec::new(
                    Bz2CompressionLevel::try_from(5u8).unwrap(),
                )),
                #[cfg(feature = "blosc")]
                Arc::new(
                    zarrs::array::codec::BloscCodec::new(
                        BloscCompressor::BloscLZ,
                        BloscCompressionLevel::try_from(5u8).unwrap(),
                        None,
                        BloscShuffleMode::NoShuffle,
                        None,
                    )
                    .unwrap(),
                ),
                #[cfg(feature = "crc32c")]
                Arc::new(zarrs::array::codec::Crc32cCodec::new()),
                #[cfg(feature = "adler32")]
                Arc::new(zarrs::array::codec::Adler32Codec::default()),
            ],
        );
        if let Err(e) = result {
            eprintln!("Compressed test failed with index_location={index_location:?}: {e}");
            panic!("Test failed: {e}");
        }
    }
}

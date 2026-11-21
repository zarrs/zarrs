#![allow(missing_docs)]
#![cfg(feature = "sharding")]
//! Subchunk caches layered over chunk caches.

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruChunkLimitThreadLocal,
    ChunkCacheDecodedLruSizeLimit, ChunkCacheDecodedLruSizeLimitThreadLocal,
    ChunkCacheEncodedLruChunkLimit, ChunkCachePartialDecoderLruChunkLimit, NoChunkCache,
    SubchunkCached, SyncArrayCache,
};
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesTarget, ArrayCached,
    ArrayError, ArrayReadOps, ArraySubset, CodecOptions, DataType, ElementOrder, FillValue,
    Resources, data_type, transmute_to_bytes_vec,
};
use zarrs::storage::ReadableWritableStorageTraits;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// The data type of a test array.
#[derive(Clone, Copy, Debug)]
enum TestDataType {
    UInt16,
    /// An optional `uint16` with a null fill value, which is null for some elements outside of fill subchunks.
    OptionalUInt16,
    /// A variable-length `string`.
    String,
}

const DATA_TYPES: [TestDataType; 3] = [
    TestDataType::UInt16,
    TestDataType::OptionalUInt16,
    TestDataType::String,
];

/// A 30x30 array of 16x16 shards with 4x4 subchunks, so the last shard of each dimension extends beyond the array.
///
/// With `nested`, each subchunk is a shard of 2x2 subchunks.
fn sharded_array<TStorage: ?Sized + ReadableWritableStorageTraits + 'static>(
    store: Arc<TStorage>,
    nested: bool,
    data_type: TestDataType,
) -> Result<Arc<Array<TStorage>>, Box<dyn Error>> {
    let (data_type_value, fill_value): (DataType, FillValue) = match data_type {
        TestDataType::UInt16 => (data_type::uint16(), 0u16.into()),
        TestDataType::OptionalUInt16 => (data_type::uint16().to_optional(), None::<u16>.into()),
        TestDataType::String => (data_type::string(), String::new().into()),
    };
    let mut sharding = ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type_value);
    if nested {
        sharding.array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type_value).build_arc(),
        );
    }
    let mut builder = ArrayBuilder::new(vec![30, 30], vec![16, 16], data_type_value, fill_value);
    builder.array_to_bytes_codec(sharding.build_arc());
    let array = builder.build_arc(store, "/array")?;
    // Shard [1, 1] is not stored, and the top left subchunk of shard [0, 0] is the fill value (`None`)
    let values = (0..900u16).map(|i| {
        let (row, col) = (i / 30, i % 30);
        let fill = (row < 4 && col < 4) || (row >= 16 && col >= 16);
        (!fill).then_some(i + 1)
    });
    let (subset, resources) = (array.subset_all(), Resources::default());
    match data_type {
        TestDataType::UInt16 => array.store_array_subset(
            &subset,
            values.map(|v| v.unwrap_or(0)).collect::<Vec<u16>>(),
            &resources,
        ),
        TestDataType::OptionalUInt16 => array.store_array_subset(
            &subset,
            values
                .map(|v| v.filter(|v| v % 7 != 0))
                .collect::<Vec<Option<u16>>>(),
            &resources,
        ),
        TestDataType::String => array.store_array_subset(
            &subset,
            values
                .map(|v| v.map_or_else(String::new, |v| v.to_string()))
                .collect::<Vec<String>>(),
            &resources,
        ),
    }?;
    Ok(array)
}

/// Read regions, chunks and subchunks of `array`, twice, so the second reads are served by cached subchunks.
fn reads<A: ArrayReadOps>(array: &A) -> Result<Vec<Option<ArrayBytes<'static>>>, Box<dyn Error>> {
    let resources = Resources::default();
    let mut bytes = Vec::new();
    for _ in 0..2 {
        for subset in [
            ArraySubset::new_with_ranges(&[0..30, 0..30]),
            ArraySubset::new_with_ranges(&[1..3, 1..3]),
            ArraySubset::new_with_ranges(&[3..9, 2..13]),
            ArraySubset::new_with_ranges(&[14..19, 10..30]),
            ArraySubset::new_with_ranges(&[28..30, 28..30]),
        ] {
            bytes.push(Some(array.retrieve_array_subset(&subset, &resources)?));
        }
        for chunk_indices in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            bytes.push(Some(array.retrieve_chunk(&chunk_indices, &resources)?));
            bytes.push(array.retrieve_chunk_if_exists(&chunk_indices, &resources)?);
            let subset = ArraySubset::new_with_ranges(&[2..7, 3..11]);
            bytes.push(Some(array.retrieve_partial_chunk(
                &chunk_indices,
                &subset,
                &resources,
            )?));
            bytes.push(Some(
                array
                    .partial_decoder(&chunk_indices, &resources)?
                    .partial_decode(&subset, &CodecOptions::default(), &resources)?
                    .into_owned(),
            ));
        }
        for subchunk_indices in [[0, 0], [1, 2], [3, 3], [7, 7]] {
            bytes.push(Some(
                array.retrieve_subchunk(&subchunk_indices, &resources)?,
            ));
        }
    }
    Ok(bytes)
}

/// Check that reads through `cache` match reads of `array`, in C order and in F order, which decodes into strided
/// targets.
fn check_reads<TStorage, C>(array: &Arc<Array<TStorage>>, cache: C) -> TestResult
where
    TStorage: ?Sized + ReadableWritableStorageTraits + 'static,
    C: SyncArrayCache + 'static,
{
    let cached = ArrayCached::new(array.clone(), cache);
    assert_eq!(reads(&cached)?, reads(array.as_ref())?);
    assert_eq!(
        reads(&cached.with_element_order(ElementOrder::F))?,
        reads(&array.with_element_order(ElementOrder::F))?
    );
    Ok(())
}

/// Read `subset` of a `uint16` `array` into a target of one dimension, rather than of the shape of `subset`.
fn retrieve_flat<A: ArrayReadOps>(
    array: &A,
    subset: &ArraySubset,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let mut output = vec![0u16; subset.num_elements_usize()];
    {
        let shape = [subset.num_elements()];
        let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(bytemuck::cast_slice_mut::<
            u16,
            u8,
        >(&mut output));
        let view = unsafe {
            // SAFETY: this is the only view over output and covers it exactly.
            ArrayBytesFixedDisjointView::new(
                output_slice,
                2,
                &shape,
                ArraySubset::new_with_shape(shape),
            )?
        };
        array.retrieve_array_subset_into(
            subset,
            &mut ArrayBytesTarget::Fixed(view),
            &Resources::default(),
        )?;
    }
    Ok(output)
}

#[test]
fn subchunk_cache_reads_into_a_target_of_another_shape() -> TestResult {
    let array = sharded_array(
        Arc::new(MemoryStore::default()),
        false,
        TestDataType::UInt16,
    )?;
    let cached = ArrayCached::new(
        array.clone(),
        SubchunkCached::new(
            0,
            ChunkCacheDecodedLruChunkLimit::new(64),
            ChunkCachePartialDecoderLruChunkLimit::new(4),
        ),
    );
    // Part of a chunk, and parts of several chunks
    for subset in [
        ArraySubset::new_with_ranges(&[3..9, 2..13]),
        ArraySubset::new_with_ranges(&[14..19, 10..30]),
    ] {
        for _ in 0..2 {
            assert_eq!(
                retrieve_flat(&cached, &subset)?,
                retrieve_flat(array.as_ref(), &subset)?
            );
        }
    }
    Ok(())
}

#[test]
fn subchunk_cache_reads_match_the_array() -> TestResult {
    for (nested, data_type) in [false, true]
        .into_iter()
        .flat_map(|nested| DATA_TYPES.map(|data_type| (nested, data_type)))
    {
        let array = sharded_array(Arc::new(MemoryStore::default()), nested, data_type)?;
        let subchunks = || ChunkCacheDecodedLruChunkLimit::new(64);
        check_reads(
            &array,
            SubchunkCached::new(
                0,
                subchunks(),
                ChunkCachePartialDecoderLruChunkLimit::new(4),
            ),
        )?;
        check_reads(
            &array,
            SubchunkCached::new(0, subchunks(), ChunkCacheEncodedLruChunkLimit::new(4)),
        )?;
        check_reads(&array, SubchunkCached::new(0, subchunks(), NoChunkCache))?;
        check_reads(
            &array,
            SubchunkCached::new(0, subchunks(), ChunkCacheDecodedLruChunkLimit::new(4)),
        )?;
        // A tiny cache evicts subchunks while reading
        check_reads(
            &array,
            SubchunkCached::new(
                0,
                ChunkCacheDecodedLruChunkLimit::new(1),
                ChunkCachePartialDecoderLruChunkLimit::new(1),
            ),
        )?;
        if nested {
            check_reads(
                &array,
                SubchunkCached::new(
                    1,
                    subchunks(),
                    ChunkCachePartialDecoderLruChunkLimit::new(4),
                ),
            )?;
            // Nested subchunk caches: level one subchunks over level zero subchunks
            check_reads(
                &array,
                SubchunkCached::new(
                    1,
                    subchunks(),
                    SubchunkCached::new(
                        0,
                        subchunks(),
                        ChunkCachePartialDecoderLruChunkLimit::new(4),
                    ),
                ),
            )?;
        }
    }
    Ok(())
}

#[test]
fn subchunk_cache_serves_repeated_reads_without_storage() -> TestResult {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let array = sharded_array(store.clone(), false, TestDataType::UInt16)?;
    let cached = ArrayCached::new(
        array.clone(),
        SubchunkCached::new(
            0,
            ChunkCacheDecodedLruChunkLimit::new(64),
            ChunkCachePartialDecoderLruChunkLimit::new(4),
        ),
    );
    let subset = ArraySubset::new_with_ranges(&[5..7, 5..7]);
    let expected = array.retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())?;

    store.reset();
    assert_eq!(
        cached.retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())?,
        expected
    );
    assert!(store.reads() > 0);
    assert_eq!(cached.cache().subchunks().len(), 1);

    // A cached subchunk is served without storage reads
    store.reset();
    assert_eq!(
        cached.retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())?,
        expected
    );
    assert_eq!(store.reads(), 0);

    // Another subchunk of the shard reads only that subchunk, as the chunk cache holds the shard index
    store.reset();
    cached.retrieve_array_subset::<Vec<u16>>(
        &ArraySubset::new_with_ranges(&[9..10, 9..10]),
        &Resources::default(),
    )?;
    assert_eq!(store.reads(), 1);
    assert_eq!(cached.cache().subchunks().len(), 2);
    Ok(())
}

#[test]
fn subchunk_cache_opens_a_chunk_once_per_read() -> TestResult {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let array = sharded_array(store.clone(), false, TestDataType::UInt16)?;
    let cached = ArrayCached::new(
        array.clone(),
        SubchunkCached::new(0, ChunkCacheDecodedLruChunkLimit::new(64), NoChunkCache),
    );
    let reads = |subset: &ArraySubset| -> Result<usize, Box<dyn Error>> {
        store.reset();
        array.retrieve_array_subset::<Vec<u16>>(subset, &Resources::default())?;
        Ok(store.reads())
    };
    // Reading 4 subchunks of one shard through the cache costs the same as reading them from the array: the shard
    // index is read once
    let subset = ArraySubset::new_with_ranges(&[4..12, 4..12]);
    let expected_reads = reads(&subset)?;
    store.reset();
    cached.retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())?;
    assert_eq!(store.reads(), expected_reads);
    assert_eq!(cached.cache().subchunks().len(), 4);
    Ok(())
}

#[test]
fn subchunk_cache_reads_a_cached_fill_chunk_without_storage() -> TestResult {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    // A 32x32 array of 16x16 shards with 4x4 subchunks, so no shard extends beyond the array, with only shard [0, 0]
    // stored
    let data_type = data_type::uint16();
    let mut builder = ArrayBuilder::new(vec![32, 32], vec![16, 16], data_type.clone(), 0u16);
    builder.array_to_bytes_codec(
        ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type).build_arc(),
    );
    let array = builder.build_arc(store.clone(), "/array")?;
    array.store_chunk(&[0, 0], vec![1u16; 256], &Resources::default())?;
    let cached = ArrayCached::new(
        array,
        SubchunkCached::new(0, ChunkCacheDecodedLruChunkLimit::new(64), NoChunkCache),
    );
    cached.retrieve_chunk::<Vec<u16>>(&[1, 1], &Resources::default())?;
    assert_eq!(cached.cache().subchunks().len(), 16);

    // A chunk of cached fill subchunks is the fill value whether or not it is stored, so it is read without storage
    store.reset();
    assert_eq!(
        cached.retrieve_chunk::<Vec<u16>>(&[1, 1], &Resources::default())?,
        vec![0u16; 256]
    );
    assert_eq!(
        cached.retrieve_array_subset::<Vec<u16>>(
            &ArraySubset::new_with_ranges(&[16..32, 16..32]),
            &Resources::default()
        )?,
        vec![0u16; 256]
    );
    assert_eq!(store.reads(), 0);

    // Whereas whether it exists is not cached
    assert_eq!(
        cached.retrieve_chunk_if_exists::<Vec<u16>>(&[1, 1], &Resources::default())?,
        None
    );
    assert!(store.reads() > 0);
    Ok(())
}

#[test]
fn subchunk_cache_invalidates_subchunks_of_written_chunks() -> TestResult {
    let array = sharded_array(
        Arc::new(MemoryStore::default()),
        false,
        TestDataType::UInt16,
    )?;
    let cached = ArrayCached::new(
        array.clone(),
        SubchunkCached::new(
            0,
            ChunkCacheDecodedLruChunkLimit::new(256),
            ChunkCachePartialDecoderLruChunkLimit::new(4),
        ),
    );
    let read_all =
        || cached.retrieve_array_subset::<Vec<u16>>(&array.subset_all(), &Resources::default());
    let check = || -> TestResult {
        assert_eq!(
            read_all()?,
            array.retrieve_array_subset::<Vec<u16>>(&array.subset_all(), &Resources::default())?
        );
        Ok(())
    };
    let subchunks = || cached.cache().subchunks().len();
    read_all()?;
    let all = subchunks();
    assert_eq!(all, 64);

    // A subset write may re-encode the chunk, so it evicts all subchunks of that chunk
    cached.store_array_subset(
        &ArraySubset::new_with_ranges(&[5..6, 5..10]),
        &[7u16; 5],
        &Resources::default(),
    )?;
    assert_eq!(subchunks(), all - 16);
    check()?;

    cached.store_partial_chunk(
        &[0, 1],
        &ArraySubset::new_with_ranges(&[0..1, 0..1]),
        &[9u16],
        &Resources::default(),
    )?;
    assert_eq!(subchunks(), all - 16);
    check()?;

    // A chunk write evicts the subchunks of the chunk
    cached.store_chunk(&[1, 0], vec![3u16; 256], &Resources::default())?;
    assert_eq!(subchunks(), all - 16);
    check()?;

    cached.erase_chunk(&[1, 0])?;
    assert_eq!(subchunks(), all - 16);
    check()?;

    // A partial encoder also evicts the subchunks of its chunk
    let encoder = cached.partial_encoder(&[0, 0], &Resources::default())?;
    encoder.partial_encode(
        &ArraySubset::new_with_ranges(&[0..4, 0..4]),
        &ArrayBytes::new_flen(transmute_to_bytes_vec(vec![5u16; 16])),
        &CodecOptions::default(),
        &Resources::default(),
    )?;
    assert_eq!(subchunks(), all - 16);
    check()?;

    // Writes across chunks evict the subchunks of each of them
    cached.store_array_subset(
        &ArraySubset::new_with_ranges(&[15..17, 0..2]),
        &[4u16; 4],
        &Resources::default(),
    )?;
    assert_eq!(subchunks(), all - 32);
    check()?;

    cached.store_chunks(
        &ArraySubset::new_with_ranges(&[0..1, 0..2]),
        vec![6u16; 512],
        &Resources::default(),
    )?;
    assert_eq!(subchunks(), all - 32);
    check()?;

    cached.erase_chunks(
        &ArraySubset::new_with_ranges(&[0..2, 1..2]),
        &Resources::default(),
    )?;
    assert_eq!(subchunks(), all - 32);
    check()?;

    // As does invalidating chunks that are not a region
    cached.invalidate_chunks(&[vec![0, 1], vec![1, 0]]);
    assert_eq!(subchunks(), all - 32);
    check()?;

    // A metadata write evicts everything
    cached.store_metadata()?;
    assert_eq!(subchunks(), 0);
    check()?;
    Ok(())
}

#[test]
fn subchunk_cache_size_limit_holds_no_fill_subchunk_bytes() -> TestResult {
    let array = sharded_array(
        Arc::new(MemoryStore::default()),
        false,
        TestDataType::UInt16,
    )?;
    // Room for one decoded 4x4 subchunk of u16
    let cached = ArrayCached::new(
        array.clone(),
        SubchunkCached::new(
            0,
            ChunkCacheDecodedLruSizeLimit::new(32),
            ChunkCachePartialDecoderLruChunkLimit::new(4),
        ),
    );
    // Shard [1, 1] is not stored, so its 16 subchunks are fill and all fit alongside a decoded subchunk
    cached.retrieve_array_subset::<Vec<u16>>(
        &ArraySubset::new_with_ranges(&[16..30, 16..30]),
        &Resources::default(),
    )?;
    cached.retrieve_array_subset::<Vec<u16>>(
        &ArraySubset::new_with_ranges(&[5..6, 5..6]),
        &Resources::default(),
    )?;
    assert_eq!(cached.cache().subchunks().len(), 1 + 16);
    assert_eq!(
        cached.retrieve_chunk_if_exists::<Vec<u16>>(&[1, 1], &Resources::default())?,
        None
    );
    Ok(())
}

#[test]
fn subchunk_cache_requires_a_global_subchunk_grid() -> TestResult {
    let array = ArrayBuilder::new(vec![8], vec![4], data_type::uint8(), 0u8)
        .build_arc(Arc::new(MemoryStore::default()), "/array")?;
    let cached = ArrayCached::new(
        array,
        SubchunkCached::new(0, ChunkCacheDecodedLruChunkLimit::new(4), NoChunkCache),
    );
    assert!(matches!(
        cached.retrieve_chunk::<Vec<u8>>(&[0], &Resources::default()),
        Err(ArrayError::MissingSubchunkGrid)
    ));
    Ok(())
}

#[test]
fn thread_local_subchunk_reads_remain_coherent_after_writes() -> TestResult {
    fn check<S: ChunkCache<Value = zarrs::array::chunk_cache::ChunkCacheTypeDecoded> + 'static>(
        subchunks: S,
        target: usize,
    ) -> TestResult {
        let array = ArrayBuilder::new(vec![16], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(MemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(array, SubchunkCached::new(0, subchunks, NoChunkCache));
        let resources = Resources::default().with_compute_target(target);
        cached.store_chunks(&[0..2], &[1u8; 16], &resources)?;
        assert_eq!(
            cached.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
            vec![1; 8]
        );
        assert_eq!(cached.cache().subchunks().len(), 4);
        cached.store_chunk(&[0], &[2u8; 8], &resources)?;
        assert_eq!(
            cached.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
            vec![2; 8]
        );
        assert_eq!(
            cached.retrieve_array_subset::<Vec<u8>>(&[0..16], &resources)?,
            [vec![2; 8], vec![1; 8]].concat()
        );
        assert_eq!(cached.cache().subchunks().len(), 8);
        cached.store_array_subset(&[0..16], &[3u8; 16], &resources)?;
        assert_eq!(
            cached.retrieve_array_subset::<Vec<u8>>(&[0..16], &resources)?,
            vec![3; 16]
        );
        Ok(())
    }
    for target in [0, 2] {
        check(ChunkCacheDecodedLruChunkLimitThreadLocal::new(16), target)?;
        check(ChunkCacheDecodedLruSizeLimitThreadLocal::new(128), target)?;
    }
    Ok(())
}

#[test]
fn nested_thread_local_subchunk_caches_stay_on_the_calling_thread() -> TestResult {
    let array = sharded_array(Arc::new(MemoryStore::default()), true, TestDataType::UInt16)?;
    let cached = ArrayCached::new(
        array,
        SubchunkCached::new(
            1,
            ChunkCacheDecodedLruChunkLimit::new(256),
            SubchunkCached::new(
                0,
                ChunkCacheDecodedLruChunkLimitThreadLocal::new(64),
                NoChunkCache,
            ),
        ),
    );
    let resources = Resources::default().with_compute_target(0);
    cached.retrieve_chunk::<Vec<u16>>(&[0, 0], &resources)?;
    assert_eq!(cached.cache().chunks().subchunks().len(), 16);
    cached.store_chunk(&[0, 0], vec![2u16; 256], &resources)?;
    assert_eq!(
        cached.retrieve_chunk::<Vec<u16>>(&[0, 0], &resources)?,
        vec![2; 256]
    );
    Ok(())
}

#[test]
fn thread_local_subchunk_reads_with_nested_unlimited_decoding_finish() -> TestResult {
    let (send, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                let data_type = data_type::uint16();
                let mut outer = ShardingCodecBuilder::new(vec![nz(8), nz(8)], &data_type);
                outer.array_to_bytes_codec(
                    ShardingCodecBuilder::new(vec![nz(1), nz(1)], &data_type).build_arc(),
                );
                let mut builder = ArrayBuilder::new(vec![64, 64], vec![64, 64], data_type, 0u16);
                builder.array_to_bytes_codec(outer.build_arc());
                let array = builder
                    .build_arc(Arc::new(MemoryStore::default()), "/array")
                    .unwrap();
                let resources = Resources::default().with_compute_target(0);
                array
                    .store_chunk(&[0, 0], vec![1u16; 4096], &resources)
                    .unwrap();
                for _ in 0..8 {
                    let cached = ArrayCached::new(
                        array.clone(),
                        SubchunkCached::new(
                            0,
                            ChunkCacheDecodedLruChunkLimitThreadLocal::new(256),
                            ChunkCachePartialDecoderLruChunkLimit::new(1),
                        ),
                    );
                    assert_eq!(
                        cached
                            .retrieve_chunk::<Vec<u16>>(&[0, 0], &resources)
                            .unwrap(),
                        vec![1; 4096]
                    );
                }
            });
        send.send(()).unwrap();
    });
    receive.recv_timeout(std::time::Duration::from_secs(10))?;
    worker.join().unwrap();
    Ok(())
}

#[test]
fn retained_partial_decoders_do_not_repopulate_invalidated_subchunks() -> TestResult {
    fn check<C: SyncArrayCache>(chunks: C) -> TestResult {
        let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(MemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCached::new(0, ChunkCacheDecodedLruChunkLimit::new(8), chunks),
        );
        let resources = Resources::default();
        cached.store_chunk(&[0], &[1u8; 8], &resources)?;
        for value in [2u8, 3] {
            let decoder = cached.partial_decoder(&[0], &resources)?;
            cached.store_chunk(&[0], &[value; 8], &resources)?;
            decoder.partial_decode(&[0..8], &CodecOptions::default(), &resources)?;
            let mut output = [0u8; 8];
            let shape = [8];
            let view = unsafe {
                // SAFETY: this is the only view of output and covers it exactly.
                ArrayBytesFixedDisjointView::new(
                    unsafe_cell_slice::UnsafeCellSlice::new(&mut output),
                    1,
                    &shape,
                    ArraySubset::new_with_shape(shape),
                )?
            };
            decoder.partial_decode_into(
                &[0..8],
                &mut ArrayBytesTarget::Fixed(view),
                &CodecOptions::default(),
                &resources,
            )?;
            assert_eq!(output, [value - 1; 8]);
            assert_eq!(
                cached.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
                vec![value; 8]
            );
        }
        Ok(())
    }
    check(ChunkCacheDecodedLruChunkLimit::new(2))?;
    check(ChunkCacheEncodedLruChunkLimit::new(2))
}

#[test]
fn retained_partial_decoders_read_cached_subchunks_after_writes_to_other_chunks() -> TestResult {
    let array = ArrayBuilder::new(vec![16], vec![8], data_type::uint8(), 0u8)
        .subchunk_shape(vec![2])
        .build_arc(Arc::new(MemoryStore::default()), "/array")?;
    let cached = ArrayCached::new(
        array,
        SubchunkCached::new(0, ChunkCacheDecodedLruChunkLimit::new(8), NoChunkCache),
    );
    let resources = Resources::default();
    cached.store_chunks(&[0..2], &[1u8; 16], &resources)?;
    let decoder = cached.partial_decoder(&[0], &resources)?;
    cached.store_chunk(&[1], &[2u8; 8], &resources)?;
    decoder.partial_decode(&[0..8], &CodecOptions::default(), &resources)?;
    assert_eq!(cached.cache().subchunks().len(), 4);
    Ok(())
}

#[test]
fn invalidation_after_writes_bypassing_the_cache_retires_partial_decoders() -> TestResult {
    let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
        .subchunk_shape(vec![2])
        .build_arc(Arc::new(MemoryStore::default()), "/array")?;
    let cached = ArrayCached::new(
        array.clone(),
        SubchunkCached::new(
            0,
            ChunkCacheDecodedLruChunkLimit::new(8),
            ChunkCacheEncodedLruChunkLimit::new(2),
        ),
    );
    let resources = Resources::default();
    for value in [1u8, 2, 3] {
        array.store_chunk(&[0], &[value - 1; 8], &resources)?;
        let decoder = cached.partial_decoder(&[0], &resources)?;
        array.store_chunk(&[0], &[value; 8], &resources)?;
        match value {
            1 => cached.invalidate(),
            2 => cached.invalidate_chunk(&[0]),
            _ => cached.invalidate_chunks(&[0..1]),
        }
        assert_eq!(
            decoder.partial_decode(&[0..8], &CodecOptions::default(), &resources)?,
            ArrayBytes::new_flen(vec![value - 1; 8])
        );
        assert!(cached.cache().subchunks().is_empty());
        assert_eq!(
            cached.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
            vec![value; 8]
        );
    }
    Ok(())
}

#[test]
fn subchunk_cache_accepts_empty_selections_with_arbitrary_origins() -> TestResult {
    let array = ArrayBuilder::new(vec![32, 32], vec![16, 16], data_type::uint16(), 0u16)
        .subchunk_shape(vec![4, 4])
        .build_arc(Arc::new(MemoryStore::default()), "/array")?;
    let cached = ArrayCached::new(
        array,
        SubchunkCached::new(
            0,
            ChunkCacheDecodedLruChunkLimit::new(8),
            ChunkCacheEncodedLruChunkLimit::new(2),
        ),
    );
    let subset = ArraySubset::new_with_start_shape(vec![u64::MAX, 0], vec![0, 2])?;
    let resources = Resources::default();
    assert!(
        cached
            .retrieve_partial_chunk::<Vec<u16>>(&[1, 0], &subset, &resources)?
            .is_empty()
    );
    // Without retrieving the chunk
    assert!(cached.cache().chunks().is_empty());
    assert_eq!(
        cached
            .partial_decoder(&[1, 0], &resources)?
            .partial_decode(&subset, &CodecOptions::default(), &resources)?
            .size(),
        0
    );
    assert!(cached.cache().subchunks().is_empty());
    Ok(())
}

fn lossy_sharded_array() -> Result<Arc<Array<MemoryStore>>, Box<dyn Error>> {
    use zarrs::array::codec::FixedScaleOffsetCodec;
    use zarrs::metadata_ext::codec::fixedscaleoffset::FixedScaleOffsetCodecConfiguration;
    let configuration: FixedScaleOffsetCodecConfiguration = serde_json::from_str(
        r#"{"offset":-18797.0078125,"scale":610.6317749023438,"dtype":"f4","astype":"f4"}"#,
    )?;
    let data_type = data_type::float32();
    let mut sharding = ShardingCodecBuilder::new(vec![nz(2)], &data_type);
    sharding.array_to_array_codecs(vec![Arc::new(
        FixedScaleOffsetCodec::new_with_configuration(&configuration)?,
    )]);
    let mut builder = ArrayBuilder::new(vec![4], vec![4], data_type, 0.0f32);
    builder.array_to_bytes_codec(sharding.build_arc());
    Ok(builder.build_arc(Arc::new(MemoryStore::default()), "/array")?)
}

#[test]
fn partial_writes_invalidate_subchunks_changed_by_lossy_reencoding() -> TestResult {
    for array_subset in [false, true] {
        let array = lossy_sharded_array()?;
        let resources = Resources::default();
        array.store_chunk(&[0], &[234_551.36_f32; 4], &resources)?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCached::new(
                0,
                ChunkCacheDecodedLruChunkLimit::new(8),
                ChunkCachePartialDecoderLruChunkLimit::new(2),
            ),
        );
        let before = cached.retrieve_chunk::<Vec<f32>>(&[0], &resources)?;
        if array_subset {
            cached.store_array_subset(&[0..1], &[1.0f32], &resources)?;
        } else {
            cached.store_partial_chunk(&[0], &[0..1], &[1.0f32], &resources)?;
        }
        let expected = array.retrieve_chunk::<Vec<f32>>(&[0], &resources)?;
        assert_ne!(
            before[2], expected[2],
            "re-encoding changes an unwritten subchunk"
        );
        assert_eq!(
            cached.retrieve_chunk::<Vec<f32>>(&[0], &resources)?,
            expected
        );
    }
    Ok(())
}

#[cfg(feature = "async")]
mod r#async {
    use super::*;
    use zarrs::array::AsyncArrayReadOps;
    use zarrs::array::chunk_cache::{
        AsyncArrayCache, AsyncChunkCache, AsyncChunkCacheDecodedLruChunkLimit,
        AsyncChunkCacheEncodedLruChunkLimit, AsyncChunkCachePartialDecoderLruChunkLimit,
    };
    use zarrs::storage::AsyncReadableWritableStorageTraits;
    use zarrs::storage::store::AsyncMemoryStore;

    /// The asynchronous counterpart of [`sharded_array`], with the same data.
    async fn async_sharded_array<TStorage>(
        store: Arc<TStorage>,
        nested: bool,
        data_type: TestDataType,
    ) -> Result<Arc<Array<TStorage>>, Box<dyn Error>>
    where
        TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static,
    {
        let reference = sharded_array(Arc::new(MemoryStore::default()), nested, data_type)?;
        let array = Arc::new(Array::new_with_metadata(
            store,
            "/array",
            reference.metadata().clone(),
        )?);
        array
            .async_store_array_subset(
                &array.subset_all(),
                reference.retrieve_array_subset::<ArrayBytes>(
                    &reference.subset_all(),
                    &Resources::default(),
                )?,
                &Resources::default(),
            )
            .await?;
        Ok(array)
    }

    /// The asynchronous counterpart of [`reads`].
    async fn async_reads<A: AsyncArrayReadOps>(
        array: &A,
    ) -> Result<Vec<Option<ArrayBytes<'static>>>, Box<dyn Error>> {
        let resources = Resources::default();
        let mut bytes = Vec::new();
        for _ in 0..2 {
            for subset in [
                ArraySubset::new_with_ranges(&[0..30, 0..30]),
                ArraySubset::new_with_ranges(&[3..9, 2..13]),
                ArraySubset::new_with_ranges(&[14..19, 10..30]),
            ] {
                bytes.push(Some(
                    array
                        .async_retrieve_array_subset(&subset, &resources)
                        .await?,
                ));
            }
            for chunk_indices in [[0, 0], [0, 1], [1, 0], [1, 1]] {
                bytes.push(Some(
                    array
                        .async_retrieve_chunk(&chunk_indices, &resources)
                        .await?,
                ));
                bytes.push(
                    array
                        .async_retrieve_chunk_if_exists(&chunk_indices, &resources)
                        .await?,
                );
                let subset = ArraySubset::new_with_ranges(&[2..7, 3..11]);
                bytes.push(Some(
                    array
                        .async_partial_decoder(&chunk_indices, &resources)
                        .await?
                        .partial_decode(&subset, &CodecOptions::default(), &resources)
                        .await?
                        .into_owned(),
                ));
            }
        }
        Ok(bytes)
    }

    async fn async_check_reads<TStorage, C>(array: &Arc<Array<TStorage>>, cache: C) -> TestResult
    where
        TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static,
        C: AsyncArrayCache + 'static,
    {
        let cached = ArrayCached::new(array.clone(), cache);
        assert_eq!(
            async_reads(&cached).await?,
            async_reads(array.as_ref()).await?
        );
        assert_eq!(
            async_reads(&cached.with_element_order(ElementOrder::F)).await?,
            async_reads(&array.with_element_order(ElementOrder::F)).await?
        );
        Ok(())
    }

    #[tokio::test]
    async fn async_subchunk_cache_reads_match_the_array() -> TestResult {
        for (nested, data_type) in [false, true]
            .into_iter()
            .flat_map(|nested| DATA_TYPES.map(|data_type| (nested, data_type)))
        {
            let array =
                async_sharded_array(Arc::new(AsyncMemoryStore::default()), nested, data_type)
                    .await?;
            let subchunks = || AsyncChunkCacheDecodedLruChunkLimit::new(64);
            async_check_reads(
                &array,
                SubchunkCached::new(
                    0,
                    subchunks(),
                    AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
                ),
            )
            .await?;
            async_check_reads(
                &array,
                SubchunkCached::new(0, subchunks(), AsyncChunkCacheEncodedLruChunkLimit::new(4)),
            )
            .await?;
            async_check_reads(&array, SubchunkCached::new(0, subchunks(), NoChunkCache)).await?;
            if nested {
                async_check_reads(
                    &array,
                    SubchunkCached::new(
                        1,
                        subchunks(),
                        SubchunkCached::new(
                            0,
                            subchunks(),
                            AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
                        ),
                    ),
                )
                .await?;
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn async_subchunk_cache_serves_repeated_reads_without_storage() -> TestResult {
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            AsyncMemoryStore::default(),
        )));
        let array = async_sharded_array(store.clone(), false, TestDataType::UInt16).await?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCached::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(64),
                AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
            ),
        );
        let subset = ArraySubset::new_with_ranges(&[5..7, 5..7]);
        let expected = array
            .async_retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())
            .await?;

        store.reset();
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())
                .await?,
            expected
        );
        assert!(store.reads() > 0);
        store.reset();
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())
                .await?,
            expected
        );
        assert_eq!(store.reads(), 0);
        assert_eq!(cached.cache().subchunks().len().await, 1);
        Ok(())
    }

    #[tokio::test]
    async fn async_subchunk_cache_invalidates_subchunks_of_written_chunks() -> TestResult {
        let array = async_sharded_array(
            Arc::new(AsyncMemoryStore::default()),
            false,
            TestDataType::UInt16,
        )
        .await?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCached::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(256),
                AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
            ),
        );
        let subset_all = array.subset_all();
        cached
            .async_retrieve_array_subset::<Vec<u16>>(&subset_all, &Resources::default())
            .await?;
        let all = cached.cache().subchunks().len().await;
        assert_eq!(all, 64);

        cached
            .async_store_array_subset(
                &ArraySubset::new_with_ranges(&[5..6, 5..10]),
                &[7u16; 5],
                &Resources::default(),
            )
            .await?;
        assert_eq!(cached.cache().subchunks().len().await, all - 16);
        cached
            .async_store_chunk(&[1, 0], vec![3u16; 256], &Resources::default())
            .await?;
        assert_eq!(cached.cache().subchunks().len().await, all - 32);
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&subset_all, &Resources::default())
                .await?,
            array
                .async_retrieve_array_subset::<Vec<u16>>(&subset_all, &Resources::default())
                .await?
        );

        // Writes across chunks evict the subchunks of each of them
        cached
            .async_store_array_subset(
                &ArraySubset::new_with_ranges(&[15..17, 0..2]),
                &[4u16; 4],
                &Resources::default(),
            )
            .await?;
        assert_eq!(cached.cache().subchunks().len().await, all - 32);
        cached
            .async_retrieve_array_subset::<Vec<u16>>(&subset_all, &Resources::default())
            .await?;
        cached
            .async_invalidate_chunks(&[vec![0, 1], vec![1, 0]])
            .await;
        assert_eq!(cached.cache().subchunks().len().await, all - 32);
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&subset_all, &Resources::default())
                .await?,
            array
                .async_retrieve_array_subset::<Vec<u16>>(&subset_all, &Resources::default())
                .await?
        );
        cached.async_store_metadata().await?;
        assert_eq!(cached.cache().subchunks().len().await, 0);
        Ok(())
    }

    #[tokio::test]
    async fn retained_async_decoders_do_not_repopulate_invalidated_subchunks() -> TestResult {
        async fn check<C: AsyncArrayCache>(chunks: C) -> TestResult {
            let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
                .subchunk_shape(vec![2])
                .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
            let cached = ArrayCached::new(
                array,
                SubchunkCached::new(0, AsyncChunkCacheDecodedLruChunkLimit::new(8), chunks),
            );
            let resources = Resources::default();
            cached
                .async_store_chunk(&[0], &[1u8; 8], &resources)
                .await?;
            for value in [2u8, 3] {
                let decoder = cached.async_partial_decoder(&[0], &resources).await?;
                cached
                    .async_store_chunk(&[0], &[value; 8], &resources)
                    .await?;
                decoder
                    .partial_decode(&[0..8], &CodecOptions::default(), &resources)
                    .await?;
                let mut output = [0u8; 8];
                let shape = [8];
                let view = unsafe {
                    // SAFETY: this is the only view of output and covers it exactly.
                    ArrayBytesFixedDisjointView::new(
                        unsafe_cell_slice::UnsafeCellSlice::new(&mut output),
                        1,
                        &shape,
                        ArraySubset::new_with_shape(shape),
                    )?
                };
                decoder
                    .partial_decode_into(
                        &[0..8],
                        &mut ArrayBytesTarget::Fixed(view),
                        &CodecOptions::default(),
                        &resources,
                    )
                    .await?;
                assert_eq!(output, [value - 1; 8]);
                assert_eq!(
                    cached
                        .async_retrieve_chunk::<Vec<u8>>(&[0], &resources)
                        .await?,
                    vec![value; 8]
                );
            }
            Ok(())
        }
        check(AsyncChunkCacheDecodedLruChunkLimit::new(2)).await?;
        check(AsyncChunkCacheEncodedLruChunkLimit::new(2)).await
    }

    #[tokio::test]
    async fn async_invalidation_after_writes_bypassing_the_cache_retires_partial_decoders()
    -> TestResult {
        let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCached::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(8),
                AsyncChunkCacheEncodedLruChunkLimit::new(2),
            ),
        );
        let resources = Resources::default();
        for value in [1u8, 2, 3] {
            array
                .async_store_chunk(&[0], &[value - 1; 8], &resources)
                .await?;
            let decoder = cached.async_partial_decoder(&[0], &resources).await?;
            array
                .async_store_chunk(&[0], &[value; 8], &resources)
                .await?;
            match value {
                1 => cached.async_invalidate().await,
                2 => cached.async_invalidate_chunk(&[0]).await,
                _ => cached.async_invalidate_chunks(&[0..1]).await,
            }
            assert_eq!(
                decoder
                    .partial_decode(&[0..8], &CodecOptions::default(), &resources)
                    .await?,
                ArrayBytes::new_flen(vec![value - 1; 8])
            );
            assert!(cached.cache().subchunks().is_empty().await);
            assert_eq!(
                cached
                    .async_retrieve_chunk::<Vec<u8>>(&[0], &resources)
                    .await?,
                vec![value; 8]
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn async_subchunk_cache_accepts_empty_selections_with_arbitrary_origins() -> TestResult {
        let array = ArrayBuilder::new(vec![32, 32], vec![16, 16], data_type::uint16(), 0u16)
            .subchunk_shape(vec![4, 4])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCached::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(8),
                AsyncChunkCacheEncodedLruChunkLimit::new(2),
            ),
        );
        let subset = ArraySubset::new_with_start_shape(vec![u64::MAX, 0], vec![0, 2])?;
        let resources = Resources::default();
        assert!(
            cached
                .async_retrieve_partial_chunk::<Vec<u16>>(&[1, 0], &subset, &resources)
                .await?
                .is_empty()
        );
        // Without retrieving the chunk
        assert!(cached.cache().chunks().is_empty().await);
        assert_eq!(
            cached
                .async_partial_decoder(&[1, 0], &resources)
                .await?
                .partial_decode(&subset, &CodecOptions::default(), &resources)
                .await?
                .size(),
            0
        );
        assert!(cached.cache().subchunks().is_empty().await);
        Ok(())
    }

    #[tokio::test]
    async fn async_partial_writes_invalidate_subchunks_changed_by_lossy_reencoding() -> TestResult {
        for array_subset in [false, true] {
            let reference = lossy_sharded_array()?;
            let array = Arc::new(Array::new_with_metadata(
                Arc::new(AsyncMemoryStore::default()),
                "/array",
                reference.metadata().clone(),
            )?);
            let resources = Resources::default();
            array
                .async_store_chunk(&[0], &[234_551.36_f32; 4], &resources)
                .await?;
            let cached = ArrayCached::new(
                array.clone(),
                SubchunkCached::new(
                    0,
                    AsyncChunkCacheDecodedLruChunkLimit::new(8),
                    AsyncChunkCachePartialDecoderLruChunkLimit::new(2),
                ),
            );
            let before = cached
                .async_retrieve_chunk::<Vec<f32>>(&[0], &resources)
                .await?;
            if array_subset {
                cached
                    .async_store_array_subset(&[0..1], &[1.0f32], &resources)
                    .await?;
            } else {
                cached
                    .async_store_partial_chunk(&[0], &[0..1], &[1.0f32], &resources)
                    .await?;
            }
            let expected = array
                .async_retrieve_chunk::<Vec<f32>>(&[0], &resources)
                .await?;
            assert_ne!(
                before[2], expected[2],
                "re-encoding changes an unwritten subchunk"
            );
            assert_eq!(
                cached
                    .async_retrieve_chunk::<Vec<f32>>(&[0], &resources)
                    .await?,
                expected
            );
        }
        Ok(())
    }
}

#![allow(missing_docs)]
#![cfg(feature = "sharding")]
//! Subchunk caches layered over chunk caches.
//!
//! Tests are written in their asynchronous form, from which `ambisync` derives the synchronous tests.

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

#[cfg(feature = "async")]
use zarrs::array::AsyncArrayReadOps;
#[cfg(feature = "async")]
use zarrs::array::chunk_cache::{
    AsyncArrayCache, AsyncChunkCache, AsyncChunkCacheDecodedLruChunkLimit,
    AsyncChunkCacheDecodedLruSizeLimit, AsyncChunkCacheEncodedLruChunkLimit, AsyncChunkCacheInit,
    AsyncChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruSizeLimit,
    ChunkCacheEncodedLruChunkLimit, ChunkCachePartialDecoderLruChunkLimit, NoCache, SubchunkCache,
    SyncArrayCache,
};
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesTarget, ArrayCached,
    ArrayError, ArrayIndices, ArrayReadOps, ArraySubset, CodecOptions, DataType, ElementOrder,
    FillValue, Indexer, Resources, data_type,
};
#[cfg(feature = "async")]
use zarrs::storage::AsyncReadableWritableStorageTraits;
use zarrs::storage::ReadableWritableStorageTraits;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
#[cfg(feature = "async")]
use zarrs::storage::store::AsyncMemoryStore;
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

/// A hook run once by a [`HookedCache`].
type Hook = Box<dyn FnOnce() + Send>;

/// A chunk cache that runs a hook before its next chunk invalidation, and a hook after its next retrieval.
struct HookedCache<C> {
    cache: C,
    before_invalidate: Mutex<Option<Hook>>,
    after_retrieve: Mutex<Option<Hook>>,
}

impl<C> HookedCache<C> {
    fn new(cache: C) -> Self {
        Self {
            cache,
            before_invalidate: Mutex::new(None),
            after_retrieve: Mutex::new(None),
        }
    }

    fn before_invalidate(&self, hook: impl FnOnce() + Send + 'static) {
        *self.before_invalidate.lock().unwrap() = Some(Box::new(hook));
    }

    fn after_retrieve(&self, hook: impl FnOnce() + Send + 'static) {
        *self.after_retrieve.lock().unwrap() = Some(Box::new(hook));
    }
}

/// Run and clear the hook in `hook`, if any.
fn run_hook(hook: &Mutex<Option<Hook>>) {
    let hook = hook.lock().unwrap().take();
    if let Some(hook) = hook {
        hook();
    }
}

/// A fixed length target of one dimension over `output`.
fn flat_target(
    output: &mut [u8],
    data_type_size: usize,
) -> Result<ArrayBytesTarget<'_>, Box<dyn Error>> {
    let shape = [(output.len() / data_type_size) as u64];
    let view = unsafe {
        // SAFETY: this is the only view of output and covers it exactly.
        ArrayBytesFixedDisjointView::new(
            unsafe_cell_slice::UnsafeCellSlice::new(output),
            data_type_size,
            &shape,
            ArraySubset::new_with_shape(shape),
        )?
    };
    Ok(ArrayBytesTarget::Fixed(view))
}

#[ambisync::scope(
    sync(
        fns("async_{}"),
        types("Async{}", AsyncArrayCache => SyncArrayCache)
    ),
    async(feature = "async")
)]
mod subchunk_cache {
    use super::*;

    #[ambisync(async(flavor = async_trait, send = cfg(not(target_arch = "wasm32"))))]
    impl<C: AsyncChunkCache> AsyncChunkCache for HookedCache<C> {
        type Value = C::Value;

        #[sync_signature(fn try_get_or_insert_with(
            &self,
            chunk_indices: ArrayIndices,
            init: &mut dyn FnMut() -> Result<Self::Value, ArrayError>,
        ) -> Result<Self::Value, Arc<ArrayError>>)]
        async fn try_get_or_insert_with(
            &self,
            chunk_indices: ArrayIndices,
            init: AsyncChunkCacheInit<'_, Self::Value>,
        ) -> Result<Self::Value, Arc<ArrayError>> {
            let value = self.cache.try_get_or_insert_with(chunk_indices, init).await;
            run_hook(&self.after_retrieve);
            value
        }

        async fn invalidate(&self) -> usize {
            self.cache.invalidate().await
        }

        async fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
            run_hook(&self.before_invalidate);
            self.cache.invalidate_chunk(chunk_indices).await
        }

        async fn invalidate_chunks(&self, chunks: &dyn Indexer) -> usize {
            run_hook(&self.before_invalidate);
            self.cache.invalidate_chunks(chunks).await
        }

        async fn len(&self) -> usize {
            self.cache.len().await
        }

        async fn size(&self) -> usize {
            self.cache.size().await
        }
    }

    /// A 30x30 array of 16x16 shards with 4x4 subchunks, so the last shard of each dimension extends beyond the
    /// array.
    ///
    /// With `nested`, each subchunk is a shard of 2x2 subchunks.
    #[ambisync]
    async fn async_sharded_array<TStorage>(
        store: Arc<TStorage>,
        nested: bool,
        data_type: TestDataType,
    ) -> Result<Arc<Array<TStorage>>, Box<dyn Error>>
    where
        TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static,
    {
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
        let mut builder =
            ArrayBuilder::new(vec![30, 30], vec![16, 16], data_type_value, fill_value);
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
            TestDataType::UInt16 => {
                array
                    .async_store_array_subset(
                        &subset,
                        values.map(|v| v.unwrap_or(0)).collect::<Vec<u16>>(),
                        &resources,
                    )
                    .await
            }
            TestDataType::OptionalUInt16 => {
                array
                    .async_store_array_subset(
                        &subset,
                        values
                            .map(|v| v.filter(|v| v % 7 != 0))
                            .collect::<Vec<Option<u16>>>(),
                        &resources,
                    )
                    .await
            }
            TestDataType::String => {
                array
                    .async_store_array_subset(
                        &subset,
                        values
                            .map(|v| v.map_or_else(String::new, |v| v.to_string()))
                            .collect::<Vec<String>>(),
                        &resources,
                    )
                    .await
            }
        }?;
        Ok(array)
    }

    /// Read regions, chunks and subchunks of `array`, twice, so the second reads are served by cached subchunks.
    #[ambisync]
    async fn async_reads<A: AsyncArrayReadOps>(
        array: &A,
    ) -> Result<Vec<Option<ArrayBytes<'static>>>, Box<dyn Error>> {
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
                        .async_retrieve_partial_chunk(&chunk_indices, &subset, &resources)
                        .await?,
                ));
                bytes.push(Some(
                    array
                        .async_partial_decoder(&chunk_indices, &resources)
                        .await?
                        .partial_decode(&subset, &CodecOptions::default(), &resources)
                        .await?
                        .into_owned(),
                ));
            }
            for subchunk_indices in [[0, 0], [1, 2], [3, 3], [7, 7]] {
                bytes.push(Some(
                    array
                        .async_retrieve_subchunk(&subchunk_indices, &resources)
                        .await?,
                ));
            }
        }
        Ok(bytes)
    }

    /// Check that reads through `cache` match reads of `array`, in C order and in F order, which decodes into strided
    /// targets.
    #[ambisync]
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

    /// Read `subset` of a `uint16` `array` into a target of one dimension, rather than of the shape of `subset`.
    #[ambisync]
    async fn async_retrieve_flat<A: AsyncArrayReadOps>(
        array: &A,
        subset: &ArraySubset,
    ) -> Result<Vec<u16>, Box<dyn Error>> {
        let mut output = vec![0u16; subset.num_elements_usize()];
        array
            .async_retrieve_array_subset_into(
                subset,
                &mut flat_target(bytemuck::cast_slice_mut(&mut output), 2)?,
                &Resources::default(),
            )
            .await?;
        Ok(output)
    }

    #[ambisync::test]
    async fn async_subchunk_cache_reads_into_a_target_of_another_shape() -> TestResult {
        let array = async_sharded_array(
            Arc::new(AsyncMemoryStore::default()),
            false,
            TestDataType::UInt16,
        )
        .await?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCache::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(64),
                AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
            ),
        );
        // Part of a chunk, and parts of several chunks
        for subset in [
            ArraySubset::new_with_ranges(&[3..9, 2..13]),
            ArraySubset::new_with_ranges(&[14..19, 10..30]),
        ] {
            for _ in 0..2 {
                assert_eq!(
                    async_retrieve_flat(&cached, &subset).await?,
                    async_retrieve_flat(array.as_ref(), &subset).await?
                );
            }
        }
        Ok(())
    }

    #[ambisync::test]
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
                SubchunkCache::new(
                    0,
                    subchunks(),
                    AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
                ),
            )
            .await?;
            async_check_reads(
                &array,
                SubchunkCache::new(0, subchunks(), AsyncChunkCacheEncodedLruChunkLimit::new(4)),
            )
            .await?;
            async_check_reads(&array, SubchunkCache::new(0, subchunks(), NoCache)).await?;
            async_check_reads(
                &array,
                SubchunkCache::new(0, subchunks(), AsyncChunkCacheDecodedLruChunkLimit::new(4)),
            )
            .await?;
            // A tiny cache evicts subchunks while reading
            async_check_reads(
                &array,
                SubchunkCache::new(
                    0,
                    AsyncChunkCacheDecodedLruChunkLimit::new(1),
                    AsyncChunkCachePartialDecoderLruChunkLimit::new(1),
                ),
            )
            .await?;
            if nested {
                async_check_reads(
                    &array,
                    SubchunkCache::new(
                        1,
                        subchunks(),
                        AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
                    ),
                )
                .await?;
                // Nested subchunk caches: level one subchunks over level zero subchunks
                async_check_reads(
                    &array,
                    SubchunkCache::new(
                        1,
                        subchunks(),
                        SubchunkCache::new(
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

    #[ambisync::test]
    async fn async_subchunk_cache_serves_repeated_reads_without_storage() -> TestResult {
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            AsyncMemoryStore::default(),
        )));
        let array = async_sharded_array(store.clone(), false, TestDataType::UInt16).await?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCache::new(0, AsyncChunkCacheDecodedLruChunkLimit::new(64), NoCache),
        );
        // 4 subchunks of one shard
        let subset = ArraySubset::new_with_ranges(&[4..12, 4..12]);
        store.reset();
        let expected = array
            .async_retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())
            .await?;
        let expected_reads = store.reads();

        // Reading them through the cache costs the same as reading them from the array: the shard index is read once
        store.reset();
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())
                .await?,
            expected
        );
        assert_eq!(store.reads(), expected_reads);
        assert_eq!(cached.cache().subchunks().len().await, 4);

        // Cached subchunks are served without storage reads
        store.reset();
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())
                .await?,
            expected
        );
        assert_eq!(store.reads(), 0);
        Ok(())
    }

    #[ambisync::test]
    async fn async_subchunk_cache_reads_a_cached_fill_chunk_without_storage() -> TestResult {
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            AsyncMemoryStore::default(),
        )));
        // A 32x32 array of 16x16 shards with 4x4 subchunks, so no shard extends beyond the array, with only shard
        // [0, 0] stored
        let data_type = data_type::uint16();
        let mut builder = ArrayBuilder::new(vec![32, 32], vec![16, 16], data_type.clone(), 0u16);
        builder.array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type).build_arc(),
        );
        let array = builder.build_arc(store.clone(), "/array")?;
        array
            .async_store_chunk(&[0, 0], vec![1u16; 256], &Resources::default())
            .await?;
        let cached = ArrayCached::new(
            array,
            SubchunkCache::new(0, AsyncChunkCacheDecodedLruChunkLimit::new(64), NoCache),
        );
        cached
            .async_retrieve_chunk::<Vec<u16>>(&[1, 1], &Resources::default())
            .await?;
        assert_eq!(cached.cache().subchunks().len().await, 16);

        // A chunk of cached fill subchunks is the fill value whether or not it is stored, so it is read without
        // storage
        store.reset();
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<u16>>(&[1, 1], &Resources::default())
                .await?,
            vec![0u16; 256]
        );
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(
                    &ArraySubset::new_with_ranges(&[16..32, 16..32]),
                    &Resources::default()
                )
                .await?,
            vec![0u16; 256]
        );
        assert_eq!(store.reads(), 0);

        // Whereas whether it exists is not cached
        assert_eq!(
            cached
                .async_retrieve_chunk_if_exists::<Vec<u16>>(&[1, 1], &Resources::default())
                .await?,
            None
        );
        assert!(store.reads() > 0);
        Ok(())
    }

    #[ambisync::test]
    async fn async_subchunk_cache_invalidates_subchunks_of_written_chunks() -> TestResult {
        let array = async_sharded_array(
            Arc::new(AsyncMemoryStore::default()),
            false,
            TestDataType::UInt16,
        )
        .await?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCache::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(256),
                AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
            ),
        );
        let read_all = async || {
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&array.subset_all(), &Resources::default())
                .await
        };
        let check = async || -> TestResult {
            assert_eq!(
                read_all().await?,
                array
                    .async_retrieve_array_subset::<Vec<u16>>(
                        &array.subset_all(),
                        &Resources::default()
                    )
                    .await?
            );
            Ok(())
        };
        let subchunks = async || cached.cache().subchunks().len().await;
        read_all().await?;
        let all = subchunks().await;
        assert_eq!(all, 64);

        // A subset write may re-encode the chunk, so it evicts all subchunks of that chunk
        cached
            .async_store_array_subset(
                &ArraySubset::new_with_ranges(&[5..6, 5..10]),
                &[7u16; 5],
                &Resources::default(),
            )
            .await?;
        assert_eq!(subchunks().await, all - 16);
        check().await?;

        // A chunk write evicts the subchunks of the chunk
        cached
            .async_store_chunk(&[1, 0], vec![3u16; 256], &Resources::default())
            .await?;
        assert_eq!(subchunks().await, all - 16);
        check().await?;

        // Writes across chunks evict the subchunks of each of them
        cached
            .async_store_array_subset(
                &ArraySubset::new_with_ranges(&[15..17, 0..2]),
                &[4u16; 4],
                &Resources::default(),
            )
            .await?;
        assert_eq!(subchunks().await, all - 32);
        check().await?;

        // As does invalidating chunks that are not a region
        cached
            .async_invalidate_chunks(&[vec![0, 1], vec![1, 0]])
            .await;
        assert_eq!(subchunks().await, all - 32);
        check().await?;

        // A metadata write evicts everything
        cached.async_store_metadata().await?;
        assert_eq!(subchunks().await, 0);
        check().await?;
        Ok(())
    }

    #[ambisync::test]
    async fn async_subchunk_cache_size_limit_holds_no_fill_subchunk_bytes() -> TestResult {
        let array = async_sharded_array(
            Arc::new(AsyncMemoryStore::default()),
            false,
            TestDataType::UInt16,
        )
        .await?;
        // Room for one decoded 4x4 subchunk of u16
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCache::new(
                0,
                AsyncChunkCacheDecodedLruSizeLimit::new(32),
                AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
            ),
        );
        // Shard [1, 1] is not stored, so its 16 subchunks are fill and all fit alongside a decoded subchunk
        cached
            .async_retrieve_array_subset::<Vec<u16>>(
                &ArraySubset::new_with_ranges(&[16..30, 16..30]),
                &Resources::default(),
            )
            .await?;
        cached
            .async_retrieve_array_subset::<Vec<u16>>(
                &ArraySubset::new_with_ranges(&[5..6, 5..6]),
                &Resources::default(),
            )
            .await?;
        assert_eq!(cached.cache().subchunks().len().await, 1 + 16);
        Ok(())
    }

    #[ambisync::test]
    async fn async_subchunk_cache_requires_a_global_subchunk_grid() -> TestResult {
        let array = ArrayBuilder::new(vec![8], vec![4], data_type::uint8(), 0u8)
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCache::new(0, AsyncChunkCacheDecodedLruChunkLimit::new(4), NoCache),
        );
        let chunk = cached
            .async_retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
            .await;
        assert!(matches!(chunk, Err(ArrayError::MissingSubchunkGrid)));
        Ok(())
    }

    /// Check that partial decoders opened before writes through a subchunk cache over `chunks` decode the chunk as
    /// it was, without refilling the subchunk cache.
    #[ambisync]
    async fn async_check_retained_partial_decoders<C: AsyncArrayCache>(chunks: C) -> TestResult {
        let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCache::new(0, AsyncChunkCacheDecodedLruChunkLimit::new(8), chunks),
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
            assert_eq!(
                decoder
                    .partial_decode(&[0..8], &CodecOptions::default(), &resources)
                    .await?,
                ArrayBytes::new_flen(vec![value - 1; 8])
            );
            let mut output = [0u8; 8];
            decoder
                .partial_decode_into(
                    &[0..8],
                    &mut flat_target(&mut output, 1)?,
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

    #[ambisync::test]
    async fn async_retained_partial_decoders_do_not_repopulate_invalidated_subchunks() -> TestResult
    {
        async_check_retained_partial_decoders(AsyncChunkCacheDecodedLruChunkLimit::new(2)).await?;
        async_check_retained_partial_decoders(AsyncChunkCacheEncodedLruChunkLimit::new(2)).await
    }

    #[ambisync::test]
    async fn async_retained_partial_decoders_read_cached_subchunks_after_writes_to_other_chunks()
    -> TestResult {
        let array = ArrayBuilder::new(vec![16], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCache::new(0, AsyncChunkCacheDecodedLruChunkLimit::new(8), NoCache),
        );
        let resources = Resources::default();
        cached
            .async_store_chunks(&[0..2], &[1u8; 16], &resources)
            .await?;
        let decoder = cached.async_partial_decoder(&[0], &resources).await?;
        cached
            .async_store_chunk(&[1], &[2u8; 8], &resources)
            .await?;
        decoder
            .partial_decode(&[0..8], &CodecOptions::default(), &resources)
            .await?;
        assert_eq!(cached.cache().subchunks().len().await, 4);
        Ok(())
    }

    #[ambisync::test]
    async fn async_invalidation_after_writes_bypassing_the_cache_retires_partial_decoders()
    -> TestResult {
        let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array.clone(),
            SubchunkCache::new(
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

    #[ambisync::test]
    async fn async_subchunk_cache_accepts_empty_selections_with_arbitrary_origins() -> TestResult {
        let array = ArrayBuilder::new(vec![32, 32], vec![16, 16], data_type::uint16(), 0u16)
            .subchunk_shape(vec![4, 4])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCache::new(
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

    #[ambisync::test]
    async fn async_reads_during_invalidation_do_not_refill_subchunks_from_the_chunk_cache()
    -> TestResult {
        let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCache::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(8),
                HookedCache::new(AsyncChunkCacheEncodedLruChunkLimit::new(2)),
            ),
        );
        let resources = Resources::default();
        cached
            .async_store_chunk(&[0], &[1u8; 8], &resources)
            .await?;
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<u8>>(&[0], &resources)
                .await?,
            vec![1; 8]
        );

        // A read while the chunk cache still holds the stored chunk, after it is written
        let reader = cached.clone();
        // Hooks are synchronous, as the futures of array operations are not provably `Send`
        cached.cache().chunks().before_invalidate(move || {
            ambisync::alt!(
                sync => reader.retrieve_chunk::<Vec<u8>>(&[0], &Resources::default()),
                async => futures::executor::block_on(
                    reader.async_retrieve_chunk::<Vec<u8>>(&[0], &Resources::default()),
                ),
            )
            .unwrap();
        });
        cached
            .async_store_chunk(&[0], &[2u8; 8], &resources)
            .await?;
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<u8>>(&[0], &resources)
                .await?,
            vec![2; 8]
        );
        Ok(())
    }

    #[ambisync::test]
    async fn async_partial_decoders_opened_during_invalidation_do_not_refill_subchunks()
    -> TestResult {
        let array = ArrayBuilder::new(vec![8], vec![8], data_type::uint8(), 0u8)
            .subchunk_shape(vec![2])
            .build_arc(Arc::new(AsyncMemoryStore::default()), "/array")?;
        let cached = ArrayCached::new(
            array,
            SubchunkCache::new(
                0,
                AsyncChunkCacheDecodedLruChunkLimit::new(8),
                HookedCache::new(AsyncChunkCacheEncodedLruChunkLimit::new(2)),
            ),
        );
        let resources = Resources::default();
        cached
            .async_store_chunk(&[0], &[1u8; 8], &resources)
            .await?;

        // A write after a partial decoder retrieves the chunk it decodes
        let writer = cached.clone();
        cached.cache().chunks().after_retrieve(move || {
            ambisync::alt!(
                sync => writer.store_chunk(&[0], &[2u8; 8], &Resources::default()),
                async => futures::executor::block_on(
                    writer.async_store_chunk(&[0], &[2u8; 8], &Resources::default()),
                ),
            )
            .unwrap();
        });
        let decoder = cached.async_partial_decoder(&[0], &resources).await?;
        assert_eq!(
            decoder
                .partial_decode(&[0..8], &CodecOptions::default(), &resources)
                .await?,
            ArrayBytes::new_flen(vec![1u8; 8])
        );
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<u8>>(&[0], &resources)
                .await?,
            vec![2; 8]
        );
        Ok(())
    }
}

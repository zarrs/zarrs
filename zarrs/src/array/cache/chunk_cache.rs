use std::sync::Arc;

#[cfg(feature = "async")]
use super::{
    AsyncChunkCacheInit, AsyncChunkCacheTraits, AsyncChunkCacheType,
    ChunkCacheTypeAsyncPartialDecoder,
};
use super::{
    ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded, ChunkCacheTypePartialDecoder,
    SyncChunkCacheTraits, SyncChunkCacheType,
};
#[cfg(feature = "async")]
use crate::array::Indexer;
use crate::array::{ArrayError, ArrayIndices};

type ChunkIndices = ArrayIndices;

mod backend;

/// The constructors of a chunk cache.
macro_rules! impl_chunk_cache_constructors {
    () => {
        /// Create a cache that holds at most `chunk_capacity` chunks.
        ///
        /// The cache is split into shards for concurrency, each holding a share of the chunk capacity, so it may evict
        /// a chunk before it holds `chunk_capacity` chunks, or hold slightly more.
        #[must_use]
        pub fn new_chunk_limit(chunk_capacity: u64) -> Self {
            let cache = backend::Cache::new_with_chunk_capacity(chunk_capacity);
            Self { cache }
        }

        /// Create a cache that holds chunks up to a total of `size_capacity` bytes.
        ///
        /// Each chunk counts as at least one byte, so chunks that hold no bytes (e.g. fill value chunks) are evicted
        /// too. A chunk taking up nearly all of `size_capacity` on its own is not cached.
        #[must_use]
        pub fn new_size_limit(size_capacity: u64) -> Self {
            let cache = backend::Cache::new_with_size_capacity(size_capacity);
            Self { cache }
        }
    };
}

/// A chunk cache with a capacity in chunks or bytes.
///
/// The eviction policy is an implementation detail. It is currently [S3-FIFO](https://s3fifo.com/), which retains
/// frequently retrieved chunks through scans of the array.
pub struct ChunkCache<CT: SyncChunkCacheType> {
    cache: backend::Cache<CT>,
}

impl<CT: SyncChunkCacheType> ChunkCache<CT> {
    impl_chunk_cache_constructors!();
}

impl<CT: SyncChunkCacheType> SyncChunkCacheTraits for ChunkCache<CT> {
    type Value = CT;

    fn try_get_or_insert_with(
        &self,
        chunk_indices: ChunkIndices,
        f: &mut dyn FnMut() -> Result<Self::Value, ArrayError>,
    ) -> Result<Self::Value, Arc<ArrayError>> {
        self.cache.try_get_or_insert_with(chunk_indices, f)
    }

    fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
        self.cache.remove(chunk_indices)
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn size(&self) -> usize {
        self.cache.size()
    }

    fn invalidate(&self) -> usize {
        self.cache.clear()
    }
}

/// An asynchronous chunk cache with a capacity in chunks or bytes.
///
/// See [`ChunkCache`] for its eviction policy.
#[cfg(feature = "async")]
pub struct AsyncChunkCache<CT: AsyncChunkCacheType> {
    cache: backend::Cache<CT>,
}

#[cfg(feature = "async")]
impl<CT: AsyncChunkCacheType> AsyncChunkCache<CT> {
    impl_chunk_cache_constructors!();
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<CT: AsyncChunkCacheType> AsyncChunkCacheTraits for AsyncChunkCache<CT> {
    type Value = CT;

    async fn try_get_or_insert_with(
        &self,
        chunk_indices: ChunkIndices,
        init: AsyncChunkCacheInit<'_, Self::Value>,
    ) -> Result<Self::Value, Arc<ArrayError>> {
        match self
            .cache
            .inner()
            .get_value_or_guard_async(&chunk_indices)
            .await
        {
            Ok(value) => value.map(|cached| cached.value),
            Err(guard) => {
                let value = init
                    .await
                    .map(|value| self.cache.wrap(value))
                    .map_err(Arc::new);
                // An error means that the chunk was invalidated while it was initialised, so it is not cached.
                let _ = guard.insert(value.clone());
                value.map(|cached| cached.value)
            }
        }
    }

    async fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
        self.cache.remove(chunk_indices)
    }

    async fn invalidate_chunks(&self, chunks: &dyn Indexer) -> usize {
        chunks
            .iter_indices()
            .filter(|chunk_indices| self.cache.remove(chunk_indices))
            .count()
    }

    async fn len(&self) -> usize {
        self.cache.len()
    }

    async fn size(&self) -> usize {
        self.cache.size()
    }

    async fn invalidate(&self) -> usize {
        self.cache.clear()
    }
}

/// An encoded chunk cache.
pub type ChunkCacheEncoded = ChunkCache<ChunkCacheTypeEncoded>;

/// An decoded chunk cache.
pub type ChunkCacheDecoded = ChunkCache<ChunkCacheTypeDecoded>;

/// An partial decoder chunk cache.
pub type ChunkCachePartialDecoder = ChunkCache<ChunkCacheTypePartialDecoder>;

/// An asynchronous encoded chunk cache.
#[cfg(feature = "async")]
pub type AsyncChunkCacheEncoded = AsyncChunkCache<ChunkCacheTypeEncoded>;

/// An asynchronous decoded chunk cache.
#[cfg(feature = "async")]
pub type AsyncChunkCacheDecoded = AsyncChunkCache<ChunkCacheTypeDecoded>;

/// An asynchronous partial decoder chunk cache.
#[cfg(feature = "async")]
pub type AsyncChunkCachePartialDecoder = AsyncChunkCache<ChunkCacheTypeAsyncPartialDecoder>;

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use zarrs_codec::{ArrayBytes, ArrayPartialDecoderTraits, CodecError, CodecOptions, Resources};
    use zarrs_storage::Bytes;

    use super::*;
    use crate::array::DataType;
    use crate::array::cache::{ChunkCacheEncoded, ChunkCachePartialDecoder};

    /// Retrieve `count` chunks of `size` bytes through `cache`.
    fn retrieve(cache: &ChunkCacheEncoded, count: u64, size: usize) {
        for i in 0..count {
            SyncChunkCacheTraits::try_get_or_insert_with(
                cache,
                ArrayIndices::from([i].as_slice()),
                &mut || Ok(Some(Bytes::from(vec![0u8; size]))),
            )
            .unwrap();
        }
    }

    #[test]
    fn chunk_cache_chunk_limit() {
        let cache = ChunkCacheEncoded::new_chunk_limit(4);
        retrieve(&cache, 10, 10);
        assert!((1..=4).contains(&cache.len()));

        let cache = ChunkCacheEncoded::new_chunk_limit(1);
        retrieve(&cache, 3, 10);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn chunk_cache_size_limit() {
        let cache = ChunkCacheEncoded::new_size_limit(100);
        retrieve(&cache, 20, 10);
        assert!(cache.len() > 0);
        assert!(cache.size() <= 100);

        // A chunk larger than the size limit is not cached
        let cache = ChunkCacheEncoded::new_size_limit(100);
        retrieve(&cache, 1, 200);
        assert_eq!(cache.len(), 0);
    }

    /// Retrieve the chunks `indices` of `size` bytes through `cache`, returning the number retrieved uncached.
    fn retrieve_counting(
        cache: &ChunkCacheEncoded,
        indices: std::ops::Range<u64>,
        size: usize,
    ) -> usize {
        let mut calls = 0;
        for i in indices {
            SyncChunkCacheTraits::try_get_or_insert_with(
                cache,
                ArrayIndices::from([i].as_slice()),
                &mut || {
                    calls += 1;
                    Ok(Some(Bytes::from(vec![0u8; size])))
                },
            )
            .unwrap();
        }
        calls
    }

    #[test]
    fn chunk_cache_size_limit_admits_small_chunk_working_set() {
        // A cache of 4096 chunks of 4 KiB, whose record of evicted chunks must hold more than the 1500 chunks of
        // region B for them to be admitted as frequently retrieved
        const SIZE: usize = 4096;
        let cache = ChunkCacheEncoded::new_size_limit(4096 * SIZE as u64);
        let a = 0..4096;
        let b = 10_000..11_500;
        retrieve_counting(&cache, a.clone(), SIZE);
        retrieve_counting(&cache, a, SIZE);
        let misses: Vec<usize> = (0..4)
            .map(|_| retrieve_counting(&cache, b.clone(), SIZE))
            .collect();
        assert!(misses[3] < 150, "{misses:?}");
    }

    #[test]
    fn chunk_cache_shares_retrieval_error() {
        const WAITERS: usize = 7;
        let chunk_indices = ArrayIndices::from([0].as_slice());
        for cache in [
            ChunkCacheEncoded::new_chunk_limit(4),
            ChunkCacheEncoded::new_size_limit(100),
        ] {
            let calls = AtomicUsize::new(0);
            let (loader_started_tx, loader_started_rx) = std::sync::mpsc::channel();
            let waiters_ready = std::sync::Barrier::new(WAITERS + 1);
            std::thread::scope(|scope| {
                let loader = scope.spawn(|| {
                    SyncChunkCacheTraits::try_get_or_insert_with(
                        &cache,
                        chunk_indices.clone(),
                        &mut || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            loader_started_tx.send(()).unwrap();
                            // Wait until every waiter is about to enter `try_get_or_insert_with`, then give
                            // them time to register on the placeholder before failing.
                            waiters_ready.wait();
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            Err(ArrayError::Other("failed".to_string()))
                        },
                    )
                });

                loader_started_rx.recv().unwrap();
                let waiters: Vec<_> = (0..WAITERS)
                    .map(|_| {
                        scope.spawn(|| {
                            waiters_ready.wait();
                            SyncChunkCacheTraits::try_get_or_insert_with(
                                &cache,
                                chunk_indices.clone(),
                                &mut || {
                                    calls.fetch_add(1, Ordering::Relaxed);
                                    Err(ArrayError::Other("unexpected retry".to_string()))
                                },
                            )
                        })
                    })
                    .collect();

                let loader_err = loader.join().unwrap().unwrap_err();
                for waiter in waiters {
                    let waiter_err = waiter.join().unwrap().unwrap_err();
                    assert!(Arc::ptr_eq(&loader_err, &waiter_err));
                }
            });
            assert_eq!(calls.into_inner(), 1);
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.size(), 0);
            assert!(!cache.invalidate_chunk(&chunk_indices));

            // The error is not cached; an immediate retrieval retries it.
            retrieve(&cache, 1, 10);
            assert_eq!(cache.len(), 1);
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_chunk_cache_shares_error_and_retries_immediately() {
        use std::pin::pin;
        use std::task::Poll;

        let cache = AsyncChunkCacheEncoded::new_chunk_limit(4);
        let key = ArrayIndices::from([0].as_slice());
        let (started_tx, started_rx) = futures::channel::oneshot::channel();
        let (release_tx, release_rx) = futures::channel::oneshot::channel();
        let calls = AtomicUsize::new(0);
        let mut started_tx = Some(started_tx);

        let mut loader_init = pin!(async {
            calls.fetch_add(1, Ordering::Relaxed);
            started_tx.take().unwrap().send(()).unwrap();
            release_rx.await.unwrap();
            Err(ArrayError::Other("failed".to_string()))
        });
        let mut loader = pin!(AsyncChunkCacheTraits::try_get_or_insert_with(
            &cache,
            key.clone(),
            loader_init.as_mut(),
        ));

        // Poll loader until it installs the placeholder and blocks on `release_rx`.
        let mut started_rx = pin!(started_rx);
        futures::future::poll_fn(|cx| {
            assert!(matches!(loader.as_mut().poll(cx), Poll::Pending));
            started_rx.as_mut().poll(cx)
        })
        .await
        .unwrap();

        // Poll waiter once so it deterministically registers on the placeholder.
        let mut waiter_init = pin!(async {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(Bytes::from_static(b"unexpected")))
        });
        let mut waiter = pin!(AsyncChunkCacheTraits::try_get_or_insert_with(
            &cache,
            key.clone(),
            waiter_init.as_mut(),
        ));
        futures::future::poll_fn(|cx| {
            assert!(matches!(waiter.as_mut().poll(cx), Poll::Pending));
            Poll::Ready(())
        })
        .await;

        release_tx.send(()).unwrap();
        let loader_err = loader.await.unwrap_err();
        let waiter_err = waiter.await.unwrap_err();
        assert!(Arc::ptr_eq(&loader_err, &waiter_err));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.len().await, 0);

        // Immediate retry must invoke the initialiser rather than observing a cached error.
        let mut retry_init = pin!(async { Ok(Some(Bytes::from_static(b"ok"))) });
        let value = AsyncChunkCacheTraits::try_get_or_insert_with(&cache, key, retry_init.as_mut())
            .await
            .unwrap();
        assert_eq!(value, Some(Bytes::from_static(b"ok")));
        assert_eq!(cache.len().await, 1);
    }

    struct GrowingPartialDecoder {
        held: AtomicUsize,
    }

    impl zarrs_codec::ArrayPartialDecoderNoSubchunkingTraits for GrowingPartialDecoder {}

    impl ArrayPartialDecoderTraits for GrowingPartialDecoder {
        fn data_type(&self) -> &DataType {
            unreachable!()
        }

        fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
            Ok(true)
        }

        fn size_held(&self) -> usize {
            self.held.load(Ordering::Relaxed)
        }

        fn partial_decode(
            &self,
            _indexer: &dyn crate::array::Indexer,
            _options: &CodecOptions,
            _resources: &Resources,
        ) -> Result<ArrayBytes<'_>, CodecError> {
            unreachable!()
        }

        fn supports_partial_decode(&self) -> bool {
            false
        }
    }

    #[test]
    fn chunk_cache_partial_decoder_weight_growth_does_not_panic() {
        let cache = ChunkCachePartialDecoder::new_size_limit(100);
        let decoder = Arc::new(GrowingPartialDecoder {
            held: AtomicUsize::new(10),
        });
        let cached = SyncChunkCacheTraits::try_get_or_insert_with(
            &cache,
            ArrayIndices::from([0].as_slice()),
            &mut || Ok(decoder.clone()),
        )
        .unwrap();
        assert_eq!(cached.size_held(), 10);
        // Simulate a partial decoder retaining more bytes after insertion.
        decoder.held.store(500, Ordering::Relaxed);
        assert_eq!(cache.size(), 500);
        // Invalidation and eviction must not panic or underflow cache weight accounting.
        assert!(cache.invalidate_chunk(&[0]));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.size(), 0);

        // Also test eviction of a grown decoder.
        let decoder2 = Arc::new(GrowingPartialDecoder {
            held: AtomicUsize::new(10),
        });
        SyncChunkCacheTraits::try_get_or_insert_with(
            &cache,
            ArrayIndices::from([1].as_slice()),
            &mut || Ok(decoder2.clone()),
        )
        .unwrap();
        decoder2.held.store(500, Ordering::Relaxed);
        for i in 2..20u64 {
            SyncChunkCacheTraits::try_get_or_insert_with(
                &cache,
                ArrayIndices::from([i].as_slice()),
                &mut || {
                    Ok(Arc::new(GrowingPartialDecoder {
                        held: AtomicUsize::new(50),
                    }))
                },
            )
            .unwrap();
        }
    }

    struct DropQueriesCache {
        cache: Arc<ChunkCachePartialDecoder>,
        held: usize,
    }

    impl Drop for DropQueriesCache {
        fn drop(&mut self) {
            let _ = self.cache.len();
        }
    }

    impl zarrs_codec::ArrayPartialDecoderNoSubchunkingTraits for DropQueriesCache {}

    impl ArrayPartialDecoderTraits for DropQueriesCache {
        fn data_type(&self) -> &DataType {
            unreachable!()
        }

        fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
            Ok(true)
        }

        fn size_held(&self) -> usize {
            self.held
        }

        fn partial_decode(
            &self,
            _indexer: &dyn crate::array::Indexer,
            _options: &CodecOptions,
            _resources: &Resources,
        ) -> Result<ArrayBytes<'_>, CodecError> {
            unreachable!()
        }

        fn supports_partial_decode(&self) -> bool {
            false
        }
    }

    #[test]
    fn chunk_cache_drops_values_outside_shard_locks() {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let cache = Arc::new(ChunkCachePartialDecoder::new_size_limit(100));
            // Insert 5 small entries, then a single large entry that evicts >= 3 of them at once.
            for i in 0..5u64 {
                SyncChunkCacheTraits::try_get_or_insert_with(
                    cache.as_ref(),
                    ArrayIndices::from([i].as_slice()),
                    &mut || {
                        Ok(Arc::new(DropQueriesCache {
                            cache: cache.clone(),
                            held: 10,
                        }))
                    },
                )
                .unwrap();
            }
            SyncChunkCacheTraits::try_get_or_insert_with(
                cache.as_ref(),
                ArrayIndices::from([5].as_slice()),
                &mut || {
                    Ok(Arc::new(DropQueriesCache {
                        cache: cache.clone(),
                        held: 90,
                    }))
                },
            )
            .unwrap();

            // `invalidate()` (`clear`) must also drop values outside shard locks.
            let _ = cache.invalidate();
            assert_eq!(cache.len(), 0);
            sender.send(()).unwrap();
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("deadlock");
    }
}

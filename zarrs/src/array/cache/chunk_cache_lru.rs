#[cfg(feature = "async")]
use std::future::Future;
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
use crate::array::{ArrayError, ArrayIndices, Indexer};

type ChunkIndices = ArrayIndices;

trait CacheTraits<CT: ChunkCacheType> {
    fn len(&self) -> usize;

    /// Return the sum of the sizes of the cached values.
    fn size(&self) -> usize;

    fn remove(&self, chunk_indices: &[u64]) -> bool;

    /// Remove the entries at the indices of `indexer`, returning the number removed.
    fn remove_many(&self, indexer: &dyn Indexer) -> usize {
        indexer
            .iter_indices()
            .filter(|chunk_indices| self.remove(chunk_indices))
            .count()
    }

    fn clear(&self) -> usize;

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>;
}

trait CacheChunkLimitTraits {
    fn new_with_chunk_capacity(chunk_capacity: u64) -> Self;
}

trait CacheSizeLimitTraits {
    fn new_with_size_capacity(size_capacity: u64) -> Self;
}

#[cfg(feature = "async")]
trait AsyncCacheTraits<CT: ChunkCacheType> {
    fn len(&self) -> impl Future<Output = usize>;

    /// Return the sum of the sizes of the cached values.
    fn size(&self) -> impl Future<Output = usize>;

    fn remove(&self, chunk_indices: &[u64]) -> impl Future<Output = bool>;

    /// Remove the entries at the indices of `indexer`, returning the number removed.
    fn remove_many(&self, indexer: &dyn Indexer) -> impl Future<Output = usize> {
        async move {
            let mut removed = 0;
            for chunk_indices in indexer.iter_indices() {
                removed += usize::from(self.remove(&chunk_indices).await);
            }
            removed
        }
    }

    fn clear(&self) -> impl Future<Output = usize>;

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
        init: F,
    ) -> impl Future<Output = Result<CT, Arc<ArrayError>>>
    where
        F: Future<Output = Result<CT, ArrayError>>;
}

#[path = "chunk_cache_lru_quick_cache.rs"]
mod backend;

/// A chunk cache with a fixed chunk capacity.
///
/// The eviction policy is an implementation detail. It is currently [S3-FIFO](https://s3fifo.com/), which retains
/// frequently retrieved chunks through scans of the array.
pub struct ChunkCacheLruChunkLimit<CT: SyncChunkCacheType> {
    cache: backend::Cache<CT>,
}

impl<CT: SyncChunkCacheType> ChunkCacheLruChunkLimit<CT> {
    /// Create a new [`ChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(chunk_capacity: u64) -> Self {
        let cache = backend::Cache::new_with_chunk_capacity(chunk_capacity);
        Self { cache }
    }
}

/// A chunk cache with a fixed size capacity.
pub struct ChunkCacheLruSizeLimit<CT: SyncChunkCacheType> {
    cache: backend::Cache<CT>,
}

impl<CT: SyncChunkCacheType> ChunkCacheLruSizeLimit<CT> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    ///
    /// Each chunk counts as at least one byte, so chunks that hold no bytes (e.g. fill value chunks) are evicted too.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = backend::Cache::new_with_size_capacity(capacity);
        Self { cache }
    }
}

/// The [`SyncChunkCacheTraits`] methods of a chunk cache.
macro_rules! impl_ChunkCacheLruCommon {
    () => {
        type Value = CT;

        fn try_get_or_insert_with(
            &self,
            chunk_indices: ChunkIndices,
            f: &mut dyn FnMut() -> Result<Self::Value, ArrayError>,
        ) -> Result<Self::Value, Arc<ArrayError>> {
            CacheTraits::try_get_or_insert_with(&self.cache, chunk_indices, f)
        }

        fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
            CacheTraits::remove(&self.cache, chunk_indices)
        }

        fn invalidate_chunks(&self, chunks: &dyn Indexer) -> usize {
            CacheTraits::remove_many(&self.cache, chunks)
        }

        fn len(&self) -> usize {
            CacheTraits::len(&self.cache)
        }

        fn size(&self) -> usize {
            CacheTraits::size(&self.cache)
        }

        fn invalidate(&self) -> usize {
            CacheTraits::clear(&self.cache)
        }
    };
}

impl<CT: SyncChunkCacheType> SyncChunkCacheTraits for ChunkCacheLruChunkLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: SyncChunkCacheType> SyncChunkCacheTraits for ChunkCacheLruSizeLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

/// An asynchronous chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub struct AsyncChunkCacheLruChunkLimit<CT: AsyncChunkCacheType> {
    cache: backend::Cache<CT>,
}

#[cfg(feature = "async")]
impl<CT: AsyncChunkCacheType> AsyncChunkCacheLruChunkLimit<CT> {
    /// Create a new [`AsyncChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(chunk_capacity: u64) -> Self {
        let cache = backend::Cache::new_with_chunk_capacity(chunk_capacity);
        Self { cache }
    }
}

/// An asynchronous chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub struct AsyncChunkCacheLruSizeLimit<CT: AsyncChunkCacheType> {
    cache: backend::Cache<CT>,
}

#[cfg(feature = "async")]
impl<CT: AsyncChunkCacheType> AsyncChunkCacheLruSizeLimit<CT> {
    /// Create a new [`AsyncChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    ///
    /// Each chunk counts as at least one byte, so chunks that hold no bytes (e.g. fill value chunks) are evicted too.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = backend::Cache::new_with_size_capacity(capacity);
        Self { cache }
    }
}

#[cfg(feature = "async")]
/// Implement [`AsyncChunkCacheTraits`] for a chunk cache type.
macro_rules! impl_AsyncChunkCacheLru {
    ($ty:ident) => {
        #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
        #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
        impl<CT: AsyncChunkCacheType> AsyncChunkCacheTraits for $ty<CT> {
            type Value = CT;

            async fn try_get_or_insert_with(
                &self,
                chunk_indices: ChunkIndices,
                init: AsyncChunkCacheInit<'_, Self::Value>,
            ) -> Result<Self::Value, Arc<ArrayError>> {
                AsyncCacheTraits::try_get_or_insert_with(&self.cache, chunk_indices, init).await
            }

            async fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
                AsyncCacheTraits::remove(&self.cache, chunk_indices).await
            }

            async fn invalidate_chunks(&self, chunks: &dyn Indexer) -> usize {
                AsyncCacheTraits::remove_many(&self.cache, chunks).await
            }

            async fn len(&self) -> usize {
                AsyncCacheTraits::len(&self.cache).await
            }

            async fn size(&self) -> usize {
                AsyncCacheTraits::size(&self.cache).await
            }

            async fn invalidate(&self) -> usize {
                AsyncCacheTraits::clear(&self.cache).await
            }
        }
    };
}

#[cfg(feature = "async")]
impl_AsyncChunkCacheLru!(AsyncChunkCacheLruChunkLimit);

#[cfg(feature = "async")]
impl_AsyncChunkCacheLru!(AsyncChunkCacheLruSizeLimit);

/// An encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An encoded chunk cache with a fixed size capacity.
pub type ChunkCacheEncodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// A decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

/// A decoded chunk cache with a fixed size capacity.
pub type ChunkCacheDecodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

/// A partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimit =
    ChunkCacheLruChunkLimit<ChunkCacheTypePartialDecoder>;

/// A partial decoder chunk cache with a fixed size capacity.
pub type ChunkCachePartialDecoderLruSizeLimit =
    ChunkCacheLruSizeLimit<ChunkCacheTypePartialDecoder>;

/// An asynchronous encoded chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheEncodedLruChunkLimit = AsyncChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An asynchronous encoded chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheEncodedLruSizeLimit = AsyncChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// An asynchronous decoded chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheDecodedLruChunkLimit = AsyncChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

/// An asynchronous decoded chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheDecodedLruSizeLimit = AsyncChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

/// An asynchronous partial decoder chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCachePartialDecoderLruChunkLimit =
    AsyncChunkCacheLruChunkLimit<ChunkCacheTypeAsyncPartialDecoder>;

/// An asynchronous partial decoder chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCachePartialDecoderLruSizeLimit =
    AsyncChunkCacheLruSizeLimit<ChunkCacheTypeAsyncPartialDecoder>;

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use zarrs_codec::{ArrayBytes, ArrayPartialDecoderTraits, CodecError, CodecOptions, Resources};
    use zarrs_storage::Bytes;

    use super::*;
    use crate::array::DataType;

    /// A chunk cache of encoded chunks.
    type ChunkCacheEncoded = dyn SyncChunkCacheTraits<Value = ChunkCacheTypeEncoded>;

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
        let cache = ChunkCacheEncodedLruSizeLimit::new(4096 * SIZE as u64);
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
            &ChunkCacheEncodedLruChunkLimit::new(4) as &ChunkCacheEncoded,
            &ChunkCacheEncodedLruSizeLimit::new(100),
        ] {
            let calls = AtomicUsize::new(0);
            let (loader_started_tx, loader_started_rx) = std::sync::mpsc::channel();
            let waiters_ready = std::sync::Barrier::new(WAITERS + 1);
            std::thread::scope(|scope| {
                let loader = scope.spawn(|| {
                    SyncChunkCacheTraits::try_get_or_insert_with(
                        cache,
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
                                cache,
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
            retrieve(cache, 1, 10);
            assert_eq!(cache.len(), 1);
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_chunk_cache_shares_error_and_retries_immediately() {
        use std::pin::pin;
        use std::task::Poll;

        let cache = AsyncChunkCacheEncodedLruChunkLimit::new(4);
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
        let cache = ChunkCachePartialDecoderLruSizeLimit::new(100);
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
        cache: Arc<ChunkCachePartialDecoderLruSizeLimit>,
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
            let cache = Arc::new(ChunkCachePartialDecoderLruSizeLimit::new(100));
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

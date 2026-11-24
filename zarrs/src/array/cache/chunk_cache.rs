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
        fn from_config(config: backend::Config) -> Self {
            Self {
                cache: config.build(),
                config,
            }
        }

        /// Create a cache that holds at most `chunk_capacity` chunks.
        ///
        /// The cache is split into shards for concurrency, each holding a share of the chunk capacity, so it may evict
        /// a chunk before it holds `chunk_capacity` chunks, or hold slightly more.
        #[must_use]
        pub fn new_chunk_limit(chunk_capacity: u64) -> Self {
            Self::from_config(backend::Config::chunk_limit(chunk_capacity))
        }

        /// Create a cache that holds chunks up to a total of `size_capacity` bytes.
        ///
        /// Each chunk counts as at least one byte, so chunks that hold no bytes (e.g. fill value chunks) are evicted
        /// too. A chunk larger than the hot allocation of `size_capacity` (97% by default) is not cached; see
        /// [`with_hot_allocation`](Self::with_hot_allocation).
        ///
        /// `chunk_size_hint` is the expected size of a chunk in bytes. It sizes the record of recently evicted chunks
        /// (see [`with_ghost_allocation`](Self::with_ghost_allocation)), and splits the cache into shards for
        /// concurrency, each holding a share of `size_capacity` of at least 32 chunks of the hinted size. A chunk
        /// larger than the hot allocation of a shard is then not cached. Without a hint, the cache is a single shard,
        /// which is slower under concurrent retrievals but caches any chunk up to its hot allocation, and the record
        /// expects 4 KiB chunks.
        #[must_use]
        pub fn new_size_limit(size_capacity: u64, chunk_size_hint: Option<u64>) -> Self {
            Self::from_config(backend::Config::size_limit(size_capacity, chunk_size_hint))
        }

        /// Return an empty cache that reserves the fraction `hot_allocation` of its capacity for frequently retrieved
        /// chunks.
        ///
        /// The default is 0.97. Lower values favour recently retrieved chunks, as an LRU cache does; useful values
        /// are between 0.5 and 0.99. Values are clamped to `[0, 1]`, and NaN keeps the current value.
        ///
        /// A cache with a size limit does not cache a chunk larger than `hot_allocation` of its capacity (or of the
        /// capacity of a shard, see [`new_size_limit`](Self::new_size_limit)), so lower values also shrink the largest
        /// chunk it caches, and at 0 it caches nothing.
        #[must_use]
        pub fn with_hot_allocation(self, hot_allocation: f64) -> Self {
            Self::from_config(self.config.with_hot_allocation(hot_allocation))
        }

        /// Return an empty cache that remembers recently evicted chunks, up to the fraction `ghost_allocation` of the
        /// number of chunks it is expected to hold, and admits them as frequently retrieved if they are retrieved again.
        ///
        /// A cache with a chunk limit is expected to hold its chunk capacity. A cache with a size limit is expected to
        /// hold chunks of its chunk size hint (see [`new_size_limit`](Self::new_size_limit)), or else 4 KiB chunks
        /// (at least 1024), so that it remembers the evicted chunks of a working set of small chunks.
        ///
        /// The default is 0.5. Values are clamped to `[0, 1]`, and NaN keeps the current value. Each shard of the cache
        /// remembers at least one evicted chunk, even with an allocation of 0.
        #[must_use]
        pub fn with_ghost_allocation(self, ghost_allocation: f64) -> Self {
            Self::from_config(self.config.with_ghost_allocation(ghost_allocation))
        }
    };
}

/// A chunk cache with a capacity in chunks or bytes.
///
/// The eviction policy is [S3-FIFO](https://s3fifo.com/), which retains frequently retrieved chunks through scans of
/// the array. It can be tuned with [`with_hot_allocation`](Self::with_hot_allocation) and
/// [`with_ghost_allocation`](Self::with_ghost_allocation). Unlike an LRU cache:
/// - Once its hot allocation is full, the cache admits a newly retrieved chunk on probation and evicts it first. If no
///   other chunk is on probation, as in a cache that holds one chunk, the new chunk is evicted at once and cached only
///   when it is retrieved again, so the cache retrieves each new chunk twice.
/// - A cache with a size limit does not cache a chunk larger than its hot allocation (97% by default) of the capacity of
///   a shard. A cache sized to hold exactly `n` chunks holds fewer, and one sized to hold one chunk caches nothing, so
///   leave headroom.
///
/// Invalidating the whole cache drops its chunks while holding its internal locks, so a cached partial decoder must not
/// access the cache when dropped.
pub struct ChunkCache<CT: SyncChunkCacheType> {
    cache: backend::Cache<CT>,
    config: backend::Config,
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
    config: backend::Config,
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
    fn chunk_cache_one_slot_replaces_first_chunk() {
        for (cache, chunk_size) in [
            (ChunkCacheEncoded::new_chunk_limit(1), 10),
            (ChunkCacheEncoded::new_size_limit(100, Some(90)), 90),
            // A ghost allocation of 0 still keeps one ghost slot
            (
                ChunkCacheEncoded::new_chunk_limit(1).with_ghost_allocation(0.0),
                10,
            ),
            (
                ChunkCacheEncoded::new_size_limit(100, Some(90)).with_ghost_allocation(0.0),
                90,
            ),
            (
                ChunkCacheEncoded::new_size_limit(100, None).with_ghost_allocation(0.0),
                90,
            ),
        ] {
            assert_eq!(retrieve_counting(&cache, 0..1, chunk_size), 1);
            // First access to B evicts B to the ghost list; second promotes B and evicts A; third hits.
            assert_eq!(retrieve_counting(&cache, 1..2, chunk_size), 1);
            assert_eq!(retrieve_counting(&cache, 1..2, chunk_size), 1);
            assert_eq!(retrieve_counting(&cache, 1..2, chunk_size), 0);
            assert_eq!(cache.len(), 1);
        }
    }

    #[test]
    fn chunk_cache_size_limit() {
        let cache = ChunkCacheEncoded::new_size_limit(100, None);
        retrieve(&cache, 20, 10);
        assert!(cache.len() > 0);
        assert!(cache.size() <= 100);

        // A chunk larger than the size limit is not cached
        let cache = ChunkCacheEncoded::new_size_limit(100, None);
        retrieve(&cache, 1, 200);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn chunk_cache_allocations() {
        // Out of range and NaN allocations do not panic
        for allocation in [f64::NAN, f64::NEG_INFINITY, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let cache = ChunkCacheEncoded::new_chunk_limit(4)
                .with_hot_allocation(allocation)
                .with_ghost_allocation(allocation);
            retrieve(&cache, 10, 10);
            assert!((1..=4).contains(&cache.len()));

            let cache = ChunkCacheEncoded::new_size_limit(100, None)
                .with_hot_allocation(allocation)
                .with_ghost_allocation(allocation);
            retrieve(&cache, 20, 10);
            assert!(cache.size() <= 100);
            // A hot allocation of 0 (or clamped to 0) caches no chunk with a size limit
            if allocation > 0.0 || allocation.is_nan() {
                assert!(cache.len() > 0);
            } else {
                assert_eq!(cache.len(), 0);
            }
        }

        // A size-limited cache does not cache chunks larger than its hot allocation
        let cache = ChunkCacheEncoded::new_size_limit(100, None).with_hot_allocation(0.5);
        retrieve(&cache, 1, 60);
        assert_eq!(cache.len(), 0);
        retrieve(&cache, 1, 40);
        assert_eq!(cache.len(), 1);
    }

    /// Retrieve chunk `index` through `cache`, initialising it with `f` if it is uncached.
    fn retrieve_with(
        cache: &ChunkCacheEncoded,
        index: u64,
        f: &mut dyn FnMut() -> Result<ChunkCacheTypeEncoded, ArrayError>,
    ) -> ChunkCacheTypeEncoded {
        SyncChunkCacheTraits::try_get_or_insert_with(
            cache,
            ArrayIndices::from([index].as_slice()),
            f,
        )
        .unwrap()
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
        let cache = ChunkCacheEncoded::new_size_limit(4096 * SIZE as u64, None);
        let a = 0..4096;
        let b = 10_000..11_500;
        retrieve_counting(&cache, a.clone(), SIZE);
        retrieve_counting(&cache, a, SIZE);
        let misses: Vec<usize> = (0..4)
            .map(|_| retrieve_counting(&cache, b.clone(), SIZE))
            .collect();
        assert!(misses[3] < 150, "{misses:?}");
    }

    /// Run `f` on another thread, and panic if it does not complete within 30 s.
    fn complete_within_30s(f: impl FnOnce() + Send + 'static) {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            f();
            sender.send(()).unwrap();
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("deadlock");
    }

    #[test]
    fn chunk_cache_rayon_worker_does_not_wait_on_its_own_retrieval() {
        use rayon::prelude::*;
        complete_within_30s(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let cache = ChunkCacheEncoded::new_chunk_limit(4);
            pool.install(|| {
                (0..64).into_par_iter().for_each(|_| {
                    SyncChunkCacheTraits::try_get_or_insert_with(
                        &cache,
                        ArrayIndices::from([0].as_slice()),
                        &mut || {
                            // Runs other tasks retrieving the same chunk on this thread
                            rayon::yield_now();
                            Ok(None)
                        },
                    )
                    .unwrap();
                });
            });
        });
    }

    #[test]
    fn chunk_cache_rayon_worker_initialising_a_chunk_waits_for_another_chunk() {
        complete_within_30s(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let cache = ChunkCacheEncoded::new_chunk_limit(4);
            let calls = AtomicUsize::new(0);
            let (started, wait_started) = std::sync::mpsc::channel();
            let retrieve = |chunk: u64, f: &mut dyn FnMut() -> Result<_, ArrayError>| {
                SyncChunkCacheTraits::try_get_or_insert_with(
                    &cache,
                    ArrayIndices::from([chunk].as_slice()),
                    f,
                )
                .unwrap()
            };
            let (retrieve, calls) = (&retrieve, &calls);
            pool.scope(|scope| {
                scope.spawn(move |_| {
                    retrieve(1, &mut || {
                        calls.fetch_add(1, Ordering::Relaxed);
                        started.send(()).unwrap();
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        Ok(Some(Bytes::from_static(b"first")))
                    });
                });
                scope.spawn(move |_| {
                    // While initialising chunk 0, wait for the other retrieval of chunk 1
                    retrieve(0, &mut || {
                        wait_started.recv().unwrap();
                        let value = retrieve(1, &mut || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            Ok(Some(Bytes::from_static(b"second")))
                        });
                        assert_eq!(value, Some(Bytes::from_static(b"first")));
                        Ok(None)
                    });
                });
            });
            assert_eq!(calls.load(Ordering::Relaxed), 1);
        });
    }

    #[test]
    fn chunk_cache_rayon_worker_stops_waiting_on_another_retrieval() {
        complete_within_30s(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let cache = ChunkCacheEncoded::new_chunk_limit(4);
            let (started, wait_started) = std::sync::mpsc::channel();
            let (retrieved, wait_retrieved) = std::sync::mpsc::channel();
            let retrieve = |f: &mut dyn FnMut() -> Result<_, ArrayError>| {
                SyncChunkCacheTraits::try_get_or_insert_with(
                    &cache,
                    ArrayIndices::from([0].as_slice()),
                    f,
                )
                .unwrap()
            };
            pool.scope(|scope| {
                // A retrieval that completes only after another retrieval of the same chunk
                scope.spawn(move |_| {
                    retrieve(&mut || {
                        started.send(()).unwrap();
                        wait_retrieved.recv().unwrap();
                        Ok(Some(Bytes::from_static(b"first")))
                    });
                });
                scope.spawn(move |_| {
                    wait_started.recv().unwrap();
                    let value = retrieve(&mut || Ok(Some(Bytes::from_static(b"second"))));
                    assert_eq!(value, Some(Bytes::from_static(b"second")));
                    retrieved.send(()).unwrap();
                });
            });
            assert_eq!(cache.len(), 1);
        });
    }

    #[test]
    fn chunk_cache_chunk_size_hint() {
        // Chunks of about the hinted size are cached
        let cache = ChunkCacheEncoded::new_size_limit(1 << 20, Some(1024));
        retrieve(&cache, 2048, 1024);
        assert!(cache.len() > 512);
        assert!(cache.size() <= 1 << 20);

        // Without a hint the cache is one shard, so a chunk up to its hot allocation is cached. With a hint it is split
        // into shards of at most half the capacity, so a chunk larger than a shard is not cached.
        let cache = ChunkCacheEncoded::new_size_limit(1 << 20, None);
        retrieve(&cache, 1, 600 << 10);
        assert_eq!(cache.len(), 1);
        let cache = ChunkCacheEncoded::new_size_limit(1 << 20, Some(1024));
        retrieve(&cache, 1, 600 << 10);
        assert_eq!(cache.len(), 0);

        // Hinted caches stay within their documented byte ceiling even when capacity is odd
        let cache = ChunkCacheEncoded::new_size_limit(63, Some(1));
        retrieve(&cache, 100, 1);
        assert!(cache.size() <= 63);
        let cache = ChunkCacheEncoded::new_size_limit(65, Some(1));
        retrieve(&cache, 100, 1);
        assert!(cache.size() <= 65);

        // A zero hint does not panic
        let cache = ChunkCacheEncoded::new_size_limit(100, Some(0));
        retrieve(&cache, 20, 10);
        assert!(cache.size() <= 100);
    }

    #[test]
    fn chunk_cache_shares_retrieval_error() {
        const WAITERS: usize = 7;
        let chunk_indices = ArrayIndices::from([0].as_slice());
        for cache in [
            ChunkCacheEncoded::new_chunk_limit(4),
            ChunkCacheEncoded::new_size_limit(100, None),
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

    #[test]
    fn chunk_cache_invalidate_discards_retrieval_in_progress() {
        let cache = ChunkCacheEncoded::new_chunk_limit(4);
        let (started, wait_started) = std::sync::mpsc::channel();
        let (invalidated, wait_invalidated) = std::sync::mpsc::channel();
        let cache = &cache;
        std::thread::scope(|scope| {
            scope.spawn(move || {
                retrieve_with(cache, 0, &mut || {
                    started.send(()).unwrap();
                    wait_invalidated.recv().unwrap();
                    Ok(Some(Bytes::from_static(b"stale")))
                });
            });
            wait_started.recv().unwrap();
            cache.invalidate();
            invalidated.send(()).unwrap();
        });
        assert_eq!(cache.len(), 0);
        assert_eq!(retrieve_counting(cache, 0..1, 10), 1);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_chunk_cache_invalidate_discards_retrieval_in_progress() {
        use std::pin::pin;
        use std::task::Poll;

        let cache = AsyncChunkCacheEncoded::new_chunk_limit(4);
        let key = ArrayIndices::from([0].as_slice());
        let (release_tx, release_rx) = futures::channel::oneshot::channel::<()>();
        let mut loader_init = pin!(async {
            release_rx.await.unwrap();
            Ok(Some(Bytes::from_static(b"stale")))
        });
        let mut loader = pin!(AsyncChunkCacheTraits::try_get_or_insert_with(
            &cache,
            key.clone(),
            loader_init.as_mut(),
        ));
        // Poll the loader once so it installs the placeholder and blocks on `release_rx`.
        futures::future::poll_fn(|cx| {
            assert!(matches!(loader.as_mut().poll(cx), Poll::Pending));
            Poll::Ready(())
        })
        .await;

        cache.invalidate().await;
        release_tx.send(()).unwrap();
        loader.await.unwrap();
        assert_eq!(cache.len().await, 0);
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
        let cache = ChunkCachePartialDecoder::new_size_limit(100, None);
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
            let cache = Arc::new(ChunkCachePartialDecoder::new_size_limit(100, None));
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
            sender.send(()).unwrap();
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("deadlock");
    }

    #[test]
    #[ignore = "TODO: quick_cache `Cache::clear` drops values under the shard locks"]
    fn chunk_cache_invalidate_drops_values_outside_shard_locks() {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let cache = Arc::new(ChunkCachePartialDecoder::new_size_limit(100, None));
            SyncChunkCacheTraits::try_get_or_insert_with(
                cache.as_ref(),
                ArrayIndices::from([0].as_slice()),
                &mut || {
                    Ok(Arc::new(DropQueriesCache {
                        cache: cache.clone(),
                        held: 10,
                    }))
                },
            )
            .unwrap();
            assert_eq!(cache.invalidate(), 1);
            assert_eq!(cache.len(), 0);
            sender.send(()).unwrap();
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("deadlock");
    }

    #[test]
    fn chunk_cache_sync_miss_during_thread_local_teardown() {
        struct DropReadsCache(ChunkCacheEncoded);

        impl Drop for DropReadsCache {
            fn drop(&mut self) {
                let _ = SyncChunkCacheTraits::try_get_or_insert_with(
                    &self.0,
                    ArrayIndices::from([0].as_slice()),
                    &mut || Ok(None),
                );
            }
        }

        thread_local! {
            static EARLY_TLS: std::cell::RefCell<Option<DropReadsCache>> = const { std::cell::RefCell::new(None) };
        }

        std::thread::spawn(|| {
            // Register EARLY_TLS before any cache access so it is destroyed after `INITIALISING`.
            EARLY_TLS.with_borrow_mut(|slot| {
                *slot = Some(DropReadsCache(ChunkCacheEncoded::new_chunk_limit(2)));
            });
            let cache = ChunkCacheEncoded::new_chunk_limit(2);
            SyncChunkCacheTraits::try_get_or_insert_with(
                &cache,
                ArrayIndices::from([0].as_slice()),
                &mut || Ok(None),
            )
            .unwrap();
        })
        .join()
        .unwrap();
    }
}

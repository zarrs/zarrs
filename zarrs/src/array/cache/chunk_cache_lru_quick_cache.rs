//! The chunk cache backend, a [`quick_cache`] cache.
//!
//! Concurrent retrievals of an uncached chunk are coalesced: one caller initialises the chunk while the others wait.
//! If initialisation fails, the error is returned to that caller and every waiting caller, and is not cached.

use quick_cache::sync::GuardResult;
use quick_cache::{DefaultHashBuilder, Lifecycle, OptionsBuilder, Weighter};

use super::{
    Arc, ArrayError, CacheChunkLimitTraits, CacheSizeLimitTraits, CacheTraits, ChunkCacheType,
    ChunkIndices,
};
#[cfg(feature = "async")]
use super::{AsyncCacheTraits, Future};

/// The smallest typical chunk size in bytes, which sets the number of chunks a cache with a size limit is expected to
/// hold.
const SIZE_LIMIT_MIN_CHUNK_SIZE: u64 = 4096;

/// The minimum number of chunks a cache with a size limit is expected to hold.
const SIZE_LIMIT_MIN_ESTIMATED_CHUNKS: usize = 1024;

/// Return the number of chunks a cache with a size limit of `size_capacity` bytes is expected to hold, which sizes its
/// record of recently evicted chunks.
///
/// This assumes small chunks, so that the record can hold the evicted chunks of a working set of small chunks. The
/// record fills only as chunks are evicted, so with larger chunks it holds more evicted chunks than the cache holds
/// chunks, at worst about 1.5% of `size_capacity` in record entries.
fn size_limit_estimated_chunks(size_capacity: u64) -> usize {
    usize::try_from(size_capacity / SIZE_LIMIT_MIN_CHUNK_SIZE)
        .unwrap_or(usize::MAX)
        .max(SIZE_LIMIT_MIN_ESTIMATED_CHUNKS)
}

/// A cached chunk paired with its weight at insertion.
///
/// `quick_cache` requires a resident entry's weight to remain constant until eviction; storing the weight here keeps
/// accounting sound even if a cached partial decoder retains additional bytes on subsequent reads.
#[derive(Clone)]
struct CachedValue<CT> {
    value: CT,
    weight: u64,
}

/// The value passed through a placeholder: the cached chunk, or the error of a failed initialisation.
type Value<CT> = Result<CachedValue<CT>, Arc<ArrayError>>;

/// Weighs each cached chunk by its insertion weight, and an error as `u64::MAX` so it is never admitted.
#[derive(Clone, Copy, Default)]
pub(super) struct ChunkWeighter;

impl<CT: ChunkCacheType> Weighter<ChunkIndices, Value<CT>> for ChunkWeighter {
    fn weight(&self, _chunk_indices: &ChunkIndices, value: &Value<CT>) -> u64 {
        match value {
            Ok(cached) => cached.weight,
            // Strictly greater than `weight_target_hot` (capped at `u64::MAX - 1`), so `quick_cache`
            // hands the error to waiting callers and removes the placeholder atomically without ever
            // inserting it as resident.
            Err(_) => u64::MAX,
        }
    }
}

/// Deferred drops accumulated during a cache operation, released after the shard lock is dropped.
pub(super) struct DeferredDrops<K, V> {
    inline: [Option<(K, V)>; 2],
    overflow: Vec<(K, V)>,
}

impl<K, V> Default for DeferredDrops<K, V> {
    fn default() -> Self {
        Self {
            inline: [None, None],
            overflow: Vec::new(),
        }
    }
}

/// Lifecycle that defers dropping all evicted items until after the shard write lock is released.
///
/// Unlike `quick_cache::sync::DefaultLifecycle`, which drops the third and subsequent evicted items under the lock,
/// this spills beyond two items into a `Vec` so custom value destructors can never deadlock on the shard lock.
#[derive(Clone, Default)]
pub(super) struct DropOutsideLockLifecycle;

impl<K, V> Lifecycle<K, V> for DropOutsideLockLifecycle {
    type RequestState = DeferredDrops<K, V>;

    #[inline]
    fn on_evict(&self, state: &mut Self::RequestState, key: K, val: V) {
        if std::mem::needs_drop::<(K, V)>() {
            if state.inline[0].is_none() {
                state.inline[0] = Some((key, val));
            } else if state.inline[1].is_none() {
                state.inline[1] = Some((key, val));
            } else {
                state.overflow.push((key, val));
            }
        }
    }
}

type InnerCache<CT> = quick_cache::sync::Cache<
    ChunkIndices,
    Value<CT>,
    ChunkWeighter,
    DefaultHashBuilder,
    DropOutsideLockLifecycle,
>;

pub(super) struct Cache<CT> {
    cache: InnerCache<CT>,
    by_size: bool,
}

fn new_cache<CT: ChunkCacheType>(
    options: &mut OptionsBuilder,
    weight_capacity: u64,
    by_size: bool,
) -> Cache<CT> {
    // Reserve `u64::MAX` as strictly overweight so failed initialisations are never admitted.
    let weight_capacity = weight_capacity.min(u64::MAX - 1);
    let options = options
        .weight_capacity(weight_capacity)
        .build()
        .expect("the weight capacity and estimated items capacity are set");
    let cache = InnerCache::with_options(
        options,
        ChunkWeighter,
        DefaultHashBuilder::default(),
        DropOutsideLockLifecycle,
    );
    Cache { cache, by_size }
}

fn clear<CT: ChunkCacheType>(cache: &Cache<CT>) -> usize {
    // `drain()` releases each shard write lock before yielding items so values drop outside the lock.
    cache.cache.drain().count()
}

fn size<CT: ChunkCacheType>(cache: &Cache<CT>) -> usize {
    cache
        .cache
        .iter()
        .filter_map(|(_, value)| value.ok().map(|cached| cached.value.size()))
        .sum()
}

impl<CT: ChunkCacheType> Cache<CT> {
    fn wrap(&self, value: CT) -> CachedValue<CT> {
        let weight = if self.by_size {
            // A zero-weight chunk is never evicted, so weigh each chunk at least one byte.
            u64::try_from(value.size()).unwrap_or(u64::MAX).max(1)
        } else {
            1
        };
        CachedValue { value, weight }
    }
}

impl<CT: ChunkCacheType> CacheTraits<CT> for Cache<CT> {
    fn len(&self) -> usize {
        self.cache.len()
    }

    fn size(&self) -> usize {
        size(self)
    }

    fn remove(&self, chunk_indices: &[u64]) -> bool {
        self.cache.remove(chunk_indices).is_some()
    }

    fn clear(&self) -> usize {
        clear(self)
    }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        match self.cache.get_value_or_guard(&chunk_indices, None) {
            GuardResult::Value(value) => value.map(|cached| cached.value),
            GuardResult::Guard(guard) => {
                let value = f().map(|value| self.wrap(value)).map_err(Arc::new);
                // An error means that the chunk was invalidated while it was initialised, so it is not cached.
                let _ = guard.insert(value.clone());
                value.map(|cached| cached.value)
            }
            // Waiting without a timeout does not time out, but initialise the chunk uncached if it does
            GuardResult::Timeout => f().map_err(Arc::new),
        }
    }
}

#[cfg(feature = "async")]
impl<CT: ChunkCacheType> AsyncCacheTraits<CT> for Cache<CT> {
    async fn len(&self) -> usize {
        self.cache.len()
    }

    async fn size(&self) -> usize {
        size(self)
    }

    async fn remove(&self, chunk_indices: &[u64]) -> bool {
        self.cache.remove(chunk_indices).is_some()
    }

    async fn clear(&self) -> usize {
        clear(self)
    }

    async fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
        init: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: Future<Output = Result<CT, ArrayError>>,
    {
        match self.cache.get_value_or_guard_async(&chunk_indices).await {
            Ok(value) => value.map(|cached| cached.value),
            Err(guard) => {
                let value = init.await.map(|value| self.wrap(value)).map_err(Arc::new);
                // An error means that the chunk was invalidated while it was initialised, so it is not cached.
                let _ = guard.insert(value.clone());
                value.map(|cached| cached.value)
            }
        }
    }
}

impl<CT: ChunkCacheType> CacheChunkLimitTraits for Cache<CT> {
    fn new_with_chunk_capacity(chunk_capacity: u64) -> Self {
        // Shards are sized to hold at least 32 chunks each.
        let estimated_chunks = usize::try_from(chunk_capacity).unwrap_or(usize::MAX);
        new_cache(
            OptionsBuilder::new().estimated_items_capacity(estimated_chunks),
            chunk_capacity,
            false,
        )
    }
}

impl<CT: ChunkCacheType> CacheSizeLimitTraits for Cache<CT> {
    fn new_with_size_capacity(size_capacity: u64) -> Self {
        // A single shard, since a chunk larger than the capacity of a shard is never cached and the chunk size is
        // not known in advance.
        new_cache(
            OptionsBuilder::new()
                .estimated_items_capacity(size_limit_estimated_chunks(size_capacity))
                .shards(1),
            size_capacity,
            true,
        )
    }
}

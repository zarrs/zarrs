//! The chunk cache backend, a [`quick_cache`] cache.
//!
//! Concurrent retrievals of an uncached chunk are coalesced: one caller initialises the chunk while the others wait.
//! If initialisation fails, the error is returned to that caller and every waiting caller, and is not cached.
//!
//! A rayon worker thread must not block indefinitely on another caller's initialisation: the initialisation may be
//! lower on its own stack (a worker waiting on a `join` inside an initialisation runs other tasks, which may request
//! the same chunk), or wait on a task held by a worker that is itself blocked. So a rayon worker does not wait for
//! a chunk it is already initialising, and waits at most [`RAYON_WAIT_TIMEOUT`] (1 s) for another caller's
//! initialisation, then initialises the chunk uncached. A wait cycle between workers initialising different chunks
//! thus stalls for at most 1 s, and rayon workers do not share an initialisation that takes longer than 1 s.

#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;
use std::time::Duration;

use quick_cache::sync::GuardResult;
use quick_cache::{DefaultHashBuilder, Lifecycle, OptionsBuilder, Weighter};

use super::{Arc, ArrayError, ChunkCacheType, ChunkIndices};

/// The smallest typical chunk size in bytes, which sets the number of chunks a cache with a size limit is expected to
/// hold.
const SIZE_LIMIT_MIN_CHUNK_SIZE: u64 = 4096;

/// The minimum number of chunks a cache with a size limit is expected to hold.
const SIZE_LIMIT_MIN_ESTIMATED_CHUNKS: usize = 1024;

/// Matches `quick_cache`'s default ghost allocation.
const DEFAULT_GHOST_ALLOCATION: f64 = 0.5;

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

/// Return the default shard count before shrinking for small caches, matching `quick_cache`.
fn default_shards() -> usize {
    let parallelism = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(1 << 16);
    (parallelism * 4).next_power_of_two()
}

/// A cached chunk paired with its weight at insertion.
///
/// `quick_cache` requires a resident entry's weight to remain constant until eviction; storing the weight here keeps
/// accounting sound even if a cached partial decoder retains additional bytes on subsequent reads.
#[derive(Clone)]
pub(super) struct CachedValue<CT> {
    pub(super) value: CT,
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

pub(super) type InnerCache<CT> = quick_cache::sync::Cache<
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

/// The configuration of a cache, kept so that it can be rebuilt with other eviction options.
#[derive(Clone, Copy)]
pub(super) struct Config {
    weight_capacity: u64,
    by_size: bool,
    hot_allocation: Option<f64>,
    ghost_allocation: Option<f64>,
    chunk_size_hint: Option<u64>,
}

/// Clamp an allocation to `[0, 1]`, or return `None` for NaN, since `quick_cache` panics on either.
fn clamp_allocation(allocation: f64) -> Option<f64> {
    (!allocation.is_nan()).then(|| allocation.clamp(0.0, 1.0))
}

impl Config {
    pub(super) fn chunk_limit(chunk_capacity: u64) -> Self {
        Self {
            weight_capacity: chunk_capacity,
            by_size: false,
            hot_allocation: None,
            ghost_allocation: None,
            chunk_size_hint: None,
        }
    }

    pub(super) fn size_limit(size_capacity: u64, chunk_size_hint: Option<u64>) -> Self {
        Self {
            weight_capacity: size_capacity,
            by_size: true,
            hot_allocation: None,
            ghost_allocation: None,
            chunk_size_hint: chunk_size_hint.map(|chunk_size| chunk_size.max(1)),
        }
    }

    pub(super) fn with_hot_allocation(mut self, hot_allocation: f64) -> Self {
        self.hot_allocation = clamp_allocation(hot_allocation).or(self.hot_allocation);
        self
    }

    pub(super) fn with_ghost_allocation(mut self, ghost_allocation: f64) -> Self {
        self.ghost_allocation = clamp_allocation(ghost_allocation).or(self.ghost_allocation);
        self
    }

    pub(super) fn build<CT: ChunkCacheType>(&self) -> Cache<CT> {
        // Reserve `u64::MAX` as strictly overweight so failed initialisations are never admitted.
        let max_weight_capacity = self.weight_capacity.min(u64::MAX - 1);
        let mut ghost_allocation = self.ghost_allocation.unwrap_or(DEFAULT_GHOST_ALLOCATION);

        let (num_shards, weight_capacity, estimated_items_capacity) =
            if let (true, Some(chunk_size)) = (self.by_size, self.chunk_size_hint) {
                // Shards hold at least 32 chunks of the hinted size each.
                let items = usize::try_from(max_weight_capacity / chunk_size)
                    .unwrap_or(usize::MAX)
                    .max(1);
                let mut num_shards = default_shards();
                while num_shards > 1 && items / num_shards < 32 {
                    num_shards /= 2;
                }
                // Round down to a multiple of `num_shards` so per-shard ceiling division in `quick_cache`
                // cannot exceed `self.weight_capacity`.
                let weight_capacity =
                    max_weight_capacity - (max_weight_capacity % num_shards as u64);
                (num_shards, weight_capacity, items)
            } else if self.by_size {
                // A single shard, since a chunk larger than the capacity of a shard is never cached and the chunk size
                // is not known in advance.
                (
                    1,
                    max_weight_capacity,
                    size_limit_estimated_chunks(max_weight_capacity),
                )
            } else {
                // Shards are sized to hold at least 32 chunks each.
                let items = usize::try_from(max_weight_capacity).unwrap_or(usize::MAX);
                let mut num_shards = default_shards();
                while num_shards > 1 && items / num_shards < 32 {
                    num_shards /= 2;
                }
                (num_shards, max_weight_capacity, items)
            };

        // Keep at least one ghost slot per shard of a non-empty cache, even with a ghost allocation of 0. Otherwise a
        // shard whose hot share is full (e.g. `new_chunk_limit(1)`) evicts each new chunk without a record that could
        // promote it, so it never caches another chunk. `quick_cache` holds `floor(shard_items * ghost_allocation)`
        // ghosts per shard; the 1.5 keeps that at least 1 despite rounding.
        if weight_capacity > 0 {
            let shard_items = estimated_items_capacity.div_ceil(num_shards).max(1);
            if (shard_items as f64 * ghost_allocation) as usize == 0 {
                ghost_allocation = (1.5 / shard_items as f64).min(1.0);
            }
        }

        let mut options = OptionsBuilder::new();
        options
            .shards(num_shards)
            .weight_capacity(weight_capacity)
            .estimated_items_capacity(estimated_items_capacity);
        if let Some(hot_allocation) = self.hot_allocation {
            options.hot_allocation(hot_allocation);
        }
        options.ghost_allocation(ghost_allocation);
        let options = options
            .build()
            .expect("the weight capacity and estimated items capacity are set");
        let cache = InnerCache::with_options(
            options,
            ChunkWeighter,
            DefaultHashBuilder::default(),
            DropOutsideLockLifecycle,
        );
        Cache {
            cache,
            by_size: self.by_size,
        }
    }
}

/// How long a rayon worker waits for another caller's initialisation of a chunk before initialising it uncached.
#[cfg(not(target_arch = "wasm32"))]
const RAYON_WAIT_TIMEOUT: Duration = Duration::from_secs(1);

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    /// The cache addresses and chunk indices of the chunks this thread is initialising, innermost last.
    static INITIALISING: RefCell<Vec<(usize, ChunkIndices)>> = const { RefCell::new(Vec::new()) };
}

/// Records that this thread is initialising a chunk while it lives.
#[cfg(not(target_arch = "wasm32"))]
struct Initialising(bool);

#[cfg(not(target_arch = "wasm32"))]
impl Initialising {
    fn new(cache: usize, chunk_indices: &ChunkIndices) -> Self {
        let pushed = INITIALISING
            .try_with(|chunks| chunks.borrow_mut().push((cache, chunk_indices.clone())))
            .is_ok();
        Self(pushed)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Initialising {
    fn drop(&mut self) {
        if self.0 {
            let _ = INITIALISING.try_with(|chunks| {
                chunks.borrow_mut().pop();
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
struct Initialising;

#[cfg(target_arch = "wasm32")]
impl Initialising {
    fn new(_cache: usize, _chunk_indices: &ChunkIndices) -> Self {
        Self
    }
}

/// Return how long to wait for another caller's initialisation of the chunk at `chunk_indices` of the cache at address
/// `cache`, or [`None`] to wait until it completes.
#[cfg(not(target_arch = "wasm32"))]
fn wait_timeout(cache: usize, chunk_indices: &ChunkIndices) -> Option<Duration> {
    if rayon::current_thread_index().is_none() {
        None
    } else if INITIALISING
        .try_with(|chunks| {
            chunks
                .borrow()
                .iter()
                .any(|(initialising_cache, initialising_chunk)| {
                    *initialising_cache == cache && initialising_chunk == chunk_indices
                })
        })
        .unwrap_or(false)
    {
        Some(Duration::ZERO)
    } else {
        Some(RAYON_WAIT_TIMEOUT)
    }
}

/// `Instant` is unsupported on `wasm32-unknown-unknown`, and rayon does not run a thread pool there.
#[cfg(target_arch = "wasm32")]
fn wait_timeout(_cache: usize, _chunk_indices: &ChunkIndices) -> Option<Duration> {
    None
}

impl<CT: ChunkCacheType> Cache<CT> {
    pub(super) fn wrap(&self, value: CT) -> CachedValue<CT> {
        let weight = if self.by_size {
            // A zero-weight chunk is never evicted, so weigh each chunk at least one byte.
            u64::try_from(value.size()).unwrap_or(u64::MAX).max(1)
        } else {
            1
        };
        CachedValue { value, weight }
    }

    #[cfg(feature = "async")]
    pub(super) fn inner(&self) -> &InnerCache<CT> {
        &self.cache
    }

    pub(super) fn len(&self) -> usize {
        self.cache.len()
    }

    pub(super) fn size(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|(_, value)| value.ok().map(|cached| cached.value.size()))
            .sum()
    }

    pub(super) fn remove(&self, chunk_indices: &[u64]) -> bool {
        self.cache.remove(chunk_indices).is_some()
    }

    pub(super) fn clear(&self) -> usize {
        // `clear()` invalidates placeholders, so a retrieval in progress is not cached, as with `remove`. `drain()` does
        // not, as it removes resident chunks only.
        // TODO: `clear()` drops values under the shard write locks, so a value whose `Drop` accesses this cache
        // deadlocks. Fix upstream in `quick_cache` rather than working around it here.
        let count = self.cache.len();
        self.cache.clear();
        count
    }

    pub(super) fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        if let Some(value) = self.cache.get(&chunk_indices) {
            return value.map(|cached| cached.value);
        }
        let address = std::ptr::from_ref(self) as usize;
        let timeout = wait_timeout(address, &chunk_indices);
        match self.cache.get_value_or_guard(&chunk_indices, timeout) {
            GuardResult::Value(value) => value.map(|cached| cached.value),
            GuardResult::Guard(guard) => {
                let value = {
                    let _initialising = timeout
                        .is_some()
                        .then(|| Initialising::new(address, &chunk_indices));
                    f().map(|value| self.wrap(value)).map_err(Arc::new)
                };
                // An error means that the chunk was invalidated while it was initialised, so it is not cached.
                let _ = guard.insert(value.clone());
                value.map(|cached| cached.value)
            }
            GuardResult::Timeout => {
                let _initialising = timeout
                    .is_some()
                    .then(|| Initialising::new(address, &chunk_indices));
                f().map_err(Arc::new)
            }
        }
    }
}

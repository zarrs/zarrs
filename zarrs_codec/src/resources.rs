//! Resources available to an operation.

use std::sync::Arc;
use zarrs_storage::ReadCoalescing;

#[cfg(not(target_family = "wasm"))]
use crate::plan::IoPool;

/// The default per-operation I/O target.
const DEFAULT_IO_TARGET: usize = 32;

/// The default limit on fetched but not yet consumed bytes (256 MiB).
const DEFAULT_FETCHED_BYTES_LIMIT: usize = 256 * 1024 * 1024;

/// How synchronous operations dispatch reads. See [`Resources::read_dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadDispatch {
    /// Read on the compute thread that will decode the bytes read.
    ///
    /// This avoids a thread hop, but a read occupies a compute thread while it waits.
    Inline,
    /// Read on an I/O thread pool ([`Resources::io_pool`]).
    ///
    /// This allows more concurrent reads than compute threads, and compute threads never wait for reads.
    Pooled,
}

/// Resources available to an encode/decode or array operation.
///
/// These are passed at each call and control how much of the system an operation may use.
/// They are distinct from [`CodecOptions`](super::CodecOptions), which control the behaviour of an operation.
///
/// Compute (encoding and decoding) and I/O (storage requests) have separate limits.
///
/// The default values are:
/// - `compute_target`: number of threads available to Rayon
/// - `io_target`: `32`
/// - `io_limiter`: none
/// - `fetched_bytes_limit`: 256 MiB
/// - `read_dispatch`: none (adapt to the latency of reads)
/// - `read_coalescing`: none (use the hint of the store)
/// - `io_pool`: none (use the global I/O pool)
#[derive(Debug, Clone)]
pub struct Resources {
    compute_target: usize,
    io_target: usize,
    io_limiter: Option<IoLimiter>,
    fetched_bytes_limit: usize,
    read_dispatch: Option<ReadDispatch>,
    read_coalescing: Option<ReadCoalescing>,
    #[cfg(not(target_family = "wasm"))]
    io_pool: Option<Arc<IoPool>>,
}

impl Default for Resources {
    fn default() -> Self {
        Self {
            compute_target: rayon::current_num_threads(),
            io_target: DEFAULT_IO_TARGET,
            io_limiter: None,
            fetched_bytes_limit: DEFAULT_FETCHED_BYTES_LIMIT,
            read_dispatch: None,
            read_coalescing: None,
            #[cfg(not(target_family = "wasm"))]
            io_pool: None,
        }
    }
}

impl Resources {
    /// Return the compute target.
    ///
    /// This is the number of concurrent compute (encoding/decoding) operations to target for an operation.
    /// Limiting concurrent operations is needed to reduce memory usage and improve performance.
    /// Concurrency is unconstrained if the compute target is set to zero.
    #[must_use]
    pub fn compute_target(&self) -> usize {
        self.compute_target
    }

    /// Set the compute target.
    pub fn set_compute_target(&mut self, compute_target: usize) -> &mut Self {
        self.compute_target = compute_target;
        self
    }

    /// Set the compute target.
    #[must_use]
    pub fn with_compute_target(mut self, compute_target: usize) -> Self {
        self.compute_target = compute_target;
        self
    }

    /// Return the I/O target.
    ///
    /// This is the maximum number of concurrent storage requests of an operation.
    /// Concurrency is unconstrained if the I/O target is set to zero.
    ///
    /// Use an [`IoLimiter`] to limit the storage requests of many operations together.
    #[must_use]
    pub fn io_target(&self) -> usize {
        self.io_target
    }

    /// Set the I/O target.
    pub fn set_io_target(&mut self, io_target: usize) -> &mut Self {
        self.io_target = io_target;
        self
    }

    /// Set the I/O target.
    #[must_use]
    pub fn with_io_target(mut self, io_target: usize) -> Self {
        self.io_target = io_target;
        self
    }

    /// Return the I/O limiter, if set.
    ///
    /// Every storage request of an operation holds a permit of the I/O limiter.
    /// Operations sharing a limiter, including concurrent synchronous and asynchronous operations,
    /// together make at most as many concurrent storage requests as it has permits.
    #[must_use]
    pub fn io_limiter(&self) -> Option<&IoLimiter> {
        self.io_limiter.as_ref()
    }

    /// Set the I/O limiter.
    pub fn set_io_limiter(&mut self, io_limiter: Option<IoLimiter>) -> &mut Self {
        self.io_limiter = io_limiter;
        self
    }

    /// Set the I/O limiter.
    #[must_use]
    pub fn with_io_limiter(mut self, io_limiter: Option<IoLimiter>) -> Self {
        self.io_limiter = io_limiter;
        self
    }

    /// Acquire a permit of the I/O limiter, if set, blocking the current thread until one is available.
    ///
    /// On wasm, which cannot block on a permit held by another operation, no permit is acquired.
    pub fn io_permit_blocking(&self) -> Option<IoPermit> {
        #[cfg(not(target_family = "wasm"))]
        return self.io_limiter.as_ref().map(IoLimiter::acquire_blocking);
        #[cfg(target_family = "wasm")]
        None
    }

    /// Acquire a permit of the I/O limiter, if set, waiting asynchronously until one is available.
    pub async fn io_permit(&self) -> Option<IoPermit> {
        match &self.io_limiter {
            Some(limiter) => Some(limiter.acquire().await),
            None => None,
        }
    }

    /// The compute target, or [`usize::MAX`] if it is unconstrained (zero).
    #[must_use]
    pub fn compute_limit(&self) -> usize {
        unconstrained_if_zero(self.compute_target)
    }

    /// The I/O target, or [`usize::MAX`] if it is unconstrained (zero).
    #[must_use]
    pub fn io_limit(&self) -> usize {
        unconstrained_if_zero(self.io_target)
    }

    /// These resources with an equal share (at least one) of the compute target between `shares` concurrent operations.
    ///
    /// An unconstrained compute target remains unconstrained.
    #[must_use]
    pub fn with_compute_share(&self, shares: usize) -> Self {
        let mut resources = self.clone();
        if self.compute_target != 0 {
            resources.compute_target = (self.compute_target / shares.max(1)).max(1);
        }
        resources
    }

    /// Return the fetched bytes limit.
    ///
    /// This is a soft limit on the bytes that an operation has retrieved from storage but not yet decoded.
    /// An operation stops issuing new storage requests while it would exceed the limit.
    /// A single request larger than the limit is still issued once nothing else is buffered,
    /// so the limit never prevents progress.
    ///
    /// The size of a request is estimated from the operation (e.g. the size of a subchunk in a shard),
    /// otherwise from the mean size of the requests of the operation so far.
    #[must_use]
    pub fn fetched_bytes_limit(&self) -> usize {
        self.fetched_bytes_limit
    }

    /// Set the fetched bytes limit.
    pub fn set_fetched_bytes_limit(&mut self, fetched_bytes_limit: usize) -> &mut Self {
        self.fetched_bytes_limit = fetched_bytes_limit;
        self
    }

    /// Set the fetched bytes limit.
    #[must_use]
    pub fn with_fetched_bytes_limit(mut self, fetched_bytes_limit: usize) -> Self {
        self.fetched_bytes_limit = fetched_bytes_limit;
        self
    }

    /// Return the read dispatch override, if set.
    ///
    /// Unless this is set, synchronous operations read on compute threads while reads are fast, and
    /// on the I/O pool once reads are slower than handing them to the pool.
    /// For example, set [`ReadDispatch::Pooled`] for storage whose first reads are known to be slow.
    #[must_use]
    pub fn read_dispatch(&self) -> Option<ReadDispatch> {
        self.read_dispatch
    }

    /// Set the read dispatch override.
    pub fn set_read_dispatch(&mut self, read_dispatch: Option<ReadDispatch>) -> &mut Self {
        self.read_dispatch = read_dispatch;
        self
    }

    /// Set the read dispatch override.
    #[must_use]
    pub fn with_read_dispatch(mut self, read_dispatch: Option<ReadDispatch>) -> Self {
        self.read_dispatch = read_dispatch;
        self
    }

    /// Return the read coalescing override, if set.
    ///
    /// Reads of nearby byte ranges of the same value that are queued together (e.g. the subchunks of a shard) are
    /// coalesced into one storage request as set by the [`ReadCoalescing`] hint of the store
    /// ([`ReadableStorageTraits::read_coalescing`](zarrs_storage::ReadableStorageTraits::read_coalescing)), unless
    /// overridden. Set a [`ReadCoalescing`] with a maximum size of zero to disable coalescing.
    #[must_use]
    pub fn read_coalescing(&self) -> Option<ReadCoalescing> {
        self.read_coalescing
    }

    /// Set the read coalescing override.
    pub fn set_read_coalescing(&mut self, read_coalescing: Option<ReadCoalescing>) -> &mut Self {
        self.read_coalescing = read_coalescing;
        self
    }

    /// Set the read coalescing override.
    #[must_use]
    pub fn with_read_coalescing(mut self, read_coalescing: Option<ReadCoalescing>) -> Self {
        self.read_coalescing = read_coalescing;
        self
    }

    /// Return the I/O pool for pooled reads of synchronous operations, if set.
    ///
    /// If unset, the global I/O pool ([`IoPool::global`]) is used.
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn io_pool(&self) -> Option<&IoPool> {
        self.io_pool.as_deref()
    }

    /// Set the I/O pool.
    #[cfg(not(target_family = "wasm"))]
    pub fn set_io_pool(&mut self, io_pool: Option<Arc<IoPool>>) -> &mut Self {
        self.io_pool = io_pool;
        self
    }

    /// Set the I/O pool.
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn with_io_pool(mut self, io_pool: Option<Arc<IoPool>>) -> Self {
        self.io_pool = io_pool;
        self
    }
}

fn unconstrained_if_zero(target: usize) -> usize {
    if target == 0 { usize::MAX } else { target }
}

/// A limit on concurrent storage requests that is shared between operations.
///
/// Clones share the same permits.
/// Permits can be acquired by both synchronous and asynchronous operations.
#[derive(Debug, Clone)]
pub struct IoLimiter(Arc<async_lock::Semaphore>);

impl IoLimiter {
    /// Create an I/O limiter that allows `permits` concurrent storage requests.
    ///
    /// # Panics
    /// Panics if `permits` is zero.
    #[must_use]
    pub fn new(permits: usize) -> Self {
        assert!(permits > 0, "an I/O limiter must have at least one permit");
        Self(Arc::new(async_lock::Semaphore::new(permits)))
    }

    /// Acquire a permit, waiting asynchronously until one is available.
    pub async fn acquire(&self) -> IoPermit {
        IoPermit(self.0.acquire_arc().await)
    }

    /// Acquire a permit, blocking the current thread until one is available.
    #[cfg(not(target_family = "wasm"))]
    pub fn acquire_blocking(&self) -> IoPermit {
        IoPermit(self.0.acquire_arc_blocking())
    }

    /// Acquire a permit if one is available.
    pub fn try_acquire(&self) -> Option<IoPermit> {
        self.0.try_acquire_arc().map(IoPermit)
    }
}

/// A permit of an [`IoLimiter`], released when dropped.
#[derive(Debug)]
#[must_use]
pub struct IoPermit(#[allow(dead_code)] async_lock::SemaphoreGuardArc);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_limiter_is_shared_by_clones() {
        let resources = Resources::default().with_io_limiter(Some(IoLimiter::new(2)));
        let clone = resources.clone();
        let a = resources.io_limiter().unwrap().try_acquire().unwrap();
        let _b = clone.io_limiter().unwrap().try_acquire().unwrap();
        assert!(resources.io_limiter().unwrap().try_acquire().is_none());
        drop(a);
        assert!(clone.io_limiter().unwrap().try_acquire().is_some());
    }

    #[test]
    fn io_limiter_sync_and_async() {
        let limiter = IoLimiter::new(1);
        let permit = futures::executor::block_on(limiter.acquire());
        assert!(limiter.try_acquire().is_none());
        drop(permit);
        let _permit = limiter.acquire_blocking();
        assert!(limiter.try_acquire().is_none());
    }
}

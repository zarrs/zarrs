//! Shared benchmark utilities.
#![allow(dead_code, unreachable_pub, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use zarrs::storage::byte_range::ByteRangeIterator;
#[cfg(feature = "async")]
use zarrs::storage::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableStorageTraits,
    AsyncWritableStorageTraits,
};
use zarrs::storage::{
    ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};
use zarrs_storage::CowBytes;

/// A storage adapter that adds a fixed latency to every read and write request.
///
/// It also records the number of requests and the peak number of requests in flight,
/// so benchmarks can report the I/O concurrency an operation achieved.
///
/// Synchronous requests block the calling thread with [`std::thread::sleep`].
/// Asynchronous requests wait on [`tokio::time::sleep`], so they require a tokio runtime with the time driver.
#[derive(Debug)]
pub struct LatencyStore<TStorage: ?Sized> {
    storage: Arc<TStorage>,
    latency: Duration,
    requests: AtomicUsize,
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

impl<TStorage: ?Sized> LatencyStore<TStorage> {
    /// Wrap `storage`, adding `latency` to each request.
    pub fn new(storage: Arc<TStorage>, latency: Duration) -> Self {
        Self {
            storage,
            latency,
            requests: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            peak_in_flight: AtomicUsize::new(0),
        }
    }

    /// The number of requests since the last [`reset`](Self::reset).
    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }

    /// The peak number of concurrent requests since the last [`reset`](Self::reset).
    pub fn peak_in_flight(&self) -> usize {
        self.peak_in_flight.load(Ordering::Relaxed)
    }

    /// Reset the request counters.
    pub fn reset(&self) {
        self.requests.store(0, Ordering::Relaxed);
        self.peak_in_flight.store(0, Ordering::Relaxed);
    }

    fn begin(&self) -> InFlight<'_> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let in_flight = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_in_flight.fetch_max(in_flight, Ordering::Relaxed);
        InFlight(&self.in_flight)
    }

    fn wait_sync(&self) -> InFlight<'_> {
        let guard = self.begin();
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency);
        }
        guard
    }

    #[cfg(feature = "async")]
    async fn wait_async(&self) -> InFlight<'_> {
        let guard = self.begin();
        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }
        guard
    }
}

struct InFlight<'a>(&'a AtomicUsize);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ReadableStorageTraits for LatencyStore<TStorage> {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let _guard = self.wait_sync();
        self.storage.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let _guard = self.wait_sync();
        self.storage.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.storage.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

impl<TStorage: ?Sized + ListableStorageTraits> ListableStorageTraits for LatencyStore<TStorage> {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.storage.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.storage.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.storage.list_dir(prefix)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.storage.size_prefix(prefix)
    }
}

impl<TStorage: ?Sized + WritableStorageTraits> WritableStorageTraits for LatencyStore<TStorage> {
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        let _guard = self.wait_sync();
        self.storage.set(key, value)
    }

    fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        let _guard = self.wait_sync();
        self.storage.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        let _guard = self.wait_sync();
        self.storage.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let _guard = self.wait_sync();
        self.storage.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.storage.supports_set_partial()
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<TStorage: ?Sized + AsyncReadableStorageTraits> AsyncReadableStorageTraits
    for LatencyStore<TStorage>
{
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let _guard = self.wait_async().await;
        self.storage.get(key).await
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        let _guard = self.wait_async().await;
        self.storage.get_partial_many(key, byte_ranges).await
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.storage.size_key(key).await
    }

    fn supports_get_partial(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<TStorage: ?Sized + AsyncListableStorageTraits> AsyncListableStorageTraits
    for LatencyStore<TStorage>
{
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        self.storage.list().await
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.storage.list_prefix(prefix).await
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.storage.list_dir(prefix).await
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.storage.size_prefix(prefix).await
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<TStorage: ?Sized + AsyncWritableStorageTraits> AsyncWritableStorageTraits
    for LatencyStore<TStorage>
{
    async fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        let _guard = self.wait_async().await;
        self.storage.set(key, value).await
    }

    async fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        let _guard = self.wait_async().await;
        self.storage.set_partial_many(key, offset_values).await
    }

    async fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        let _guard = self.wait_async().await;
        self.storage.erase(key).await
    }

    async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let _guard = self.wait_async().await;
        self.storage.erase_prefix(prefix).await
    }

    fn supports_set_partial(&self) -> bool {
        self.storage.supports_set_partial()
    }
}

/// Deterministic pseudo-random bytes from a simple LCG.
pub fn random_bytes(n: usize) -> Vec<u8> {
    let mut state: u64 = 0x1234_5678_9abc_def0;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// Compressible pseudo-random `u16` values (smooth with noise in the low bits).
pub fn compressible_u16(n: usize) -> Vec<u16> {
    random_bytes(n)
        .into_iter()
        .enumerate()
        .map(|(i, noise)| ((i / 64) as u16).wrapping_mul(7) ^ u16::from(noise & 0x0f))
        .collect()
}

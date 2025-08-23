//! A storage transformer which records performance metrics.

use crate::{
    byte_range::ByteRange, Bytes, ListableStorageTraits, MaybeBytes, ReadableStorageTraits,
    StorageError, StoreKey, StoreKeyOffsetValue, StoreKeyRange, StoreKeys, StoreKeysPrefixes,
    StorePrefix, WritableStorageTraits,
};

#[cfg(feature = "async")]
use crate::{
    AsyncBytes, AsyncListableStorageTraits, AsyncReadableStorageTraits, AsyncWritableStorageTraits,
    MaybeAsyncBytes,
};

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// The performance metrics storage transformer. Accumulates metrics, such as bytes read and written.
///
/// It is intended to aid in testing by allowing the application to validate that metrics (e.g., bytes read/written, total read/write operations) match expected values for specific operations.
///
/// ### Example
/// ```rust
/// # use std::sync::{Arc, Mutex};
/// # use zarrs_storage::store::MemoryStore;
/// # use zarrs_storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
/// let store = Arc::new(MemoryStore::new());
/// let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));
/// // do some store operations...
/// // assert_eq!(store.bytes_read(), ...);
/// // assert_eq!(store.bytes_written(), ...);
/// // assert_eq!(store.reads(), ...);
/// // assert_eq!(store.writes(), ...);
/// // assert_eq!(store.keys_erased(), ...);
/// ```
#[derive(Debug)]
pub struct PerformanceMetricsStorageAdapter<TStorage: ?Sized> {
    storage: Arc<TStorage>,
    bytes_read: AtomicUsize,
    bytes_written: AtomicUsize,
    reads: AtomicUsize,
    writes: AtomicUsize,
    keys_erased: AtomicUsize,
}

impl<TStorage: ?Sized> PerformanceMetricsStorageAdapter<TStorage> {
    /// Create a new performance metrics storage transformer.
    #[must_use]
    pub fn new(storage: Arc<TStorage>) -> Self {
        Self {
            storage,
            bytes_read: AtomicUsize::default(),
            bytes_written: AtomicUsize::default(),
            reads: AtomicUsize::default(),
            writes: AtomicUsize::default(),
            keys_erased: AtomicUsize::default(),
        }
    }

    /// Reset the performance metrics.
    pub fn reset(&self) {
        self.bytes_read.store(0, Ordering::Relaxed);
        self.bytes_written.store(0, Ordering::Relaxed);
        self.reads.store(0, Ordering::Relaxed);
        self.writes.store(0, Ordering::Relaxed);
    }

    /// Returns the number of bytes read.
    pub fn bytes_read(&self) -> usize {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Returns the number of bytes written.
    pub fn bytes_written(&self) -> usize {
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// Returns the number of read requests.
    pub fn reads(&self) -> usize {
        self.reads.load(Ordering::Relaxed)
    }

    /// Returns the number of write requests.
    pub fn writes(&self) -> usize {
        self.writes.load(Ordering::Relaxed)
    }

    /// Returns the number of key erase requests.
    ///
    /// Includes keys erased that may not have existed, and excludes prefix erase requests.
    pub fn keys_erased(&self) -> usize {
        self.keys_erased.load(Ordering::Relaxed)
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ReadableStorageTraits
    for PerformanceMetricsStorageAdapter<TStorage>
{
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let value = self.storage.get(key);
        let bytes_read = value
            .as_ref()
            .map_or(0, |v| v.as_ref().map_or(0, Bytes::len));
        self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
        self.reads.fetch_add(1, Ordering::Relaxed);
        value
    }

    fn get_partial_values_key(
        &self,
        key: &StoreKey,
        byte_ranges: &mut (dyn Iterator<Item = ByteRange> + Send),
    ) -> Result<Option<Vec<Bytes>>, StorageError> {
        let size_hint_lower_bound = byte_ranges.size_hint().0;
        let values = self.storage.get_partial_values_key(key, byte_ranges)?;
        if let Some(values) = &values {
            let bytes_read = values.iter().map(Bytes::len).sum();
            self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
            self.reads.fetch_add(values.len(), Ordering::Relaxed);
        } else if size_hint_lower_bound > 0 {
            // If the key is found to be empty, consider that as a read
            self.reads.fetch_add(1, Ordering::Relaxed);
        }
        Ok(values)
    }

    fn get_partial_values(
        &self,
        key_ranges: &[StoreKeyRange],
    ) -> Result<Vec<MaybeBytes>, StorageError> {
        let values = self.storage.get_partial_values(key_ranges)?;
        let bytes_read = values
            .iter()
            .map(|value| value.as_ref().map_or(0, Bytes::len))
            .sum::<usize>();
        self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
        self.reads.fetch_add(key_ranges.len(), Ordering::Relaxed);
        Ok(values)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.storage.size_key(key)
    }
}

impl<TStorage: ?Sized + ListableStorageTraits> ListableStorageTraits
    for PerformanceMetricsStorageAdapter<TStorage>
{
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.storage.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.storage.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.storage.list_dir(prefix)
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.storage.size()
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.storage.size_prefix(prefix)
    }
}

impl<TStorage: ?Sized + WritableStorageTraits> WritableStorageTraits
    for PerformanceMetricsStorageAdapter<TStorage>
{
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        self.bytes_written.fetch_add(value.len(), Ordering::Relaxed);
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.storage.set(key, value)
    }

    fn set_partial_values(
        &self,
        key_offset_values: &[StoreKeyOffsetValue],
    ) -> Result<(), StorageError> {
        let bytes_written = key_offset_values
            .iter()
            .map(|ksv| ksv.value().len())
            .sum::<usize>();
        self.bytes_written
            .fetch_add(bytes_written, Ordering::Relaxed);
        self.writes
            .fetch_add(key_offset_values.len(), Ordering::Relaxed);
        self.storage.set_partial_values(key_offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.keys_erased.fetch_add(1, Ordering::Relaxed);
        self.storage.erase(key)
    }

    fn erase_values(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        self.keys_erased.fetch_add(keys.len(), Ordering::Relaxed);
        self.storage.erase_values(keys)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.storage.erase_prefix(prefix)
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<TStorage: ?Sized + AsyncReadableStorageTraits> AsyncReadableStorageTraits
    for PerformanceMetricsStorageAdapter<TStorage>
{
    async fn get(&self, key: &StoreKey) -> Result<MaybeAsyncBytes, StorageError> {
        let value = self.storage.get(key).await;
        let bytes_read = value
            .as_ref()
            .map_or(0, |v| v.as_ref().map_or(0, AsyncBytes::len));
        self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
        self.reads.fetch_add(1, Ordering::Relaxed);
        value
    }

    async fn get_partial_values_key(
        &self,
        key: &StoreKey,
        byte_ranges: &mut (dyn Iterator<Item = ByteRange> + Send),
    ) -> Result<Option<Vec<AsyncBytes>>, StorageError> {
        let size_hint_lower_bound = byte_ranges.size_hint().0;
        let values = self
            .storage
            .get_partial_values_key(key, byte_ranges)
            .await?;
        if let Some(values) = &values {
            let bytes_read = values.iter().map(AsyncBytes::len).sum();
            self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
            self.reads.fetch_add(values.len(), Ordering::Relaxed);
        } else if size_hint_lower_bound > 0 {
            // If the key is found to be empty, consider that as a read
            self.reads.fetch_add(1, Ordering::Relaxed);
        }
        Ok(values)
    }

    async fn get_partial_values(
        &self,
        key_ranges: &[StoreKeyRange],
    ) -> Result<Vec<MaybeAsyncBytes>, StorageError> {
        let values = self.storage.get_partial_values(key_ranges).await?;
        let bytes_read = values
            .iter()
            .map(|value| value.as_ref().map_or(0, AsyncBytes::len))
            .sum::<usize>();
        self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
        self.reads.fetch_add(key_ranges.len(), Ordering::Relaxed);
        Ok(values)
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.storage.size_key(key).await
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<TStorage: ?Sized + AsyncListableStorageTraits> AsyncListableStorageTraits
    for PerformanceMetricsStorageAdapter<TStorage>
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

    async fn size(&self) -> Result<u64, StorageError> {
        self.storage.size().await
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.storage.size_prefix(prefix).await
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<TStorage: ?Sized + AsyncWritableStorageTraits> AsyncWritableStorageTraits
    for PerformanceMetricsStorageAdapter<TStorage>
{
    async fn set(&self, key: &StoreKey, value: AsyncBytes) -> Result<(), StorageError> {
        self.bytes_written.fetch_add(value.len(), Ordering::Relaxed);
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.storage.set(key, value).await
    }

    async fn set_partial_values(
        &self,
        key_offset_values: &[StoreKeyOffsetValue],
    ) -> Result<(), StorageError> {
        let bytes_written = key_offset_values
            .iter()
            .map(|ksv| ksv.value().len())
            .sum::<usize>();
        self.bytes_written
            .fetch_add(bytes_written, Ordering::Relaxed);
        self.writes
            .fetch_add(key_offset_values.len(), Ordering::Relaxed);
        self.storage.set_partial_values(key_offset_values).await
    }

    async fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.storage.erase(key).await
    }

    async fn erase_values(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        self.storage.erase_values(keys).await
    }

    async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.storage.erase_prefix(prefix).await
    }
}

#[cfg(test)]
mod tests {
    use crate::store::MemoryStore;
    use crate::store_test;
    use std::sync::Arc;

    use super::*;

    #[test]
    fn performance_metrics() {
        let store = Arc::new(MemoryStore::new());
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));
        store_test::store_write(&store).unwrap();
        store_test::store_read(&store).unwrap();
        store_test::store_list(&store).unwrap();
        assert!(store.bytes_read() >= 12);
        assert!(store.bytes_written() >= 10);
        assert!(store.reads() >= 8);
        assert!(store.writes() >= 14);
        assert!(store.keys_erased() >= 4);
        store.reset();
        assert_eq!(store.bytes_read(), 0);
    }
}

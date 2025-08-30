//! A storage transformer which prints function calls.

use std::{
    io::Write,
    sync::{Arc, Mutex},
};

use itertools::Itertools;

use crate::{
    byte_range::{ByteRange, ByteRangeIterator},
    Bytes, ListableStorageTraits, MaybeBytes, MaybeSend, MaybeSync, ReadableStorageTraits,
    StorageError, StoreKey, StoreKeyOffsetValue, StoreKeyRange, StoreKeys, StoreKeysPrefixes,
    StorePrefix, WritableStorageTraits,
};

#[cfg(feature = "async")]
use crate::{
    AsyncBytes, AsyncListableStorageTraits, AsyncReadableStorageTraits, AsyncWritableStorageTraits,
    MaybeAsyncBytes,
};

/// This trait combines `Write`, `MaybeSend`, and `MaybeSync`
/// as they cannot be combined together directly in function signatures.
pub trait WriteMaybeSendSync: Write + MaybeSend + MaybeSync {}
impl<T: Write + MaybeSend + MaybeSync> WriteMaybeSendSync for T {}

/// The usage log storage transformer. Logs storage method calls.
///
/// It is intended to aid in debugging and optimising performance by revealing storage access patterns.
///
/// ### Example (log to stdout)
/// ```rust
/// # use std::sync::{Arc, Mutex};
/// # use zarrs_storage::store::MemoryStore;
/// # use zarrs_storage::storage_adapter::usage_log::UsageLogStorageAdapter;
/// let store = Arc::new(MemoryStore::new());
/// let log_writer = Arc::new(Mutex::new(
///     // std::io::BufWriter::new(
///     std::io::stdout(),
///     //    )
/// ));
/// let store = Arc::new(UsageLogStorageAdapter::new(store, log_writer, || {
///     chrono::Utc::now().format("[%T%.3f] ").to_string()
/// }));
/// ````
///
/// Applying array methods with the above [`UsageLogStorageAdapter`] prints outputs like:
/// ```text
/// [23:41:19.885] set(group/array/c/1/0, len=140) -> Ok(())
/// [23:41:19.885] get_partial_values_key(group/array/c/0/0, [-36..-0]) -> len=Ok([36])
/// [23:41:19.886] get_partial_values_key(group/array/c/0/0, [52..104]) -> len=Ok([52])
/// [23:41:19.887] get(group/array/c/1/0) -> len=Ok(140)
/// [23:41:19.891] get(zarr.json) -> len=Ok(0)
/// [23:41:19.891] list_dir() -> (keys:[], prefixes:[group/])
/// [23:41:19.891] get(group/zarr.json) -> len=Ok(86)
/// [23:41:19.891] list_dir(group/) -> (keys:[group/zarr.json], prefixes:[group/array/])
/// [23:41:19.891] get(group/array/zarr.json) -> len=Ok(1315)
/// [23:41:19.892] list() -> [group/array/c/0/0, group/array/c/1/0, group/array/zarr.json, group/zarr.json]
/// ```
pub struct UsageLogStorageAdapter<TStorage: ?Sized> {
    storage: Arc<TStorage>,
    handle: Arc<Mutex<dyn WriteMaybeSendSync>>,
    prefix_func: fn() -> String,
}

impl<TStorage: ?Sized> core::fmt::Debug for UsageLogStorageAdapter<TStorage> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        writeln!(f, "usage log")
    }
}

impl<TStorage: ?Sized> UsageLogStorageAdapter<TStorage> {
    /// Create a new usage log storage adapter.
    pub fn new(
        storage: Arc<TStorage>,
        handle: Arc<Mutex<dyn WriteMaybeSendSync>>,
        prefix_func: fn() -> String,
    ) -> Self {
        Self {
            storage,
            handle,
            prefix_func,
        }
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ReadableStorageTraits
    for UsageLogStorageAdapter<TStorage>
{
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let result = self.storage.get(key);
        writeln!(
            self.handle.lock().unwrap(),
            "{}get({key}) -> len={:?}",
            (self.prefix_func)(),
            result.as_ref().map(|v| v.as_ref().map_or(0, Bytes::len))
        )?;
        result
    }

    fn get_partial_values_key(
        &self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator,
    ) -> Result<Option<Vec<Bytes>>, StorageError> {
        let byte_ranges = byte_ranges.collect::<Vec<ByteRange>>();
        let result = self
            .storage
            .get_partial_values_key(key, Box::new(byte_ranges.iter().copied()));
        writeln!(
            self.handle.lock().unwrap(),
            "{}get_partial_values_key({key}, [{}]) -> len={:?}",
            (self.prefix_func)(),
            byte_ranges.iter().format(", "),
            result.as_ref().map(|v| {
                v.as_ref()
                    .map_or(vec![], |v| v.iter().map(Bytes::len).collect_vec())
            })
        )?;
        result
    }

    fn get_partial_values(
        &self,
        key_ranges: &[StoreKeyRange],
    ) -> Result<Vec<MaybeBytes>, StorageError> {
        let result = self.storage.get_partial_values(key_ranges);
        writeln!(
            self.handle.lock().unwrap(),
            "{}get_partial_values([{}]) -> len={:?}",
            (self.prefix_func)(),
            key_ranges.iter().format(", "),
            result
                .as_ref()
                .map(|v| { v.iter().map(|v| v.iter().map(Bytes::len).collect_vec()) })
        )?;
        result
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let result = self.storage.size_key(key);
        writeln!(
            self.handle.lock().unwrap(),
            "{}size_key({key}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }
}

impl<TStorage: ?Sized + ListableStorageTraits> ListableStorageTraits
    for UsageLogStorageAdapter<TStorage>
{
    fn list(&self) -> Result<StoreKeys, StorageError> {
        let result = self.storage.list();
        writeln!(
            self.handle.lock().unwrap(),
            "{}list() -> [{}]",
            (self.prefix_func)(),
            result.as_ref().unwrap_or(&vec![]).iter().format(", ")
        )?;
        result
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let result = self.storage.list_prefix(prefix);
        writeln!(
            self.handle.lock().unwrap(),
            "{}list_prefix({prefix}) -> [{}]",
            (self.prefix_func)(),
            result.as_ref().unwrap_or(&vec![]).iter().format(", ")
        )?;
        result
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let result = self.storage.list_dir(prefix);
        writeln!(
            self.handle.lock().unwrap(),
            "{}list_dir({prefix}) -> (keys:[{}], prefixes:[{}])",
            (self.prefix_func)(),
            result.as_ref().map_or(String::new(), |skp| skp
                .keys()
                .iter()
                .format(", ")
                .to_string()),
            result.as_ref().map_or(String::new(), |skp| skp
                .prefixes()
                .iter()
                .format(", ")
                .to_string()),
        )?;
        result
    }

    fn size(&self) -> Result<u64, StorageError> {
        let result = self.storage.size();
        writeln!(
            self.handle.lock().unwrap(),
            "{}size() -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let result: Result<u64, StorageError> = self.storage.size_prefix(prefix);
        writeln!(
            self.handle.lock().unwrap(),
            "{}size_prefix({prefix}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }
}

impl<TStorage: ?Sized + WritableStorageTraits> WritableStorageTraits
    for UsageLogStorageAdapter<TStorage>
{
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let len = value.len();
        let result = self.storage.set(key, value);
        writeln!(
            self.handle.lock().unwrap(),
            "{}set({key}, len={}) -> {result:?}",
            (self.prefix_func)(),
            len
        )?;
        result
    }

    fn set_partial_values(
        &self,
        key_offset_values: &[StoreKeyOffsetValue],
    ) -> Result<(), StorageError> {
        struct DebugStoreKeyOffsetValue<'a>(&'a StoreKeyOffsetValue<'a>);
        impl core::fmt::Debug for DebugStoreKeyOffsetValue<'_> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(
                    f,
                    "({} offset={} len={})",
                    self.0.key(),
                    self.0.offset(),
                    self.0.value().len()
                )
            }
        }
        let result = self.storage.set_partial_values(key_offset_values);
        writeln!(
            self.handle.lock().unwrap(),
            "{}set_partial_values({:?}) -> {result:?}",
            (self.prefix_func)(),
            key_offset_values
                .iter()
                .map(DebugStoreKeyOffsetValue)
                .collect_vec()
        )?;
        result
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        let result = self.storage.erase(key);
        writeln!(
            self.handle.lock().unwrap(),
            "{}erase({key}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }

    fn erase_values(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        let result = self.storage.erase_values(keys);
        writeln!(
            self.handle.lock().unwrap(),
            "{}erase_values([{}]) -> {result:?}",
            keys.iter().format(", "),
            (self.prefix_func)()
        )?;
        result
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let result = self.storage.erase_prefix(prefix);
        writeln!(
            self.handle.lock().unwrap(),
            "{}erase_prefix({prefix}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncReadableStorageTraits> AsyncReadableStorageTraits
    for UsageLogStorageAdapter<TStorage>
{
    async fn get(&self, key: &StoreKey) -> Result<MaybeAsyncBytes, StorageError> {
        let result = self.storage.get(key).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}get({key}) -> len={:?}",
            (self.prefix_func)(),
            result
                .as_ref()
                .map(|v| v.as_ref().map_or(0, AsyncBytes::len))
        )?;
        result
    }

    async fn get_partial_values_key<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<Option<Vec<AsyncBytes>>, StorageError> {
        let byte_ranges: Vec<ByteRange> = byte_ranges.collect::<Vec<ByteRange>>();
        let result = self
            .storage
            .get_partial_values_key(key, Box::new(byte_ranges.iter().copied()))
            .await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}get_partial_values_key({key}, [{}]) -> len={:?}",
            (self.prefix_func)(),
            byte_ranges.iter().format(", "),
            result.as_ref().map(|v| {
                v.as_ref()
                    .map_or(vec![], |v| v.iter().map(AsyncBytes::len).collect_vec())
            })
        )?;
        result
    }

    async fn get_partial_values(
        &self,
        key_ranges: &[StoreKeyRange],
    ) -> Result<Vec<MaybeAsyncBytes>, StorageError> {
        let result = self.storage.get_partial_values(key_ranges).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}get_partial_values([{}]) -> len={:?}",
            (self.prefix_func)(),
            key_ranges.iter().format(", "),
            result.as_ref().map(|v| {
                v.iter()
                    .map(|v| v.iter().map(AsyncBytes::len).collect_vec())
            })
        )?;
        result
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let result = self.storage.size_key(key).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}size_key({key}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncListableStorageTraits> AsyncListableStorageTraits
    for UsageLogStorageAdapter<TStorage>
{
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        let keys = self.storage.list().await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}list() -> [{}]",
            (self.prefix_func)(),
            keys.as_ref().unwrap_or(&vec![]).iter().format(", "),
        )?;
        keys
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let result = self.storage.list_prefix(prefix).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}list_prefix({prefix}) -> [{}]",
            (self.prefix_func)(),
            result.as_ref().unwrap_or(&vec![]).iter().format(", ")
        )?;
        result
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let result = self.storage.list_dir(prefix).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}list_dir({prefix}) -> (keys:[{}], prefixes:[{}])",
            (self.prefix_func)(),
            result.as_ref().map_or(String::new(), |skp| skp
                .keys()
                .iter()
                .format(", ")
                .to_string()),
            result.as_ref().map_or(String::new(), |skp| skp
                .prefixes()
                .iter()
                .format(", ")
                .to_string()),
        )?;
        result
    }

    async fn size(&self) -> Result<u64, StorageError> {
        let result = self.storage.size().await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}size() -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let result: Result<u64, StorageError> = self.storage.size_prefix(prefix).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}size_prefix({prefix}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncWritableStorageTraits> AsyncWritableStorageTraits
    for UsageLogStorageAdapter<TStorage>
{
    async fn set(&self, key: &StoreKey, value: AsyncBytes) -> Result<(), StorageError> {
        let len = value.len();
        let result = self.storage.set(key, value).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}set({key}, len={len}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }

    async fn set_partial_values(
        &self,
        key_offset_values: &[StoreKeyOffsetValue],
    ) -> Result<(), StorageError> {
        let result = self.storage.set_partial_values(key_offset_values).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}set_partial_values({key_offset_values:?}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }

    async fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        let result = self.storage.erase(key).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}erase({key}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }

    async fn erase_values(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        let result = self.storage.erase_values(keys).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}erase_values([{}]) -> {result:?}",
            (self.prefix_func)(),
            keys.iter().format(", ")
        )?;
        result
    }

    async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let result = self.storage.erase_prefix(prefix).await;
        writeln!(
            self.handle.lock().unwrap(),
            "{}erase_prefix({prefix}) -> {result:?}",
            (self.prefix_func)()
        )?;
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{store::MemoryStore, store_test};
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn usage_log() {
        let store = Arc::new(MemoryStore::new());
        let log_writer = Arc::new(Mutex::new(
            // std::io::BufWriter::new(
            std::io::stdout(),
            //    )
        ));
        let store = Arc::new(UsageLogStorageAdapter::new(store, log_writer, || {
            chrono::Utc::now().format("[%T%.3f] ").to_string()
        }));
        store_test::store_write(&store).unwrap();
        store_test::store_read(&store).unwrap();
        store_test::store_list(&store).unwrap();
    }
}

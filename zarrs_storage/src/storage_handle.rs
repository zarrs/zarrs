use std::sync::Arc;

use super::{
    byte_range::ByteRangeIterator, Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StorePrefix, WritableStorageTraits,
};
#[cfg(feature = "async")]
use super::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableStorageTraits,
    AsyncWritableStorageTraits,
};
use crate::OffsetBytesIterator;

/// A storage handle.
///
/// This is a handle to borrowed storage which can be owned and cloned, even if the storage it references is unsized.
#[derive(Clone)]
pub struct StorageHandle<TStorage: ?Sized>(Arc<TStorage>);

impl<TStorage: ?Sized> StorageHandle<TStorage> {
    /// Create a new storage handle.
    pub const fn new(storage: Arc<TStorage>) -> Self {
        Self(storage)
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ReadableStorageTraits for StorageHandle<TStorage> {
    fn get(&self, key: &super::StoreKey) -> Result<MaybeBytes, super::StorageError> {
        self.0.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &super::StoreKey) -> Result<Option<u64>, super::StorageError> {
        self.0.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

impl<TStorage: ?Sized + ListableStorageTraits> ListableStorageTraits for StorageHandle<TStorage> {
    fn list(&self) -> Result<super::StoreKeys, super::StorageError> {
        self.0.list()
    }

    fn list_prefix(
        &self,
        prefix: &super::StorePrefix,
    ) -> Result<super::StoreKeys, super::StorageError> {
        self.0.list_prefix(prefix)
    }

    fn list_dir(
        &self,
        prefix: &super::StorePrefix,
    ) -> Result<super::StoreKeysPrefixes, super::StorageError> {
        self.0.list_dir(prefix)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.0.size_prefix(prefix)
    }

    fn size(&self) -> Result<u64, super::StorageError> {
        self.0.size()
    }
}

impl<TStorage: ?Sized + WritableStorageTraits> WritableStorageTraits for StorageHandle<TStorage> {
    fn set(&self, key: &super::StoreKey, value: Bytes) -> Result<(), super::StorageError> {
        self.0.set(key, value)
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), super::StorageError> {
        self.0.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &super::StoreKey) -> Result<(), super::StorageError> {
        self.0.erase(key)
    }

    fn erase_many(&self, keys: &[super::StoreKey]) -> Result<(), super::StorageError> {
        self.0.erase_many(keys)
    }

    fn erase_prefix(&self, prefix: &super::StorePrefix) -> Result<(), super::StorageError> {
        self.0.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.0.supports_set_partial()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncReadableStorageTraits> AsyncReadableStorageTraits
    for StorageHandle<TStorage>
{
    async fn get(&self, key: &super::StoreKey) -> Result<MaybeBytes, super::StorageError> {
        self.0.get(key).await
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges).await
    }

    async fn size_key(&self, key: &super::StoreKey) -> Result<Option<u64>, super::StorageError> {
        self.0.size_key(key).await
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncListableStorageTraits> AsyncListableStorageTraits
    for StorageHandle<TStorage>
{
    async fn list(&self) -> Result<super::StoreKeys, super::StorageError> {
        self.0.list().await
    }

    async fn list_prefix(
        &self,
        prefix: &super::StorePrefix,
    ) -> Result<super::StoreKeys, super::StorageError> {
        self.0.list_prefix(prefix).await
    }

    async fn list_dir(
        &self,
        prefix: &super::StorePrefix,
    ) -> Result<super::StoreKeysPrefixes, super::StorageError> {
        self.0.list_dir(prefix).await
    }

    async fn size_prefix(&self, prefix: &super::StorePrefix) -> Result<u64, super::StorageError> {
        self.0.size_prefix(prefix).await
    }

    async fn size(&self) -> Result<u64, super::StorageError> {
        self.0.size().await
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncWritableStorageTraits> AsyncWritableStorageTraits
    for StorageHandle<TStorage>
{
    async fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        self.0.set(key, value).await
    }

    async fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), super::StorageError> {
        self.0.set_partial_many(key, offset_values).await
    }

    async fn erase(&self, key: &super::StoreKey) -> Result<(), super::StorageError> {
        self.0.erase(key).await
    }

    async fn erase_many(&self, keys: &[super::StoreKey]) -> Result<(), super::StorageError> {
        self.0.erase_many(keys).await
    }

    async fn erase_prefix(&self, prefix: &super::StorePrefix) -> Result<(), super::StorageError> {
        self.0.erase_prefix(prefix).await
    }

    fn supports_set_partial(&self) -> bool {
        self.0.supports_set_partial()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::{store::MemoryStore, StorageHandle};

    #[test]
    fn memory_storage_handle() -> Result<(), Box<dyn Error>> {
        let store = Arc::new(MemoryStore::new());
        let store = Arc::new(StorageHandle::new(store));
        crate::store_test::store_write(&store)?;
        crate::store_test::store_read(&store)?;
        crate::store_test::store_list(&store)?;
        Ok(())
    }
}

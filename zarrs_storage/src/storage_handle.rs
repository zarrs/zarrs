use std::sync::Arc;

use super::{
    byte_range::ByteRangeIterator, Bytes, ListableStorageTraits, MaybeBytes, ReadableStorageTraits,
    StorageError, StoreKey, StorePrefix, WritableStorageTraits,
};

#[cfg(feature = "async")]
use super::{AsyncListableStorageTraits, AsyncReadableStorageTraits, AsyncWritableStorageTraits};

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

    fn get_partial_values_key(
        &self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator,
    ) -> Result<Option<Vec<Bytes>>, StorageError> {
        self.0.get_partial_values_key(key, byte_ranges)
    }

    fn get_partial_values(
        &self,
        key_ranges: &[super::StoreKeyRange],
    ) -> Result<Vec<MaybeBytes>, StorageError> {
        self.0.get_partial_values(key_ranges)
    }

    fn size_key(&self, key: &super::StoreKey) -> Result<Option<u64>, super::StorageError> {
        self.0.size_key(key)
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

    fn set_partial_values(
        &self,
        key_offset_values: &[super::StoreKeyOffsetValue],
    ) -> Result<(), super::StorageError> {
        self.0.set_partial_values(key_offset_values)
    }

    fn erase(&self, key: &super::StoreKey) -> Result<(), super::StorageError> {
        self.0.erase(key)
    }

    fn erase_values(&self, keys: &[super::StoreKey]) -> Result<(), super::StorageError> {
        self.0.erase_values(keys)
    }

    fn erase_prefix(&self, prefix: &super::StorePrefix) -> Result<(), super::StorageError> {
        self.0.erase_prefix(prefix)
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

    async fn get_partial_values_key<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<Option<Vec<Bytes>>, StorageError> {
        self.0.get_partial_values_key(key, byte_ranges).await
    }

    async fn get_partial_values(
        &self,
        key_ranges: &[super::StoreKeyRange],
    ) -> Result<Vec<MaybeBytes>, StorageError> {
        self.0.get_partial_values(key_ranges).await
    }

    async fn size_key(&self, key: &super::StoreKey) -> Result<Option<u64>, super::StorageError> {
        self.0.size_key(key).await
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

    async fn set_partial_values(
        &self,
        key_offset_values: &[super::StoreKeyOffsetValue],
    ) -> Result<(), super::StorageError> {
        self.0.set_partial_values(key_offset_values).await
    }

    async fn erase(&self, key: &super::StoreKey) -> Result<(), super::StorageError> {
        self.0.erase(key).await
    }

    async fn erase_values(&self, keys: &[super::StoreKey]) -> Result<(), super::StorageError> {
        self.0.erase_values(keys).await
    }

    async fn erase_prefix(&self, prefix: &super::StorePrefix) -> Result<(), super::StorageError> {
        self.0.erase_prefix(prefix).await
    }
}

#[cfg(test)]
mod tests {
    use crate::{store::MemoryStore, StorageHandle};

    use super::*;
    use std::error::Error;

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

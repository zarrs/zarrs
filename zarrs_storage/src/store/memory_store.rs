//! An in-memory store.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::RwLock;

use bytes::BytesMut;
#[cfg(feature = "async")]
use futures::{stream, StreamExt};

use crate::byte_range::{ByteRange, ByteRangeIterator, InvalidByteRangeError};
#[cfg(feature = "async")]
use crate::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableStorageTraits,
    AsyncWritableStorageTraits,
};
use crate::{
    Bytes, CowBytes, ListableStorageTraits, MaybeBytes, OffsetBytesIterator, ReadableStorageTraits,
    StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, SyncMaybeBytesIterator,
    WritableStorageTraits,
};

/// An in-memory store.
///
/// Supports synchronous operations and, with the `async` feature, asynchronous operations.
#[derive(Debug)]
pub struct MemoryStore {
    data_map: RwLock<BTreeMap<StoreKey, Bytes>>,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore {
    /// Create a new memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_map: RwLock::default(),
        }
    }

    fn get_partial_many_iter<'a>(
        &self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<Option<impl Iterator<Item = Result<Bytes, StorageError>> + 'a>, StorageError> {
        let Some(data) = ReadableStorageTraits::get(self, key)? else {
            return Ok(None);
        };
        Ok(Some(byte_ranges.map(move |byte_range| {
            let range = byte_range
                .checked_range_usize(data.len())
                .ok_or_else(|| InvalidByteRangeError::new(byte_range, data.len() as u64))?;
            Ok(data.slice(range))
        })))
    }
}

impl ReadableStorageTraits for MemoryStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let data_map = self
            .data_map
            .read()
            .map_err(|_| StorageError::from("memory store lock is poisoned"))?;
        Ok(data_map.get(key).cloned())
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<SyncMaybeBytesIterator<'a>, StorageError> {
        Ok(self
            .get_partial_many_iter(key, byte_ranges)?
            .map(|iter| Box::new(iter) as _))
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        let data_map = self.data_map.read().unwrap();
        let Some(data) = data_map.get(key) else {
            return Ok(None);
        };
        let range = byte_range
            .checked_range_usize(data.len())
            .ok_or_else(|| InvalidByteRangeError::new(byte_range, data.len() as u64))?;
        Ok(Some(data.slice(range)))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let data_map = self.data_map.read().unwrap();
        Ok(data_map.get(key).map(|entry| entry.len() as u64))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl WritableStorageTraits for MemoryStore {
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        // A full set replaces any existing value, so store the handle directly to avoid a copy.
        // The value is converted, and a replaced value dropped, outside of the lock.
        let value: Bytes = value.into();
        let replaced = {
            let mut data_map = self.data_map.write().unwrap();
            if let Some(entry) = data_map.get_mut(key) {
                Some(std::mem::replace(entry, value))
            } else {
                data_map.insert(key.clone(), value);
                None
            }
        };
        drop(replaced);
        Ok(())
    }

    fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        let mut data_map = self.data_map.write().unwrap();
        let entry = data_map.entry(key.clone()).or_default();

        // Take ownership so try_into_mut can succeed when there are no other clones.
        let mut data = std::mem::take(entry)
            .try_into_mut()
            .unwrap_or_else(|bytes: Bytes| BytesMut::from(bytes.as_ref()));
        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            let end = offset + value.len();
            if data.len() < end {
                data.resize(end, 0);
            }
            data[offset..end].copy_from_slice(&value);
        }
        *entry = data.freeze();
        Ok(())
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        // The erased value is dropped outside of the lock
        let erased = self.data_map.write().unwrap().remove(key);
        drop(erased);
        Ok(())
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let mut data_map = self
            .data_map
            .write()
            .map_err(|_| StorageError::from("memory store lock is poisoned"))?;
        data_map.retain(|key, _| !key.has_prefix(prefix));
        Ok(())
    }

    fn supports_set_partial(&self) -> bool {
        true
    }
}

impl ListableStorageTraits for MemoryStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        let data_map = self.data_map.read().unwrap();
        Ok(data_map.keys().cloned().collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let data_map = self.data_map.read().unwrap();
        Ok(data_map
            .keys()
            .filter(|&key| key.has_prefix(prefix))
            .cloned()
            .collect())
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let mut keys: StoreKeys = vec![];
        let mut prefixes: BTreeSet<StorePrefix> = BTreeSet::default();
        let data_map = self.data_map.read().unwrap();
        for key in data_map.keys() {
            if key.has_prefix(prefix) {
                let key_strip = key.as_str().strip_prefix(prefix.as_str()).unwrap();
                let key_strip = key_strip.strip_prefix('/').unwrap_or(key_strip);
                let components: Vec<_> = key_strip.split('/').collect();
                if components.len() > 1 {
                    prefixes.insert(StorePrefix::new(
                        prefix.as_str().to_string() + components[0] + "/",
                    )?);
                } else {
                    let parent = key.parent();
                    if parent.eq(prefix) {
                        keys.push(key.clone());
                    }
                }
            }
        }
        drop(data_map);
        Ok(StoreKeysPrefixes::new(keys, prefixes.into_iter().collect()))
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let data_map = self
            .data_map
            .read()
            .map_err(|_| StorageError::from("memory store lock is poisoned"))?;
        data_map
            .iter()
            .filter(|(key, _)| key.has_prefix(prefix))
            .try_fold(0u64, |size, (_, value)| {
                size.checked_add(value.len() as u64)
                    .ok_or_else(|| StorageError::from("memory store size exceeds u64::MAX"))
            })
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncReadableStorageTraits for MemoryStore {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        ReadableStorageTraits::get(self, key)
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        let Some(iter) = self.get_partial_many_iter(key, byte_ranges)? else {
            return Ok(None);
        };
        let result: Vec<_> = iter.collect();
        Ok(Some(stream::iter(result).boxed()))
    }

    async fn get_partial<'a>(
        &'a self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        ReadableStorageTraits::get_partial(self, key, byte_range)
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        ReadableStorageTraits::size_key(self, key)
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncWritableStorageTraits for MemoryStore {
    async fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        // Insert directly so new keys require only one lookup under the write lock.
        let key = key.clone();
        let value: Bytes = value.into();
        let replaced = self
            .data_map
            .write()
            .map_err(|_| StorageError::from("memory store lock is poisoned"))?
            .insert(key, value);
        drop(replaced);
        Ok(())
    }

    async fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        WritableStorageTraits::set_partial_many(self, key, offset_values)
    }

    async fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        WritableStorageTraits::erase(self, key)
    }

    async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        WritableStorageTraits::erase_prefix(self, prefix)
    }

    fn supports_set_partial(&self) -> bool {
        true
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncListableStorageTraits for MemoryStore {
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        ListableStorageTraits::list(self)
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        ListableStorageTraits::list_prefix(self, prefix)
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        ListableStorageTraits::list_dir(self, prefix)
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        ListableStorageTraits::size_prefix(self, prefix)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use super::MemoryStore;
    use crate::byte_range::ByteRange;
    use crate::{
        ReadableStorageTraits, ReadableWritableListableStorageTraits, StoreKey,
        WritableStorageTraits,
    };

    #[test]
    fn memory() -> Result<(), Box<dyn Error>> {
        let store = MemoryStore::new();
        crate::store_test::store_write(&store)?;
        crate::store_test::store_read(&store)?;
        crate::store_test::store_list(&store)?;
        crate::store_test::store_list_size(&store)?;
        Ok(())
    }

    #[test]
    fn memory_get_partial_out_of_range() -> Result<(), Box<dyn Error>> {
        let store = MemoryStore::new();
        let key: StoreKey = "a/b".try_into()?;
        store.set(&key, vec![0, 1, 2].into())?;
        for byte_range in [
            ByteRange::Suffix(4),
            ByteRange::FromStart(4, None),
            ByteRange::FromStart(u64::MAX, Some(2)),
        ] {
            assert!(store.get_partial(&key, byte_range).is_err());
        }
        Ok(())
    }

    #[test]
    fn memory_set_partial_many_multiple_offsets() -> Result<(), Box<dyn Error>> {
        let store = MemoryStore::new();
        let key: StoreKey = "a/b".try_into()?;
        store.set(&key, vec![0, 0, 0, 0].into())?;
        store.set_partial_many(
            &key,
            Box::new([(0, vec![1, 2].into()), (3, vec![3].into())].into_iter()),
        )?;
        assert_eq!(store.get(&key)?, Some(vec![1, 2, 0, 3].into()));
        Ok(())
    }

    #[test]
    fn memory_set_partial_many_grows_and_creates() -> Result<(), Box<dyn Error>> {
        let store = MemoryStore::new();
        // A partial write to an absent key zero fills up to the offset.
        let key: StoreKey = "absent".try_into()?;
        store.set_partial(&key, 2, vec![7, 8].into())?;
        assert_eq!(store.get(&key)?, Some(vec![0, 0, 7, 8].into()));

        // A partial write past the end grows the value rather than truncating it.
        store.set_partial(&key, 5, vec![9].into())?;
        assert_eq!(store.get(&key)?, Some(vec![0, 0, 7, 8, 0, 9].into()));
        Ok(())
    }

    #[test]
    fn memory_upcast1() -> Result<(), Box<dyn Error>> {
        let store: Arc<dyn ReadableWritableListableStorageTraits> = Arc::new(MemoryStore::new());
        crate::store_test::store_write(&store.clone().writable())?;
        crate::store_test::store_read(&store.clone().readable())?;
        crate::store_test::store_list(&store.clone().listable())?;
        crate::store_test::store_list_size(&store.clone().listable())?;
        Ok(())
    }

    #[test]
    fn memory_upcast2() -> Result<(), Box<dyn Error>> {
        let store: Arc<dyn ReadableWritableListableStorageTraits> = Arc::new(MemoryStore::new());
        crate::store_test::store_write(&store.clone().readable_writable().writable())?;
        crate::store_test::store_read(&store.clone().readable_writable().readable())?;
        Ok(())
    }

    #[test]
    fn memory_upcast3() -> Result<(), Box<dyn Error>> {
        let store: Arc<dyn ReadableWritableListableStorageTraits> = Arc::new(MemoryStore::new());
        crate::store_test::store_write(&store.clone().writable())?;
        crate::store_test::store_read(&store.clone().readable_listable().readable())?;
        crate::store_test::store_list(&store.clone().readable_listable().listable())?;
        crate::store_test::store_list_size(&store.clone().readable_listable().listable())?;
        Ok(())
    }

    #[cfg(feature = "async")]
    mod async_tests {
        use std::error::Error;
        use std::sync::Arc;

        use futures::executor::block_on;

        use super::MemoryStore;
        use crate::byte_range::ByteRange;
        use crate::{
            AsyncReadableStorageTraits, AsyncReadableWritableListableStorageTraits,
            AsyncWritableStorageTraits, Bytes, StoreKey,
        };

        #[test]
        fn async_memory() -> Result<(), Box<dyn Error>> {
            block_on(async {
                let store = MemoryStore::new();
                crate::store_test::async_store_write(&store).await?;
                crate::store_test::async_store_read(&store).await?;
                crate::store_test::async_store_list(&store).await?;
                crate::store_test::async_store_list_size(&store).await?;
                Ok(())
            })
        }

        #[test]
        fn async_memory_invalid_byte_ranges() {
            block_on(async {
                let store = MemoryStore::new();
                let key = StoreKey::new("key").unwrap();
                store
                    .set(&key, Bytes::from_static(b"value").into())
                    .await
                    .unwrap();

                assert!(store
                    .get_partial(&key, ByteRange::FromStart(6, None))
                    .await
                    .is_err());
                assert!(store.get_partial(&key, ByteRange::Suffix(6)).await.is_err());
            });
        }

        #[test]
        fn async_memory_upcast() -> Result<(), Box<dyn Error>> {
            block_on(async {
                let store: Arc<dyn AsyncReadableWritableListableStorageTraits> =
                    Arc::new(MemoryStore::new());
                crate::store_test::async_store_write(&store.clone().writable()).await?;
                crate::store_test::async_store_read(&store.clone().readable()).await?;
                crate::store_test::async_store_list(&store.clone().listable()).await?;
                crate::store_test::async_store_list_size(&store.clone().listable()).await?;
                Ok(())
            })
        }
    }
}

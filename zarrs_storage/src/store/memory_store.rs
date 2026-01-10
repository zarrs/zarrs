//! A synchronous in-memory store.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use bytes::BytesMut;

use crate::byte_range::{ByteOffset, ByteRangeIterator, InvalidByteRangeError};
use crate::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};

/// A synchronous in-memory store.
#[derive(Debug)]
pub struct MemoryStore {
    data_map: Mutex<BTreeMap<StoreKey, BytesMut>>,
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
            data_map: Mutex::default(),
        }
    }

    fn set_impl(&self, key: &StoreKey, value: &[u8], offset: ByteOffset, truncate: bool) {
        let mut data_map = self.data_map.lock().unwrap();
        let data = data_map.entry(key.clone()).or_default();

        if offset == 0 && data.is_empty() {
            data.extend_from_slice(value);
        } else {
            let length = usize::try_from(offset + value.len() as u64).unwrap();
            if data.len() < length {
                data.resize(length, 0);
            } else if truncate {
                data.truncate(length);
            }
            let offset = usize::try_from(offset).unwrap();
            data[offset..offset + value.len()].copy_from_slice(value);
        }
    }
}

impl ReadableStorageTraits for MemoryStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let data_map = self.data_map.lock().unwrap();
        let data = data_map.get(key);
        if let Some(data) = data {
            Ok(Some(data.clone().freeze()))
        } else {
            Ok(None)
        }
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let data_map = self.data_map.lock().unwrap();
        let data = data_map.get(key);
        if let Some(data) = data {
            let data = data.clone().freeze();
            let out = Box::new(byte_ranges.map(move |byte_range| {
                let start = usize::try_from(byte_range.start(data.len() as u64)).unwrap();
                let end = usize::try_from(byte_range.end(data.len() as u64)).unwrap();
                if end > data.len() {
                    Err(InvalidByteRangeError::new(byte_range, data.len() as u64).into())
                } else {
                    Ok(data.slice(start..end))
                }
            }));
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let data_map = self.data_map.lock().unwrap();
        data_map
            .get(key)
            .map_or_else(|| Ok(None), |entry| Ok(Some(entry.len() as u64)))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl WritableStorageTraits for MemoryStore {
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        Self::set_impl(self, key, &value, 0, true);
        Ok(())
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        for (offset, value) in offset_values {
            self.set_impl(key, &value, offset, false);
        }
        Ok(())
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        let mut data_map = self.data_map.lock().unwrap();
        data_map.remove(key);
        Ok(())
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let mut data_map = self.data_map.lock().unwrap();
        let keys: Vec<StoreKey> = data_map.keys().cloned().collect();
        for key in keys {
            if key.has_prefix(prefix) {
                data_map.remove(&key);
            }
        }
        Ok(())
    }

    fn supports_set_partial(&self) -> bool {
        true
    }
}

impl ListableStorageTraits for MemoryStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        let data_map = self.data_map.lock().unwrap();
        Ok(data_map.keys().cloned().collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let data_map = self.data_map.lock().unwrap();
        Ok(data_map
            .keys()
            .filter(|&key| key.has_prefix(prefix))
            .cloned()
            .collect())
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let mut keys: StoreKeys = vec![];
        let mut prefixes: BTreeSet<StorePrefix> = BTreeSet::default();
        let data_map = self.data_map.lock().unwrap();
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
        let prefixes: Vec<StorePrefix> = prefixes.iter().cloned().collect();
        Ok(StoreKeysPrefixes { keys, prefixes })
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let mut size = 0;
        for key in self.list_prefix(prefix)? {
            if let Some(size_key) = self.size_key(&key)? {
                size += size_key;
            }
        }
        Ok(size)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use super::*;
    use crate::ReadableWritableListableStorageTraits;

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
}

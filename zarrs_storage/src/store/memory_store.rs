//! A synchronous in-memory store.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use bytes::BytesMut;

use crate::byte_range::{ByteRangeIterator, InvalidByteRangeError};
use crate::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};

/// A synchronous in-memory store.
#[derive(Debug)]
pub struct MemoryStore {
    data_map: Mutex<BTreeMap<StoreKey, Bytes>>,
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
}

impl ReadableStorageTraits for MemoryStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let data_map = self.data_map.lock().unwrap();
        Ok(data_map.get(key).cloned())
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let data_map = self.data_map.lock().unwrap();
        let data = data_map.get(key);
        if let Some(data) = data {
            let data = data.clone();
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
        // A full set replaces any existing value, so store the handle directly to avoid a copy.
        let mut data_map = self.data_map.lock().unwrap();
        data_map.insert(key.clone(), value);
        Ok(())
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        let mut data_map = self.data_map.lock().unwrap();
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
}

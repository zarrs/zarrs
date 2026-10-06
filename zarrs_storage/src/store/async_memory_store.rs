//! An asynchronous in-memory store.

use std::collections::{BTreeMap, BTreeSet};

use bytes::BytesMut;
use futures::lock::Mutex;
use futures::{stream, StreamExt};

use crate::byte_range::{ByteOffset, ByteRange, ByteRangeIterator, InvalidByteRangeError};
use crate::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableStorageTraits,
    AsyncWritableStorageTraits, Bytes, MaybeBytes, OffsetBytesIterator, StorageError, StoreKey,
    StoreKeys, StoreKeysPrefixes, StorePrefix,
};

/// An asynchronous in-memory store.
///
/// This store uses [`futures::lock::Mutex`], so it is not tied to a particular async runtime.
#[derive(Debug)]
pub struct AsyncMemoryStore {
    data_map: Mutex<BTreeMap<StoreKey, Bytes>>,
}

impl Default for AsyncMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncMemoryStore {
    /// Create a new asynchronous memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_map: Mutex::new(BTreeMap::new()),
        }
    }

    fn set_impl(
        data_map: &mut BTreeMap<StoreKey, Bytes>,
        key: &StoreKey,
        value: &[u8],
        offset: ByteOffset,
        truncate: bool,
    ) {
        let entry = data_map.entry(key.clone()).or_default();

        if offset == 0 && entry.is_empty() {
            *entry = Bytes::copy_from_slice(value);
        } else {
            let length = usize::try_from(offset + value.len() as u64).unwrap();
            // Take ownership so try_into_mut can succeed when there are no other clones.
            let mut data = std::mem::take(entry)
                .try_into_mut()
                .unwrap_or_else(|bytes: Bytes| BytesMut::from(bytes.as_ref()));
            if data.len() < length {
                data.resize(length, 0);
            } else if truncate {
                data.truncate(length);
            }
            let offset = usize::try_from(offset).unwrap();
            data[offset..offset + value.len()].copy_from_slice(value);
            *entry = data.freeze();
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncReadableStorageTraits for AsyncMemoryStore {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let data_map = self.data_map.lock().await;
        Ok(data_map.get(key).cloned())
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        let data_map = self.data_map.lock().await;
        let Some(data) = data_map.get(key).cloned() else {
            return Ok(None);
        };
        drop(data_map);

        let result = byte_ranges
            .map(|byte_range| {
                let data_len = data.len() as u64;
                let valid = match byte_range {
                    ByteRange::FromStart(offset, length) => length
                        .map_or(Some(offset), |length| offset.checked_add(length))
                        .is_some_and(|end| end <= data_len),
                    ByteRange::Suffix(length) => length <= data_len,
                };
                if !valid {
                    return Err(InvalidByteRangeError::new(byte_range, data_len).into());
                }

                let start = usize::try_from(byte_range.start(data_len)).unwrap();
                let end = usize::try_from(byte_range.end(data_len)).unwrap();
                Ok(data.slice(start..end))
            })
            .collect::<Vec<_>>();
        Ok(Some(stream::iter(result).boxed()))
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let data_map = self.data_map.lock().await;
        Ok(data_map.get(key).map(|entry| entry.len() as u64))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncWritableStorageTraits for AsyncMemoryStore {
    async fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let mut data_map = self.data_map.lock().await;
        Self::set_impl(&mut data_map, key, &value, 0, true);
        Ok(())
    }

    async fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        let mut data_map = self.data_map.lock().await;
        for (offset, value) in offset_values {
            Self::set_impl(&mut data_map, key, &value, offset, false);
        }
        Ok(())
    }

    async fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        let mut data_map = self.data_map.lock().await;
        data_map.remove(key);
        Ok(())
    }

    async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let mut data_map = self.data_map.lock().await;
        data_map.retain(|key, _| !key.has_prefix(prefix));
        Ok(())
    }

    fn supports_set_partial(&self) -> bool {
        true
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncListableStorageTraits for AsyncMemoryStore {
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        let data_map = self.data_map.lock().await;
        Ok(data_map.keys().cloned().collect())
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let data_map = self.data_map.lock().await;
        Ok(data_map
            .keys()
            .filter(|&key| key.has_prefix(prefix))
            .cloned()
            .collect())
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let mut keys: StoreKeys = vec![];
        let mut prefixes: BTreeSet<StorePrefix> = BTreeSet::new();
        let data_map = self.data_map.lock().await;
        for key in data_map.keys() {
            if key.has_prefix(prefix) {
                let key_strip = key.as_str().strip_prefix(prefix.as_str()).unwrap();
                let key_strip = key_strip.strip_prefix('/').unwrap_or(key_strip);
                let components: Vec<_> = key_strip.split('/').collect();
                if components.len() > 1 {
                    prefixes.insert(StorePrefix::new(
                        prefix.as_str().to_string() + components[0] + "/",
                    )?);
                } else if key.parent().eq(prefix) {
                    keys.push(key.clone());
                }
            }
        }
        Ok(StoreKeysPrefixes::new(keys, prefixes.into_iter().collect()))
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let data_map = self.data_map.lock().await;
        Ok(data_map
            .iter()
            .filter(|(key, _)| key.has_prefix(prefix))
            .map(|(_, value)| value.len() as u64)
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use futures::executor::block_on;

    use super::*;
    use crate::AsyncReadableWritableListableStorageTraits;

    #[test]
    fn async_memory() -> Result<(), Box<dyn Error>> {
        block_on(async {
            let store = AsyncMemoryStore::new();
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
            let store = AsyncMemoryStore::new();
            let key = StoreKey::new("key").unwrap();
            store.set(&key, Bytes::from_static(b"value")).await.unwrap();

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
                Arc::new(AsyncMemoryStore::new());
            crate::store_test::async_store_write(&store.clone().writable()).await?;
            crate::store_test::async_store_read(&store.clone().readable()).await?;
            crate::store_test::async_store_list(&store.clone().listable()).await?;
            crate::store_test::async_store_list_size(&store.clone().listable()).await?;
            Ok(())
        })
    }
}

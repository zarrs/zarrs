//! A storage adapter for atomic writes through temporary keys.

use std::sync::Arc;

use crate::{
    AtomicRenameStorageTraits, Bytes, ListableStorageTraits, MaybeBytesIterator,
    OffsetBytesIterator, ReadableStorageTraits, StorageError, StoreKey, StoreKeys,
    StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};

/// A storage adapter that writes complete values to a temporary key before atomically renaming
/// them to their destination.
///
/// Temporary keys use the destination key with a `.tmp` suffix. An existing temporary key causes
/// the write to fail rather than overwrite evidence of an interrupted write. A failed rename
/// leaves the temporary key in place so it can be identified through listing operations.
#[derive(Debug)]
pub struct AtomicWriteStorageAdapter<TStorage: ?Sized> {
    storage: Arc<TStorage>,
}

impl<TStorage: ?Sized> AtomicWriteStorageAdapter<TStorage> {
    /// Create a new atomic write storage adapter.
    #[must_use]
    pub fn new(storage: Arc<TStorage>) -> Self {
        Self { storage }
    }

    /// Return the temporary key used for `key`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] for the root key, because it has no sibling temporary key.
    pub fn temporary_key(key: &StoreKey) -> Result<StoreKey, StorageError> {
        if key == &StoreKey::root() {
            return Err(StorageError::Other(
                "atomic writes do not support the root store key".to_string(),
            ));
        }
        // SAFETY: Appending ".tmp" to a valid store key always produces a valid store key.
        Ok(unsafe { StoreKey::new_unchecked(format!("{}.tmp", key.as_str())) })
    }
}

impl<
        TStorage: ?Sized
            + AtomicRenameStorageTraits
            + ReadableStorageTraits
            + WritableStorageTraits
            + 'static,
    > AtomicWriteStorageAdapter<TStorage>
{
    fn set_via_temporary_key(
        &self,
        key: &StoreKey,
        temporary_key: &StoreKey,
        value: Bytes,
    ) -> Result<(), StorageError> {
        if self.storage.size_key(temporary_key)?.is_some() {
            return Err(StorageError::Other(format!(
                "temporary key {temporary_key} already exists"
            )));
        }
        self.storage.set(temporary_key, value)?;
        self.storage.rename(temporary_key, key)
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ReadableStorageTraits
    for AtomicWriteStorageAdapter<TStorage>
{
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: crate::byte_range::ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.storage.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.storage.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

impl<TStorage: ?Sized + ListableStorageTraits> ListableStorageTraits
    for AtomicWriteStorageAdapter<TStorage>
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

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.storage.size_prefix(prefix)
    }
}

impl<
        TStorage: ?Sized
            + AtomicRenameStorageTraits
            + ReadableStorageTraits
            + WritableStorageTraits
            + 'static,
    > WritableStorageTraits for AtomicWriteStorageAdapter<TStorage>
{
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let temporary_key = Self::temporary_key(key)?;
        self.set_via_temporary_key(key, &temporary_key, value)
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        let temporary_key = Self::temporary_key(key)?;

        let bytes_out = self.storage.get(key)?.unwrap_or_default();
        let mut bytes_out: bytes::BytesMut = bytes_out.into();
        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if bytes_out.len() < offset + value.len() {
                bytes_out.resize(offset + value.len(), 0);
            }
            bytes_out[offset..offset + value.len()].copy_from_slice(&value);
        }

        self.set_via_temporary_key(key, &temporary_key, bytes_out.freeze())
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.storage.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.storage.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::byte_range::ByteRangeIterator;
    use crate::store::MemoryStore;
    use crate::MaybeBytes;

    use super::*;

    #[derive(Debug, Default)]
    struct RenameStore {
        inner: MemoryStore,
    }

    impl ReadableStorageTraits for RenameStore {
        fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: ByteRangeIterator<'a>,
        ) -> Result<MaybeBytesIterator<'a>, StorageError> {
            self.inner.get_partial_many(key, byte_ranges)
        }

        fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.inner.size_key(key)
        }

        fn supports_get_partial(&self) -> bool {
            self.inner.supports_get_partial()
        }
    }

    impl WritableStorageTraits for RenameStore {
        fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
            self.inner.set(key, value)
        }

        fn set_partial_many(
            &self,
            key: &StoreKey,
            offset_values: OffsetBytesIterator,
        ) -> Result<(), StorageError> {
            self.inner.set_partial_many(key, offset_values)
        }

        fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
            self.inner.erase(key)
        }

        fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
            self.inner.erase_prefix(prefix)
        }

        fn supports_set_partial(&self) -> bool {
            self.inner.supports_set_partial()
        }
    }

    impl AtomicRenameStorageTraits for RenameStore {
        fn rename(&self, source: &StoreKey, destination: &StoreKey) -> Result<(), StorageError> {
            let value: MaybeBytes = self.inner.get(source)?;
            let value = value.ok_or_else(|| StorageError::Other(format!("{source} is missing")))?;
            self.inner.set(destination, value)?;
            self.inner.erase(source)
        }
    }

    #[test]
    fn writes_full_and_partial_values() {
        let adapter = AtomicWriteStorageAdapter::new(Arc::new(RenameStore::default()));
        let key = StoreKey::new("key").unwrap();
        adapter.set(&key, Bytes::from_static(b"00")).unwrap();
        adapter
            .set_partial(&key, 1, Bytes::from_static(b"B"))
            .unwrap();

        assert_eq!(adapter.get(&key).unwrap(), Some(Bytes::from_static(b"0B")));
    }

    #[test]
    fn rejects_existing_temporary_key() {
        let storage = Arc::new(RenameStore::default());
        let key = StoreKey::new("key").unwrap();
        let temporary_key = AtomicWriteStorageAdapter::<RenameStore>::temporary_key(&key).unwrap();
        storage
            .set(&temporary_key, Bytes::from_static(b"incomplete"))
            .unwrap();
        let adapter = AtomicWriteStorageAdapter::new(storage);

        assert_eq!(
            adapter
                .set(&key, Bytes::from_static(b"replacement"))
                .unwrap_err()
                .to_string(),
            "temporary key key.tmp already exists"
        );
    }

    #[test]
    fn rejects_root_key() {
        let adapter = AtomicWriteStorageAdapter::new(Arc::new(RenameStore::default()));

        assert_eq!(
            adapter
                .set(&StoreKey::root(), Bytes::from_static(b"value"))
                .unwrap_err()
                .to_string(),
            "atomic writes do not support the root store key"
        );
    }
}

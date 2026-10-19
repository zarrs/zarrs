//! A storage adapter for atomic writes through temporary keys.

use std::sync::Arc;

use crate::{
    AtomicRenameStorageTraits, CowBytes, ListableStorageTraits, MaybeBytesIterator,
    OffsetBytesIterator, ReadableStorageTraits, StorageError, StoreKey, StoreKeys,
    StoreKeysPrefixes, StorePrefix, StoreWriter, WritableStorageTraits,
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
        value: CowBytes<'_>,
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

    fn read_coalescing(&self) -> Option<crate::ReadCoalescing> {
        self.storage.read_coalescing()
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
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        let temporary_key = Self::temporary_key(key)?;
        self.set_via_temporary_key(key, &temporary_key, value)
    }

    fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
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

        self.set_via_temporary_key(key, &temporary_key, bytes_out.into())
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

    fn set_multipart<'s>(
        &'s self,
        key: &StoreKey,
        size_hint: Option<u64>,
    ) -> Result<Box<dyn StoreWriter + 's>, StorageError> {
        let temporary_key = Self::temporary_key(key)?;
        if self.storage.size_key(&temporary_key)?.is_some() {
            return Err(StorageError::Other(format!(
                "temporary key {temporary_key} already exists"
            )));
        }
        let writer = self.storage.set_multipart(&temporary_key, size_hint)?;
        Ok(Box::new(AtomicStoreWriter {
            writer,
            storage: &*self.storage,
            key: key.clone(),
            temporary_key,
        }))
    }

    fn supports_multipart(&self) -> bool {
        self.storage.supports_multipart()
    }

    fn supports_multipart_patch(&self) -> bool {
        self.storage.supports_multipart_patch()
    }
}

/// A multipart writer to a temporary key, which is renamed to its destination on finish.
struct AtomicStoreWriter<'s, TStorage: ?Sized> {
    writer: Box<dyn StoreWriter + 's>,
    storage: &'s TStorage,
    key: StoreKey,
    temporary_key: StoreKey,
}

impl<TStorage: ?Sized + AtomicRenameStorageTraits> StoreWriter for AtomicStoreWriter<'_, TStorage> {
    fn write(&mut self, part: CowBytes<'_>) -> Result<(), StorageError> {
        self.writer.write(part)
    }

    fn write_many(&mut self, parts: Vec<CowBytes<'_>>) -> Result<(), StorageError> {
        self.writer.write_many(parts)
    }

    fn patch(&mut self, offset: u64, bytes: CowBytes<'_>) -> Result<(), StorageError> {
        self.writer.patch(offset, bytes)
    }

    fn finish(self: Box<Self>) -> Result<(), StorageError> {
        self.writer.finish()?;
        self.storage.rename(&self.temporary_key, &self.key)
    }

    fn abort(self: Box<Self>) -> Result<(), StorageError> {
        self.writer.abort()
    }
}

#[cfg(test)]
mod tests {
    use crate::byte_range::ByteRangeIterator;
    use crate::store::MemoryStore;
    use crate::{Bytes, MaybeBytes};

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
        fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
            self.inner.set(key, value)
        }

        fn set_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            offset_values: OffsetBytesIterator<'a>,
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

    impl ListableStorageTraits for RenameStore {
        fn list(&self) -> Result<StoreKeys, StorageError> {
            self.inner.list()
        }

        fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
            self.inner.list_prefix(prefix)
        }

        fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
            self.inner.list_dir(prefix)
        }

        fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
            self.inner.size_prefix(prefix)
        }
    }

    impl AtomicRenameStorageTraits for RenameStore {
        fn rename(&self, source: &StoreKey, destination: &StoreKey) -> Result<(), StorageError> {
            let value: MaybeBytes = self.inner.get(source)?;
            let value = value.ok_or_else(|| StorageError::Other(format!("{source} is missing")))?;
            self.inner.set(destination, value.into())?;
            self.inner.erase(source)
        }
    }

    #[test]
    fn writes_full_and_partial_values() {
        let adapter = AtomicWriteStorageAdapter::new(Arc::new(RenameStore::default()));
        let key = StoreKey::new("key").unwrap();
        adapter.set(&key, Bytes::from_static(b"00").into()).unwrap();
        adapter
            .set_partial(&key, 1, Bytes::from_static(b"B").into())
            .unwrap();

        assert_eq!(adapter.get(&key).unwrap(), Some(Bytes::from_static(b"0B")));
    }

    #[test]
    fn multipart() {
        let adapter = AtomicWriteStorageAdapter::new(Arc::new(RenameStore::default()));
        crate::store_test::store_multipart(&adapter).unwrap();
    }

    #[test]
    fn multipart_writes_to_the_temporary_key() {
        let storage = Arc::new(RenameStore::default());
        let adapter = AtomicWriteStorageAdapter::new(storage.clone());
        let key = StoreKey::new("key").unwrap();
        let temporary_key = AtomicWriteStorageAdapter::<RenameStore>::temporary_key(&key).unwrap();
        let mut writer = adapter.set_multipart(&key, None).unwrap();
        writer.write(Bytes::from_static(b"ab").into()).unwrap();
        writer.write(Bytes::from_static(b"c").into()).unwrap();
        writer.finish().unwrap();
        assert_eq!(adapter.get(&key).unwrap(), Some(Bytes::from_static(b"abc")));
        assert_eq!(storage.get(&temporary_key).unwrap(), None);

        storage
            .set(&temporary_key, Bytes::from_static(b"incomplete").into())
            .unwrap();
        assert_eq!(
            adapter.set_multipart(&key, None).err().unwrap().to_string(),
            "temporary key key.tmp already exists"
        );
    }

    #[test]
    fn rejects_existing_temporary_key() {
        let storage = Arc::new(RenameStore::default());
        let key = StoreKey::new("key").unwrap();
        let temporary_key = AtomicWriteStorageAdapter::<RenameStore>::temporary_key(&key).unwrap();
        storage
            .set(&temporary_key, Bytes::from_static(b"incomplete").into())
            .unwrap();
        let adapter = AtomicWriteStorageAdapter::new(storage);

        assert_eq!(
            adapter
                .set(&key, Bytes::from_static(b"replacement").into())
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
                .set(&StoreKey::root(), Bytes::from_static(b"value").into())
                .unwrap_err()
                .to_string(),
            "atomic writes do not support the root store key"
        );
    }
}

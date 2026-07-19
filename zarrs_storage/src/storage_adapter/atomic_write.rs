//! A storage adapter for atomic writes through temporary keys.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{
    Bytes, ListableStorageTraits, MaybeBytesIterator, OffsetBytesIterator, ReadableStorageTraits,
    StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};

/// Storage that can atomically rename one key to another.
///
/// Implementations must replace `destination` atomically if it already exists. The source and
/// destination must be on the same storage system.
pub trait AtomicRenameStorageTraits {
    /// Atomically rename `source` to `destination`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the rename fails.
    fn rename(&self, source: &StoreKey, destination: &StoreKey) -> Result<(), StorageError>;
}

/// A storage adapter that writes complete values to a temporary key before atomically renaming
/// them to their destination.
///
/// Temporary keys use the destination key with a `.tmp` suffix. An existing temporary key causes
/// the write to fail rather than overwrite evidence of an interrupted write. A failed rename
/// leaves the temporary key in place so it can be identified through listing operations.
/// Concurrent writes to the same destination or temporary key through one adapter are serialised.
#[derive(Debug)]
pub struct AtomicWriteStorageAdapter<TStorage: ?Sized> {
    storage: Arc<TStorage>,
    writes: Mutex<HashMap<StoreKey, Arc<Mutex<()>>>>,
}

impl<TStorage: ?Sized> AtomicWriteStorageAdapter<TStorage> {
    /// Create a new atomic write storage adapter.
    #[must_use]
    pub fn new(storage: Arc<TStorage>) -> Self {
        Self {
            storage,
            writes: Mutex::default(),
        }
    }

    /// Return the temporary key used for `key`.
    #[must_use]
    pub fn temporary_key(key: &StoreKey) -> StoreKey {
        // SAFETY: Appending ".tmp" to a valid store key always produces a valid store key.
        unsafe { StoreKey::new_unchecked(format!("{}.tmp", key.as_str())) }
    }

    fn get_write_mutexes(
        &self,
        key: &StoreKey,
        temporary_key: &StoreKey,
    ) -> (Arc<Mutex<()>>, Arc<Mutex<()>>) {
        let (first_key, second_key) = if key < temporary_key {
            (key, temporary_key)
        } else {
            (temporary_key, key)
        };
        let mut writes = self.writes.lock().unwrap();
        let first = writes
            .entry(first_key.clone())
            .or_insert_with(|| Arc::new(Mutex::default()))
            .clone();
        let second = writes
            .entry(second_key.clone())
            .or_insert_with(|| Arc::new(Mutex::default()))
            .clone();
        (first, second)
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
    fn set_while_locked(
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
        let temporary_key = Self::temporary_key(key);
        let (first_mutex, second_mutex) = self.get_write_mutexes(key, &temporary_key);
        let _first_lock = first_mutex.lock().unwrap();
        let _second_lock = second_mutex.lock().unwrap();
        self.set_while_locked(key, &temporary_key, value)
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        let temporary_key = Self::temporary_key(key);
        let (first_mutex, second_mutex) = self.get_write_mutexes(key, &temporary_key);
        let _first_lock = first_mutex.lock().unwrap();
        let _second_lock = second_mutex.lock().unwrap();

        let bytes_out = self.storage.get(key)?.unwrap_or_default();
        let mut bytes_out: bytes::BytesMut = bytes_out.into();
        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if bytes_out.len() < offset + value.len() {
                bytes_out.resize(offset + value.len(), 0);
            }
            bytes_out[offset..offset + value.len()].copy_from_slice(&value);
        }

        self.set_while_locked(key, &temporary_key, bytes_out.freeze())
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
    use std::sync::{mpsc, Condvar};
    use std::thread;
    use std::time::Duration;

    use crate::byte_range::ByteRangeIterator;
    use crate::store::MemoryStore;
    use crate::MaybeBytes;

    use super::*;

    #[derive(Debug, Default)]
    struct BlockingReadStore {
        inner: MemoryStore,
        read_count: Mutex<usize>,
        read_count_changed: Condvar,
        release_first_read: Mutex<bool>,
        release_first_read_changed: Condvar,
    }

    impl BlockingReadStore {
        fn before_read(&self) {
            let mut read_count = self.read_count.lock().unwrap();
            *read_count += 1;
            let is_first_read = *read_count == 1;
            self.read_count_changed.notify_all();
            drop(read_count);

            if is_first_read {
                let release = self.release_first_read.lock().unwrap();
                drop(
                    self.release_first_read_changed
                        .wait_while(release, |release| !*release)
                        .unwrap(),
                );
            }
        }

        fn wait_for_read_count(&self, expected: usize, timeout: Duration) -> bool {
            let read_count = self.read_count.lock().unwrap();
            let (read_count, _) = self
                .read_count_changed
                .wait_timeout_while(read_count, timeout, |read_count| *read_count < expected)
                .unwrap();
            *read_count >= expected
        }

        fn release_first_read(&self) {
            *self.release_first_read.lock().unwrap() = true;
            self.release_first_read_changed.notify_all();
        }
    }

    impl ReadableStorageTraits for BlockingReadStore {
        fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: ByteRangeIterator<'a>,
        ) -> Result<MaybeBytesIterator<'a>, StorageError> {
            self.before_read();
            self.inner.get_partial_many(key, byte_ranges)
        }

        fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.inner.size_key(key)
        }

        fn supports_get_partial(&self) -> bool {
            self.inner.supports_get_partial()
        }
    }

    impl WritableStorageTraits for BlockingReadStore {
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

    impl AtomicRenameStorageTraits for BlockingReadStore {
        fn rename(&self, source: &StoreKey, destination: &StoreKey) -> Result<(), StorageError> {
            let value: MaybeBytes = self.inner.get(source)?;
            let value = value.ok_or_else(|| StorageError::Other(format!("{source} is missing")))?;
            self.inner.set(destination, value)?;
            self.inner.erase(source)
        }
    }

    #[test]
    fn temporary_key_is_reserved_as_a_destination() {
        let adapter = AtomicWriteStorageAdapter::new(Arc::new(MemoryStore::new()));
        let key = StoreKey::new("key").unwrap();
        let temporary_key = AtomicWriteStorageAdapter::<MemoryStore>::temporary_key(&key);
        let nested_temporary_key =
            AtomicWriteStorageAdapter::<MemoryStore>::temporary_key(&temporary_key);

        let key_mutexes = adapter.get_write_mutexes(&key, &temporary_key);
        let temporary_key_mutexes =
            adapter.get_write_mutexes(&temporary_key, &nested_temporary_key);
        assert!(
            Arc::ptr_eq(&key_mutexes.0, &temporary_key_mutexes.0)
                || Arc::ptr_eq(&key_mutexes.0, &temporary_key_mutexes.1)
                || Arc::ptr_eq(&key_mutexes.1, &temporary_key_mutexes.0)
                || Arc::ptr_eq(&key_mutexes.1, &temporary_key_mutexes.1)
        );
    }

    #[test]
    fn partial_writes_are_serialised_before_reading() {
        let storage = Arc::new(BlockingReadStore::default());
        let key = StoreKey::new("key").unwrap();
        storage.inner.set(&key, Bytes::from_static(b"00")).unwrap();
        let adapter = Arc::new(AtomicWriteStorageAdapter::new(storage.clone()));

        let first_adapter = adapter.clone();
        let first_key = key.clone();
        let first = thread::spawn(move || {
            first_adapter.set_partial(&first_key, 0, Bytes::from_static(b"A"))
        });
        assert!(storage.wait_for_read_count(1, Duration::from_secs(1)));

        let (second_started_tx, second_started_rx) = mpsc::channel();
        let second_adapter = adapter.clone();
        let second_key = key.clone();
        let second = thread::spawn(move || {
            second_started_tx.send(()).unwrap();
            second_adapter.set_partial(&second_key, 1, Bytes::from_static(b"B"))
        });
        second_started_rx.recv().unwrap();

        assert!(!storage.wait_for_read_count(2, Duration::from_millis(250)));
        storage.release_first_read();
        first.join().unwrap().unwrap();
        second.join().unwrap().unwrap();
        assert_eq!(adapter.get(&key).unwrap(), Some(Bytes::from_static(b"AB")));
    }
}

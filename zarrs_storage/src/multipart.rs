//! Multipart (streaming) writes of a store value.
//!
//! A multipart write sets a value as a sequence of parts, with [`WritableStorageTraits::set_multipart`] or
//! [`AsyncWritableStorageTraits::set_multipart`](crate::AsyncWritableStorageTraits::set_multipart).
//! The value is visible only once the writer is finished, and a writer that is aborted (or dropped) leaves any
//! existing value in place.
//!
//! A store that cannot write a value in parts collects them with a [`BufferedStoreWriter`] and sets the value on
//! finish, so streaming a value to it gains nothing over setting it whole.
//! The `supports_multipart` methods of the storage traits tell these apart.

use crate::{CowBytes, MaybeSend, StorageError, StoreKey, WritableStorageTraits};

/// A writer of a store value in parts.
///
/// Created with [`WritableStorageTraits::set_multipart`].
pub trait StoreWriter: MaybeSend {
    /// Append the next part of the value.
    ///
    /// Parts may be of any size, and a store rechunks them as it requires.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn write(&mut self, part: CowBytes<'_>) -> Result<(), StorageError>;

    /// Append several parts of the value, in order.
    ///
    /// The default implementation writes each part with [`write`](StoreWriter::write).
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn write_many(&mut self, parts: Vec<CowBytes<'_>>) -> Result<(), StorageError> {
        parts.into_iter().try_for_each(|part| self.write(part))
    }

    /// Overwrite the bytes already written at `offset` with `bytes`, before the value is finished.
    ///
    /// Supported if the store [`supports_multipart_patch`](WritableStorageTraits::supports_multipart_patch).
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] by default, or a [`StorageError`] if the bytes were not written yet or
    /// there is an underlying storage error.
    fn patch(&mut self, offset: u64, bytes: CowBytes<'_>) -> Result<(), StorageError> {
        let _ = (offset, bytes);
        Err(patch_unsupported())
    }

    /// Complete the value, making it visible in the store.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn finish(self: Box<Self>) -> Result<(), StorageError>;

    /// Abandon the value, leaving any existing value in place.
    ///
    /// A writer dropped without being finished is aborted on a best effort basis, ignoring errors.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn abort(self: Box<Self>) -> Result<(), StorageError>;
}

/// An asynchronous writer of a store value in parts.
///
/// Created with [`AsyncWritableStorageTraits::set_multipart`](crate::AsyncWritableStorageTraits::set_multipart).
#[cfg(feature = "async")]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait AsyncStoreWriter: MaybeSend {
    /// Append the next part of the value.
    ///
    /// Parts may be of any size, and a store rechunks them as it requires.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn write(&mut self, part: CowBytes<'_>) -> Result<(), StorageError>;

    /// Append several parts of the value, in order.
    ///
    /// The default implementation writes each part with [`write`](AsyncStoreWriter::write).
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn write_many(&mut self, parts: Vec<CowBytes<'_>>) -> Result<(), StorageError> {
        for part in parts {
            self.write(part).await?;
        }
        Ok(())
    }

    /// Overwrite the bytes already written at `offset` with `bytes`, before the value is finished.
    ///
    /// Supported if the store
    /// [`supports_multipart_patch`](crate::AsyncWritableStorageTraits::supports_multipart_patch).
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] by default, or a [`StorageError`] if the bytes were not written yet or
    /// there is an underlying storage error.
    async fn patch(&mut self, offset: u64, bytes: CowBytes<'_>) -> Result<(), StorageError> {
        let _ = (offset, bytes);
        Err(patch_unsupported())
    }

    /// Complete the value, making it visible in the store.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn finish(self: Box<Self>) -> Result<(), StorageError>;

    /// Abandon the value, leaving any existing value in place.
    ///
    /// A writer dropped without being finished is aborted on a best effort basis, ignoring errors.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn abort(self: Box<Self>) -> Result<(), StorageError>;
}

/// The error of a multipart writer that does not support [`StoreWriter::patch`].
fn patch_unsupported() -> StorageError {
    StorageError::Unsupported("the multipart writer does not support patching".to_string())
}

/// The parts of a value collected by a buffered writer.
#[derive(Default)]
struct BufferedParts(Vec<CowBytes<'static>>);

impl BufferedParts {
    /// Append `part`, copying it only if it is borrowed.
    fn push(&mut self, part: CowBytes<'_>) {
        self.0.push(part.into_static());
    }

    /// Append `parts`, copying only those that are borrowed.
    fn extend(&mut self, parts: Vec<CowBytes<'_>>) {
        self.0.reserve(parts.len());
        self.0.extend(parts.into_iter().map(CowBytes::into_static));
    }

    /// The value: a single part as is, or several parts concatenated into one exact-size allocation.
    fn concatenate(self) -> CowBytes<'static> {
        let mut parts = self.0;
        if parts.len() <= 1 {
            return parts.pop().unwrap_or_default();
        }
        let len = parts.iter().map(CowBytes::len).sum();
        let mut value = Vec::with_capacity(len);
        for part in parts {
            // Each part is dropped once it is copied.
            value.extend_from_slice(&part);
        }
        CowBytes::from(value)
    }
}

/// A [`StoreWriter`] that collects the parts of a value and sets it on finish.
///
/// This is the default writer of [`WritableStorageTraits::set_multipart`].
/// A single part is set as is, and several parts are concatenated into one allocation.
/// Borrowed parts are copied when they are written.
pub struct BufferedStoreWriter<'s, TStorage: ?Sized> {
    storage: &'s TStorage,
    key: StoreKey,
    parts: BufferedParts,
}

impl<'s, TStorage: ?Sized> BufferedStoreWriter<'s, TStorage> {
    /// Create a writer of the value at `key` in `storage`.
    #[must_use]
    pub fn new(storage: &'s TStorage, key: StoreKey) -> Self {
        Self {
            storage,
            key,
            parts: BufferedParts::default(),
        }
    }
}

impl<TStorage: ?Sized + WritableStorageTraits> StoreWriter for BufferedStoreWriter<'_, TStorage> {
    fn write(&mut self, part: CowBytes<'_>) -> Result<(), StorageError> {
        self.parts.push(part);
        Ok(())
    }

    fn write_many(&mut self, parts: Vec<CowBytes<'_>>) -> Result<(), StorageError> {
        self.parts.extend(parts);
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), StorageError> {
        let Self {
            storage,
            key,
            parts,
        } = *self;
        storage.set(&key, parts.concatenate())
    }

    fn abort(self: Box<Self>) -> Result<(), StorageError> {
        Ok(())
    }
}

/// An [`AsyncStoreWriter`] that collects the parts of a value and sets it on finish.
///
/// This is the default writer of
/// [`AsyncWritableStorageTraits::set_multipart`](crate::AsyncWritableStorageTraits::set_multipart).
/// A single part is set as is, and several parts are concatenated into one allocation.
/// Borrowed parts are copied when they are written.
#[cfg(feature = "async")]
pub struct AsyncBufferedStoreWriter<'s, TStorage: ?Sized> {
    storage: &'s TStorage,
    key: StoreKey,
    parts: BufferedParts,
}

#[cfg(feature = "async")]
impl<'s, TStorage: ?Sized> AsyncBufferedStoreWriter<'s, TStorage> {
    /// Create a writer of the value at `key` in `storage`.
    #[must_use]
    pub fn new(storage: &'s TStorage, key: StoreKey) -> Self {
        Self {
            storage,
            key,
            parts: BufferedParts::default(),
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl<TStorage: ?Sized + crate::AsyncWritableStorageTraits> AsyncStoreWriter
    for AsyncBufferedStoreWriter<'_, TStorage>
{
    async fn write(&mut self, part: CowBytes<'_>) -> Result<(), StorageError> {
        self.parts.push(part);
        Ok(())
    }

    async fn write_many(&mut self, parts: Vec<CowBytes<'_>>) -> Result<(), StorageError> {
        self.parts.extend(parts);
        Ok(())
    }

    async fn finish(self: Box<Self>) -> Result<(), StorageError> {
        let Self {
            storage,
            key,
            parts,
        } = *self;
        storage.set(&key, parts.concatenate()).await
    }

    async fn abort(self: Box<Self>) -> Result<(), StorageError> {
        Ok(())
    }
}

/// A value appended part by part into one buffer, reserved from a size hint.
pub(crate) struct AppendBuffer(Vec<u8>);

impl AppendBuffer {
    /// Create a buffer with capacity for `size_hint` bytes, an upper bound on the size of the value.
    pub(crate) fn new(size_hint: Option<u64>) -> Self {
        let capacity = size_hint.map_or(0, |size| usize::try_from(size).unwrap_or(0));
        Self(Vec::with_capacity(capacity))
    }

    pub(crate) fn push(&mut self, part: &[u8]) {
        self.0.extend_from_slice(part);
    }

    pub(crate) fn extend(&mut self, parts: &[CowBytes<'_>]) {
        self.0.reserve(parts.iter().map(CowBytes::len).sum());
        for part in parts {
            self.0.extend_from_slice(part);
        }
    }

    pub(crate) fn into_value(self) -> CowBytes<'static> {
        // `Bytes` adopts the allocation, including any unused capacity, without copying.
        CowBytes::from(self.0)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::byte_range::ByteRangeIterator;
    use crate::store::MemoryStore;
    use crate::{
        Bytes, ListableStorageTraits, MaybeBytesIterator, OffsetBytesIterator,
        ReadableStorageTraits, StoreKeys, StoreKeysPrefixes, StorePrefix,
    };

    /// A [`MemoryStore`] with the default (buffered) multipart writer.
    #[derive(Default)]
    struct BufferedMemoryStore(MemoryStore);

    impl ReadableStorageTraits for BufferedMemoryStore {
        fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: ByteRangeIterator<'a>,
        ) -> Result<MaybeBytesIterator<'a>, StorageError> {
            self.0.get_partial_many(key, byte_ranges)
        }

        fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.0.size_key(key)
        }

        fn supports_get_partial(&self) -> bool {
            true
        }
    }

    impl WritableStorageTraits for BufferedMemoryStore {
        fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
            self.0.set(key, value)
        }

        fn set_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            offset_values: OffsetBytesIterator<'a>,
        ) -> Result<(), StorageError> {
            self.0.set_partial_many(key, offset_values)
        }

        fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
            self.0.erase(key)
        }

        fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
            self.0.erase_prefix(prefix)
        }

        fn supports_set_partial(&self) -> bool {
            true
        }
    }

    impl ListableStorageTraits for BufferedMemoryStore {
        fn list(&self) -> Result<StoreKeys, StorageError> {
            self.0.list()
        }

        fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
            self.0.list_prefix(prefix)
        }

        fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
            self.0.list_dir(prefix)
        }

        fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
            self.0.size_prefix(prefix)
        }
    }

    #[test]
    fn buffered_store_writer() -> Result<(), Box<dyn Error>> {
        let store = BufferedMemoryStore::default();
        assert!(!store.supports_multipart());
        crate::store_test::store_multipart(&store)
    }

    #[test]
    fn buffered_store_writer_single_part_is_not_copied() -> Result<(), Box<dyn Error>> {
        let store = BufferedMemoryStore::default();
        let key = StoreKey::new("a")?;
        let part = Bytes::from(vec![1u8, 2, 3]);
        let mut writer = store.set_multipart(&key, None)?;
        writer.write_many(vec![part.clone().into()])?;
        writer.finish()?;
        assert_eq!(store.get(&key)?.unwrap().as_ptr(), part.as_ptr());
        Ok(())
    }

    #[test]
    fn append_buffer_reserves_from_the_size_hint() {
        let mut buffer = AppendBuffer::new(Some(6));
        let capacity = buffer.0.capacity();
        buffer.push(&[1, 2]);
        buffer.extend(&[vec![3u8].into(), vec![4, 5, 6].into()]);
        assert_eq!(buffer.0.capacity(), capacity);
        assert_eq!(&*buffer.into_value(), &[1, 2, 3, 4, 5, 6]);
    }
}

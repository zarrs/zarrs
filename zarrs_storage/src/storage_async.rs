use std::sync::Arc;

use auto_impl::auto_impl;
use bytes::BytesMut;
use futures::StreamExt;

use super::{
    byte_range::ByteRangeIterator, MaybeSend, MaybeSync, StorageError, StoreKey, StoreKeys,
    StoreKeysPrefixes, StorePrefix, StorePrefixes,
};
use crate::{
    byte_range::ByteRange, AsyncMaybeBytesIterator, Bytes, MaybeBytes, OffsetBytesIterator,
};

/// Async readable storage traits.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
#[auto_impl(Arc)]
pub trait AsyncReadableStorageTraits: MaybeSend + MaybeSync {
    /// Retrieve the value (bytes) associated with a given [`StoreKey`].
    ///
    /// Returns [`None`] if the key is not found.
    ///
    /// # Errors
    ///
    /// Returns a [`StorageError`] if the store key does not exist or there is an error with the underlying store.
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.get_partial(key, ByteRange::FromStart(0, None)).await
    }

    /// Retrieve partial bytes from a list of byte ranges for a store key.
    ///
    /// Returns [`None`] if the key is not found.
    ///
    /// # Errors
    ///
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError>;

    /// Retrieve partial bytes from a single byte range for a store key.
    ///
    /// Returns [`None`] if the key is not found.
    ///
    /// # Errors
    ///
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn get_partial<'a>(
        &'a self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        let mut result = self
            .get_partial_many(key, Box::new([byte_range].into_iter()))
            .await?;
        if let Some(result) = &mut result {
            let bytes = result.next().await.expect("one byte range")?;
            debug_assert!(result.next().await.is_none());
            Ok(Some(bytes))
        } else {
            Ok(None)
        }
    }

    /// Return the size in bytes of the value at `key`.
    ///
    /// Returns [`None`] if the key is not found.
    ///
    /// # Errors
    ///
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError>;

    /// Returns whether this store supports partial reads.
    ///
    /// If this returns `true`, the store can efficiently handle `get_partial` and `get_partial_many` operations.
    /// If this returns `false`, partial reads will fall back to a full read operation.
    fn supports_get_partial(&self) -> bool;
}

/// Async listable storage traits.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
#[auto_impl(Arc)]
pub trait AsyncListableStorageTraits: MaybeSend + MaybeSync {
    /// Retrieve all [`StoreKeys`] in the store.
    ///
    /// # Errors
    ///
    /// Returns a [`StorageError`] if there is an underlying error with the store.
    async fn list(&self) -> Result<StoreKeys, StorageError>;

    /// Retrieve all [`StoreKeys`] with a given [`StorePrefix`].
    ///
    /// # Errors
    ///
    /// Returns a [`StorageError`] if the prefix is not a directory or there is an underlying error with the store.
    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError>;

    /// Retrieve all [`StoreKeys`] and [`StorePrefix`] which are direct children of [`StorePrefix`].
    ///
    /// # Errors
    ///
    /// Returns a [`StorageError`] if the prefix is not a directory or there is an underlying error with the store.
    ///
    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError>;

    /// Return the size in bytes of all keys under `prefix`.
    ///
    /// # Errors
    ///
    /// Returns a `StorageError` if the store does not support `size()` or there is an underlying error with the store.
    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError>;

    /// Return the size in bytes of the storage.
    ///
    /// # Errors
    ///
    /// Returns a `StorageError` if the store does not support `size()` or there is an underlying error with the store.
    async fn size(&self) -> Result<u64, StorageError> {
        self.size_prefix(&StorePrefix::root()).await
    }
}

/// Set partial values for an asynchronous store.
///
/// This method reads entire values, updates them, and replaces them.
/// Stores can use this internally if they do not support updating/appending without replacement.
///
/// # Errors
/// Returns a [`StorageError`] if an underlying store operation fails.
///
/// # Panics
/// Panics if a key ends beyond `usize::MAX`.
pub async fn async_store_set_partial_many<T: AsyncReadableWritableStorageTraits>(
    store: &T,
    key: &StoreKey,
    offset_values: OffsetBytesIterator<'_>,
) -> Result<(), StorageError> {
    // Read the store key
    let bytes_out = store.get(key).await?.unwrap_or_default();
    let mut bytes_out: BytesMut = bytes_out.into();

    // Update the store key
    for (offset, value) in offset_values {
        let offset = usize::try_from(offset).unwrap();
        if bytes_out.len() < offset + value.len() {
            bytes_out.resize(offset + value.len(), 0);
        }
        bytes_out[offset..offset + value.len()].copy_from_slice(&value);
    }

    // Write the store key
    store.set(key, bytes_out.into()).await
}

/// Async writable storage traits.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
#[auto_impl(Arc)]
pub trait AsyncWritableStorageTraits: MaybeSend + MaybeSync {
    /// Store bytes at a [`StoreKey`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] on failure to store.
    async fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError>;

    /// Store bytes from an offset and value.
    ///
    /// # Errors
    /// Returns a [`StorageError`] on failure to store.
    async fn set_partial(
        &self,
        key: &StoreKey,
        offset: u64,
        value: Bytes,
    ) -> Result<(), StorageError> {
        self.set_partial_many(key, Box::new([(offset, value)].into_iter()))
            .await
    }

    /// Store bytes from a [`OffsetBytesIterator`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] on failure to store.
    async fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError>;

    /// Erase a [`StoreKey`].
    ///
    /// Succeeds if the key does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn erase(&self, key: &StoreKey) -> Result<(), StorageError>;

    /// Erase a list of [`StoreKey`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn erase_many(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        let futures_erase = keys.iter().map(|key| self.erase(key));
        futures::future::join_all(futures_erase)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    /// Erase all [`StoreKey`] under [`StorePrefix`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError>;

    /// Returns whether this store supports partial writes.
    ///
    /// If this returns `true`, the store can efficiently handle `set_partial` and `set_partial_many` operations.
    /// If this returns `false`, partial sets will fall back to a full read and write operation.
    fn supports_set_partial(&self) -> bool;
}

/// A supertrait of [`AsyncReadableStorageTraits`] and [`AsyncWritableStorageTraits`].
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait AsyncReadableWritableStorageTraits:
    AsyncReadableStorageTraits + AsyncWritableStorageTraits
{
    /// Return a readable version of the store.
    fn readable(self: Arc<Self>) -> Arc<dyn AsyncReadableStorageTraits>;

    /// Return a writable version of the store.
    fn writable(self: Arc<Self>) -> Arc<dyn AsyncWritableStorageTraits>;
}

impl<T> AsyncReadableWritableStorageTraits for T
where
    T: AsyncReadableStorageTraits + AsyncWritableStorageTraits + 'static,
{
    fn readable(self: Arc<Self>) -> Arc<dyn AsyncReadableStorageTraits> {
        self.clone()
    }

    fn writable(self: Arc<Self>) -> Arc<dyn AsyncWritableStorageTraits> {
        self.clone()
    }
}

/// A supertrait of [`AsyncReadableStorageTraits`] and [`AsyncListableStorageTraits`].
pub trait AsyncReadableListableStorageTraits:
    AsyncReadableStorageTraits + AsyncListableStorageTraits
{
    /// Return a readable version of the store.
    fn readable(self: Arc<Self>) -> Arc<dyn AsyncReadableStorageTraits>;

    /// Return a listable version of the store.
    fn listable(self: Arc<Self>) -> Arc<dyn AsyncListableStorageTraits>;
}

impl<T> AsyncReadableListableStorageTraits for T
where
    T: AsyncReadableStorageTraits + AsyncListableStorageTraits + 'static,
{
    fn readable(self: Arc<Self>) -> Arc<dyn AsyncReadableStorageTraits> {
        self.clone()
    }

    fn listable(self: Arc<Self>) -> Arc<dyn AsyncListableStorageTraits> {
        self.clone()
    }
}

/// A supertrait of [`AsyncReadableWritableStorageTraits`] and [`AsyncListableStorageTraits`].
pub trait AsyncReadableWritableListableStorageTraits:
    AsyncReadableWritableStorageTraits + AsyncListableStorageTraits
{
    /// Return a readable and writable version of the store.
    fn readable_writable(self: Arc<Self>) -> Arc<dyn AsyncReadableWritableStorageTraits>;

    /// Return a readable and listable version of the store.
    fn readable_listable(self: Arc<Self>) -> Arc<dyn AsyncReadableListableStorageTraits>;

    /// Return a listable version of the store.
    fn listable(self: Arc<Self>) -> Arc<dyn AsyncListableStorageTraits>;
}

impl<T> AsyncReadableWritableListableStorageTraits for T
where
    T: AsyncReadableWritableStorageTraits + AsyncListableStorageTraits + 'static,
{
    fn readable_writable(self: Arc<Self>) -> Arc<dyn AsyncReadableWritableStorageTraits> {
        self.clone()
    }

    fn readable_listable(self: Arc<Self>) -> Arc<dyn AsyncReadableListableStorageTraits> {
        self.clone()
    }

    fn listable(self: Arc<Self>) -> Arc<dyn AsyncListableStorageTraits> {
        self.clone()
    }
}

/// Asynchronously discover the children of a store prefix.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub async fn async_discover_children<
    TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
>(
    storage: &Arc<TStorage>,
    prefix: &StorePrefix,
) -> Result<StorePrefixes, StorageError> {
    let children: Result<Vec<_>, _> = storage
        .list_dir(prefix)
        .await?
        .prefixes()
        .iter()
        .filter(|v| !v.as_str().starts_with("__"))
        .map(|v| StorePrefix::new(v.as_str()))
        .collect();
    Ok(children?)
}

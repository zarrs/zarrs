//! An async to sync storage adapter.
//!
//! This adapter has footguns, see [`AsyncToSyncStorageAdapter`].
//!
//! The docs for the [`AsyncToSyncBlockOn`] trait include an example implementation for the `tokio` runtime.

use crate::{
    byte_range::ByteRangeIterator, AsyncListableStorageTraits, AsyncReadableStorageTraits,
    AsyncWritableStorageTraits, Bytes, ListableStorageTraits, MaybeBytesIterator, MaybeSend,
    MaybeSync, ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes,
    StorePrefix, WritableStorageTraits,
};

use futures::StreamExt;
use std::sync::Arc;

/// Trait for an asynchronous runtime implementing `block_on`.
///
/// ### Example `tokio` implementation of [`AsyncToSyncBlockOn`].
/// ```rust,ignore
/// # use zarrs_storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;
/// struct TokioBlockOn(tokio::runtime::Handle);
///
/// impl AsyncToSyncBlockOn for TokioBlockOn {
///     fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
///         self.0.block_on(future)
///     }
/// }
pub trait AsyncToSyncBlockOn: MaybeSend + MaybeSync {
    /// Runs a future to completion.
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output;
}

/// An async to sync storage adapter.
///
/// The [`AsyncToSyncBlockOn`] implementation must be compatible with the asynchonous store.
/// Incompatibility may result in runtime errors. For example:
/// > there is no reactor running, must be called from the context of a Tokio 1.x runtime
///
/// An [`AsyncToSyncStorageAdapter`] will panic if called within an asynchronous execution context!
pub struct AsyncToSyncStorageAdapter<TStorage: ?Sized, TBlockOn: AsyncToSyncBlockOn> {
    storage: Arc<TStorage>,
    block_on: TBlockOn,
}

impl<TStorage: ?Sized, TBlockOn: AsyncToSyncBlockOn> AsyncToSyncStorageAdapter<TStorage, TBlockOn> {
    /// Create a new async to sync storage adapter.
    #[must_use]
    pub fn new(storage: Arc<TStorage>, block_on: TBlockOn) -> Self {
        Self { storage, block_on }
    }

    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.block_on.block_on(future)
    }
}

impl<TStorage: ?Sized + AsyncReadableStorageTraits, TBlockOn: AsyncToSyncBlockOn>
    ReadableStorageTraits for AsyncToSyncStorageAdapter<TStorage, TBlockOn>
{
    fn get_byte_ranges<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let results = self.block_on(self.storage.get_byte_ranges(key, byte_ranges))?;
        if let Some(results) = results {
            let results = self.block_on(results.collect::<Vec<_>>());
            Ok(Some(Box::new(results.into_iter())))
        } else {
            Ok(None)
        }
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.block_on(self.storage.size_key(key))
    }
}

impl<TStorage: ?Sized + AsyncListableStorageTraits, TBlockOn: AsyncToSyncBlockOn>
    ListableStorageTraits for AsyncToSyncStorageAdapter<TStorage, TBlockOn>
{
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.block_on(self.storage.list())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.block_on(self.storage.list_prefix(prefix))
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.block_on(self.storage.list_dir(prefix))
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.block_on(self.storage.size_prefix(prefix))
    }
}

impl<TStorage: ?Sized + AsyncWritableStorageTraits, TBlockOn: AsyncToSyncBlockOn>
    WritableStorageTraits for AsyncToSyncStorageAdapter<TStorage, TBlockOn>
{
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        self.block_on(self.storage.set(key, value))
    }

    fn set_partial_values(
        &self,
        key_offset_values: &[crate::StoreKeyOffsetValue],
    ) -> Result<(), StorageError> {
        self.block_on(self.storage.set_partial_values(key_offset_values))
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.block_on(self.storage.erase(key))
    }

    fn erase_values(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        self.block_on(self.storage.erase_values(keys))
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.block_on(self.storage.erase_prefix(prefix))
    }
}

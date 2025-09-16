//! A sync to async storage adapter.
//!
//! The docs for the [`SyncToAsyncSpawnBlocking`] trait include an example implementation for the `tokio` runtime.

use crate::{
    byte_range::ByteRangeIterator, AsyncListableStorageTraits, AsyncMaybeBytesIterator,
    AsyncReadableStorageTraits, AsyncWritableStorageTraits, Bytes, ListableStorageTraits,
    MaybeSend, MaybeSync, OffsetBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
    StoreKeys, StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};

use futures::stream;
use std::sync::Arc;

/// Trait for spawning synchronous work onto an asynchronous runtime.
///
/// ### Example `tokio` implementation of [`SyncToAsyncSpawnBlocking`].
/// ```rust,ignore
/// # use zarrs_storage::storage_adapter::sync_to_async::SyncToAsyncSpawnBlocking;
/// struct TokioSpawnBlocking;
///
/// impl SyncToAsyncSpawnBlocking for TokioSpawnBlocking {
///     fn spawn_blocking<F, R>(&self, f: F) -> impl std::future::Future<Output = R> + Send
///     where
///         F: FnOnce() -> R + Send + 'static,
///         R: Send + 'static,
///     {
///         async move {
///             tokio::task::spawn_blocking(f).await.unwrap()
///         }
///     }
/// }
/// ```
pub trait SyncToAsyncSpawnBlocking: MaybeSend + MaybeSync {
    /// Spawns a blocking task.
    fn spawn_blocking<F, R>(&self, f: F) -> impl std::future::Future<Output = R> + MaybeSend
    where
        F: FnOnce() -> R + MaybeSend + 'static,
        R: MaybeSend + 'static;
}

/// A sync to async storage adapter.
///
/// A [`SyncToAsyncStorageAdapter`] uses `spawn_blocking` to run synchronous operations
/// asynchronously without blocking the async runtime.
pub struct SyncToAsyncStorageAdapter<TStorage: ?Sized, TSpawnBlocking: SyncToAsyncSpawnBlocking> {
    storage: Arc<TStorage>,
    spawn_blocking: TSpawnBlocking,
}

impl<TStorage: ?Sized, TSpawnBlocking: SyncToAsyncSpawnBlocking>
    SyncToAsyncStorageAdapter<TStorage, TSpawnBlocking>
{
    /// Create a new sync to async storage adapter.
    #[must_use]
    pub fn new(storage: Arc<TStorage>, spawn_blocking: TSpawnBlocking) -> Self {
        Self {
            storage,
            spawn_blocking,
        }
    }

    async fn spawn_blocking<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + MaybeSend + 'static,
        R: MaybeSend + 'static,
    {
        self.spawn_blocking.spawn_blocking(f).await
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        TSpawnBlocking: SyncToAsyncSpawnBlocking,
    > AsyncReadableStorageTraits for SyncToAsyncStorageAdapter<TStorage, TSpawnBlocking>
{
    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        let key = key.clone();
        let byte_ranges: Vec<_> = byte_ranges.collect();
        let storage = self.storage.clone();

        let results = self
            .spawn_blocking(
                move || -> Result<Option<Vec<Result<Bytes, StorageError>>>, StorageError> {
                    let iterator =
                        storage.get_partial_many(&key, Box::new(byte_ranges.into_iter()))?;
                    match iterator {
                        Some(iterator) => Ok(Some(iterator.collect::<Vec<_>>())),
                        None => Ok(None),
                    }
                },
            )
            .await?;

        if let Some(results) = results {
            Ok(Some(Box::pin(stream::iter(results))))
        } else {
            Ok(None)
        }
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let key = key.clone();
        let storage = self.storage.clone();

        self.spawn_blocking(move || storage.size_key(&key)).await
    }

    fn supports_get_partial(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<
        TStorage: ?Sized + ListableStorageTraits + 'static,
        TSpawnBlocking: SyncToAsyncSpawnBlocking,
    > AsyncListableStorageTraits for SyncToAsyncStorageAdapter<TStorage, TSpawnBlocking>
{
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.list()).await
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let prefix = prefix.clone();
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.list_prefix(&prefix))
            .await
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let prefix = prefix.clone();
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.list_dir(&prefix)).await
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let prefix = prefix.clone();
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.size_prefix(&prefix))
            .await
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<
        TStorage: ?Sized + WritableStorageTraits + 'static,
        TSpawnBlocking: SyncToAsyncSpawnBlocking,
    > AsyncWritableStorageTraits for SyncToAsyncStorageAdapter<TStorage, TSpawnBlocking>
{
    async fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let key = key.clone();
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.set(&key, value)).await
    }

    async fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        let key = key.clone();
        let offset_values: Vec<_> = offset_values.collect();
        let storage = self.storage.clone();

        self.spawn_blocking(move || {
            storage.set_partial_many(&key, Box::new(offset_values.into_iter()))
        })
        .await
    }

    async fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        let key = key.clone();
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.erase(&key)).await
    }

    async fn erase_many(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        let keys = keys.to_vec();
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.erase_many(&keys)).await
    }

    async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        let prefix = prefix.clone();
        let storage = self.storage.clone();
        self.spawn_blocking(move || storage.erase_prefix(&prefix))
            .await
    }

    fn supports_set_partial(&self) -> bool {
        self.storage.supports_set_partial()
    }
}

//! A sync to async storage adapter.
//!
//! The docs for the [`SyncToAsyncSpawnBlocking`] trait include an example implementation for the `tokio` runtime.

use std::sync::Arc;

use futures::stream;

use crate::byte_range::ByteRangeIterator;
use crate::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableStorageTraits,
    AsyncStoreWriter, AsyncWritableStorageTraits, Bytes, CowBytes, ListableStorageTraits,
    MaybeSend, MaybeSync, OffsetBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
    StoreKeys, StoreKeysPrefixes, StorePrefix, StoreWriter, WritableStorageTraits,
};

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

    fn read_coalescing(&self) -> Option<crate::ReadCoalescing> {
        self.storage.read_coalescing()
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
    async fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        let key = key.clone();
        let storage = self.storage.clone();
        // The value must outlive this method to cross onto the blocking thread.
        let value = value.into_static();
        self.spawn_blocking(move || storage.set(&key, value)).await
    }

    async fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        let key = key.clone();
        // The values must outlive this method to cross onto the blocking thread.
        let offset_values: Vec<_> = offset_values
            .map(|(offset, value)| (offset, value.into_static()))
            .collect();
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

    async fn set_multipart<'s>(
        &'s self,
        key: &StoreKey,
        size_hint: Option<u64>,
    ) -> Result<Box<dyn AsyncStoreWriter + 's>, StorageError> {
        let key = key.clone();
        let storage = self.storage.clone();
        let writer = self
            .spawn_blocking(move || OwnedStoreWriter::new(storage, &key, size_hint))
            .await?;
        Ok(Box::new(SyncToAsyncStoreWriter {
            writer: Some(writer),
            spawn_blocking: &self.spawn_blocking,
        }))
    }

    fn supports_multipart(&self) -> bool {
        self.storage.supports_multipart()
    }

    fn supports_multipart_patch(&self) -> bool {
        self.storage.supports_multipart_patch()
    }
}

/// A synchronous multipart writer that owns the storage it borrows, so it can cross onto a blocking thread.
struct OwnedStoreWriter<TStorage: ?Sized> {
    // Declared before `storage`, so it is dropped first.
    writer: Box<dyn StoreWriter + 'static>,
    storage: Arc<TStorage>,
}

impl<TStorage: ?Sized + WritableStorageTraits + 'static> OwnedStoreWriter<TStorage> {
    fn new(
        storage: Arc<TStorage>,
        key: &StoreKey,
        size_hint: Option<u64>,
    ) -> Result<Self, StorageError> {
        let writer = storage.set_multipart(key, size_hint)?;
        // SAFETY: The writer borrows the storage behind the `Arc`, which does not move while `storage` holds it.
        // The writer never leaves this struct except to be consumed by `finish` or `abort`, before `storage` is
        // dropped, and the `writer` field is dropped before the `storage` field.
        let writer = unsafe {
            std::mem::transmute::<Box<dyn StoreWriter + '_>, Box<dyn StoreWriter + 'static>>(writer)
        };
        Ok(Self { writer, storage })
    }

    fn finish(self) -> Result<(), StorageError> {
        let Self { writer, storage } = self;
        let result = writer.finish();
        drop(storage);
        result
    }

    fn abort(self) -> Result<(), StorageError> {
        let Self { writer, storage } = self;
        let result = writer.abort();
        drop(storage);
        result
    }
}

/// An asynchronous multipart writer that runs each call of a synchronous writer on a blocking thread.
struct SyncToAsyncStoreWriter<'s, TStorage: ?Sized, TSpawnBlocking> {
    /// The writer, or [`None`] if a call was cancelled while the writer was on a blocking thread.
    writer: Option<OwnedStoreWriter<TStorage>>,
    spawn_blocking: &'s TSpawnBlocking,
}

impl<
        TStorage: ?Sized + WritableStorageTraits + 'static,
        TSpawnBlocking: SyncToAsyncSpawnBlocking,
    > SyncToAsyncStoreWriter<'_, TStorage, TSpawnBlocking>
{
    fn take(&mut self) -> Result<OwnedStoreWriter<TStorage>, StorageError> {
        self.writer.take().ok_or_else(|| {
            StorageError::Other("a previous call of the multipart writer was cancelled".to_string())
        })
    }

    /// Run `f` with the writer on a blocking thread.
    async fn with_writer(
        &mut self,
        f: impl FnOnce(&mut dyn StoreWriter) -> Result<(), StorageError> + MaybeSend + 'static,
    ) -> Result<(), StorageError> {
        let mut writer = self.take()?;
        let (writer, result) = self
            .spawn_blocking
            .spawn_blocking(move || {
                let result = f(writer.writer.as_mut());
                (writer, result)
            })
            .await;
        self.writer = Some(writer);
        result
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<
        TStorage: ?Sized + WritableStorageTraits + 'static,
        TSpawnBlocking: SyncToAsyncSpawnBlocking,
    > AsyncStoreWriter for SyncToAsyncStoreWriter<'_, TStorage, TSpawnBlocking>
{
    async fn write(&mut self, part: CowBytes<'_>) -> Result<(), StorageError> {
        // The part must outlive this method to cross onto the blocking thread.
        let part = part.into_static();
        self.with_writer(move |writer| writer.write(part)).await
    }

    async fn write_many(&mut self, parts: Vec<CowBytes<'_>>) -> Result<(), StorageError> {
        // The parts must outlive this method to cross onto the blocking thread.
        let parts: Vec<_> = parts.into_iter().map(CowBytes::into_static).collect();
        self.with_writer(move |writer| writer.write_many(parts))
            .await
    }

    async fn patch(&mut self, offset: u64, bytes: CowBytes<'_>) -> Result<(), StorageError> {
        // The bytes must outlive this method to cross onto the blocking thread.
        let bytes = bytes.into_static();
        self.with_writer(move |writer| writer.patch(offset, bytes))
            .await
    }

    async fn finish(mut self: Box<Self>) -> Result<(), StorageError> {
        let writer = self.take()?;
        self.spawn_blocking
            .spawn_blocking(move || writer.finish())
            .await
    }

    async fn abort(mut self: Box<Self>) -> Result<(), StorageError> {
        let writer = self.take()?;
        self.spawn_blocking
            .spawn_blocking(move || writer.abort())
            .await
    }
}

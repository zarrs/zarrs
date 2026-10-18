//! The storage of an array, transformed by its storage transformers.

use std::sync::Arc;

#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncReadableStorage, AsyncReadableStorageTraits, AsyncReadableWritableStorage,
    AsyncReadableWritableStorageTraits, AsyncWritableStorage, AsyncWritableStorageTraits,
};
use zarrs_storage::{
    ReadableStorage, ReadableStorageTraits, ReadableWritableStorage, ReadableWritableStorageTraits,
    StorageError, StorageHandle, WritableStorage, WritableStorageTraits,
};

use crate::array::Array;

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Array<TStorage> {
    /// The readable storage of the array, transformed by its storage transformers.
    pub(crate) fn readable_storage(&self) -> Result<ReadableStorage, StorageError> {
        self.storage_transformers()
            .create_readable_transformer(Arc::new(StorageHandle::new(self.storage.clone())))
    }
}

impl<TStorage: ?Sized + WritableStorageTraits + 'static> Array<TStorage> {
    /// The writable storage of the array, transformed by its storage transformers.
    pub(crate) fn writable_storage(&self) -> Result<WritableStorage, StorageError> {
        self.storage_transformers()
            .create_writable_transformer(Arc::new(StorageHandle::new(self.storage.clone())))
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// The readable and writable storage of the array, transformed by its storage transformers.
    pub(crate) fn readable_writable_storage(
        &self,
    ) -> Result<ReadableWritableStorage, StorageError> {
        self.storage_transformers()
            .create_readable_writable_transformer(Arc::new(StorageHandle::new(
                self.storage.clone(),
            )))
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
    /// The asynchronous readable storage of the array, transformed by its storage transformers.
    pub(crate) async fn async_readable_storage(
        &self,
    ) -> Result<AsyncReadableStorage, StorageError> {
        self.storage_transformers()
            .create_async_readable_transformer(Arc::new(StorageHandle::new(self.storage.clone())))
            .await
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncWritableStorageTraits + 'static> Array<TStorage> {
    /// The asynchronous writable storage of the array, transformed by its storage transformers.
    pub(crate) async fn async_writable_storage(
        &self,
    ) -> Result<AsyncWritableStorage, StorageError> {
        self.storage_transformers()
            .create_async_writable_transformer(Arc::new(StorageHandle::new(self.storage.clone())))
            .await
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// The asynchronous readable and writable storage of the array, transformed by its storage transformers.
    pub(crate) async fn async_readable_writable_storage(
        &self,
    ) -> Result<AsyncReadableWritableStorage, StorageError> {
        self.storage_transformers()
            .create_async_readable_writable_transformer(Arc::new(StorageHandle::new(
                self.storage.clone(),
            )))
            .await
    }
}

//! The storage of an array, transformed by its storage transformers.
//!
//! All array I/O, including storing and erasing metadata, goes through the transformed storage.
//! The untransformed [`storage_handle`](Array::storage_handle) only exists to create it.
//! The one exception is reading metadata in `Array::[async_]open`, because the storage
//! transformers of an array are only known after its metadata is read.
//! A storage transformer that remaps metadata keys therefore makes the array unopenable.

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

impl<TStorage: ?Sized> Array<TStorage> {
    /// The untransformed storage of the array.
    pub(crate) fn storage_handle(&self) -> Arc<StorageHandle<TStorage>> {
        Arc::new(StorageHandle::new(self.storage.clone()))
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Array<TStorage> {
    /// The readable storage of the array, transformed by its storage transformers.
    pub(crate) fn readable_storage(&self) -> Result<ReadableStorage, StorageError> {
        self.storage_transformers()
            .create_readable_transformer(self.storage_handle())
    }
}

impl<TStorage: ?Sized + WritableStorageTraits + 'static> Array<TStorage> {
    /// The writable storage of the array, transformed by its storage transformers.
    pub(crate) fn writable_storage(&self) -> Result<WritableStorage, StorageError> {
        self.storage_transformers()
            .create_writable_transformer(self.storage_handle())
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// The readable and writable storage of the array, transformed by its storage transformers.
    pub(crate) fn readable_writable_storage(
        &self,
    ) -> Result<ReadableWritableStorage, StorageError> {
        self.storage_transformers()
            .create_readable_writable_transformer(self.storage_handle())
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
    /// The asynchronous readable storage of the array, transformed by its storage transformers.
    pub(crate) async fn async_readable_storage(
        &self,
    ) -> Result<AsyncReadableStorage, StorageError> {
        self.storage_transformers()
            .create_async_readable_transformer(self.storage_handle())
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
            .create_async_writable_transformer(self.storage_handle())
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
            .create_async_readable_writable_transformer(self.storage_handle())
            .await
    }
}

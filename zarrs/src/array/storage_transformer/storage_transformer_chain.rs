//! A sequence of storage transformers.

use derive_more::From;
use zarrs_storage::StorageError;

use crate::{
    metadata::v3::MetadataV3,
    node::NodePath,
    plugin::PluginCreateError,
    storage::{ListableStorage, ReadableStorage, ReadableWritableStorage, WritableStorage},
};

#[cfg(feature = "async")]
use crate::storage::{
    AsyncListableStorage, AsyncReadableStorage, AsyncReadableWritableStorage, AsyncWritableStorage,
};

use super::{try_create_storage_transformer, StorageTransformer};

/// Configuration for a storage transformer chain.
#[derive(Debug, Clone, Default, From)]
pub struct StorageTransformerChain(Vec<StorageTransformer>);

impl StorageTransformerChain {
    /// Create a storage transformer chain from a list of storage transformers.
    #[must_use]
    pub fn new(storage_transformers: Vec<StorageTransformer>) -> Self {
        Self(storage_transformers)
    }

    /// Create a storage transformer chain from configurations.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if there is a configuration issue or attempt to create an unregistered storage transformer.
    pub fn from_metadata(
        metadatas: &[MetadataV3],
        path: &NodePath,
    ) -> Result<Self, PluginCreateError> {
        let mut storage_transformers = Vec::with_capacity(metadatas.len());
        for metadata in metadatas {
            let storage_transformer = match try_create_storage_transformer(metadata, path) {
                Ok(storage_transformer) => Ok(storage_transformer),
                Err(err) => {
                    if metadata.must_understand() {
                        Err(err)
                    } else {
                        continue;
                    }
                }
            }?;
            storage_transformers.push(storage_transformer);
        }
        Ok(Self(storage_transformers))
    }

    /// Create storage transformer chain metadata.
    #[must_use]
    pub fn create_metadatas(&self) -> Vec<MetadataV3> {
        self.0
            .iter()
            .map(|storage_transformer| storage_transformer.create_metadata())
            .collect()
    }
}

impl StorageTransformerChain {
    /// Create a readable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub fn create_readable_transformer(
        &self,
        mut storage: ReadableStorage,
    ) -> Result<ReadableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer.clone().create_readable_transformer(storage)?;
        }
        Ok(storage)
    }

    /// Create a writable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub fn create_writable_transformer(
        &self,
        mut storage: WritableStorage,
    ) -> Result<WritableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer.clone().create_writable_transformer(storage)?;
        }
        Ok(storage)
    }

    /// Create a readable and writable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub fn create_readable_writable_transformer(
        &self,
        mut storage: ReadableWritableStorage,
    ) -> Result<ReadableWritableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer
                .clone()
                .create_readable_writable_transformer(storage)?;
        }
        Ok(storage)
    }

    /// Create a listable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub fn create_listable_transformer(
        &self,
        mut storage: ListableStorage,
    ) -> Result<ListableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer.clone().create_listable_transformer(storage)?;
        }
        Ok(storage)
    }

    #[cfg(feature = "async")]
    /// Create an asynchronous readable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub async fn create_async_readable_transformer(
        &self,
        mut storage: AsyncReadableStorage,
    ) -> Result<AsyncReadableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer
                .clone()
                .create_async_readable_transformer(storage)
                .await?;
        }
        Ok(storage)
    }

    #[cfg(feature = "async")]
    /// Create an asynchronous writable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub async fn create_async_writable_transformer(
        &self,
        mut storage: AsyncWritableStorage,
    ) -> Result<AsyncWritableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer
                .clone()
                .create_async_writable_transformer(storage)
                .await?;
        }
        Ok(storage)
    }

    #[cfg(feature = "async")]
    /// Create an asynchronous redable and writable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub async fn create_async_readable_writable_transformer(
        &self,
        mut storage: AsyncReadableWritableStorage,
    ) -> Result<AsyncReadableWritableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer
                .clone()
                .create_async_readable_writable_transformer(storage)
                .await?;
        }
        Ok(storage)
    }

    #[cfg(feature = "async")]
    /// Create an asynchronous listable storage transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    pub async fn create_async_listable_transformer(
        &self,
        mut storage: AsyncListableStorage,
    ) -> Result<AsyncListableStorage, StorageError> {
        for transformer in &self.0 {
            storage = transformer
                .clone()
                .create_async_listable_transformer(storage)
                .await?;
        }
        Ok(storage)
    }
}

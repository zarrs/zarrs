//! A sequence of storage transformers.

use derive_more::From;

use super::{StorageTransformer, try_create_storage_transformer};
use crate::node::NodePath;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::PluginCreateError;
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncListableStorage, AsyncReadableStorage, AsyncReadableWritableStorage, AsyncWritableStorage,
};
use zarrs_storage::{
    ListableStorage, ReadableStorage, ReadableWritableStorage, StorageError, WritableStorage,
};

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
    ///
    /// # Panics
    /// Panics if any storage transformer does not have a V3 name.
    #[must_use]
    pub fn create_metadatas(&self) -> Vec<MetadataV3> {
        self.0
            .iter()
            .map(|storage_transformer| {
                let name = storage_transformer
                    .name_v3()
                    .expect("storage transformer must have a V3 name")
                    .into_owned();
                MetadataV3::new_with_configuration(name, storage_transformer.configuration())
            })
            .collect()
    }
}

#[ambisync::paired(
    sync(
        fns("create_async_{} => create_{}"),
        types(
            AsyncReadableStorage => ReadableStorage,
            AsyncWritableStorage => WritableStorage,
            AsyncReadableWritableStorage => ReadableWritableStorage,
            AsyncListableStorage => ListableStorage,
        ),
    ),
    async(feature = "async"),
)]
impl StorageTransformerChain {
    /// Create a readable storage transformer.
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

    /// Create a writable storage transformer.
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

    /// Create a readable and writable storage transformer.
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

    /// Create a listable storage transformer.
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

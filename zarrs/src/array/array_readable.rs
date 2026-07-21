use std::sync::Arc;

use ambisync::ambisync;

use super::{Array, ArrayCreateError, ArrayMetadata, ArrayMetadataV2, ArrayMetadataV3};
use crate::config::MetadataRetrieveVersion;
use crate::node::{NodePath, meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{ReadableStorageTraits, StorageError};

#[ambisync(
    sync(
        fns("async_{}"),
        types(AsyncReadableStorageTraits => ReadableStorageTraits),
    ),
    async(feature = "async"),
)]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
    /// Open an existing array in `storage` at `path` with the default
    /// [`MetadataRetrieveVersion`].
    ///
    /// The metadata is read from the store.
    ///
    /// # Errors
    /// Returns [`ArrayCreateError`] if there is a storage error or any metadata is invalid.
    pub async fn async_open(storage: Arc<TStorage>, path: &str) -> Result<Self, ArrayCreateError> {
        Self::async_open_opt(storage, path, &MetadataRetrieveVersion::Default).await
    }

    /// Open an existing array in `storage` at `path` with a non-default
    /// [`MetadataRetrieveVersion`].
    ///
    /// The metadata is read from the store.
    ///
    /// # Errors
    /// Returns [`ArrayCreateError`] if there is a storage error or any metadata is invalid.
    pub async fn async_open_opt(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, ArrayCreateError> {
        let metadata = Self::async_open_metadata(&storage, path, version).await?;
        Self::validate_metadata(&metadata)?;
        Self::new_with_metadata(storage, path, metadata)
    }

    async fn async_open_metadata(
        storage: &Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<ArrayMetadata, ArrayCreateError> {
        let node_path = NodePath::new(path)?;

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V3 = version {
            // Try V3
            let key_v3 = meta_key_v3(&node_path);
            if let Some(metadata) = storage.get(&key_v3).await? {
                let metadata: ArrayMetadataV3 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v3, err.to_string()))?;
                return Ok(ArrayMetadata::V3(metadata));
            }
        }

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V2 = version {
            // Try V2
            let key_v2 = meta_key_v2_array(&node_path);
            if let Some(metadata) = storage.get(&key_v2).await? {
                let mut metadata: ArrayMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v2, err.to_string()))?;

                let attributes_key = meta_key_v2_attributes(&node_path);
                let attributes = storage.get(&attributes_key).await?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }

                return Ok(ArrayMetadata::V2(metadata));
            }
        }

        Err(ArrayCreateError::MissingMetadata)
    }
}

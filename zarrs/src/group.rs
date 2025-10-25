//! Zarr groups.
//!
//! A Zarr group is a node in a Zarr hierarchy.
//! It can have associated attributes and may have child nodes (groups or [`arrays`](crate::array)).
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group>.
//!
//! Use [`GroupBuilder`] to setup a new group, or use [`Group::open`] to read and/or write an existing group.
//!
//! ## Group Metadata
//! Group metadata **must be explicitly stored** with [`store_metadata`](Group::store_metadata) or [`store_metadata_opt`](Group::store_metadata_opt) if a group is newly created or its metadata has been mutated.
//! Support for implicit groups was removed from Zarr V3 after provisional acceptance.
//!
//! Below is an example of a `zarr.json` file for a group:
//! ```json
//! {
//!     "zarr_format": 3,
//!     "node_type": "group",
//!     "attributes": {
//!         "spam": "ham",
//!         "eggs": 42,
//!     }
//! }
//! ```
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata> for more information on group metadata.

mod group_builder;
mod group_metadata_options;

use std::sync::Arc;

use derive_more::Display;
use thiserror::Error;
use zarrs_metadata::NodeMetadata;
use zarrs_metadata::{v2::GroupMetadataV2, v3::AdditionalFieldV3};
use zarrs_metadata_ext::group::consolidated_metadata::ConsolidatedMetadata;
use zarrs_storage::ListableStorageTraits;

use crate::{
    array::{AdditionalFieldUnsupportedError, Array, ArrayCreateError},
    config::{
        global_config, MetadataConvertVersion, MetadataEraseVersion, MetadataRetrieveVersion,
    },
    node::{
        get_all_nodes_of, get_child_nodes, meta_key_v2_attributes, meta_key_v2_group, meta_key_v3,
        Node, NodeCreateError, NodePath, NodePathError,
    },
    storage::{ReadableStorageTraits, StorageError, StorageHandle, WritableStorageTraits},
};
use zarrs_metadata_ext::v2_to_v3::group_metadata_v2_to_v3;

#[cfg(feature = "async")]
use crate::node::async_get_child_nodes;
#[cfg(feature = "async")]
use crate::storage::{
    AsyncListableStorageTraits, AsyncReadableStorageTraits, AsyncWritableStorageTraits,
};

pub use self::group_builder::GroupBuilder;
pub use crate::metadata::{v3::GroupMetadataV3, GroupMetadata};
pub use group_metadata_options::GroupMetadataOptions;

/// A group.
#[derive(Clone, Debug, Display)]
#[display(
    "group at {path} with metadata {}",
    "serde_json::to_string(metadata).unwrap_or_default()"
)]
pub struct Group<TStorage: ?Sized> {
    /// The storage.
    #[allow(dead_code)]
    storage: Arc<TStorage>,
    /// The path of the group in the store.
    #[allow(dead_code)]
    path: NodePath,
    /// The metadata.
    metadata: GroupMetadata,
}

impl<TStorage: ?Sized> Group<TStorage> {
    /// Create a group in `storage` at `path` with `metadata`.
    /// This does **not** write to the store, use [`store_metadata`](Group<WritableStorageTraits>::store_metadata) to write `metadata` to `storage`.
    ///
    /// # Errors
    ///
    /// Returns [`GroupCreateError`] if any metadata is invalid.
    pub fn new_with_metadata(
        storage: Arc<TStorage>,
        path: &str,
        metadata: GroupMetadata,
    ) -> Result<Self, GroupCreateError> {
        let path = NodePath::new(path)?;
        Ok(Self {
            storage,
            path,
            metadata,
        })
    }

    /// Get the underlying storage backing the group.
    #[must_use]
    pub fn storage(&self) -> Arc<TStorage> {
        self.storage.clone()
    }

    /// Get path.
    #[must_use]
    pub const fn path(&self) -> &NodePath {
        &self.path
    }

    /// Get attributes.
    #[must_use]
    pub const fn attributes(&self) -> &serde_json::Map<String, serde_json::Value> {
        match &self.metadata {
            GroupMetadata::V3(metadata) => &metadata.attributes,
            GroupMetadata::V2(metadata) => &metadata.attributes,
        }
    }

    /// Mutably borrow the group attributes.
    #[must_use]
    pub fn attributes_mut(&mut self) -> &mut serde_json::Map<String, serde_json::Value> {
        match &mut self.metadata {
            GroupMetadata::V3(metadata) => &mut metadata.attributes,
            GroupMetadata::V2(metadata) => &mut metadata.attributes,
        }
    }

    /// Return the underlying group metadata.
    #[must_use]
    pub fn metadata(&self) -> &GroupMetadata {
        &self.metadata
    }

    /// Return a new [`GroupMetadata`] with [`GroupMetadataOptions`] applied.
    ///
    /// This method is used internally by [`Group::store_metadata`] and [`Group::store_metadata_opt`].
    #[must_use]
    pub fn metadata_opt(&self, options: &GroupMetadataOptions) -> GroupMetadata {
        use GroupMetadata as GM;
        use MetadataConvertVersion as V;
        let metadata = self.metadata.clone();

        match (metadata, options.metadata_convert_version()) {
            (GM::V3(metadata), V::Default | V::V3) => GM::V3(metadata),
            (GM::V2(metadata), V::Default) => GM::V2(metadata),
            (GM::V2(metadata), V::V3) => GM::V3(group_metadata_v2_to_v3(&metadata)),
        }
    }

    /// Get the consolidated metadata. Returns [`None`] if `consolidated_metadata` is absent.
    ///
    /// Consolidated metadata is not currently supported for Zarr V2 groups.
    #[must_use]
    pub fn consolidated_metadata(&self) -> Option<ConsolidatedMetadata> {
        if let GroupMetadata::V3(group_metadata) = &self.metadata {
            if let Some(consolidated_metadata) = group_metadata
                .additional_fields
                .get("consolidated_metadata")
            {
                if let Ok(consolidated_metadata) = serde_json::from_value::<ConsolidatedMetadata>(
                    consolidated_metadata.as_value().clone(),
                ) {
                    return Some(consolidated_metadata);
                }
            }
            None
        } else {
            None
        }
    }

    /// Set the consolidated metadata.
    ///
    /// Consolidated metadata is not currently supported for Zarr V2 groups, and this function is a no-op.
    pub fn set_consolidated_metadata(
        &mut self,
        consolidated_metadata: Option<ConsolidatedMetadata>,
    ) -> &mut Self {
        if let GroupMetadata::V3(group_metadata) = &mut self.metadata {
            if let Some(consolidated_metadata) = consolidated_metadata {
                group_metadata.additional_fields.insert(
                    "consolidated_metadata".to_string(),
                    AdditionalFieldV3::new(consolidated_metadata, false),
                );
            } else {
                group_metadata
                    .additional_fields
                    .remove("consolidated_metadata");
            }
        }
        self
    }

    /// Convert the group to Zarr V3.
    ///
    /// If the group is already Zarr V3, this is a no-op.
    #[must_use]
    pub fn to_v3(self) -> Self {
        if let GroupMetadata::V2(metadata) = self.metadata {
            let metadata: GroupMetadata = group_metadata_v2_to_v3(&metadata).into();
            Self {
                storage: self.storage,
                path: self.path,
                metadata,
            }
        } else {
            self
        }
    }

    /// Reject the group if it contains unsupported extensions or additional fields with `"must_understand": true`.
    fn validate_metadata(metadata: &GroupMetadata) -> Result<(), GroupCreateError> {
        match &metadata {
            GroupMetadata::V2(_) => {}
            GroupMetadata::V3(_metadata) => {
                // for extension in &metadata.extensions {
                //     if extension.must_understand() {
                //         return Err(GroupCreateError::AdditionalFieldUnsupportedError(
                //             AdditionalFieldUnsupportedError::new(
                //                 extension.name().to_string(),
                //                 extension
                //                     .configuration()
                //                     .map(|configuration| {
                //                         serde_json::Value::Object(configuration.clone().into())
                //                     })
                //                     .unwrap_or_default(),
                //             ),
                //         ));
                //     }
                // }
            }
        }

        match &metadata {
            GroupMetadata::V2(_metadata) => {}
            GroupMetadata::V3(metadata) => {
                let additional_fields = &metadata.additional_fields;
                for (name, field) in additional_fields {
                    if field.must_understand() {
                        return Err(GroupCreateError::AdditionalFieldUnsupportedError(
                            AdditionalFieldUnsupportedError::new(
                                name.clone(),
                                field.as_value().clone(),
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> Group<TStorage> {
    /// Open a group in `storage` at `path` with [`MetadataRetrieveVersion`].
    /// The metadata is read from the store.
    ///
    /// # Errors
    /// Returns [`GroupCreateError`] if there is a storage error or any metadata is invalid.
    pub fn open(storage: Arc<TStorage>, path: &str) -> Result<Self, GroupCreateError> {
        Self::open_opt(storage, path, &MetadataRetrieveVersion::Default)
    }

    /// Open a group in `storage` at `path` with non-default [`MetadataRetrieveVersion`].
    /// The metadata is read from the store.
    ///
    /// # Errors
    /// Returns [`GroupCreateError`] if there is a storage error or any metadata is invalid.
    pub fn open_opt(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, GroupCreateError> {
        let metadata = Self::open_metadata(&storage, path, version)?;
        Self::validate_metadata(&metadata)?;
        Self::new_with_metadata(storage, path, metadata)
    }

    fn open_metadata(
        storage: &Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<GroupMetadata, GroupCreateError> {
        let node_path = path.try_into()?;
        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V3 = version {
            // Try Zarr V3
            let key_v3 = meta_key_v3(&node_path);
            if let Some(metadata) = storage.get(&key_v3)? {
                let metadata: GroupMetadataV3 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v3, err.to_string()))?;
                return Ok(GroupMetadata::V3(metadata));
            }
        }

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V2 = version {
            // Try Zarr V2
            let key_v2 = meta_key_v2_group(&node_path);
            if let Some(metadata) = storage.get(&key_v2)? {
                let mut metadata: GroupMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v2, err.to_string()))?;
                let attributes_key = meta_key_v2_attributes(&node_path);
                let attributes = storage.get(&attributes_key)?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }
                return Ok(GroupMetadata::V2(metadata));
            }
        }

        // No metadata has been found
        Err(GroupCreateError::MissingMetadata)
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits> Group<TStorage> {
    /// Return the children of the group
    ///
    /// The `recursive` argument determines whether the returned `Node`s will have their
    /// children (and their children, etc.) populated.
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is a metadata related error, or an underlying store error.
    pub fn children(&self, recursive: bool) -> Result<Vec<Node>, NodeCreateError> {
        get_child_nodes(&self.storage, &self.path, recursive)
    }

    /// Return all the Nodes under the group, recursively
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is a metadata related error, or an underlying store error.
    pub fn traverse(&self) -> Result<Vec<(NodePath, NodeMetadata)>, NodeCreateError> {
        get_all_nodes_of(&self.storage, &self.path, &MetadataRetrieveVersion::Default)
    }

    /// Return the children of the group that are [`Group`]s
    ///
    /// # Errors
    /// Returns [`GroupCreateError`] if there is a storage error or any metadata is invalid.
    pub fn child_groups(&self) -> Result<Vec<Self>, GroupCreateError> {
        self.children(false)?
            .into_iter()
            .filter_map(|node| {
                let path = node.path().to_string();
                let metadata: NodeMetadata = node.into();
                match metadata {
                    NodeMetadata::Group(metadata) => Some(Group::new_with_metadata(
                        self.storage.clone(),
                        path.as_str(),
                        metadata,
                    )),
                    NodeMetadata::Array(_) => None,
                }
            })
            .collect()
    }

    /// Return the children of the group that are [`Array`]s
    ///
    /// # Errors
    /// Returns [`ArrayCreateError`] if there is a storage error or any metadata is invalid.
    pub fn child_arrays(&self) -> Result<Vec<Array<TStorage>>, ArrayCreateError> {
        self.children(false)?
            .into_iter()
            .filter_map(|node| {
                let path = node.path().to_string();
                let metadata: NodeMetadata = node.into();
                match metadata {
                    NodeMetadata::Array(metadata) => Some(Array::new_with_metadata(
                        self.storage.clone(),
                        path.as_str(),
                        metadata,
                    )),
                    NodeMetadata::Group(_) => None,
                }
            })
            .collect()
    }

    /// Return the paths of the groups children
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is an underlying error with the store.
    pub fn child_paths(&self) -> Result<Vec<NodePath>, NodeCreateError> {
        let paths = self.children(false)?.into_iter().map(Into::into).collect();
        Ok(paths)
    }

    /// Return the paths of the groups children if the child is a group
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is an underlying error with the store.
    pub fn child_group_paths(&self) -> Result<Vec<NodePath>, NodeCreateError> {
        let paths = self
            .children(false)?
            .into_iter()
            .filter_map(|node| match node.metadata() {
                NodeMetadata::Group(_) => Some(node.into()),
                NodeMetadata::Array(_) => None,
            })
            .collect();
        Ok(paths)
    }

    /// Return the paths of the groups children if the child is an array
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is an underlying error with the store.
    pub fn child_array_paths(&self) -> Result<Vec<NodePath>, NodeCreateError> {
        let paths = self
            .children(false)?
            .into_iter()
            .filter_map(|node| match node.metadata() {
                NodeMetadata::Array(_) => Some(node.into()),
                NodeMetadata::Group(_) => None,
            })
            .collect();
        Ok(paths)
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncReadableStorageTraits> Group<TStorage> {
    /// Async variant of [`open`](Group::open).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_open(storage: Arc<TStorage>, path: &str) -> Result<Self, GroupCreateError> {
        Self::async_open_opt(storage, path, &MetadataRetrieveVersion::Default).await
    }

    /// Async variant of [`open_opt`](Group::open_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_open_opt(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, GroupCreateError> {
        let metadata = Self::async_open_metadata(storage.clone(), path, version).await?;
        Self::validate_metadata(&metadata)?;
        Self::new_with_metadata(storage, path, metadata)
    }

    async fn async_open_metadata(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<GroupMetadata, GroupCreateError> {
        let node_path = path.try_into()?;

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V3 = version {
            // Try Zarr V3
            let key_v3 = meta_key_v3(&node_path);
            if let Some(metadata) = storage.get(&key_v3).await? {
                let metadata: GroupMetadataV3 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v3, err.to_string()))?;
                return Ok(GroupMetadata::V3(metadata));
            }
        }

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V2 = version {
            // Try Zarr V2
            let key_v2 = meta_key_v2_group(&node_path);
            if let Some(metadata) = storage.get(&key_v2).await? {
                let mut metadata: GroupMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v2, err.to_string()))?;
                let attributes_key = meta_key_v2_attributes(&node_path);
                let attributes = storage.get(&attributes_key).await?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }
                return Ok(GroupMetadata::V2(metadata));
            }
        }

        // No metadata has been found
        Err(GroupCreateError::MissingMetadata)
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits> Group<TStorage> {
    /// Return the children of the group
    ///
    /// The `recursive` argument determines whether the returned `Node`s will have their
    /// children (and their children, etc.) populated.
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is a metadata related error, or an underlying store error.
    pub async fn async_children(&self, recursive: bool) -> Result<Vec<Node>, NodeCreateError> {
        async_get_child_nodes(&self.storage, &self.path, recursive).await
    }

    /// Return the children of the group that are [`Group`]s
    ///
    /// # Errors
    /// Returns [`GroupCreateError`] if there is a storage error or any metadata is invalid.
    pub async fn async_child_groups(&self) -> Result<Vec<Self>, GroupCreateError> {
        self.async_children(false)
            .await?
            .into_iter()
            .filter_map(|node| {
                let path = node.path().to_string();
                let metadata: NodeMetadata = node.into();
                match metadata {
                    NodeMetadata::Group(metadata) => Some(Group::new_with_metadata(
                        self.storage.clone(),
                        path.as_str(),
                        metadata,
                    )),
                    NodeMetadata::Array(_) => None,
                }
            })
            .collect()
    }

    /// Return the children of the group that are [`Array`]s
    ///
    /// # Errors
    /// Returns [`ArrayCreateError`] if there is a storage error or any metadata is invalid.
    pub async fn async_child_arrays(&self) -> Result<Vec<Array<TStorage>>, ArrayCreateError> {
        self.async_children(false)
            .await?
            .into_iter()
            .filter_map(|node| {
                let path = node.path().to_string();
                let metadata: NodeMetadata = node.into();
                match metadata {
                    NodeMetadata::Array(metadata) => Some(Array::new_with_metadata(
                        self.storage.clone(),
                        path.as_str(),
                        metadata,
                    )),
                    NodeMetadata::Group(_) => None,
                }
            })
            .collect()
    }

    /// Return the paths of the groups children
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is an underlying error with the store.
    pub async fn async_child_paths(&self) -> Result<Vec<NodePath>, NodeCreateError> {
        let paths = self
            .async_children(false)
            .await?
            .into_iter()
            .map(Into::into)
            .collect();
        Ok(paths)
    }

    /// Return the paths of the groups children if the child is a group
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is an underlying error with the store.
    pub async fn async_child_group_paths(&self) -> Result<Vec<NodePath>, NodeCreateError> {
        let paths = self
            .async_children(false)
            .await?
            .into_iter()
            .filter_map(|node| match node.metadata() {
                NodeMetadata::Group(_) => Some(node.into()),
                NodeMetadata::Array(_) => None,
            })
            .collect();
        Ok(paths)
    }

    /// Return the paths of the groups children if the child is an array
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if there is an underlying error with the store.
    pub async fn async_child_array_paths(&self) -> Result<Vec<NodePath>, NodeCreateError> {
        let paths = self
            .async_children(false)
            .await?
            .into_iter()
            .filter_map(|node| match node.metadata() {
                NodeMetadata::Array(_) => Some(node.into()),
                NodeMetadata::Group(_) => None,
            })
            .collect();
        Ok(paths)
    }
}

/// A group creation error.
#[derive(Clone, Debug, Error)]
pub enum GroupCreateError {
    /// An invalid node path
    #[error(transparent)]
    NodePathError(#[from] NodePathError),
    /// Unsupported additional field.
    #[error(transparent)]
    AdditionalFieldUnsupportedError(AdditionalFieldUnsupportedError),
    /// Storage error.
    #[error(transparent)]
    StorageError(#[from] StorageError),
    /// Missing metadata.
    #[error("group metadata is missing")]
    MissingMetadata,
}

impl<TStorage: ?Sized + ReadableStorageTraits> Group<TStorage> {}

impl<TStorage: ?Sized + WritableStorageTraits> Group<TStorage> {
    /// Store metadata with default [`GroupMetadataOptions`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    pub fn store_metadata(&self) -> Result<(), StorageError> {
        self.store_metadata_opt(&GroupMetadataOptions::default())
    }

    /// Store metadata with non-default [`GroupMetadataOptions`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    pub fn store_metadata_opt(&self, options: &GroupMetadataOptions) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));

        // Get the metadata with options applied and store
        let metadata = self.metadata_opt(options);

        // Write the metadata
        let path = self.path();
        match metadata {
            GroupMetadata::V3(metadata) => {
                let key = meta_key_v3(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_handle.set(&key, json.into())
            }
            GroupMetadata::V2(metadata) => {
                let mut metadata = metadata.clone();

                if !metadata.attributes.is_empty() {
                    // Store .zgroup
                    let key = meta_key_v2_attributes(path);
                    let json = serde_json::to_vec_pretty(&metadata.attributes).map_err(|err| {
                        StorageError::InvalidMetadata(key.clone(), err.to_string())
                    })?;
                    storage_handle.set(&key, json.into())?;

                    metadata.attributes = serde_json::Map::default();
                }

                // Store .zarray
                let key = meta_key_v2_group(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_handle.set(&key, json.into())?;
                Ok(())
            }
        }
    }

    /// Erase the metadata with default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    pub fn erase_metadata(&self) -> Result<(), StorageError> {
        let erase_version = global_config().metadata_erase_version();
        self.erase_metadata_opt(erase_version)
    }

    /// Erase the metadata with non-default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    pub fn erase_metadata_opt(&self, options: MetadataEraseVersion) -> Result<(), StorageError> {
        let storage_handle = StorageHandle::new(self.storage.clone());
        match options {
            MetadataEraseVersion::Default => match self.metadata {
                GroupMetadata::V3(_) => storage_handle.erase(&meta_key_v3(self.path())),
                GroupMetadata::V2(_) => {
                    storage_handle.erase(&meta_key_v2_group(self.path()))?;
                    storage_handle.erase(&meta_key_v2_attributes(self.path()))
                }
            },
            MetadataEraseVersion::All => {
                storage_handle.erase(&meta_key_v3(self.path()))?;
                storage_handle.erase(&meta_key_v2_group(self.path()))?;
                storage_handle.erase(&meta_key_v2_attributes(self.path()))
            }
            MetadataEraseVersion::V3 => storage_handle.erase(&meta_key_v3(self.path())),
            MetadataEraseVersion::V2 => {
                storage_handle.erase(&meta_key_v2_group(self.path()))?;
                storage_handle.erase(&meta_key_v2_attributes(self.path()))
            }
        }
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncWritableStorageTraits> Group<TStorage> {
    /// Async variant of [`store_metadata`](Group::store_metadata).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata(&self) -> Result<(), StorageError> {
        self.async_store_metadata_opt(&GroupMetadataOptions::default())
            .await
    }

    /// Async variant of [`store_metadata_opt`](Group::store_metadata_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata_opt(
        &self,
        options: &GroupMetadataOptions,
    ) -> Result<(), StorageError> {
        let storage_handle = StorageHandle::new(self.storage.clone());

        // Get the metadata with options applied and store
        let metadata = self.metadata_opt(options);

        // Write the metadata
        let path = self.path();
        match metadata {
            GroupMetadata::V3(metadata) => {
                let key = meta_key_v3(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_handle.set(&key, json.into()).await
            }
            GroupMetadata::V2(metadata) => {
                let mut metadata = metadata.clone();

                if !metadata.attributes.is_empty() {
                    // Store .zgroup
                    let key = meta_key_v2_attributes(path);
                    let json = serde_json::to_vec_pretty(&metadata.attributes).map_err(|err| {
                        StorageError::InvalidMetadata(key.clone(), err.to_string())
                    })?;
                    storage_handle.set(&key, json.into()).await?;

                    metadata.attributes = serde_json::Map::default();
                }

                // Store .zarray
                let key = meta_key_v2_group(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_handle.set(&key, json.into()).await?;
                Ok(())
            }
        }
    }

    /// Async variant of [`erase_metadata`](Group::erase_metadata).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata(&self) -> Result<(), StorageError> {
        let erase_version = global_config().metadata_erase_version();
        self.async_erase_metadata_opt(erase_version).await
    }

    /// Async variant of [`erase_metadata_opt`](Group::erase_metadata_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata_opt(
        &self,
        options: MetadataEraseVersion,
    ) -> Result<(), StorageError> {
        let storage_handle = StorageHandle::new(self.storage.clone());
        match options {
            MetadataEraseVersion::Default => match self.metadata {
                GroupMetadata::V3(_) => storage_handle.erase(&meta_key_v3(self.path())).await,
                GroupMetadata::V2(_) => {
                    storage_handle
                        .erase(&meta_key_v2_group(self.path()))
                        .await?;
                    storage_handle
                        .erase(&meta_key_v2_attributes(self.path()))
                        .await
                }
            },
            MetadataEraseVersion::All => {
                storage_handle.erase(&meta_key_v3(self.path())).await?;
                storage_handle
                    .erase(&meta_key_v2_group(self.path()))
                    .await?;
                storage_handle
                    .erase(&meta_key_v2_attributes(self.path()))
                    .await
            }
            MetadataEraseVersion::V3 => storage_handle.erase(&meta_key_v3(self.path())).await,
            MetadataEraseVersion::V2 => {
                storage_handle
                    .erase(&meta_key_v2_group(self.path()))
                    .await?;
                storage_handle
                    .erase(&meta_key_v2_attributes(self.path()))
                    .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::{store::MemoryStore, StoreKey};

    use super::*;

    const JSON_VALID1: &str = r#"{
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
        "spam": "ham",
        "eggs": 42
    }
}"#;

    #[test]
    fn group_metadata_v3_1() {
        let group_metadata: GroupMetadataV3 = serde_json::from_str(JSON_VALID1).unwrap();
        let store = MemoryStore::default();
        Group::new_with_metadata(store.into(), "/", GroupMetadata::V3(group_metadata)).unwrap();
    }

    #[test]
    fn group_metadata_v3_2() {
        let group_metadata: GroupMetadataV3 = serde_json::from_str(
            r#"{
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "spam": "ham",
                "eggs": 42
            },
            "unknown": {
                "must_understand": false
            }
        }"#,
        )
        .unwrap();
        let store = MemoryStore::default();
        Group::new_with_metadata(store.into(), "/", GroupMetadata::V3(group_metadata)).unwrap();
    }

    #[test]
    fn group_metadata_v3_invalid_format() {
        let group_metadata = serde_json::from_str::<GroupMetadataV3>(
            r#"{
            "zarr_format": 2,
            "node_type": "group",
            "attributes": {
                "spam": "ham",
                "eggs": 42
            }
        }"#,
        );
        assert!(group_metadata.is_err());
    }

    #[test]
    fn group_metadata_invalid_type() {
        let group_metadata = serde_json::from_str::<GroupMetadata>(
            r#"{
            "zarr_format": 3,
            "node_type": "array",
            "attributes": {
                "spam": "ham",
                "eggs": 42
            }
        }"#,
        );
        assert!(group_metadata.is_err());
    }

    #[test]
    fn group_metadata_unknown_additional_field() {
        let group_metadata = serde_json::from_str::<GroupMetadataV3>(
            r#"{
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                  "spam": "ham",
                  "eggs": 42
                },
                "unknown": "unsupported"
            }"#,
        )
        .unwrap();
        assert!(group_metadata.additional_fields.len() == 1);
        assert!(group_metadata
            .additional_fields
            .get("unknown")
            .unwrap()
            .must_understand());

        // Permit manual creation of group with unsupported metadata
        let storage = Arc::new(MemoryStore::new());
        let group =
            Group::new_with_metadata(storage.clone(), "/", group_metadata.clone().into()).unwrap();
        group.store_metadata().unwrap();

        // Group opening should fail with unsupported metadata
        let group = Group::open(storage, "/");
        assert!(group.is_err());
    }

    #[test]
    fn group_metadata_write_read() {
        let store = std::sync::Arc::new(MemoryStore::new());
        let group_path = "/group";
        let group = GroupBuilder::new()
            .build(store.clone(), group_path)
            .unwrap();
        group.store_metadata().unwrap();

        let group_copy = Group::open(store, group_path).unwrap();
        assert_eq!(group_copy.metadata(), group.metadata());
        let group_metadata_str = group.metadata().to_string();
        println!("{}", group_metadata_str);
        assert!(
            group_metadata_str == r#"{"node_type":"group","zarr_format":3}"#
                || group_metadata_str == r#"{"zarr_format":3,"node_type":"group"}"#
        );
        // assert_eq!(
        //     group.to_string(),
        //     r#"group at /group with metadata {"node_type":"group","zarr_format":3}"#
        // );
    }

    #[test]
    fn group_metadata_invalid_path() {
        let group_metadata: GroupMetadata = serde_json::from_str(JSON_VALID1).unwrap();
        let store = MemoryStore::default();
        assert_eq!(
            Group::new_with_metadata(store.into(), "abc", group_metadata)
                .unwrap_err()
                .to_string(),
            "invalid node path abc"
        );
    }

    #[test]
    fn group_invalid_path() {
        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());
        assert_eq!(
            Group::open(store, "abc").unwrap_err().to_string(),
            "invalid node path abc"
        );
    }

    #[test]
    fn group_invalid_metadata() {
        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());
        store
            .set(&StoreKey::new("zarr.json").unwrap(), vec![0].into())
            .unwrap();
        assert_eq!(
            Group::open(store, "/").unwrap_err().to_string(),
            "error parsing metadata for zarr.json: expected value at line 1 column 1"
        );
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn group_metadata_write_read_async() {
        let store = std::sync::Arc::new(zarrs_object_store::AsyncObjectStore::new(
            object_store::memory::InMemory::new(),
        ));
        let group_path = "/group";
        let group = GroupBuilder::new()
            .build(store.clone(), group_path)
            .unwrap();
        group.async_store_metadata().await.unwrap();

        let group_copy = Group::async_open(store, group_path).await.unwrap();
        assert_eq!(group_copy.metadata(), group.metadata());
    }

    /// Implicit group support is removed since implicit groups were removed from the Zarr V3 spec
    #[test]
    fn group_implicit() {
        let store = std::sync::Arc::new(MemoryStore::new());
        let group_path = "/group";
        assert!(Group::open(store, group_path).is_err());
    }

    #[test]
    fn group_traverse() {
        let store = Arc::new(MemoryStore::new());

        let builder = GroupBuilder::default();
        let root = builder
            .build(store.clone(), NodePath::root().as_str())
            .unwrap();

        assert!(root.store_metadata().is_ok());

        assert!(builder
            .build(store.clone(), "/group")
            .unwrap()
            .store_metadata()
            .is_ok());
        assert!(builder
            .build(store.clone(), "/group/subgroup")
            .unwrap()
            .store_metadata()
            .is_ok());
        assert!(builder
            .build(store.clone(), "/group/subgroup/leafgroup")
            .unwrap()
            .store_metadata()
            .is_ok());

        println!("{:?}", root.traverse().unwrap());

        let nodes = root.traverse();
        assert!(nodes.is_ok());

        let nodes = nodes.unwrap();

        assert!(nodes.len() == 3);
        assert_eq!(
            nodes
                .iter()
                .map(|(path, _metadata)| path.as_str())
                .collect::<Vec<&str>>()
                .sort(),
            vec!["/group", "/group/subgroup", "/group/subgroup/leafgroup"].sort()
        );
    }
}

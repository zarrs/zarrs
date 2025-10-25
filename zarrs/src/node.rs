//! Zarr nodes.
//!
//! A node in a Zarr hierarchy represents either an [`Array`](crate::array::Array) or a [`Group`](crate::group::Group).
//!
//! A [`Node`] has an associated [`NodePath`], [`NodeMetadata`], and children.
//!
//! The [`Node::hierarchy_tree`] function can be used to create a string representation of a the hierarchy below a node.

mod node_name;
pub use node_name::{NodeName, NodeNameError};

mod node_path;
pub use node_path::{NodePath, NodePathError};

mod node_sync;
pub use node_sync::{get_child_nodes, get_child_nodes_opt, node_exists, node_exists_listable};

mod key;
pub use key::{
    data_key, meta_key, meta_key_v2_array, meta_key_v2_attributes, meta_key_v2_group, meta_key_v3,
};

#[cfg(feature = "async")]
mod node_async;
#[cfg(feature = "async")]
pub use node_async::{
    async_get_child_nodes, async_get_child_nodes_opt, async_node_exists, async_node_exists_listable,
};
use zarrs_metadata_ext::group::consolidated_metadata::ConsolidatedMetadataMetadata;
use zarrs_storage::StorePrefixError;

use std::{collections::HashMap, sync::Arc};

pub use crate::metadata::NodeMetadata;
use thiserror::Error;

use crate::{
    array::{ArrayCreateError, ArrayMetadata},
    config::MetadataRetrieveVersion,
    group::GroupCreateError,
    metadata::{
        v2::{ArrayMetadataV2, GroupMetadataV2},
        GroupMetadata,
    },
    storage::{ListableStorageTraits, ReadableStorageTraits, StorageError},
};

#[cfg(feature = "async")]
use crate::storage::{AsyncListableStorageTraits, AsyncReadableStorageTraits};

/// A Zarr hierarchy node.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#hierarchy>.
#[derive(Debug, Clone)]
pub struct Node {
    /// Node path.
    path: NodePath,
    /// Node metadata.
    metadata: NodeMetadata,
    /// Node children.
    ///
    /// Only group nodes can have children.
    children: Vec<Node>,
}

impl From<Node> for NodeMetadata {
    fn from(value: Node) -> Self {
        value.metadata
    }
}

impl From<Node> for NodePath {
    fn from(value: Node) -> Self {
        value.path
    }
}

/// A node creation error.
#[derive(Clone, Debug, Error)]
pub enum NodeCreateError {
    /// An invalid node path.
    #[error(transparent)]
    NodePathError(#[from] NodePathError),
    /// An invalid store prefix.
    #[error(transparent)]
    StorePrefixError(#[from] StorePrefixError),
    /// A storage error.
    #[error(transparent)]
    StorageError(#[from] StorageError),
    /// Metadata version mismatch
    #[error("Found V2 metadata in V3 key or vice-versa")]
    MetadataVersionMismatch,
    /// Missing metadata.
    #[error("Metadata is missing")]
    MissingMetadata,
}

impl From<NodeCreateError> for ArrayCreateError {
    fn from(value: NodeCreateError) -> Self {
        ArrayCreateError::from(match value {
            NodeCreateError::NodePathError(err) => StorageError::Other(err.to_string()),
            NodeCreateError::StorePrefixError(err) => StorageError::Other(err.to_string()),
            NodeCreateError::StorageError(err) => err,
            NodeCreateError::MetadataVersionMismatch => {
                StorageError::Other(NodeCreateError::MetadataVersionMismatch.to_string())
            }
            NodeCreateError::MissingMetadata => {
                StorageError::Other(NodeCreateError::MissingMetadata.to_string())
            }
        })
    }
}

impl From<NodeCreateError> for GroupCreateError {
    fn from(value: NodeCreateError) -> Self {
        GroupCreateError::from(match value {
            NodeCreateError::NodePathError(err) => StorageError::Other(err.to_string()),
            NodeCreateError::StorePrefixError(err) => StorageError::Other(err.to_string()),
            NodeCreateError::StorageError(err) => err,
            NodeCreateError::MetadataVersionMismatch => {
                StorageError::Other(NodeCreateError::MetadataVersionMismatch.to_string())
            }
            NodeCreateError::MissingMetadata => {
                StorageError::Other(NodeCreateError::MissingMetadata.to_string())
            }
        })
    }
}

impl Node {
    fn get_metadata<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        path: &NodePath,
        version: &MetadataRetrieveVersion,
    ) -> Result<NodeMetadata, NodeCreateError> {
        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V3 = version {
            // Try a Zarr V3 group/array
            let key_v3 = meta_key_v3(path);
            if let Some(metadata) = storage.get(&key_v3)? {
                let metadata: NodeMetadata = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v3, err.to_string()))?;
                match metadata {
                    NodeMetadata::Array(ArrayMetadata::V3(_))
                    | NodeMetadata::Group(GroupMetadata::V3(_)) => return Ok(metadata),
                    NodeMetadata::Array(ArrayMetadata::V2(_))
                    | NodeMetadata::Group(GroupMetadata::V2(_)) => {
                        return Err(NodeCreateError::MetadataVersionMismatch);
                    }
                }
            }
        }

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V2 = version {
            // Try a Zarr V2 array
            let array_key = meta_key_v2_array(path);
            let attributes_key = meta_key_v2_attributes(path);
            if let Some(metadata) = storage.get(&array_key)? {
                let mut metadata: ArrayMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(array_key, err.to_string()))?;
                let attributes = storage.get(&attributes_key)?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }
                return Ok(NodeMetadata::Array(ArrayMetadata::V2(metadata)));
            }

            // Try a Zarr V2 group
            let group_key = meta_key_v2_group(path);
            if let Some(metadata) = storage.get(&group_key)? {
                let mut metadata: GroupMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(group_key, err.to_string()))?;
                let attributes = storage.get(&attributes_key)?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }
                return Ok(NodeMetadata::Group(GroupMetadata::V2(metadata)));
            }
        }

        // No metadata has been found
        Err(NodeCreateError::MissingMetadata)
    }

    #[cfg(feature = "async")]
    // Identical to get_metadata.. with awaits
    // "maybe async" one day?
    async fn async_get_metadata<
        TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
    >(
        storage: &Arc<TStorage>,
        path: &NodePath,
        version: &MetadataRetrieveVersion,
    ) -> Result<NodeMetadata, NodeCreateError> {
        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V3 = version {
            // Try a Zarr V3 group/array
            let key_v3 = meta_key_v3(path);
            if let Some(metadata) = storage.get(&key_v3).await? {
                let metadata: NodeMetadata = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v3, err.to_string()))?;
                match metadata {
                    NodeMetadata::Array(ArrayMetadata::V3(_))
                    | NodeMetadata::Group(GroupMetadata::V3(_)) => return Ok(metadata),
                    NodeMetadata::Array(ArrayMetadata::V2(_))
                    | NodeMetadata::Group(GroupMetadata::V2(_)) => {
                        return Err(NodeCreateError::MetadataVersionMismatch);
                    }
                }
            }
        }

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V2 = version {
            // Try a Zarr V2 array
            let array_key = meta_key_v2_array(path);
            let attributes_key = meta_key_v2_attributes(path);
            if let Some(metadata) = storage.get(&array_key).await? {
                let mut metadata: ArrayMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(array_key, err.to_string()))?;
                let attributes = storage.get(&attributes_key).await?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }
                return Ok(NodeMetadata::Array(ArrayMetadata::V2(metadata)));
            }

            // Try a Zarr V2 group
            let group_key = meta_key_v2_group(path);
            if let Some(metadata) = storage.get(&group_key).await? {
                let mut metadata: GroupMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(group_key, err.to_string()))?;
                let attributes = storage.get(&attributes_key).await?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }
                return Ok(NodeMetadata::Group(GroupMetadata::V2(metadata)));
            }
        }

        // No metadata has been found
        Err(NodeCreateError::MissingMetadata)
    }

    /// Open a node at `path` and read metadata and children from `storage` with default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub fn open<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        path: &str,
    ) -> Result<Self, NodeCreateError> {
        Self::open_opt(storage, path, &MetadataRetrieveVersion::Default)
    }

    /// Open a node at `path` and read metadata and children from `storage` with non-default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub fn open_opt<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, NodeCreateError> {
        let path: NodePath = path.try_into()?;
        let metadata = Self::get_metadata(storage, &path, version)?;
        let children = match metadata {
            NodeMetadata::Array(_) => Vec::default(),
            // TODO: Add consolidated metadata support
            NodeMetadata::Group(_) => get_child_nodes_opt(storage, &path, true, version)?,
        };
        let node = Self {
            path,
            metadata,
            children,
        };
        Ok(node)
    }

    #[cfg(feature = "async")]
    /// Asynchronously open a node at `path` and read metadata and children from `storage` with default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub async fn async_open<
        TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
    >(
        storage: Arc<TStorage>,
        path: &str,
    ) -> Result<Self, NodeCreateError> {
        Self::async_open_opt(storage, path, &MetadataRetrieveVersion::Default).await
    }

    #[cfg(feature = "async")]
    /// Asynchronously open a node at `path` and read metadata and children from `storage` with non-default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub async fn async_open_opt<
        TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
    >(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, NodeCreateError> {
        let path: NodePath = path.try_into()?;
        let metadata = Self::async_get_metadata(&storage, &path, version).await?;
        let children = match metadata {
            NodeMetadata::Array(_) => Vec::default(),
            // TODO: Add consolidated metadata support
            NodeMetadata::Group(_) => {
                async_get_child_nodes_opt(&storage, &path, true, version).await?
            }
        };
        let node = Self {
            path,
            metadata,
            children,
        };
        Ok(node)
    }

    /// Create a new node at `path` with `metadata` and `children`.
    #[must_use]
    pub fn new_with_metadata(path: NodePath, metadata: NodeMetadata, children: Vec<Self>) -> Self {
        Self {
            path,
            metadata,
            children,
        }
    }

    /// Indicates if a node is the root.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.path.as_str().eq("/")
    }

    /// Returns the name of the node.
    #[must_use]
    pub fn name(&self) -> NodeName {
        let name = self
            .path
            .as_str()
            .split('/')
            .next_back()
            .unwrap_or_default();
        unsafe { NodeName::new_unchecked(name) }
    }

    /// Returns a reference to the path of the node.
    #[must_use]
    pub fn path(&self) -> &NodePath {
        &self.path
    }

    /// Returns a reference to the metadata of the node.
    #[must_use]
    pub fn metadata(&self) -> &NodeMetadata {
        &self.metadata
    }

    /// Returns a reference to the children of the node.
    #[must_use]
    pub fn children(&self) -> &[Self] {
        &self.children
    }

    /// Return a tree representation of a hierarchy as a string.
    ///
    /// Arrays are annotated with their shape and data type.
    /// For example:
    /// ```text
    /// a
    ///   baz [10000, 1000] float64
    ///   foo [10000, 1000] float64
    /// b
    /// ```
    #[must_use]
    pub fn hierarchy_tree(&self) -> String {
        fn print_metadata(name: &str, string: &mut String, metadata: &NodeMetadata) {
            match metadata {
                NodeMetadata::Array(array_metadata) => {
                    let s = match array_metadata {
                        ArrayMetadata::V3(array_metadata) => {
                            format!(
                                "{} {:?} {}",
                                name, array_metadata.shape, array_metadata.data_type
                            )
                        }
                        ArrayMetadata::V2(array_metadata) => {
                            format!(
                                "{} {:?} {:?}",
                                name, array_metadata.shape, array_metadata.dtype
                            )
                        }
                    };
                    string.push_str(&s);
                }
                NodeMetadata::Group(_) => {
                    string.push_str(name);
                }
            }
            string.push('\n');
        }

        fn update_tree(string: &mut String, children: &[Node], depth: usize) {
            for child in children {
                let name = child.name();
                string.push_str(&" ".repeat(depth * 2));
                print_metadata(name.as_str(), string, &child.metadata);
                update_tree(string, &child.children, depth + 1);
            }
        }

        let mut string = String::default();
        print_metadata("/", &mut string, &self.metadata);
        update_tree(&mut string, &self.children, 1);
        string
    }

    /// Consolidate metadata. Returns [`None`] for an array.
    ///
    /// [`ConsolidatedMetadataMetadata`] can be converted into [`ConsolidatedMetadata`](zarrs_metadata_ext::group::consolidated_metadata::ConsolidatedMetadata) in [`GroupMetadataV3`](crate::metadata::v3::GroupMetadataV3).
    #[must_use]
    #[allow(clippy::items_after_statements)]
    pub fn consolidate_metadata(&self) -> Option<ConsolidatedMetadataMetadata> {
        if let NodeMetadata::Array(_) = self.metadata {
            // Arrays cannot have consolidated metadata
            return None;
        }

        fn update_consolidated_metadata(
            node_path: &str,
            consolidated_metadata: &mut ConsolidatedMetadataMetadata,
            children: &[Node],
        ) {
            for child in children {
                let relative_path = child
                    .path()
                    .as_str()
                    .strip_prefix(node_path)
                    .expect("child path should always include the node path");
                let relative_path = relative_path.strip_prefix('/').unwrap_or(relative_path);
                let relative_path = relative_path.to_string();
                consolidated_metadata.insert(relative_path, child.metadata.clone());
                update_consolidated_metadata(node_path, consolidated_metadata, &child.children);
            }
        }
        let mut consolidated_metadata = HashMap::default();
        update_consolidated_metadata(
            self.path().as_str(),
            &mut consolidated_metadata,
            &self.children,
        );
        Some(consolidated_metadata)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        array::{ArrayBuilder, ArrayMetadataOptions},
        group::{GroupMetadata, GroupMetadataV3},
        storage::{store::MemoryStore, StoreKey, WritableStorageTraits},
    };

    use super::*;

    #[test]
    fn node_metadata_array() {
        const JSON_ARRAY: &str = r#"{
            "zarr_format": 3,
            "node_type": "array",
            "shape": [
              10000,
              1000
            ],
            "data_type": "float64",
            "chunk_grid": {
              "name": "regular",
              "configuration": {
                "chunk_shape": [
                  1000,
                  100
                ]
              }
            },
            "chunk_key_encoding": {
              "name": "default",
              "configuration": {
                "separator": "/"
              }
            },
            "fill_value": "NaN",
            "codecs": [
              {
                "name": "bytes",
                "configuration": {
                  "endian": "little"
                }
              },
              {
                "name": "gzip",
                "configuration": {
                  "level": 1
                }
              }
            ],
            "attributes": {
              "foo": 42,
              "bar": "apples",
              "baz": [
                1,
                2,
                3,
                4
              ]
            },
            "dimension_names": [
              "rows",
              "columns"
            ]
          }"#;
        serde_json::from_str::<NodeMetadata>(JSON_ARRAY).unwrap();
    }

    #[test]
    fn node_metadata_group() {
        const JSON_GROUP: &str = r#"{
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "spam": "ham",
            "eggs": 42
        }
    }"#;
        serde_json::from_str::<NodeMetadata>(JSON_GROUP).unwrap();
    }

    /// Implicit node support is removed since implicit groups were removed from the Zarr V3 spec
    #[test]
    fn node_implicit() {
        let store = std::sync::Arc::new(MemoryStore::new());
        let node_path = "/node";
        assert!(Node::open(&store, node_path).is_err());
    }

    #[test]
    fn node_array() {
        let store = std::sync::Arc::new(MemoryStore::new());
        let node_path = "/node";
        let array = ArrayBuilder::new(
            vec![1, 2, 3],
            vec![1, 1, 1],
            crate::array::DataType::Float32,
            0.0f32,
        )
        .build(store.clone(), node_path)
        .unwrap();
        array.store_metadata().unwrap();
        let stored_metadata = array.metadata_opt(&ArrayMetadataOptions::default());

        let node = Node::open(&store, node_path).unwrap();
        assert_eq!(node.metadata, NodeMetadata::Array(stored_metadata));
    }

    #[test]
    fn node_invalid_path() {
        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());
        let invalid_node_path = "node";
        assert_eq!(
            Node::open(&store, invalid_node_path)
                .unwrap_err()
                .to_string(),
            "invalid node path node"
        );
    }

    #[test]
    fn node_invalid_metadata() {
        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());
        store
            .set(&StoreKey::new("node/zarr.json").unwrap(), vec![0].into())
            .unwrap();
        assert_eq!(
            Node::open(&store, "/node").unwrap_err().to_string(),
            "error parsing metadata for node/zarr.json: expected value at line 1 column 1"
        );
    }

    #[test]
    fn node_invalid_child() {
        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());
        store
            .set(
                &StoreKey::new("node/array/zarr.json").unwrap(),
                vec![0].into(),
            )
            .unwrap();
        assert_eq!(
            Node::open(&store, "/node/array").unwrap_err().to_string(),
            "error parsing metadata for node/array/zarr.json: expected value at line 1 column 1"
        );
        assert_eq!(
            Node::open(&store, "/node").unwrap_err().to_string(),
            "Metadata is missing"
        );
    }

    #[test]
    fn node_root() {
        let node = Node::new_with_metadata(
            NodePath::root(),
            NodeMetadata::Group(GroupMetadata::V3(GroupMetadataV3::default())),
            vec![],
        );
        assert!(node.is_root());
    }
}

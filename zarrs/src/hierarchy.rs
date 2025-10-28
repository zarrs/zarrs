//! Zarr hierarchy.
//!
//! A Zarr hierarchy is a tree structure, where each node in the tree is either a [`Group`] or an [`Array`].
//!
//! A [`Hierarchy`] holds a mapping of [`NodePath`]s to [`NodeMetadata`].
//!
//! The [`Hierarchy::tree`] function can be used to create a string representation of the hierarchy.

use std::{collections::BTreeMap, sync::Arc};

use crate::node::get_all_nodes_of;
pub use crate::node::{Node, NodeCreateError, NodePath, NodePathError};

pub use crate::metadata::NodeMetadata;

use crate::{
    array::{Array, ArrayMetadata},
    config::MetadataRetrieveVersion,
    group::Group,
    storage::{ListableStorageTraits, ReadableStorageTraits},
};

#[cfg(feature = "async")]
use crate::{
    node::async_get_all_nodes_of,
    storage::{AsyncListableStorageTraits, AsyncReadableStorageTraits},
};

/// Zarr hierarchy.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#hierarchy>.
pub struct Hierarchy(BTreeMap<NodePath, NodeMetadata>);

type HierarchyCreateError = NodeCreateError;

impl Hierarchy {
    /// Create a new, empty hierarchy.
    fn new() -> Self {
        Hierarchy(BTreeMap::new())
    }

    fn insert(&mut self, path: NodePath, metadata: NodeMetadata) {
        self.0.insert(path, metadata);
    }

    /// Create a string representation of the hierarchy
    #[must_use]
    pub fn tree(&self) -> String {
        self.tree_of(&NodePath::root())
    }

    /// Create a string representation of the hierarchy for a `NodePath`
    #[must_use]
    pub fn tree_of(&self, parent_path: &NodePath) -> String {
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

        let mut s = String::from(parent_path.as_str());
        s.push('\n');

        let prefix = parent_path.as_str();
        let parent_depth = parent_path.as_path().components().count();

        for node in self
            .0
            .iter()
            .filter(|(path, _)| path.as_str().starts_with(prefix) && !path.as_str().eq(prefix))
            .map(|(p, md)| Node::new_with_metadata(p.clone(), md.clone(), vec![]))
        {
            let depth = node
                .path()
                .as_path()
                .components()
                .count()
                .saturating_sub(parent_depth);

            s.push_str(&" ".repeat(depth * 2));
            print_metadata(node.name().as_str(), &mut s, node.metadata());
        }
        s
    }

    /// Open a hierarchy at `path` and read metadata and children from `storage` with default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub fn open<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        path: &str,
    ) -> Result<Self, HierarchyCreateError> {
        Self::open_opt(storage, path, &MetadataRetrieveVersion::Default)
    }

    /// Open a hierarchy at a `path` and read metadata and children from `storage` with non-default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub fn open_opt<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, HierarchyCreateError> {
        let node_path = NodePath::try_from(path)?;
        let node_metadata = Node::get_metadata(storage, &node_path, version)?;
        let mut hierarchy = Hierarchy::new();

        let nodes = match node_metadata {
            NodeMetadata::Array(_) => Vec::default(),
            // TODO: Add consolidated metadata support
            NodeMetadata::Group(_) => get_all_nodes_of(storage, &node_path, version)?,
        };

        hierarchy.insert(node_path, node_metadata);
        hierarchy.extend(nodes);

        Ok(hierarchy)
    }

    /// Convenience method to create a `Hierarchy` from a `Group` with synchronous storage.
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if group metadata is invalid or there is a failure to list child nodes
    pub fn try_from_group<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        group: &Group<TStorage>,
    ) -> Result<Self, HierarchyCreateError> {
        let mut hierarchy = Hierarchy::new();
        hierarchy.insert(
            group.path().clone(),
            NodeMetadata::Group(group.metadata().clone()),
        );
        hierarchy.extend(get_all_nodes_of(
            &group.storage(),
            group.path(),
            &MetadataRetrieveVersion::Default,
        )?);
        Ok(hierarchy)
    }
    #[cfg(feature = "async")]
    /// Asynchronously open a hierarcht at `path` and read metadata and children from `storage` with default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub async fn async_open<
        TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
    >(
        storage: &Arc<TStorage>,
        path: &str,
    ) -> Result<Self, HierarchyCreateError> {
        Self::async_open_opt(storage, path, &MetadataRetrieveVersion::Default).await
    }

    #[cfg(feature = "async")]
    /// Asynchronously open a hierarchy at a `path` and read metadata and children from `storage` with non-default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub async fn async_open_opt<
        TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
    >(
        storage: &Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, HierarchyCreateError> {
        let node_path = NodePath::try_from(path)?;
        let node_metadata = Node::async_get_metadata(storage, &node_path, version).await?;
        let mut hierarchy = Hierarchy::new();

        let nodes = match node_metadata {
            NodeMetadata::Array(_) => Vec::default(),
            // TODO: Add consolidated metadata support
            NodeMetadata::Group(_) => async_get_all_nodes_of(storage, &node_path, version).await?,
        };

        hierarchy.insert(node_path, node_metadata);
        hierarchy.extend(nodes);

        Ok(hierarchy)
    }

    #[cfg(feature = "async")]
    /// Convenience method to create a `Hierarchy` from a Group with asynchronous storage.
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if group metadata is invalid or there is a failure to list child nodes
    pub async fn try_from_async_group<
        TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
    >(
        group: &Group<TStorage>,
    ) -> Result<Hierarchy, HierarchyCreateError> {
        let mut hierarchy = Hierarchy::new();
        hierarchy.insert(
            group.path().clone(),
            NodeMetadata::Group(group.metadata().clone()),
        );
        hierarchy.extend(
            async_get_all_nodes_of(
                &group.storage(),
                group.path(),
                &MetadataRetrieveVersion::Default,
            )
            .await?,
        );
        Ok(hierarchy)
    }
}

impl Extend<(NodePath, NodeMetadata)> for Hierarchy {
    fn extend<T: IntoIterator<Item = (NodePath, NodeMetadata)>>(&mut self, iter: T) {
        for (path, metadata) in iter {
            self.insert(path, metadata);
        }
    }
}

impl<TStorage: ?Sized> TryFrom<&Array<TStorage>> for Hierarchy {
    type Error = HierarchyCreateError;
    fn try_from(array: &Array<TStorage>) -> Result<Self, Self::Error> {
        let mut hierarchy = Hierarchy::new();
        hierarchy.insert(
            array.path().clone(),
            NodeMetadata::Array(array.metadata().clone()),
        );
        Ok(hierarchy)
    }
}

impl<TStorage: ?Sized> TryFrom<Array<TStorage>> for Hierarchy {
    type Error = HierarchyCreateError;
    fn try_from(array: Array<TStorage>) -> Result<Self, Self::Error> {
        (&array).try_into()
    }
}

#[cfg(test)]
mod tests {

    use zarrs_metadata::{v3::GroupMetadataV3, GroupMetadata};

    use super::*;
    use crate::{
        array::ArrayBuilder,
        group::GroupBuilder,
        storage::{store::MemoryStore, StoreKey, WritableStorageTraits},
    };

    const EXPECTED_TREE: &str = "/\n  array [10, 10] float32\n  group\n    array [10, 10] float32\n    subgroup\n      mysubarray [10, 10] float32\n";

    fn helper_create_dataset(store: &Arc<MemoryStore>) -> Group<MemoryStore> {
        let group_builder = GroupBuilder::default();

        let root = group_builder
            .build(store.clone(), &NodePath::root().as_str())
            .unwrap();
        let group = group_builder.build(store.clone(), "/group").unwrap();
        let array_builder = ArrayBuilder::new(
            vec![10, 10],
            vec![5, 5],
            crate::array::DataType::Float32,
            0.0f32,
        );

        let array = array_builder.build(store.clone(), "/array").unwrap();
        let group_array = array_builder.build(store.clone(), "/group/array").unwrap();
        let subgroup = group_builder
            .build(store.clone(), "/group/subgroup")
            .unwrap();
        let subgroup_array = array_builder
            .build(store.clone(), "/group/subgroup/mysubarray")
            .unwrap();

        root.store_metadata().unwrap();
        array.store_metadata().unwrap();
        group.store_metadata().unwrap();
        group_array.store_metadata().unwrap();
        subgroup.store_metadata().unwrap();
        subgroup_array.store_metadata().unwrap();

        root
    }

    #[test]
    fn hierarchy_tree_empty() {
        let hierarchy = Hierarchy::new();
        let tree_str = hierarchy.tree();
        assert_eq!(tree_str, "/\n");
    }

    #[test]
    fn hierarchy_try_from_array() {
        let store = Arc::new(MemoryStore::new());
        let array_builder =
            ArrayBuilder::new(vec![1], vec![1], crate::array::DataType::Float32, 0.0f32);

        let array = array_builder
            .build(store, "/store/of/data.zarr/path/to/an/array")
            .expect("Faulty test array");

        let hierarchy = Hierarchy::try_from(&array).unwrap();
        assert_eq!(hierarchy.0.len(), 1);
    }

    #[test]
    fn hierarchy_try_from_group() {
        let store = Arc::new(MemoryStore::new());
        let group = helper_create_dataset(&store);
        let hierarchy = Hierarchy::try_from_group(&group).unwrap();

        assert_eq!(hierarchy.0.len(), 6);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn hierarchy_async_try_from_group() {
        let store = std::sync::Arc::new(zarrs_object_store::AsyncObjectStore::new(
            object_store::memory::InMemory::new(),
        ));
        let group_path = "/group";
        let group_builder = GroupBuilder::new();
        let group = group_builder.build(store.clone(), group_path).unwrap();

        let subgroup = group_builder
            .build(store.clone(), "/group/subgroup")
            .unwrap();

        let array = ArrayBuilder::new(
            vec![10, 10],
            vec![5, 5],
            crate::array::DataType::Float32,
            0.0f32,
        )
        .build(store.clone(), "/group/subgroup/array")
        .unwrap();

        group.async_store_metadata().await.unwrap();
        subgroup.async_store_metadata().await.unwrap();
        array.async_store_metadata().await.unwrap();

        let hierarchy = Hierarchy::try_from_async_group(&group).await;
        assert!(hierarchy.is_ok());
        let hierarchy = hierarchy.unwrap();
        assert!("/\n  group\n    subgroup\n      array [10, 10] float32\n" == hierarchy.tree())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn hierarchy_async_try_from_invalid_async_group() {
        let store = std::sync::Arc::new(zarrs_object_store::AsyncObjectStore::new(
            object_store::memory::InMemory::new(),
        ));

        let root = Group::new_with_metadata(
            store.clone(),
            "/",
            GroupMetadata::V3(GroupMetadataV3::default()),
        )
        .unwrap();

        assert!(Hierarchy::try_from_async_group(&root).await.is_ok());

        use zarrs_storage::AsyncWritableStorageTraits;
        // Inject fauly subgroup
        store
            .set(
                &StoreKey::new("subgroup/zarr.json").unwrap(),
                vec![0].into(),
            )
            .await
            .unwrap();

        assert!(Hierarchy::try_from_async_group(&root).await.is_err());
    }

    #[test]
    fn hierarchy_try_from_invalid_group() {
        let store = Arc::new(MemoryStore::new());

        let root = Group::new_with_metadata(
            store.clone(),
            "/",
            GroupMetadata::V3(GroupMetadataV3::default()),
        )
        .unwrap();

        assert!(Hierarchy::try_from_group(&root).is_ok());

        // Inject fauly subgroup
        store
            .set(
                &StoreKey::new("subgroup/zarr.json").unwrap(),
                vec![0].into(),
            )
            .unwrap();

        assert!(Hierarchy::try_from_group(&root).is_err());
    }

    #[test]
    fn hierarchy_tree_of() {
        let store = Arc::new(MemoryStore::new());

        let group = helper_create_dataset(&store);

        let hierarchy = Hierarchy::try_from_group(&group).unwrap();

        assert_eq!(
            "/group/subgroup\n  mysubarray [10, 10] float32\n",
            hierarchy.tree_of(&NodePath::try_from("/group/subgroup").unwrap())
        );
    }

    #[test]
    fn hierarchy_tree() {
        let store = Arc::new(MemoryStore::new());

        let group = helper_create_dataset(&store);

        let hierarchy = Hierarchy::try_from_group(&group).unwrap();

        assert_eq!("/\n  array [10, 10] float32\n  group\n    array [10, 10] float32\n    subgroup\n      mysubarray [10, 10] float32\n"
                    ,hierarchy.tree());
    }
    #[test]
    fn hierarchy_open() {
        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());

        let _group = helper_create_dataset(&store);

        let h = Hierarchy::open(&store, "/").unwrap();

        assert_eq!(EXPECTED_TREE, h.tree())
    }
}

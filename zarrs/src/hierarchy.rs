//! Zarr hierarchies.
//!
//! A Zarr hierarchy is a tree structure, where each node in the tree is either a [`Group`] or an [`Array`].
//!
//! A [`Hierarchy`] holds a mapping of [`NodePath`]s to [`NodeMetadata`].
//!
//! The [`Hierarchy::tree`] function can be used to create a string representation of the hierarchy.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#hierarchy>.

use std::{collections::BTreeMap, sync::Arc};

pub use crate::metadata::NodeMetadata;
use crate::node::get_all_nodes_of;
pub use crate::node::{Node, NodeCreateError, NodePath, NodePathError};
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

/// A Zarr hierarchy.
#[derive(Debug, Clone)]
pub struct Hierarchy(BTreeMap<NodePath, NodeMetadata>);

impl std::fmt::Display for Hierarchy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tree())
    }
}

/// A hierarchy creation error.
pub type HierarchyCreateError = NodeCreateError;

impl Hierarchy {
    /// Create a new, empty hierarchy.
    fn new() -> Self {
        Hierarchy(BTreeMap::new())
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

        hierarchy.0.insert(node_path, node_metadata);
        hierarchy.0.extend(nodes);

        Ok(hierarchy)
    }

    #[cfg(feature = "async")]
    /// Asynchronously open a hierarchy at `path` and read metadata and children from `storage` with default [`MetadataRetrieveVersion`].
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

        hierarchy.0.insert(node_path, node_metadata);
        hierarchy.0.extend(nodes);

        Ok(hierarchy)
    }

    /// Convenience method to create a `Hierarchy` from a `Group` with synchronous storage.
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if group metadata is invalid or there is a failure to list child nodes.
    pub fn try_from_group<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        group: &Group<TStorage>,
    ) -> Result<Self, HierarchyCreateError> {
        let mut hierarchy = Hierarchy::new();
        hierarchy.0.insert(
            group.path().clone(),
            NodeMetadata::Group(group.metadata().clone()),
        );
        hierarchy.0.extend(get_all_nodes_of(
            &group.storage(),
            group.path(),
            &MetadataRetrieveVersion::Default,
        )?);
        Ok(hierarchy)
    }

    #[cfg(feature = "async")]
    /// Convenience method to create a `Hierarchy` from a Group with asynchronous storage.
    ///
    /// # Errors
    /// Returns [`HierarchyCreateError`] if group metadata is invalid or there is a failure to list child nodes.
    pub async fn try_from_async_group<
        TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
    >(
        group: &Group<TStorage>,
    ) -> Result<Hierarchy, HierarchyCreateError> {
        let mut hierarchy = Hierarchy::new();
        hierarchy.0.insert(
            group.path().clone(),
            NodeMetadata::Group(group.metadata().clone()),
        );
        hierarchy.0.extend(
            async_get_all_nodes_of(
                &group.storage(),
                group.path(),
                &MetadataRetrieveVersion::Default,
            )
            .await?,
        );
        Ok(hierarchy)
    }

    // /// Insert a node into the hierarchy.
    // pub fn insert(&mut self, path: NodePath, metadata: NodeMetadata) -> Option<NodeMetadata> {
    //     self.0.insert(path, metadata)
    // }

    /// Create a string representation of the hierarchy starting from the root.
    #[must_use]
    pub fn tree(&self) -> String {
        self.tree_of(&NodePath::root())
    }

    /// Create a string representation of the hierarchy starting from `path`.
    #[must_use]
    pub fn tree_of(&self, path: &NodePath) -> String {
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

        let mut s = String::from(path.as_str());
        s.push('\n');

        let prefix = path.as_str();
        let depth = path.as_path().components().count();

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
                .saturating_sub(depth);

            s.push_str(&" ".repeat(depth * 2));
            print_metadata(node.name().as_str(), &mut s, node.metadata());
        }
        s
    }
}

// impl Extend<(NodePath, NodeMetadata)> for Hierarchy {
//     fn extend<T: IntoIterator<Item = (NodePath, NodeMetadata)>>(&mut self, iter: T) {
//         for (path, metadata) in iter {
//             self.insert(path, metadata);
//         }
//     }
// }

impl<TStorage: ?Sized> TryFrom<&Array<TStorage>> for Hierarchy {
    type Error = HierarchyCreateError;
    fn try_from(array: &Array<TStorage>) -> Result<Self, Self::Error> {
        let mut hierarchy = Hierarchy::new();
        hierarchy.0.insert(
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
    use std::num::NonZeroU64;

    use super::*;
    use crate::metadata::{
        GroupMetadata,
        v2::{ArrayMetadataV2, GroupMetadataV2},
        v3::GroupMetadataV3,
    };
    #[cfg(feature = "async")]
    use crate::storage::AsyncReadableWritableListableStorageTraits;
    use crate::{
        array::ArrayBuilder,
        group::GroupBuilder,
        storage::{StoreKey, WritableStorageTraits, store::MemoryStore},
    };

    const EXPECTED_TREE: &str = "/\n  array [10, 10] float32\n  group\n    array [10, 10] float32\n    subgroup\n      mysubarray [10, 10] float32\n";

    fn helper_create_dataset(store: &Arc<MemoryStore>) -> Group<MemoryStore> {
        let group_builder = GroupBuilder::default();

        let root = group_builder
            .build(store.clone(), NodePath::root().as_str())
            .unwrap();
        let group = group_builder.build(store.clone(), "/group").unwrap();
        let array_builder = ArrayBuilder::new(
            vec![10, 10],
            vec![5, 5],
            crate::array::data_type::float32(),
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
    fn hierarchy_try_from_array() {
        let store = Arc::new(MemoryStore::new());
        let array_builder =
            ArrayBuilder::new(vec![1], vec![1], crate::array::data_type::float32(), 0.0f32);

        let array = array_builder
            .build(store, "/store/of/data.zarr/path/to/an/array")
            .expect("Faulty test array");

        let hierarchy = Hierarchy::try_from(&array).unwrap();
        assert_eq!(hierarchy.0.len(), 1);
        let hierarchy = Hierarchy::try_from(array).unwrap();
        assert_eq!(hierarchy.0.len(), 1);
    }

    #[cfg(feature = "async")]
    #[test]
    fn hierarchy_try_from_async_array() {
        let store = std::sync::Arc::new(zarrs_object_store::AsyncObjectStore::new(
            object_store::memory::InMemory::new(),
        ));
        let array_builder =
            ArrayBuilder::new(vec![1], vec![1], crate::array::data_type::float32(), 0.0f32);

        let array = array_builder
            .build(store, "/store/of/data.zarr/path/to/an/array")
            .expect("Faulty test array");

        let hierarchy = Hierarchy::try_from(&array).unwrap();
        assert_eq!(hierarchy.0.len(), 1);
        let hierarchy = Hierarchy::try_from(array).unwrap();
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
            crate::array::data_type::float32(),
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
        assert!(
            "/\n  group\n    subgroup\n      array [10, 10] float32\n" == hierarchy.to_string()
        );
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

        use crate::storage::AsyncWritableStorageTraits;
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

        assert_eq!(
            "/\n  array [10, 10] float32\n  group\n    array [10, 10] float32\n    subgroup\n      mysubarray [10, 10] float32\n",
            hierarchy.tree()
        );

        let store = Arc::new(MemoryStore::new());
        let groupv2 = Group::new_with_metadata(
            store.clone(),
            "/groupv2",
            GroupMetadata::V2(GroupMetadataV2::new()),
        )
        .expect("Unexpected issue when greating a Group for testing.");

        let arrayv2 = Array::new_with_metadata(
            store.clone(),
            "/groupv2/arrayv2",
            ArrayMetadata::V2(ArrayMetadataV2::new(
                vec![1],
                crate::array::ChunkShape::from(vec![NonZeroU64::new(1).unwrap()]),
                zarrs_metadata::v2::DataTypeMetadataV2::Simple("<f8".into()),
                zarrs_metadata::v2::FillValueMetadataV2::from(f64::NAN),
                None,
                None,
            )),
        )
        .expect("Unexpected issue when creating a v2 Array for testing.");

        let _ = groupv2.store_metadata();
        let _ = arrayv2.store_metadata();

        let h = Hierarchy::try_from_group(&groupv2);
        assert!(h.is_ok());
        assert!("/\n  groupv2\n    arrayv2 [1] Simple(\"<f8\")\n" == h.unwrap().tree());
    }

    #[test]
    fn hierarchy_tree_empty() {
        let hierarchy = Hierarchy::new();
        let tree_str = hierarchy.tree();
        assert_eq!(tree_str, "/\n");
    }

    #[test]
    fn hierarchy_open() {
        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());

        let _group = helper_create_dataset(&store);

        // Open a group node
        let h = Hierarchy::open(&store, "/").unwrap();
        assert_eq!(EXPECTED_TREE, h.tree());

        // Open an array node
        let h = Hierarchy::open(&store, "/array").unwrap();
        assert_eq!("/\n  array [10, 10] float32\n", h.tree());
    }

    #[cfg(feature = "async")]
    async fn async_helper_create_dataset<
        AStore: ?Sized + AsyncReadableWritableListableStorageTraits + 'static,
    >(
        store: &Arc<AStore>,
    ) -> Group<AStore> {
        let group_builder = GroupBuilder::default();

        let root = group_builder
            .build(store.clone(), NodePath::root().as_str())
            .unwrap();
        let group = group_builder.build(store.clone(), "/group").unwrap();
        let array_builder = ArrayBuilder::new(
            vec![10, 10],
            vec![5, 5],
            crate::array::data_type::float32(),
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

        root.async_store_metadata().await.unwrap();
        array.async_store_metadata().await.unwrap();
        group.async_store_metadata().await.unwrap();
        group_array.async_store_metadata().await.unwrap();
        subgroup.async_store_metadata().await.unwrap();
        subgroup_array.async_store_metadata().await.unwrap();

        root
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn hierarchy_async_open() {
        use crate::storage::AsyncReadableWritableListableStorage;

        let store: AsyncReadableWritableListableStorage = Arc::new(
            zarrs_object_store::AsyncObjectStore::new(object_store::memory::InMemory::new()),
        );

        let _group = async_helper_create_dataset(&store).await;

        // Open a Group node
        let h = Hierarchy::async_open(&store, "/").await;
        assert!(h.is_ok());
        assert_eq!(EXPECTED_TREE, h.unwrap().tree());

        // Open an Array node
        let h = Hierarchy::async_open(&store, "/array").await;
        assert!(h.is_ok());
        assert_eq!("/\n  array [10, 10] float32\n", h.unwrap().tree());
    }
}

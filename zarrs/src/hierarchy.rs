//! Zarr hierarchy  .
//!
//! A hierarchy in a Zarr hierarchy represents the relation of [`Node`](crate::node::Node) which can either be  of types [`Array`](crate::array::Array) or [`Group`](crate::group::Group).
//!
//! A [`Hierarchy`] holds the [`NodeMetadata`] for the respective [`NodePath`].
//!
//! The [`Hierarchy::tree`] function can be used to create a string representation of the hierarchy.

use std::{collections::BTreeMap, sync::Arc};

use zarrs_storage::discover_children;

use crate::node;
pub use crate::node::{
    Node,
    node_sync::get_child_nodes,
    NodeCreateError,
    NodePath,
    NodePathError,
};

pub use crate::metadata::NodeMetadata;

use crate::{
    array::{ArrayMetadata},
    config::MetadataRetrieveVersion,
    // group::GroupCreateError,
    metadata::{
        v2::{ArrayMetadataV2, GroupMetadataV2},
        GroupMetadata,
    },
    storage::{StorePrefix,ListableStorageTraits, ReadableStorageTraits, StorageError},
};

/// Zarr hierarchy.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#hierarchy>.
pub struct Hierarchy (BTreeMap<NodePath, NodeMetadata>);

type HierarchyCreateError = NodeCreateError;

impl Hierarchy {
    /// Create a new, empty hierarchy.
    pub fn new() -> Self {
        Hierarchy(BTreeMap::new())
    }

    fn insert(&mut self, path: NodePath, metadata: NodeMetadata) {
        self.0.insert(path, metadata);
    }

    fn insert_node(&mut self, node: &Node) {
        self.insert(node.path().clone(), node.metadata().clone());
    }

    fn insert_nodes<'a>(&mut self, nodes: impl Iterator<Item=&'a Node>) {
        for node in nodes {
            self.insert_node(node);
        }
    }

    pub fn tree(&self) -> String {
        self.tree_of(None)
    }
    /// Create a string representation of the hierarchy.
    pub fn tree_of(&self, parent_path: Option<NodePath>) -> String {
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
                NodeMetadata::Group(_) => {string.push_str(name);},
            }
            string.push('\n');
        }
        
        let mut s = String::new();
        let prefix = match &parent_path {
            Some(n) => n.as_str(),
            None => "",
        };
        for (name, metadata) in self.0.iter().filter(|(path,_)| path.as_str().starts_with(prefix)){
            if let Ok(name) = NodePath::try_from(name.as_str().strip_prefix(prefix).unwrap()) {
                print_metadata(&name.as_str(), &mut s, metadata);
            }
        }
        s
    }

    /// Recursively get all nodes in the hierarchy under root '/'.
    pub fn get_all_nodes<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>) -> Result<Vec<Node>,HierarchyCreateError> {
            Self::get_all_nodes_of(&storage, Some(NodePath::root()))
    }
    
    /// Recursively get all nodes under a given path.
    pub fn get_all_nodes_of<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        parent_path: Option<NodePath>,
    ) -> Result<Vec<Node>,HierarchyCreateError> {
        let path = match &parent_path {
            Some(n) => n,
            None => &NodePath::root(),
        };

        let mut nodes: Vec<Node> = Vec::new();
        for child in get_child_nodes(storage, path, false)? {
            nodes.extend(get_child_nodes(storage, child.path(), false)?);
            nodes.push(child);
        }
        Ok(nodes)
    }
    

    /// Open a hierarchy at `path` and read metadata and children from `storage` with default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub fn open<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        path: &str,
    ) -> Result<Self, HierarchyCreateError> {
        Self::open_opt(storage, path, &MetadataRetrieveVersion::Default)
        //Ok(Self::new())
    }

    /// Open a node at `path` and read metadata and children from `storage` with non-default [`MetadataRetrieveVersion`].
    ///
    /// # Errors
    /// Returns [`NodeCreateError`] if metadata is invalid or there is a failure to list child nodes.
    pub fn open_opt<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
        storage: &Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, HierarchyCreateError> {
        let path: NodePath = path.try_into()?;
        let metadata = Node::get_metadata(storage, &path, version)?;
        
        let mut hierarchy = Hierarchy::new();
        hierarchy.insert(path.clone(), metadata.clone());
        
        
        let nodes = match metadata {
            NodeMetadata::Array(_) => Vec::default(),
            // TODO: Add consolidated metadata support
            NodeMetadata::Group(_) => Self::get_all_nodes_of(storage, Some(path))?,
        };
        
        hierarchy.insert_nodes(nodes.iter());

        Ok(hierarchy)
    }
   
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::ArrayBuilder,
        group::GroupBuilder,
        storage::{store::MemoryStore, StoreKey, WritableStorageTraits}
    };

    const JSON_GROUP : &str= r#"{
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {}
    }"#;

    #[test]
    fn hierarchy_tree_empty() {
        let hierarchy = Hierarchy::new();
        let tree_str = hierarchy.tree();
        assert_eq!(tree_str, "0");
    }

    #[test]
    fn hierarchy_open() {
        let root_md = serde_json::from_str::<NodeMetadata>(JSON_GROUP).unwrap();

        let json = serde_json::to_vec_pretty(&root_md).unwrap();

        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());
        store
            .set(&StoreKey::new("root/zarr.json").unwrap(), json.into())
            .unwrap();



        let array_path = "/root/array";

        let array_builder = ArrayBuilder::new(
                vec![1, 2, 3],
                vec![1, 1, 1],
                crate::array::DataType::Float32,
                0.0f32,
            );

        let array = array_builder
            .build(store.clone(), array_path)
            .unwrap();
        array.store_metadata().unwrap();

        
        for i in 0..3 {
            let group_path = format!("/root/group{}",i);

            let group = GroupBuilder::new().build(store.clone(), &group_path).unwrap();
            group.store_metadata().unwrap();

            let array_path: NodePath = format!("{}/array{}", &group_path, i).as_str().try_into().unwrap();
            let array = array_builder
                .build(store.clone(), &array_path.as_str())
                .unwrap();
            
            array.store_metadata().unwrap();
        }


        
        let hierarchy = Hierarchy::open(&store, "/root").unwrap();
        let tree_str = hierarchy.tree();
        println!("{}", tree_str);
        assert_eq!(hierarchy.tree(), "/root\n/root/array [1, 2, 3] float32\n/root/group0\n/root/group0/array0 [1, 2, 3] float32\n/root/group1\n/root/group1/array1 [1, 2, 3] float32\n/root/group2\n/root/group2/array2 [1, 2, 3] float32\n");

        // println!("Node hierarchy tree:\n{}", Node::open(&store, "/root").unwrap().hierarchy_tree());

        let subgroup_path: NodePath = "/root/group1".try_into().unwrap();
        //println!("Hierarchy tree of node {}: {}", subgroup_path,
        assert_eq!(hierarchy.tree_of(Some(subgroup_path.clone())),"/array1 [1, 2, 3] float32\n");
    }
}

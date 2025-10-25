use std::sync::Arc;

use crate::{
    config::MetadataRetrieveVersion,
    storage::{
        discover_children, ListableStorageTraits, ReadableStorageTraits, StorageError, StorePrefix,
    },
};

use super::{
    meta_key_v2_array, meta_key_v2_group, meta_key_v3, Node, NodeCreateError, NodeMetadata,
    NodePath, NodePathError,
};

/// Get the child nodes.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub fn get_child_nodes<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
    storage: &Arc<TStorage>,
    path: &NodePath,
    recursive: bool,
) -> Result<Vec<Node>, NodeCreateError> {
    get_child_nodes_opt(storage, path, recursive, &MetadataRetrieveVersion::Default)
}

/// Get the child nodes.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub fn get_child_nodes_opt<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
    storage: &Arc<TStorage>,
    path: &NodePath,
    recursive: bool,
    version: &MetadataRetrieveVersion,
) -> Result<Vec<Node>, NodeCreateError> {
    let prefix: StorePrefix = path.try_into()?;
    let prefixes = discover_children(storage, &prefix)?;
    let mut nodes: Vec<Node> = Vec::new();
    for prefix in &prefixes {
        let path: NodePath = prefix
            .try_into()
            .map_err(|err: NodePathError| StorageError::Other(err.to_string()))?;
        let child_metadata = match Node::get_metadata(storage, &path, version) {
            Ok(metadata) => metadata,
            Err(NodeCreateError::MissingMetadata) => {
                log::warn!(
                        "Object at {path} is not recognized as a component of a Zarr hierarchy. Ignoring."
                    );
                continue;
            }
            Err(e) => return Err(e),
        };
        let path: NodePath = prefix
            .try_into()
            .map_err(|err: NodePathError| StorageError::Other(err.to_string()))?;
        let children = if recursive {
            match child_metadata {
                NodeMetadata::Array(_) => Vec::default(),
                NodeMetadata::Group(_) => get_child_nodes_opt(storage, &path, true, version)?,
            }
        } else {
            vec![]
        };
        nodes.push(Node::new_with_metadata(path, child_metadata, children));
    }
    Ok(nodes)
}

/// Recursively get all nodes in the hierarchy under root '/'.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
#[allow(dead_code)]
fn get_all_nodes<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
    storage: &Arc<TStorage>,
) -> Result<Vec<Node>, NodeCreateError> {
    get_all_nodes_of(storage, &NodePath::root())
}

/// Recursively get all nodes under a given path.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub fn get_all_nodes_of<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
    storage: &Arc<TStorage>,
    path: &NodePath,
) -> Result<Vec<Node>, NodeCreateError> {
    let mut nodes: Vec<Node> = Vec::new();
    for child in get_child_nodes(storage, path, false)? {
        if let NodeMetadata::Group(_) = child.metadata() {
            nodes.extend(get_all_nodes_of(storage, child.path())?);
        }
        nodes.push(child);
    }
    Ok(nodes)
}

/// Check if a node exists.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub fn node_exists<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
    storage: &Arc<TStorage>,
    path: &NodePath,
) -> Result<bool, StorageError> {
    Ok(storage.get(&meta_key_v3(path))?.is_some()
        || storage.get(&meta_key_v2_array(path))?.is_some()
        || storage.get(&meta_key_v2_group(path))?.is_some())
}

/// Check if a node exists.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub fn node_exists_listable<TStorage: ?Sized + ListableStorageTraits>(
    storage: &Arc<TStorage>,
    path: &NodePath,
) -> Result<bool, StorageError> {
    let prefix: StorePrefix = path.try_into()?;
    storage.list_prefix(&prefix).map(|keys| {
        keys.contains(&meta_key_v3(path))
            | keys.contains(&meta_key_v2_array(path))
            | keys.contains(&meta_key_v2_group(path))
    })
}

#[cfg(test)]
mod tests {

    use crate::storage::{store::MemoryStore, StoreKey, WritableStorageTraits};

    use super::*;

    #[test]
    fn warning_get_child_nodes() {
        testing_logger::setup();
        const JSON_GROUP: &str = r#"{
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "info": "The store for this group contains a fake node, which is not part of the dataset."
            }
        }"#;
        let root_md = serde_json::from_str::<NodeMetadata>(JSON_GROUP).unwrap();
        let json = serde_json::to_vec_pretty(&root_md).unwrap();

        let store: std::sync::Arc<MemoryStore> = std::sync::Arc::new(MemoryStore::new());
        store
            .set(&StoreKey::new("root/zarr.json").unwrap(), json.into())
            .unwrap();

        store
            .set(
                &StoreKey::new("root/fakenode/content/zarr.json").unwrap(),
                vec![0].into(),
            )
            .unwrap();

        let path: NodePath = "/root".try_into().unwrap();
        let nodes = get_child_nodes(&store, &path, true).unwrap();
        assert_eq!(nodes.len(), 0); // Should have 0 valid child nodes (fakenode is invalid)

        // Now make it a real node but corrupted
        store
            .set(
                &StoreKey::new("root/fakenode/zarr.json").unwrap(),
                vec![0].into(),
            )
            .unwrap();

        let path: NodePath = "/root/fakenode".try_into().unwrap();
        let res = get_child_nodes(&store, &path, true);
        assert!(res.is_err());
        assert!(!matches!(
            res.unwrap_err(),
            NodeCreateError::MissingMetadata
        ));

        testing_logger::validate(|captured_logs| {
            assert_eq!(captured_logs.len(), 1);
            assert_eq!(
                captured_logs.first().unwrap().body,
                "Object at /root/fakenode is not recognized as a component of a Zarr hierarchy. Ignoring.",
            );
        });
    }
}

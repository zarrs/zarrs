use std::sync::Arc;

use super::{
    Node, NodeCreateError, NodeMetadata, NodePath, NodePathError, meta_key_v2_array,
    meta_key_v2_group, meta_key_v3,
};
use crate::config::MetadataRetrieveVersion;
use crate::storage::{
    AsyncListableStorageTraits, AsyncReadableStorageTraits, StorageError, StorePrefix,
    async_discover_children,
};

/// Asynchronously get the child nodes.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub async fn async_get_child_nodes<TStorage>(
    storage: &Arc<TStorage>,
    path: &NodePath,
    recursive: bool,
) -> Result<Vec<Node>, NodeCreateError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
{
    async_get_child_nodes_opt(storage, path, recursive, &MetadataRetrieveVersion::Default).await
}

/// Asynchronously get the child nodes.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub async fn async_get_child_nodes_opt<TStorage>(
    storage: &Arc<TStorage>,
    path: &NodePath,
    recursive: bool,
    version: &MetadataRetrieveVersion,
) -> Result<Vec<Node>, NodeCreateError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
{
    let prefix: StorePrefix = path.try_into()?;
    let prefixes = async_discover_children(storage, &prefix).await?;
    let mut nodes: Vec<Node> = Vec::new();
    // TODO: Asynchronously get metadata of all prefixes
    for prefix in &prefixes {
        let path: NodePath = prefix
            .try_into()
            .map_err(|err: NodePathError| StorageError::Other(err.to_string()))?;
        let child_metadata = Node::async_get_metadata(storage, &path, version).await?;

        let children = if recursive {
            match child_metadata {
                NodeMetadata::Array(_) => Vec::default(),
                NodeMetadata::Group(_) => {
                    Box::pin(async_get_child_nodes_opt(storage, &path, true, version)).await?
                }
            }
        } else {
            vec![]
        };
        nodes.push(Node::new_with_metadata(path, child_metadata, children));
    }
    Ok(nodes)
}

/// Asynchronously get all nodes under a given path, recursively.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub(crate) async fn async_get_all_nodes_of<
    TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
>(
    storage: &Arc<TStorage>,
    path: &NodePath,
    version: &MetadataRetrieveVersion,
) -> Result<Vec<(NodePath, NodeMetadata)>, NodeCreateError> {
    let mut nodes: Vec<(NodePath, NodeMetadata)> = Vec::new();
    for child in async_get_child_nodes_opt(storage, path, false, version).await? {
        if let NodeMetadata::Group(_) = child.metadata() {
            nodes.extend(Box::pin(async_get_all_nodes_of(storage, child.path(), version)).await?);
        }
        nodes.push((child.path, child.metadata));
    }
    Ok(nodes)
}

/// Asynchronously check if a node exists.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub async fn async_node_exists<
    TStorage: ?Sized + AsyncReadableStorageTraits + AsyncListableStorageTraits,
>(
    storage: &Arc<TStorage>,
    path: &NodePath,
) -> Result<bool, StorageError> {
    Ok(storage.get(&meta_key_v3(path)).await?.is_some()
        || storage.get(&meta_key_v2_array(path)).await?.is_some()
        || storage.get(&meta_key_v2_group(path)).await?.is_some())
}

/// Asynchronously check if a node exists.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub async fn async_node_exists_listable<TStorage: ?Sized + AsyncListableStorageTraits>(
    storage: &Arc<TStorage>,
    path: &NodePath,
) -> Result<bool, StorageError> {
    let prefix: StorePrefix = path.try_into()?;
    storage.list_prefix(&prefix).await.map(|keys| {
        keys.contains(&meta_key_v3(path))
            | keys.contains(&meta_key_v2_array(path))
            | keys.contains(&meta_key_v2_group(path))
    })
}

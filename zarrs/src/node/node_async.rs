use std::sync::Arc;

use crate::{
    config::MetadataRetrieveVersion,
    storage::{
        async_discover_children, AsyncListableStorageTraits, AsyncReadableStorageTraits,
        StorageError, StorePrefix,
    },
};

use super::{
    meta_key_v2_array, meta_key_v2_group, meta_key_v3, Node, NodeCreateError, NodeMetadata,
    NodePath, NodePathError,
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

use std::sync::Arc;

use super::{
    Node, NodeCreateError, NodeMetadata, NodePath, NodePathError, meta_key_v2_array,
    meta_key_v2_group, meta_key_v3,
};
use crate::config::MetadataRetrieveVersion;
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncListableStorageTraits, AsyncReadableStorageTraits, async_discover_children,
};
use zarrs_storage::{
    ListableStorageTraits, ReadableStorageTraits, StorageError, StorePrefix, discover_children,
};

ambisync::scoped! {
    #![defaults(
        sync(
            fns("async_{}"),
            types(
                AsyncReadableStorageTraits => ReadableStorageTraits,
                AsyncListableStorageTraits => ListableStorageTraits,
            ),
        ),
        async(feature = "async"),
    )]

        #[ambisync]
        /// Get the child nodes.
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
            async_get_child_nodes_opt(
                storage,
                path,
                recursive,
                &MetadataRetrieveVersion::Default,
            )
            .await
        }

        /// Get the child nodes.
        ///
        /// # Errors
        /// Returns a [`StorageError`] if there is an underlying error with the store.
        #[ambisync]
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
                let child_metadata_result = Node::async_get_metadata(storage, &path, version).await;
                let child_metadata = ambisync::alt!(
                    sync => match child_metadata_result {
                        Ok(metadata) => metadata,
                        Err(NodeCreateError::MissingMetadata(_)) => {
                            log::warn!(
                                "Object at {path} is not recognized as a component of a Zarr hierarchy. Ignoring."
                            );
                            continue;
                        }
                        Err(error) => return Err(error),
                    },
                    async => child_metadata_result?,
                );

                let children = if recursive {
                    match child_metadata {
                        NodeMetadata::Array(_) => Vec::default(),
                        NodeMetadata::Group(_) => {
                            ambisync::alt!(
                                sync => async_get_child_nodes_opt(storage, &path, true, version)?,
                                async => Box::pin(async_get_child_nodes_opt(
                                    storage, &path, true, version,
                                ))
                                .await?,
                            )
                        }
                    }
                } else {
                    vec![]
                };
                nodes.push(Node::new_with_metadata(path, child_metadata, children));
            }
            Ok(nodes)
        }

        /// Recursively get all nodes under a given path.
        ///
        /// # Errors
        /// Returns a [`StorageError`] if there is an underlying error with the store.
        #[ambisync]
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
                    let descendants = ambisync::alt!(
                        sync => async_get_all_nodes_of(storage, child.path(), version)?,
                        async => Box::pin(async_get_all_nodes_of(
                            storage,
                            child.path(),
                            version,
                        ))
                        .await?,
                    );
                    nodes.extend(descendants);
                }
                nodes.push((child.path, child.metadata));
            }
            Ok(nodes)
        }

        /// Check if a node exists.
        ///
        /// # Errors
        /// Returns a [`StorageError`] if there is an underlying error with the store.
        #[ambisync]
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

        /// Check if a node exists using a list operation.
        ///
        /// # Errors
        /// Returns a [`StorageError`] if there is an underlying error with the store.
        #[ambisync]
        pub async fn async_node_exists_listable<
            TStorage: ?Sized + AsyncListableStorageTraits,
        >(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use zarrs_storage::store::MemoryStore;
    use zarrs_storage::{StoreKey, WritableStorageTraits};

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

        let store: Arc<MemoryStore> = Arc::new(MemoryStore::new());
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
        assert_eq!(nodes.len(), 0);

        store
            .set(
                &StoreKey::new("root/fakenode/zarr.json").unwrap(),
                vec![0].into(),
            )
            .unwrap();

        let path: NodePath = "/root/fakenode".try_into().unwrap();
        let result = get_child_nodes(&store, &path, true);
        assert!(result.is_err());
        assert!(!matches!(
            result.unwrap_err(),
            NodeCreateError::MissingMetadata(_)
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

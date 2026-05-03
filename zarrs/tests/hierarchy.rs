#![cfg(feature = "filesystem")]
#![allow(missing_docs)]

use std::sync::Arc;

use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::metadata_ext::group::consolidated_metadata::ConsolidatedMetadata;
use zarrs::node::Node;
#[cfg(feature = "async")]
use zarrs::storage::{AsyncListableStorageTraits, AsyncReadableStorageTraits};

fn sync_store() -> Arc<FilesystemStore> {
    Arc::new(
        FilesystemStore::new("./tests/data/hierarchy.zarr")
            .unwrap()
            .sorted(),
    )
}

#[cfg(feature = "async")]
async fn async_store() -> Arc<impl AsyncReadableStorageTraits + AsyncListableStorageTraits> {
    use object_store::local::LocalFileSystem;

    Arc::new(zarrs_object_store::AsyncObjectStore::new(
        LocalFileSystem::new_with_prefix("./tests/data/hierarchy.zarr").unwrap(),
    ))
}

#[test]
fn hierarchy_tree() {
    let store = sync_store();
    let node = Node::open(&store, "/").unwrap();
    let tree = node.hierarchy_tree();
    println!("{tree:?}");
    assert_eq!(
        tree,
        "/
  a
    baz [10000, 1000] float64
    foo [10000, 1000] float64
  b
"
    );
}

#[test]
fn consolidated_metadata() {
    let store = sync_store();
    let node = Node::open(&store, "/").unwrap();
    let consolidated_metadata = node.consolidate_metadata().unwrap();
    println!("{consolidated_metadata:#?}");

    for relative_path in ["a", "a/baz", "a/foo", "b"] {
        let consolidated = consolidated_metadata.get(relative_path).unwrap();
        let node_path = format!("/{relative_path}");
        let actual = Node::open(&store, &node_path).unwrap();
        assert_eq!(consolidated, actual.metadata());
    }

    let mut group = Group::open(store.clone(), "/").unwrap();
    assert!(group.consolidated_metadata().is_none());
    group.set_consolidated_metadata(Some(ConsolidatedMetadata {
        metadata: consolidated_metadata,
        ..Default::default()
    }));
    assert!(group.consolidated_metadata().is_some());

    let node = Node::open(&store, "/a").unwrap();
    let consolidated_metadata = node.consolidate_metadata().unwrap();
    println!("{consolidated_metadata:#?}");
    for relative_path in ["baz", "foo"] {
        let consolidated = consolidated_metadata.get(relative_path).unwrap();
        let node_path = format!("/a/{relative_path}");
        let actual = Node::open(&store, &node_path).unwrap();
        assert_eq!(consolidated, actual.metadata());
    }
}

#[test]
fn child_arrays() {
    let store = sync_store();

    // Two arrays in /a
    let group = Group::open(store.clone(), "/a").unwrap();
    let arrays = group.child_arrays().unwrap();
    let array_paths: Vec<_> = arrays.iter().map(|a| a.path().as_str()).collect();
    assert_eq!(array_paths, ["/a/baz", "/a/foo"]);

    // At root, there are no arrays
    let group = Group::open(store.clone(), "/").unwrap();
    let arrays = group.child_arrays().unwrap();
    assert!(arrays.is_empty());
}

#[test]
fn child_groups() {
    let store = sync_store();

    // At root, there are two groups: a and b
    let group = Group::open(store.clone(), "/").unwrap();
    let groups = group.child_groups().unwrap();
    let group_paths: Vec<_> = groups.iter().map(|g| g.path().as_str()).collect();
    assert_eq!(group_paths, ["/a", "/b"]);

    // In /a, there are no child groups (only arrays)
    let group = Group::open(store.clone(), "/a").unwrap();
    let groups = group.child_groups().unwrap();
    assert!(groups.is_empty());
}

#[test]
fn child_paths() {
    let store = sync_store();

    // At root, there are two child paths: a and b (both groups)
    let group = Group::open(store.clone(), "/").unwrap();
    let paths = group.child_paths().unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a", "/b"]);

    // In /a, there are two child paths: baz and foo (both arrays)
    let group = Group::open(store.clone(), "/a").unwrap();
    let paths = group.child_paths().unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a/baz", "/a/foo"]);
}

#[test]
fn child_group_paths() {
    let store = sync_store();

    // At root, there are two group paths: a and b
    let group = Group::open(store.clone(), "/").unwrap();
    let paths = group.child_group_paths().unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a", "/b"]);

    // In /a, there are no child group paths (only arrays)
    let group = Group::open(store.clone(), "/a").unwrap();
    let paths = group.child_group_paths().unwrap();
    assert!(paths.is_empty());
}

#[test]
fn child_array_paths() {
    let store = sync_store();

    // At root, there are no array paths (only groups)
    let group = Group::open(store.clone(), "/").unwrap();
    let paths = group.child_array_paths().unwrap();
    assert!(paths.is_empty());

    // In /a, there are two array paths: baz and foo
    let group = Group::open(store.clone(), "/a").unwrap();
    let paths = group.child_array_paths().unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a/baz", "/a/foo"]);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_child_arrays() {
    let store = async_store().await;

    // Two arrays in /a
    let group = Group::async_open(store.clone(), "/a").await.unwrap();
    let arrays = group.async_child_arrays().await.unwrap();
    let array_paths: Vec<_> = arrays.iter().map(|a| a.path().as_str()).collect();
    assert_eq!(array_paths, ["/a/baz", "/a/foo"]);

    // At root, there are no arrays
    let group = Group::async_open(store.clone(), "/").await.unwrap();
    let arrays = group.async_child_arrays().await.unwrap();
    assert!(arrays.is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_child_groups() {
    let store = async_store().await;

    // At root, there are two groups: a and b
    let group = Group::async_open(store.clone(), "/").await.unwrap();
    let groups = group.async_child_groups().await.unwrap();
    let group_paths: Vec<_> = groups.iter().map(|g| g.path().as_str()).collect();
    assert_eq!(group_paths, ["/a", "/b"]);

    // In /a, there are no child groups (only arrays)
    let group = Group::async_open(store.clone(), "/a").await.unwrap();
    let groups = group.async_child_groups().await.unwrap();
    assert!(groups.is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_child_paths() {
    let store = async_store().await;

    // At root, there are two child paths: a and b (both groups)
    let group = Group::async_open(store.clone(), "/").await.unwrap();
    let paths = group.async_child_paths().await.unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a", "/b"]);

    // In /a, there are two child paths: baz and foo (both arrays)
    let group = Group::async_open(store.clone(), "/a").await.unwrap();
    let paths = group.async_child_paths().await.unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a/baz", "/a/foo"]);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_child_group_paths() {
    let store = async_store().await;

    // At root, there are two group paths: a and b
    let group = Group::async_open(store.clone(), "/").await.unwrap();
    let paths = group.async_child_group_paths().await.unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a", "/b"]);

    // In /a, there are no child group paths (only arrays)
    let group = Group::async_open(store.clone(), "/a").await.unwrap();
    let paths = group.async_child_group_paths().await.unwrap();
    assert!(paths.is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_child_array_paths() {
    let store = async_store().await;

    // At root, there are no array paths (only groups)
    let group = Group::async_open(store.clone(), "/").await.unwrap();
    let paths = group.async_child_array_paths().await.unwrap();
    assert!(paths.is_empty());

    // In /a, there are two array paths: baz and foo
    let group = Group::async_open(store.clone(), "/a").await.unwrap();
    let paths = group.async_child_array_paths().await.unwrap();
    let path_strings: Vec<_> = paths
        .iter()
        .map(zarrs::hierarchy::NodePath::as_str)
        .collect();
    assert_eq!(path_strings, ["/a/baz", "/a/foo"]);
}

mod consolidated_open {
    use std::sync::Arc;

    use serial_test::serial;

    use zarrs::config::{UseConsolidatedMetadata, global_config_mut};
    use zarrs::hierarchy::{Hierarchy, NodePath};
    use zarrs::metadata::NodeMetadata;
    use zarrs::metadata_ext::group::consolidated_metadata::ConsolidatedMetadata;
    use zarrs::node::Node;
    use zarrs_storage::store::MemoryStore;
    use zarrs_storage::{StoreKey, WritableStorageTraits};

    /// Build a v3 root group with a `consolidated_metadata` block that lists a
    /// phantom child array `phantom`. The child is *not* written to storage.
    /// Opening with consolidated metadata enabled must surface `phantom`;
    /// disabling it must not.
    fn build_store_with_phantom() -> Arc<MemoryStore> {
        let store = Arc::new(MemoryStore::new());

        // Create root + child via the builder, then attach a "phantom" child
        // through consolidated metadata only.
        let root_builder = zarrs::group::GroupBuilder::default();
        let root = root_builder.build(store.clone(), "/").unwrap();

        let real_child = zarrs::array::ArrayBuilder::new(
            vec![10],
            vec![5],
            zarrs::array::data_type::float32(),
            0.0f32,
        )
        .build(store.clone(), "/real")
        .unwrap();
        real_child.store_metadata().unwrap();

        // Phantom child: only present in consolidated metadata.
        let phantom_metadata: NodeMetadata = serde_json::from_str(
            r#"{
                "zarr_format": 3,
                "node_type": "array",
                "shape": [42],
                "data_type": "float32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [42]}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
            }"#,
        )
        .unwrap();

        let mut consolidated = ConsolidatedMetadata::default();
        consolidated
            .metadata
            .insert("real".to_string(), NodeMetadata::Array(real_child.metadata().clone()));
        consolidated
            .metadata
            .insert("phantom".to_string(), phantom_metadata);

        let mut root = root;
        root.set_consolidated_metadata(Some(consolidated));
        // Manually overwrite with the populated metadata. (`store_metadata`
        // would re-serialize from the in-memory metadata, which now includes
        // the consolidated_metadata field.)
        let serialized = serde_json::to_vec(root.metadata()).unwrap();
        store
            .set(&StoreKey::new("zarr.json").unwrap(), serialized.into())
            .unwrap();

        store
    }

    fn reset_config() {
        global_config_mut().set_use_consolidated_metadata(UseConsolidatedMetadata::default());
    }

    #[test]
    #[serial]
    fn auto_uses_consolidated() {
        reset_config();
        let store = build_store_with_phantom();

        // Default policy is Auto: phantom child is reported via consolidated metadata.
        let h = Hierarchy::open(&store, "/").unwrap();
        let tree = h.tree();
        assert!(tree.contains("phantom"), "tree should include phantom: {tree}");
        assert!(tree.contains("real"), "tree should include real: {tree}");

        let node = Node::open(&store, "/").unwrap();
        let names: Vec<_> = node.children().iter().map(|c| c.name().to_string()).collect();
        assert!(names.contains(&"phantom".to_string()), "names: {names:?}");
        assert!(names.contains(&"real".to_string()), "names: {names:?}");

        reset_config();
    }

    #[test]
    #[serial]
    fn never_ignores_consolidated() {
        reset_config();
        let store = build_store_with_phantom();
        global_config_mut().set_use_consolidated_metadata(UseConsolidatedMetadata::Never);

        // With Never, only the actually-stored child is discovered.
        let h = Hierarchy::open(&store, "/").unwrap();
        let tree = h.tree();
        assert!(!tree.contains("phantom"), "tree should not include phantom: {tree}");
        assert!(tree.contains("real"), "tree should include real: {tree}");

        let node = Node::open(&store, "/").unwrap();
        let names: Vec<_> = node.children().iter().map(|c| c.name().to_string()).collect();
        assert!(!names.contains(&"phantom".to_string()), "names: {names:?}");
        assert!(names.contains(&"real".to_string()), "names: {names:?}");

        reset_config();
    }

    #[test]
    #[serial]
    fn must_errors_when_absent() {
        reset_config();

        // Build a store with a v3 group but NO consolidated_metadata.
        let store: Arc<MemoryStore> = Arc::new(MemoryStore::new());
        let root = zarrs::group::GroupBuilder::default()
            .build(store.clone(), "/")
            .unwrap();
        root.store_metadata().unwrap();

        global_config_mut().set_use_consolidated_metadata(UseConsolidatedMetadata::Must);

        let err = Hierarchy::open(&store, "/").expect_err("should fail under Must");
        assert!(
            err.to_string().contains("Consolidated metadata required but missing"),
            "unexpected error: {err}"
        );

        let err = Node::open(&store, "/").expect_err("should fail under Must");
        assert!(
            err.to_string().contains("Consolidated metadata required but missing"),
            "unexpected error: {err}"
        );

        reset_config();
    }

    #[test]
    #[serial]
    fn auto_falls_back_when_absent() {
        reset_config();
        // No consolidated metadata; Auto should fall back to listing.
        let store: Arc<MemoryStore> = Arc::new(MemoryStore::new());
        let root = zarrs::group::GroupBuilder::default()
            .build(store.clone(), "/")
            .unwrap();
        let arr = zarrs::array::ArrayBuilder::new(
            vec![10],
            vec![5],
            zarrs::array::data_type::float32(),
            0.0f32,
        )
        .build(store.clone(), "/only")
        .unwrap();
        root.store_metadata().unwrap();
        arr.store_metadata().unwrap();

        let h = Hierarchy::open(&store, "/").unwrap();
        assert!(h.tree().contains("only"));

        reset_config();
    }

    #[test]
    #[serial]
    fn nested_consolidated_tree() {
        reset_config();

        // /
        //   sub/        (group, only in consolidated_metadata)
        //     leaf      (array, only in consolidated_metadata)
        let store: Arc<MemoryStore> = Arc::new(MemoryStore::new());
        let root = zarrs::group::GroupBuilder::default()
            .build(store.clone(), "/")
            .unwrap();

        let sub_group_md: NodeMetadata = serde_json::from_str(
            r#"{"zarr_format": 3, "node_type": "group"}"#,
        )
        .unwrap();
        let leaf_array_md: NodeMetadata = serde_json::from_str(
            r#"{
                "zarr_format": 3,
                "node_type": "array",
                "shape": [4],
                "data_type": "int32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
            }"#,
        )
        .unwrap();

        let mut consolidated = ConsolidatedMetadata::default();
        consolidated.metadata.insert("sub".to_string(), sub_group_md);
        consolidated
            .metadata
            .insert("sub/leaf".to_string(), leaf_array_md);

        let mut root = root;
        root.set_consolidated_metadata(Some(consolidated));
        let serialized = serde_json::to_vec(root.metadata()).unwrap();
        store
            .set(&StoreKey::new("zarr.json").unwrap(), serialized.into())
            .unwrap();

        // Hierarchy: flat map should contain both /sub and /sub/leaf.
        let h = Hierarchy::open(&store, "/").unwrap();
        let tree = h.tree();
        assert!(tree.contains("sub"), "{tree}");
        assert!(tree.contains("leaf"), "{tree}");

        // Node: tree should be nested correctly.
        let node = Node::open(&store, "/").unwrap();
        assert_eq!(node.children().len(), 1);
        let sub = &node.children()[0];
        assert_eq!(sub.path(), &NodePath::try_from("/sub").unwrap());
        assert_eq!(sub.children().len(), 1);
        assert_eq!(
            sub.children()[0].path(),
            &NodePath::try_from("/sub/leaf").unwrap()
        );

        reset_config();
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[serial]
    async fn async_auto_uses_consolidated() {
        use object_store::memory::InMemory;
        use zarrs_object_store::AsyncObjectStore;
        use zarrs_storage::AsyncWritableStorageTraits;

        reset_config();

        let store = Arc::new(AsyncObjectStore::new(InMemory::new()));

        // Manually serialize a root v3 group with a phantom child via consolidated metadata.
        let phantom_md: NodeMetadata = serde_json::from_str(
            r#"{
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3],
                "data_type": "float32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
            }"#,
        )
        .unwrap();
        let mut consolidated = ConsolidatedMetadata::default();
        consolidated
            .metadata
            .insert("phantom".to_string(), phantom_md);

        let root_md = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "consolidated_metadata": serde_json::to_value(&consolidated).unwrap(),
        });

        store
            .set(
                &StoreKey::new("zarr.json").unwrap(),
                serde_json::to_vec(&root_md).unwrap().into(),
            )
            .await
            .unwrap();

        let h = Hierarchy::async_open(&store, "/").await.unwrap();
        assert!(h.tree().contains("phantom"));

        let node = Node::async_open(store.clone(), "/").await.unwrap();
        let names: Vec<_> = node.children().iter().map(|c| c.name().to_string()).collect();
        assert!(names.contains(&"phantom".to_string()), "names: {names:?}");

        reset_config();
    }
}

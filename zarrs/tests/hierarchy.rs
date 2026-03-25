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

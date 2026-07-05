#![cfg(feature = "url-pipeline")]
#![allow(missing_docs)]

use std::error::Error;

use zarrs_storage::{ReadableStorageTraits, StoreKey, WritableStorageTraits};

// Referencing `FilesystemStore` ensures this crate (and therefore its `inventory::submit!`
// `file:` scheme registration) is actually linked into the test binary. Cargo/rustc may
// otherwise drop an rlib's contents entirely if nothing in the binary names any of its items.
#[allow(dead_code)]
fn _ensure_zarrs_filesystem_linked(_: zarrs_filesystem::FilesystemStore) {}

#[test]
fn file_scheme_resolves_to_filesystem_store() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::TempDir::new()?;
    let url = format!("file://{}", dir.path().to_str().unwrap());

    let storage = zarrs_url_pipeline::try_resolve_readable_writable(&url)?;

    let key = StoreKey::new("a/b.txt")?;
    storage.set(&key, zarrs_storage::Bytes::from_static(b"hello world"))?;
    assert_eq!(
        storage.get(&key)?,
        Some(zarrs_storage::Bytes::from_static(b"hello world"))
    );

    // The data should really have landed on disk at the expected path.
    let expected_path = dir.path().join("a/b.txt");
    assert_eq!(std::fs::read(expected_path)?, b"hello world");

    Ok(())
}

#[test]
fn unknown_scheme_still_errors_with_filesystem_registered() {
    assert!(zarrs_url_pipeline::try_resolve_readable("s3://bucket/key").is_err());
}

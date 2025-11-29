use super::NodePath;
use crate::storage::StoreKey;

/// Return the metadata key given a node path for a specified metadata file name (e.g. zarr.json, .zarray, .zgroup, .zaatrs).
#[must_use]
fn meta_key_any(path: &NodePath, metadata_file_name: &str) -> StoreKey {
    let path = path.as_str();
    if path.eq("/") {
        unsafe { StoreKey::new_unchecked(metadata_file_name.to_string()) }
    } else {
        let path = path.strip_prefix('/').unwrap_or(path);
        unsafe { StoreKey::new_unchecked(format!("{path}/{metadata_file_name}")) }
    }
}

/// Return the Zarr V3 metadata key (zarr.json) given a node path.
// #[deprecated = "use meta_key_v3 for zarr V3 data"]
#[must_use]
pub fn meta_key(path: &NodePath) -> StoreKey {
    meta_key_v3(path)
}

/// Return the Zarr V3 metadata key (zarr.json) given a node path.
#[must_use]
pub fn meta_key_v3(path: &NodePath) -> StoreKey {
    meta_key_any(path, "zarr.json")
}

/// Return the Zarr V2 array metadata key (.zarray) given a node path.
#[must_use]
pub fn meta_key_v2_array(path: &NodePath) -> StoreKey {
    meta_key_any(path, ".zarray")
}

/// Return the Zarr V2 group metadata key (.zgroup) given a node path.
#[must_use]
pub fn meta_key_v2_group(path: &NodePath) -> StoreKey {
    meta_key_any(path, ".zgroup")
}

/// Return the Zarr V2 user-defined attributes key (.zattrs) given a node path.
#[must_use]
pub fn meta_key_v2_attributes(path: &NodePath) -> StoreKey {
    meta_key_any(path, ".zattrs")
}

/// Return the data key given a node path and a `chunk_key` of an array.
///
/// A chunk key is computed with the `encode` method of a chunk key encoder.
#[must_use]
pub fn data_key(path: &NodePath, chunk_key: &StoreKey) -> StoreKey {
    let path = path.as_str();
    let path = path.strip_prefix('/').unwrap_or(path);
    let key_path = if path.is_empty() {
        chunk_key.as_str().to_string()
    } else {
        format!("{}/{}", path, chunk_key.as_str())
    };
    unsafe { StoreKey::new_unchecked(key_path) }
}

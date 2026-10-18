//! Zarr chunk key encodings. Includes a [default](default::DefaultChunkKeyEncoding) and [v2](v2::V2ChunkKeyEncoding) implementation.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/index.html>.
//!
#![doc = include_str!("../../doc/status/chunk_key_encodings.md")]

pub mod default;
pub mod default_suffix;
pub mod v2;

pub use default::*;
pub use default_suffix::*;
pub use v2::*;

use zarrs_chunk_key_encoding::ChunkKeyEncodingError;

/// Re-export the `zarrs_chunk_key_encoding` API.
///
/// The API is mostly useful to implementors of custom chunk key encodings.
/// Users can import less used types (e.g. errors) from this crate if needed.
pub use zarrs_chunk_key_encoding as api;

/// The store key of the chunk at `chunk_indices` of the array at `path`, encoded with `encoding` in one allocation.
///
/// # Errors
/// Returns a [`ChunkKeyEncodingError`] if `chunk_indices` cannot be encoded.
pub(crate) fn chunk_store_key_path(
    path: &crate::node::NodePath,
    encoding: &zarrs_chunk_key_encoding::ChunkKeyEncoding,
    chunk_indices: &[u64],
) -> Result<String, ChunkKeyEncodingError> {
    let path = path.as_str();
    let path = path.strip_prefix('/').unwrap_or(path);
    let mut key =
        String::with_capacity(path.len() + 1 + chunk_key_capacity("c", chunk_indices, ""));
    if !path.is_empty() {
        key.push_str(path);
        key.push('/');
    }
    encoding.encode_into(chunk_indices, &mut key)?;
    Ok(key)
}

/// Join `prefix` (if not empty) and `chunk_grid_indices` with `separator`, followed by `suffix`, in one allocation.
fn join_chunk_key(
    prefix: &str,
    chunk_grid_indices: &[u64],
    separator: zarrs_metadata::ChunkKeySeparator,
    suffix: &str,
) -> String {
    let mut key = String::with_capacity(chunk_key_capacity(prefix, chunk_grid_indices, suffix));
    push_chunk_key(&mut key, prefix, chunk_grid_indices, separator, suffix);
    key
}

/// The capacity of a chunk key of `prefix`, `chunk_grid_indices` and `suffix`, with indices of a few digits.
fn chunk_key_capacity(prefix: &str, chunk_grid_indices: &[u64], suffix: &str) -> usize {
    prefix.len() + chunk_grid_indices.len() * 4 + suffix.len()
}

/// Append `prefix` (if not empty) and `chunk_grid_indices` joined with `separator`, followed by `suffix`, to `key`.
fn push_chunk_key(
    key: &mut String,
    prefix: &str,
    chunk_grid_indices: &[u64],
    separator: zarrs_metadata::ChunkKeySeparator,
    suffix: &str,
) {
    let separator: char = separator.into();
    key.push_str(prefix);
    let mut buffer = itoa::Buffer::new();
    for (i, &index) in chunk_grid_indices.iter().enumerate() {
        if i > 0 || !prefix.is_empty() {
            key.push(separator);
        }
        key.push_str(buffer.format(index));
    }
    key.push_str(suffix);
}

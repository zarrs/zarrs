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

/// Re-export the `zarrs_chunk_key_encoding` API.
///
/// The API is mostly useful to implementors of custom chunk key encodings.
/// Users can import less used types (e.g. errors) from this crate if needed.
pub use zarrs_chunk_key_encoding as api;

/// Join `prefix` (if not empty) and `chunk_grid_indices` with `separator`, followed by `suffix`, in one allocation.
fn join_chunk_key(
    prefix: &str,
    chunk_grid_indices: &[u64],
    separator: zarrs_metadata::ChunkKeySeparator,
    suffix: &str,
) -> String {
    let separator: char = separator.into();
    // Indices typically have a few digits
    let mut key = String::with_capacity(prefix.len() + chunk_grid_indices.len() * 4 + suffix.len());
    key.push_str(prefix);
    let mut buffer = itoa::Buffer::new();
    for (i, &index) in chunk_grid_indices.iter().enumerate() {
        if i > 0 || !prefix.is_empty() {
            key.push(separator);
        }
        key.push_str(buffer.format(index));
    }
    key.push_str(suffix);
    key
}

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

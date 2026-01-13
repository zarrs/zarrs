//! Zarr chunk grids.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-grids>.
//!
//! A [`ChunkGrid`](zarrs_chunk_grid::ChunkGrid) is an [`Arc`](std::sync::Arc) wrapped [`ChunkGridTraits`](zarrs_chunk_grid::ChunkGridTraits) implementation.
//! Chunk grids are Zarr extension points and they can be registered through [`inventory`] as a [`ChunkGridPlugin`](zarrs_chunk_grid::ChunkGridPlugin).
//!
#![doc = include_str!("../../doc/status/chunk_grids.md")]

pub mod rectangular;
pub mod regular;
pub mod regular_bounded;

pub use rectangular::*;
pub use regular::*;
pub use regular_bounded::*;

/// Re-export the `zarrs_chunk_grid` API.
///
/// The API is mostly useful to implementors of custom chunk grids.
/// Users can import less used types (e.g. errors) from this crate if needed.
pub use zarrs_chunk_grid as api;

#[cfg(test)]
mod tests {
    use crate::metadata::v3::MetadataV3;
    use zarrs_chunk_grid::ChunkGrid;

    #[test]
    fn chunk_grid_configuration_regular() {
        let json = r#"
    {
        "name": "regular",
        "configuration": {
            "chunk_shape": [5, 20, 400]
        }
    }"#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        ChunkGrid::from_metadata(&metadata, &[400, 400, 400]).unwrap();
    }

    #[test]
    fn chunk_grid_configuration_rectangular() {
        let json = r#"
    {
        "name": "rectangular",
        "configuration": {
            "chunk_shape": [[5, 5, 5, 15, 15, 20, 35], 10]
        }
    }"#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        ChunkGrid::from_metadata(&metadata, &[100, 100]).unwrap();
    }
}

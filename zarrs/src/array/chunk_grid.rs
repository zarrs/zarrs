//! Zarr chunk grids.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-grids>.
//!
//! A [`ChunkGrid`] is an [`Arc`] wrapped [`ChunkGridTraits`] implementation.
//! Chunk grids are Zarr extension points and they can be registered through [`inventory`] as a [`ChunkGridPlugin`].
//!
#![doc = include_str!("../../doc/status/chunk_grids.md")]

pub mod rectangular;
pub mod regular;
pub mod regular_bounded;

use std::sync::Arc;

pub use rectangular::*;
pub use regular::*;
pub use regular_bounded::*;

use derive_more::{Deref, From};
use zarrs_plugin::PluginUnsupportedError;
use zarrs_storage::{MaybeSend, MaybeSync};

use crate::array_subset::iterators::{IndicesIntoIterator, ParIndicesIntoIterator};
use crate::{
    array_subset::{ArraySubset, IncompatibleDimensionalityError},
    metadata::v3::MetadataV3,
    plugin::{Plugin, PluginCreateError},
};

use super::{ArrayIndices, ArrayShape, ChunkShape};

/// A chunk grid implementing [`ChunkGridTraits`].
#[derive(Debug, Clone, Deref, From)]
pub struct ChunkGrid(Arc<dyn ChunkGridTraits>);

/// A chunk grid plugin.
#[derive(derive_more::Deref)]
pub struct ChunkGridPlugin(Plugin<ChunkGrid, (MetadataV3, ArrayShape)>);
inventory::collect!(ChunkGridPlugin);

impl ChunkGridPlugin {
    /// Create a new [`ChunkGridPlugin`].
    pub const fn new(
        identifier: &'static str,
        match_name_fn: fn(name: &str) -> bool,
        create_fn: fn(
            metadata_and_array_shape: &(MetadataV3, ArrayShape),
        ) -> Result<ChunkGrid, PluginCreateError>,
    ) -> Self {
        Self(Plugin::new(identifier, match_name_fn, create_fn))
    }
}

impl ChunkGrid {
    /// Create a chunk grid.
    pub fn new<T: ChunkGridTraits + 'static>(chunk_grid: T) -> Self {
        let chunk_grid: Arc<dyn ChunkGridTraits> = Arc::new(chunk_grid);
        chunk_grid.into()
    }

    /// Create a chunk grid from metadata and an array shape.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if the metadata is invalid or not associated with a registered chunk grid plugin.
    pub fn from_metadata(
        metadata: &MetadataV3,
        array_shape: &[u64],
    ) -> Result<Self, PluginCreateError> {
        for plugin in inventory::iter::<ChunkGridPlugin> {
            if plugin.match_name(metadata.name()) {
                return plugin.create(&(metadata.clone(), array_shape.to_vec()));
            }
        }
        #[cfg(miri)]
        {
            // Inventory does not work in miri, so manually handle all known chunk grids
            match metadata.name() {
                chunk_grid::REGULAR => {
                    return regular::create_chunk_grid_regular(metadata);
                }
                chunk_grid::RECTANGULAR => {
                    return rectangular::create_chunk_grid_rectangular(metadata);
                }
                _ => {}
            }
        }
        Err(
            PluginUnsupportedError::new(metadata.name().to_string(), "chunk grid".to_string())
                .into(),
        )
    }
}

/// Chunk grid traits.
///
/// # Safety
/// - Chunks must be disjoint.
/// - Methods must check the dimensionality of arguments and returned indices/shapes/subsets must match the chunk grid dimensionality.
pub unsafe trait ChunkGridTraits: core::fmt::Debug + MaybeSend + MaybeSync {
    /// Create the metadata for the chunk grid.
    fn create_metadata(&self) -> MetadataV3;

    /// The dimensionality of the chunk grid.
    fn dimensionality(&self) -> usize;

    /// The array shape (i.e. number of elements).
    ///
    /// If supported by the chunk grid, zero sized dimensions are considered "unlimited".
    fn array_shape(&self) -> &ArrayShape;

    /// The grid shape (i.e. number of chunks).
    ///
    /// If supported by the chunk grid, the grid will have zero sized dimensions where the array shape is zero, which is considered "unlimited".
    fn grid_shape(&self) -> &ArrayShape;

    /// The shape of the chunk at `chunk_indices`.
    ///
    /// Returns [`None`] if the shape of the chunk at `chunk_indices` cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_indices` do not match the dimensionality of the chunk grid.
    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError>;

    /// The shape of the chunk at `chunk_indices` as an [`ArrayShape`] ([`Vec<u64>`]).
    ///
    /// Returns [`None`] if the shape of the chunk at `chunk_indices` cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_indices` do not match the dimensionality of the chunk grid.
    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError>;

    /// The origin of the chunk at `chunk_indices`.
    ///
    /// Returns [`None`] if the chunk origin cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the length of `chunk_indices` do not match the dimensionality of the chunk grid.
    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError>;

    /// Return the [`ArraySubset`] of the chunk at `chunk_indices`.
    ///
    /// Returns [`None`] if the chunk subset cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_indices` do not match the dimensionality of the chunk grid.
    fn subset(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        let chunk_origin = self.chunk_origin(chunk_indices)?;
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        if let (Some(chunk_origin), Some(chunk_shape)) = (chunk_origin, chunk_shape) {
            let ranges = chunk_origin
                .into_iter()
                .zip(chunk_shape)
                .map(|(o, s)| o..(o + s.get()));
            Ok(Some(ArraySubset::from(ranges)))
        } else {
            Ok(None)
        }
    }

    /// Return the [`ArraySubset`] of the chunks in `chunks`.
    ///
    /// Returns [`None`] if the chunk subset cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunks` do not match the dimensionality of the chunk grid.
    fn chunks_subset(
        &self,
        chunks: &ArraySubset,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        if chunks.dimensionality() != self.dimensionality() {
            Err(IncompatibleDimensionalityError::new(
                chunks.dimensionality(),
                self.dimensionality(),
            ))
        } else if let Some(end) = chunks.end_inc() {
            let start = chunks.start();
            let chunk0 = self.subset(start)?;
            let chunk1 = self.subset(&end)?;
            if let (Some(chunk0), Some(chunk1)) = (chunk0, chunk1) {
                let start = chunk0.start().to_vec();
                let shape = std::iter::zip(&start, chunk1.end_exc())
                    .map(|(&s, e)| e.saturating_sub(s))
                    .collect();
                Ok(Some(ArraySubset::new_with_start_shape(start, shape)?))
            } else {
                Ok(None)
            }
        } else {
            Ok(Some(ArraySubset::new_empty(chunks.dimensionality())))
        }
    }

    /// The indices of a chunk which has the element at `array_indices`.
    ///
    /// Returns [`None`] if the chunk indices cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `array_indices` do not match the dimensionality of the chunk grid.
    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError>;

    /// The indices within the chunk of the element at `array_indices`.
    ///
    /// Returns [`None`] if the chunk element indices cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `array_indices` do not match the dimensionality of the chunk grid.
    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError>;

    /// Check if array indices are in-bounds.
    ///
    /// Ensures array indices are within the array shape.
    /// Zero sized array dimensions are considered "unlimited" and always in-bounds.
    #[must_use]
    fn array_indices_inbounds(&self, array_indices: &[u64]) -> bool {
        array_indices.len() == self.dimensionality()
            && std::iter::zip(array_indices, self.array_shape())
                .all(|(&index, &shape)| shape == 0 || index < shape)
    }

    /// Check if chunk indices are in-bounds.
    ///
    /// Ensures chunk grid indices are within the chunk grid shape.
    /// Zero sized array dimensions are considered "unlimited" and always in-bounds.
    #[must_use]
    fn chunk_indices_inbounds(&self, chunk_indices: &[u64]) -> bool {
        chunk_indices.len() == self.dimensionality()
            && std::iter::zip(chunk_indices, self.grid_shape())
                .all(|(&index, &shape)| shape == 0 || index < shape)
    }

    /// Return an array subset indicating the chunks intersecting `array_subset`.
    ///
    /// Returns [`None`] if the intersecting chunks cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the array subset has an incorrect dimensionality.
    fn chunks_in_array_subset(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        match array_subset.end_inc() {
            Some(end) => {
                let chunks_start = self.chunk_indices(array_subset.start())?;
                let chunks_end = self.chunk_indices(&end)?;
                // .unwrap_or_else(|| self.grid_shape());

                Ok(
                    if let (Some(chunks_start), Some(chunks_end)) = (chunks_start, chunks_end) {
                        let shape = std::iter::zip(&chunks_start, chunks_end)
                            .map(|(&s, e)| e.saturating_sub(s) + 1)
                            .collect();
                        Some(ArraySubset::new_with_start_shape(chunks_start, shape)?)
                    } else {
                        None
                    },
                )
            }
            None => Ok(Some(ArraySubset::new_empty(self.dimensionality()))),
        }
    }

    /// Return a serial iterator over the chunk indices of the chunk grid.
    fn iter_chunk_indices(&self) -> IndicesIntoIterator {
        let shape = self.grid_shape().clone();
        let n_chunks = shape.iter().product::<u64>();
        let n_chunks = usize::try_from(n_chunks).unwrap();
        IndicesIntoIterator {
            subset: ArraySubset::new_with_shape(shape),
            range: 0..n_chunks,
        }
    }

    /// Return a parallel iterator over the chunk indices of the chunk grid.
    fn par_iter_chunk_indices(&self) -> ParIndicesIntoIterator {
        let shape = self.grid_shape().clone();
        let n_chunks = shape.iter().product::<u64>();
        let n_chunks = usize::try_from(n_chunks).unwrap();
        ParIndicesIntoIterator {
            subset: ArraySubset::new_with_shape(shape),
            range: 0..n_chunks,
        }
    }
}

#[allow(dead_code)]
pub(crate) trait ChunkGridTraitsIterators: ChunkGridTraits {
    /// Return a serial iterator over the chunk subsets of the chunk grid.
    fn iter_chunk_subsets(&self) -> Box<dyn Iterator<Item = ArraySubset> + '_> {
        Box::new(self.iter_chunk_indices().map(|chunk_indices| {
            self.subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("inbounds chunk")
        }))
    }

    /// Return a serial iterator over the chunk indices and subsets of the chunk grid.
    fn iter_chunk_indices_and_subsets(
        &self,
    ) -> Box<dyn Iterator<Item = (ArrayIndices, ArraySubset)> + '_> {
        Box::new(self.iter_chunk_indices().map(|chunk_indices| {
            let chunk_subset = self
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("inbounds chunk");
            (chunk_indices, chunk_subset)
        }))
    }
}

impl<T> ChunkGridTraitsIterators for T where T: ChunkGridTraits {}

#[cfg(test)]
mod tests {
    use super::*;

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

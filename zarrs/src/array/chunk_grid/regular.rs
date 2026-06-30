//! The `regular` chunk grid.
//!
//! # Compatible Implementations
//! - All Zarr V3 implementations
//!
//! ### Specification
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/chunk-grids/regular-grid/index.html>
//!
//! ### Chunk grid `name` Aliases (Zarr V3)
//! - `regular`
//!
//! ### Chunk grid `configuration` Example - [`RegularChunkGridConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!   "chunk_shape": [100, 100]
//! }
//! # "#;
//! # use zarrs::metadata_ext::chunk_grid::regular::RegularChunkGridConfiguration;
//! # let configuration: RegularChunkGridConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

use std::num::NonZeroU64;

use thiserror::Error;

use crate::array::{
    ArrayIndices, ArrayShape, ArraySubset, ChunkShape, IncompatibleDimensionError,
    IncompatibleDimensionalityError,
};
use zarrs_chunk_grid::{ChunkGrid, ChunkGridCreateError, ChunkGridPlugin, ChunkGridTraits};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{ChunkShapeNonEmpty, Configuration};
pub use zarrs_metadata_ext::chunk_grid::regular::RegularChunkGridConfiguration;

zarrs_plugin::impl_extension_aliases!(RegularChunkGrid, v3: "regular");

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new::<RegularChunkGrid>()
}

/// A `regular` chunk grid.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct RegularChunkGrid {
    array_shape: ArrayShape,
    grid_shape: ArrayShape,
    chunk_shape: ChunkShapeNonEmpty,
}

/// A [`RegularChunkGrid`] creation error.
#[derive(Clone, Debug, Error)]
#[error("regular chunk shape: {_1:?} not compatible with array shape {_0:?}")]
pub struct RegularChunkGridCreateError(ArrayShape, ChunkShapeNonEmpty);

impl From<RegularChunkGridCreateError> for IncompatibleDimensionalityError {
    fn from(value: RegularChunkGridCreateError) -> Self {
        Self::new(value.1.len(), value.0.len())
    }
}

impl From<RegularChunkGridCreateError> for ChunkGridCreateError {
    fn from(value: RegularChunkGridCreateError) -> Self {
        IncompatibleDimensionalityError::from(value).into()
    }
}

impl RegularChunkGrid {
    /// Create a new `regular` chunk grid with chunk shape `chunk_shape`.
    ///
    /// # Errors
    /// Returns a [`RegularChunkGridCreateError`] if `chunk_shape` is not compatible with the `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunk_shape: ChunkShapeNonEmpty,
    ) -> Result<Self, RegularChunkGridCreateError> {
        if array_shape.len() != chunk_shape.len() {
            return Err(RegularChunkGridCreateError(array_shape, chunk_shape));
        }

        let grid_shape = std::iter::zip(&array_shape, chunk_shape.iter())
            .map(|(a, s)| {
                let s = s.get();
                a.div_ceil(s)
            })
            .collect();
        Ok(Self {
            array_shape,
            grid_shape,
            chunk_shape,
        })
    }

    /// Return the chunk shape.
    #[must_use]
    pub fn chunk_shape(&self) -> &[NonZeroU64] {
        self.chunk_shape.as_slice()
    }
}

unsafe impl ChunkGridTraits for RegularChunkGrid {
    fn create(
        metadata: &MetadataV3,
        array_shape: &ArrayShape,
    ) -> Result<ChunkGrid, ChunkGridCreateError> {
        let configuration: RegularChunkGridConfiguration = metadata.to_typed_configuration()?;
        let chunk_grid = RegularChunkGrid::new(array_shape.clone(), configuration.chunk_shape)?;
        Ok(ChunkGrid::new(chunk_grid))
    }

    fn configuration(&self) -> Configuration {
        RegularChunkGridConfiguration {
            chunk_shape: self.chunk_shape.clone(),
        }
        .into()
    }

    fn dimensionality(&self) -> usize {
        self.chunk_shape.len()
    }

    fn array_shape(&self) -> &[u64] {
        &self.array_shape
    }

    fn grid_shape(&self) -> &[u64] {
        &self.grid_shape
    }

    fn chunk_edge_lengths(
        &self,
        dimension: usize,
    ) -> Result<Vec<NonZeroU64>, IncompatibleDimensionError> {
        if dimension >= self.dimensionality() {
            return Err(IncompatibleDimensionError::new(
                dimension,
                self.dimensionality(),
            ));
        }
        if self.array_shape[dimension] == 0 {
            return Ok(Vec::new());
        }
        let edge_length = self.chunk_shape.as_slice()[dimension];
        Ok(vec![edge_length; self.grid_shape[dimension] as usize])
    }

    fn subset(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        if self.array_shape.contains(&0) {
            return Ok(None);
        }
        if chunk_indices.len() == self.dimensionality() {
            let ranges =
                std::iter::zip(chunk_indices, self.chunk_shape.as_slice()).map(|(i, s)| {
                    let start = i * s.get();
                    start..(start + s.get())
                });
            Ok(Some(ArraySubset::from(ranges)))
        } else {
            Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            if self.array_shape.contains(&0) {
                return Ok(None);
            }
            let chunk_shape: ChunkShape = self.chunk_shape.iter().map(|s| s.get()).collect();
            Ok(Some(chunk_shape))
        } else {
            Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if self.array_shape.contains(&0) {
            return Ok(None);
        }
        if chunk_indices.len() == self.dimensionality() {
            Ok(Some(
                std::iter::zip(chunk_indices, self.chunk_shape.as_slice())
                    .map(|(i, s)| i * s.get())
                    .collect(),
            ))
        } else {
            Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if self.array_shape.contains(&0) {
            return Ok(None);
        }
        if array_indices.len() == self.dimensionality() {
            Ok(Some(
                std::iter::zip(array_indices, self.chunk_shape.as_slice())
                    .map(|(i, s)| i / s.get())
                    .collect(),
            ))
        } else {
            Err(IncompatibleDimensionalityError::new(
                array_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if array_indices.len() == self.dimensionality() {
            if self.array_shape.contains(&0) {
                return Ok(None);
            }
            Ok(Some(
                std::iter::zip(array_indices, self.chunk_shape.as_slice())
                    .map(|(i, s)| i % s.get())
                    .collect(),
            ))
        } else {
            Err(IncompatibleDimensionalityError::new(
                array_indices.len(),
                self.dimensionality(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use rayon::iter::ParallelIterator;

    use super::*;
    use crate::array::{ArrayIndicesTinyVec, ArraySubset};
    use zarrs_chunk_grid::{ChunkGrid, ChunkGridTraitsIterators};

    #[test]
    fn chunk_grid_regular_edge_lengths() {
        let array_shape: ArrayShape = vec![10, 20];
        let chunk_shape: ChunkShapeNonEmpty =
            vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(7).unwrap()];
        let grid = RegularChunkGrid::new(array_shape, chunk_shape.clone()).unwrap();

        assert_eq!(
            grid.chunk_edge_lengths(0).unwrap(),
            vec![chunk_shape[0]; 4usize]
        );
        assert_eq!(
            grid.chunk_edge_lengths(1).unwrap(),
            vec![chunk_shape[1]; 3usize]
        );

        let unlimited = RegularChunkGrid::new(vec![0, 20], chunk_shape.clone()).unwrap();
        assert_eq!(
            unlimited.chunk_edge_lengths(0).unwrap(),
            Vec::<NonZeroU64>::new()
        );
        assert_eq!(
            unlimited.chunk_edge_lengths(1).unwrap(),
            vec![chunk_shape[1]; 3usize]
        );

        let result = grid.chunk_edge_lengths(2);
        assert!(result.is_err());
    }

    #[test]
    fn chunk_grid_regular_configuration() {
        let configuration: RegularChunkGridConfiguration =
            serde_json::from_str(r#"{"chunk_shape":[1,2,3]}"#).unwrap();
        assert_eq!(
            configuration.chunk_shape,
            vec![
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(3).unwrap()
            ]
        );
        assert_eq!(
            configuration.to_string(),
            r#"regular chunk grid {"chunk_shape":[1,2,3]}"#
        );
    }

    #[test]
    fn chunk_grid_regular_metadata() {
        let metadata: MetadataV3 =
            serde_json::from_str(r#"{"name":"regular","configuration":{"chunk_shape":[1,2,3]}}"#)
                .unwrap();
        assert!(RegularChunkGrid::create(&metadata, &vec![3, 3, 3]).is_ok());
    }

    #[test]
    fn chunk_grid_regular_metadata_invalid() {
        let metadata: MetadataV3 =
            serde_json::from_str(r#"{"name":"regular","configuration":{"invalid":[1,2,3]}}"#)
                .unwrap();
        assert!(RegularChunkGrid::create(&metadata, &vec![3, 3, 3]).is_err());
        assert_eq!(
            RegularChunkGrid::create(&metadata, &vec![3, 3, 3])
                .unwrap_err()
                .to_string(),
            r"configuration is unsupported: unknown field `invalid`, expected `chunk_shape`"
        );
    }

    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn chunk_grid_regular() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];

        {
            let chunk_grid =
                RegularChunkGrid::new(array_shape.clone(), chunk_shape.clone()).unwrap();
            assert_eq!(chunk_grid.dimensionality(), 3);
            assert_eq!(chunk_grid.chunk_shape(), chunk_shape.as_slice());
            assert_eq!(
                chunk_grid.chunk_origin(&[1, 1, 1]).unwrap(),
                Some(vec![1, 2, 3])
            );
            let chunk_grid_shape = chunk_grid.grid_shape();
            assert_eq!(chunk_grid_shape, &[5, 4, 18]);
            let array_index: ArrayIndices = vec![3, 5, 50];
            assert_eq!(
                chunk_grid.chunk_indices(&array_index).unwrap(),
                Some(vec![3, 2, 16])
            );
            assert_eq!(
                chunk_grid.chunk_element_indices(&array_index).unwrap(),
                Some(vec![0, 1, 2])
            );

            assert_eq!(
                chunk_grid.chunks_subset(&[1..3, 1..2, 5..8],).unwrap(),
                Some(ArraySubset::new_with_ranges(&[1..3, 2..4, 15..24]))
            );

            assert!(chunk_grid.chunks_subset(&[1..3]).is_err());

            assert!(
                chunk_grid
                    .chunks_subset(&[0..0, 0..0, 0..0],)
                    .unwrap()
                    .unwrap()
                    .is_empty()
            );
        }

        assert!(RegularChunkGrid::new(vec![0; 1], chunk_shape.clone()).is_err());
    }

    #[test]
    fn chunk_grid_regular_out_of_bounds() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularChunkGrid::new(array_shape, chunk_shape).unwrap();

        let array_indices: ArrayIndices = vec![3, 5, 53];
        assert_eq!(
            chunk_grid.chunk_indices(&array_indices).unwrap(),
            Some(vec![3, 2, 17])
        );

        let chunk_indices: ArrayShape = vec![6, 1, 1];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert_eq!(
            chunk_grid.chunk_origin(&chunk_indices).unwrap(),
            Some(vec![6, 2, 3])
        );
    }

    #[test]
    fn chunk_grid_regular_fully_and_partially_out_of_bounds() {
        // Array [10, 12], chunks [3, 5] -> grid [4, 3]
        let array_shape: ArrayShape = vec![10, 12];
        let chunk_shape: ChunkShapeNonEmpty =
            vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(5).unwrap()];
        let chunk_grid = RegularChunkGrid::new(array_shape.clone(), chunk_shape.clone()).unwrap();
        let chunk_grid: ChunkGrid = chunk_grid.into();
        let chunk_shape: ChunkShape = chunk_shape.iter().map(|s| s.get()).collect();

        assert_eq!(chunk_grid.grid_shape(), &[4, 3]);

        // Trait-level: chunk indices within grid shape always return Some,
        // even for edge chunks that extend past the array boundary.

        // Edge chunk [3, 2]: origin [9, 10], extends to [12, 15] but array is [10, 12]
        assert_eq!(chunk_grid.chunk_origin(&[3, 2]).unwrap(), Some(vec![9, 10]));
        assert_eq!(
            chunk_grid.chunk_shape(&[3, 2]).unwrap(),
            Some(chunk_shape.clone())
        );
        assert_eq!(
            chunk_grid.subset(&[3, 2]).unwrap(),
            Some(ArraySubset::new_with_ranges(&[9..12, 10..15]))
        );

        // Interior chunk [2, 1]: origin [6, 5], fully within array
        assert_eq!(chunk_grid.chunk_origin(&[2, 1]).unwrap(), Some(vec![6, 5]));
        assert_eq!(
            chunk_grid.chunk_shape(&[2, 1]).unwrap(),
            Some(chunk_shape.clone())
        );

        // Array indices at exact array boundary → map to chunk at grid boundary
        assert_eq!(
            chunk_grid.chunk_indices(&[9, 11]).unwrap(),
            Some(vec![3, 2])
        );
        assert_eq!(
            chunk_grid.chunk_element_indices(&[9, 11]).unwrap(),
            Some(vec![0, 1])
        );

        // Array indices past array boundary — regular grid does not gate on array bounds
        let past_array = vec![10u64, 12u64];
        assert_eq!(
            chunk_grid.chunk_indices(&past_array).unwrap(),
            Some(vec![3, 2])
        );
        assert_eq!(
            chunk_grid.chunk_element_indices(&past_array).unwrap(),
            Some(vec![1, 2])
        );

        // The in-bounds checks distinguish valid from invalid indices
        assert!(chunk_grid.chunk_indices_inbounds(&[3, 2]));
        assert!(!chunk_grid.chunk_indices_inbounds(&[4, 2]));
        assert!(chunk_grid.array_indices_inbounds(&[9, 11]));
        assert!(!chunk_grid.array_indices_inbounds(&[10, 12]));
    }

    #[test]
    fn chunk_grid_regular_zero_dim() {
        let array_shape: ArrayShape = vec![5, 7, 0];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularChunkGrid::new(array_shape, chunk_shape).unwrap();
        let grid: ChunkGrid = chunk_grid.into();

        // Grid shape has 0 for zero-dim
        assert_eq!(grid.grid_shape(), &[5, 4, 0]);

        // No indices are in-bounds for zero-dim
        assert!(!grid.chunk_indices_inbounds(&[3, 1, 0]));
        assert!(!grid.array_indices_inbounds(&[3, 5, 0]));

        // Query methods return None for zero-dim
        let array_indices: ArrayIndices = vec![3, 5, 0];
        assert_eq!(grid.chunk_indices(&array_indices).unwrap(), None);
        assert_eq!(grid.chunk_origin(&[0, 0, 0]).unwrap(), None);
        assert_eq!(grid.chunk_shape(&[0, 0, 0]).unwrap(), None);
        assert_eq!(grid.subset(&[0, 0, 0]).unwrap(), None);
        assert_eq!(grid.chunk_element_indices(&array_indices).unwrap(), None);
    }

    #[test]
    fn chunk_grid_regular_iterators() {
        let array_shape: ArrayShape = vec![2, 2, 6];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularChunkGrid::new(array_shape, chunk_shape).unwrap();

        let iter = chunk_grid.iter_chunk_indices();
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![
                ArrayIndicesTinyVec::Heap(vec![0, 0, 0]),
                ArrayIndicesTinyVec::Heap(vec![0, 0, 1]),
                ArrayIndicesTinyVec::Heap(vec![1, 0, 0]),
                ArrayIndicesTinyVec::Heap(vec![1, 0, 1]),
            ]
        );

        let iter = chunk_grid.par_iter_chunk_indices();
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![
                ArrayIndicesTinyVec::Heap(vec![0, 0, 0]),
                ArrayIndicesTinyVec::Heap(vec![0, 0, 1]),
                ArrayIndicesTinyVec::Heap(vec![1, 0, 0]),
                ArrayIndicesTinyVec::Heap(vec![1, 0, 1]),
            ]
        );

        let iter = chunk_grid.iter_chunk_subsets();
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![
                ArraySubset::new_with_ranges(&[0..1, 0..2, 0..3]),
                ArraySubset::new_with_ranges(&[0..1, 0..2, 3..6]),
                ArraySubset::new_with_ranges(&[1..2, 0..2, 0..3]),
                ArraySubset::new_with_ranges(&[1..2, 0..2, 3..6]),
            ]
        );

        let iter = chunk_grid.iter_chunk_indices_and_subsets();
        #[rustfmt::skip]
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![
                (ArrayIndicesTinyVec::Heap(vec![0, 0, 0]), ArraySubset::new_with_ranges(&[0..1, 0..2, 0..3])),
                (ArrayIndicesTinyVec::Heap(vec![0, 0, 1]), ArraySubset::new_with_ranges(&[0..1, 0..2, 3..6])),
                (ArrayIndicesTinyVec::Heap(vec![1, 0, 0]), ArraySubset::new_with_ranges(&[1..2, 0..2, 0..3])),
                (ArrayIndicesTinyVec::Heap(vec![1, 0, 1]), ArraySubset::new_with_ranges(&[1..2, 0..2, 3..6])),
            ]
        );
    }
}

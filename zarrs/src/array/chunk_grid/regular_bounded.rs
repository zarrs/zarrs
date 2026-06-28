//! The `regular_bounded` chunk grid.
//!
//! This chunk grid is the same as `regular`, except that chunks at the edges of the array may be smaller than the specified chunk shape.
//!
//! <div class="warning">
//! This chunk grid is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://chunkgrid.zarrs.dev/regular_bounded>
//!
//! ### Chunk Grid `name` Aliases (Zarr V3)
//! - `zarrs.regular_bounded`
//!
//! ### Chunk Grid `configuration` Example - [`RegularBoundedChunkGridConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!   "chunk_shape": [100, 100]
//! }
//! # "#;
//! # use zarrs::array::chunk_grid::RegularBoundedChunkGridConfiguration;
//! # let configuration: RegularBoundedChunkGridConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

use std::num::NonZeroU64;

use itertools::izip;

/// Configuration parameters for a `regular_bounded` chunk grid.
pub type RegularBoundedChunkGridConfiguration = super::RegularChunkGridConfiguration; // TODO: move to zarrs_metadata_ex on stabilisation

use crate::array::{
    ArrayIndices, ArrayShape, ArraySubset, ChunkShape, IncompatibleDimensionError,
    IncompatibleDimensionalityError,
};
use zarrs_chunk_grid::{ChunkGrid, ChunkGridCreateError, ChunkGridPlugin, ChunkGridTraits};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{ChunkShapeNonEmpty, Configuration};

zarrs_plugin::impl_extension_aliases!(RegularBoundedChunkGrid,
  v3: "zarrs.regular_bounded", []
);

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new::<RegularBoundedChunkGrid>()
}

/// A `regular_bounded` chunk grid.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct RegularBoundedChunkGrid {
    array_shape: ArrayShape,
    grid_shape: ArrayShape,
    chunk_shape: ChunkShapeNonEmpty,
}

impl RegularBoundedChunkGrid {
    /// Create a new `regular_bounded` chunk grid with chunk shape `chunk_shape`.
    ///
    /// # Errors
    /// Returns a [`IncompatibleDimensionalityError`] if `chunk_shape` is not compatible with the `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunk_shape: ChunkShapeNonEmpty,
    ) -> Result<Self, IncompatibleDimensionalityError> {
        if array_shape.len() != chunk_shape.len() {
            return Err(IncompatibleDimensionalityError::new(
                chunk_shape.len(),
                array_shape.len(),
            ));
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
}

unsafe impl ChunkGridTraits for RegularBoundedChunkGrid {
    fn create(
        metadata: &MetadataV3,
        array_shape: &ArrayShape,
    ) -> Result<ChunkGrid, ChunkGridCreateError> {
        crate::warn_experimental_extension(metadata.name(), "chunk grid");
        let configuration: RegularBoundedChunkGridConfiguration =
            metadata.to_typed_configuration()?;
        let chunk_grid =
            RegularBoundedChunkGrid::new(array_shape.clone(), configuration.chunk_shape)?;
        Ok(ChunkGrid::new(chunk_grid))
    }

    fn configuration(&self) -> Configuration {
        RegularBoundedChunkGridConfiguration {
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
        let chunk_size = self.chunk_shape.as_slice()[dimension];
        let n_chunks = self.grid_shape[dimension] as usize;
        let mut result = Vec::with_capacity(n_chunks);
        result.resize(n_chunks - 1, chunk_size);
        let last_edge = self.array_shape[dimension] - (n_chunks - 1) as u64 * chunk_size.get();
        result.push(
            // SAFETY: last_edge > 0 because n_chunks = ceil(array_shape / chunk_size)
            unsafe { NonZeroU64::new_unchecked(last_edge) },
        );
        Ok(result)
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(izip!(
                self.chunk_shape.as_slice(),
                &self.array_shape,
                chunk_indices
            )
            .map(|(chunk_shape, &array_shape, chunk_indices)| {
                let start = (chunk_indices * chunk_shape.get()).min(array_shape);
                let end = (start + chunk_shape.get()).min(array_shape);
                if end > start { Some(end - start) } else { None }
            })
            .collect::<Option<Vec<_>>>())
        } else {
            Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(izip!(
                self.chunk_shape.as_slice(),
                &self.array_shape,
                chunk_indices
            )
            .map(|(chunk_shape, &array_shape, chunk_indices)| {
                let start = (chunk_indices * chunk_shape.get()).min(array_shape);
                let end = (start + chunk_shape.get()).min(array_shape);
                if end > start { Some(end - start) } else { None }
            })
            .collect::<Option<Vec<_>>>())
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
        if chunk_indices.len() == self.dimensionality() {
            Ok(izip!(
                chunk_indices,
                self.chunk_shape.as_slice(),
                &self.array_shape
            )
            .map(|(chunk_index, chunk_shape, &array_shape)| {
                let start = chunk_index * chunk_shape.get();
                if start < array_shape {
                    Some(start)
                } else {
                    None
                }
            })
            .collect::<Option<Vec<_>>>())
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
        if array_indices.len() == self.dimensionality() {
            Ok(izip!(
                array_indices,
                self.chunk_shape.as_slice(),
                &self.array_shape
            )
            .map(|(array_index, chunk_shape, array_shape)| {
                if array_index < array_shape {
                    Some(array_index / chunk_shape.get())
                } else {
                    None
                }
            })
            .collect::<Option<Vec<_>>>())
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
            Ok(izip!(
                array_indices,
                self.chunk_shape.as_slice(),
                &self.array_shape
            )
            .map(|(array_index, chunk_shape, array_shape)| {
                if array_index < array_shape {
                    Some(array_index % chunk_shape.get())
                } else {
                    None
                }
            })
            .collect::<Option<Vec<_>>>())
        } else {
            Err(IncompatibleDimensionalityError::new(
                array_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    fn subset(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            let ranges = izip!(
                self.chunk_shape.as_slice(),
                &self.array_shape,
                chunk_indices
            )
            .map(|(chunk_shape, &array_shape, chunk_indices)| {
                let start = (chunk_indices * chunk_shape.get()).min(array_shape);
                let end = (start + chunk_shape.get()).min(array_shape);
                if end > start { Some(start..end) } else { None }
            })
            .collect::<Option<Vec<_>>>();
            if let Some(ranges) = ranges {
                Ok(Some(ArraySubset::new_with_ranges(&ranges)))
            } else {
                Ok(None)
            }
        } else {
            Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
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

    #[test]
    fn chunk_grid_regular_bounded_edge_lengths() {
        let array_shape: ArrayShape = vec![10, 20];
        let chunk_shape: ChunkShapeNonEmpty =
            vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(7).unwrap()];
        let grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape.clone()).unwrap();

        assert_eq!(
            grid.chunk_edge_lengths(0).unwrap(),
            vec![
                NonZeroU64::new(3).unwrap(),
                NonZeroU64::new(3).unwrap(),
                NonZeroU64::new(3).unwrap(),
                NonZeroU64::new(1).unwrap(),
            ]
        );
        assert_eq!(
            grid.chunk_edge_lengths(1).unwrap(),
            vec![
                NonZeroU64::new(7).unwrap(),
                NonZeroU64::new(7).unwrap(),
                NonZeroU64::new(6).unwrap(),
            ]
        );

        let unlimited = RegularBoundedChunkGrid::new(vec![0, 20], chunk_shape.clone()).unwrap();
        assert_eq!(
            unlimited.chunk_edge_lengths(0).unwrap(),
            Vec::<NonZeroU64>::new()
        );

        let result = grid.chunk_edge_lengths(2);
        assert!(result.is_err());
    }

    #[test]
    fn chunk_grid_regular_bounded_metadata() {
        let metadata: MetadataV3 = serde_json::from_str(
            r#"{"name":"zarrs.regular_bounded","configuration":{"chunk_shape":[1,2,3]}}"#,
        )
        .unwrap();
        assert!(RegularBoundedChunkGrid::create(&metadata, &vec![3, 3, 3]).is_ok());
    }

    #[test]
    fn chunk_grid_regular_bounded_metadata_invalid() {
        let metadata: MetadataV3 = serde_json::from_str(
            r#"{"name":"zarrs.regular_bounded","configuration":{"invalid":[1,2,3]}}"#,
        )
        .unwrap();
        assert!(RegularBoundedChunkGrid::create(&metadata, &vec![3, 3, 3]).is_err());
        assert_eq!(
            RegularBoundedChunkGrid::create(&metadata, &vec![3, 3, 3])
                .unwrap_err()
                .to_string(),
            r"configuration is unsupported: unknown field `invalid`, expected `chunk_shape`"
        );
    }

    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn chunk_grid_regular_bounded() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];

        {
            let chunk_grid =
                RegularBoundedChunkGrid::new(array_shape.clone(), chunk_shape.clone()).unwrap();
            assert_eq!(chunk_grid.dimensionality(), 3);
            assert_eq!(chunk_grid.array_shape(), &array_shape);
            assert_eq!(
                chunk_grid.chunk_origin(&[1, 1, 1]).unwrap(),
                Some(vec![1, 2, 3])
            );
            assert_eq!(
                chunk_grid.chunk_shape(&[0, 0, 0]).unwrap(),
                Some(chunk_shape.iter().map(|n| n.get()).collect())
            );
            assert_eq!(
                chunk_grid.chunk_shape_u64(&[0, 0, 0]).unwrap(),
                Some(chunk_shape.iter().map(|u| u.get()).collect())
            );
            assert_eq!(
                chunk_grid.chunk_shape(&[0, 3, 17]).unwrap(),
                Some(vec![1u64; 3])
            );
            assert_eq!(chunk_grid.chunk_shape(&[5, 0, 0]).unwrap(), None);
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

        assert!(RegularBoundedChunkGrid::new(vec![0; 1], chunk_shape.clone()).is_err());
    }

    #[test]
    fn chunk_grid_regular_bounded_out_of_bounds() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape).unwrap();

        let array_indices: ArrayIndices = vec![3, 5, 53];
        assert_eq!(chunk_grid.chunk_indices(&array_indices).unwrap(), None);

        let chunk_indices: ArrayShape = vec![6, 1, 1];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert_eq!(chunk_grid.chunk_origin(&chunk_indices).unwrap(), None);
    }

    #[test]
    fn chunk_grid_regular_bounded_fully_and_partially_out_of_bounds() {
        // Array [10, 12], chunks [3, 5] -> grid [4, 3]
        let array_shape: ArrayShape = vec![10, 12];
        let chunk_shape: ChunkShapeNonEmpty =
            vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(5).unwrap()];
        let chunk_grid =
            RegularBoundedChunkGrid::new(array_shape.clone(), chunk_shape.clone()).unwrap();

        assert_eq!(chunk_grid.grid_shape(), &[4, 3]);

        // Fully out-of-bounds: all dims past grid extent
        assert_eq!(chunk_grid.chunk_origin(&[99, 99]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_shape(&[99, 99]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_shape_u64(&[99, 99]).unwrap(), None);
        assert_eq!(chunk_grid.subset(&[99, 99]).unwrap(), None);

        // Fully out-of-bounds: one dim past grid extent
        assert_eq!(chunk_grid.chunk_origin(&[4, 0]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_origin(&[0, 3]).unwrap(), None);

        // Fully out-of-bounds array indices
        assert_eq!(chunk_grid.chunk_indices(&[10, 12]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_indices(&[999, 999]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_element_indices(&[10, 12]).unwrap(), None);

        // Partially out-of-bounds (edge) last chunk: origin valid but extends beyond array
        // Chunk [3, 2]: origin [9, 10], full extent [12, 15], array [10, 12]
        // -> reduced shape [1, 2]
        assert_eq!(chunk_grid.chunk_origin(&[3, 2]).unwrap(), Some(vec![9, 10]));
        assert_eq!(
            chunk_grid.chunk_shape(&[3, 2]).unwrap(),
            Some(vec![1u64, 2u64])
        );
        assert_eq!(
            chunk_grid.chunk_shape_u64(&[3, 2]).unwrap(),
            Some(vec![1, 2])
        );
        assert_eq!(
            chunk_grid.subset(&[3, 2]).unwrap(),
            Some(ArraySubset::new_with_ranges(&[9..10, 10..12]))
        );

        // Partially out-of-bounds edge in first dim only
        // Chunk [3, 0]: origin [9, 0], full extent [12, 5], array [10, 12]
        // -> reduced shape [1, 5]
        assert_eq!(
            chunk_grid.chunk_shape(&[3, 0]).unwrap(),
            Some(vec![1u64, 5u64])
        );

        // Partially out-of-bounds edge in second dim only
        // Chunk [0, 2]: origin [0, 10], full extent [3, 15], array [10, 12]
        // -> reduced shape [3, 2]
        assert_eq!(
            chunk_grid.chunk_shape(&[0, 2]).unwrap(),
            Some(vec![3u64, 2u64])
        );

        // In-bounds array index at array boundary -> last chunk
        assert_eq!(
            chunk_grid.chunk_indices(&[9, 11]).unwrap(),
            Some(vec![3, 2])
        );
        assert_eq!(
            chunk_grid.chunk_element_indices(&[9, 11]).unwrap(),
            Some(vec![0, 1])
        );

        // In-bounds array index just inside boundary
        assert_eq!(chunk_grid.chunk_indices(&[8, 9]).unwrap(), Some(vec![2, 1]));
    }

    #[test]
    fn chunk_grid_regular_bounded_zero_dim() {
        let array_shape: ArrayShape = vec![5, 7, 0];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape).unwrap();

        // Grid shape has 0 for zero-dim
        assert_eq!(chunk_grid.grid_shape(), &[5, 4, 0]);

        // No indices are in-bounds for zero-dim
        assert!(!chunk_grid.chunk_indices_inbounds(&[3, 1, 0]));
        assert!(!chunk_grid.array_indices_inbounds(&[3, 5, 0]));

        // Query methods return None for zero-dim
        let array_indices: ArrayIndices = vec![3, 5, 1000];
        assert_eq!(chunk_grid.chunk_indices(&array_indices).unwrap(), None);
        assert_eq!(chunk_grid.chunk_origin(&[0, 0, 0]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_shape(&[0, 0, 0]).unwrap(), None);
    }

    #[test]
    fn chunk_grid_regular_bounded_iterators() {
        let array_shape: ArrayShape = vec![2, 2, 6];
        let chunk_shape: ChunkShapeNonEmpty = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape).unwrap();

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
    }
}

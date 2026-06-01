#![allow(dead_code)]
use std::borrow::Cow;
use std::num::NonZeroU64;

use thiserror::Error;

use crate::array::{
    ArrayIndices, ArrayShape, ChunkShape, IncompatibleDimensionError,
    IncompatibleDimensionalityError,
};
use zarrs_chunk_grid::{ChunkGrid, ChunkGridTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{ExtensionName, PluginCreateError, ZarrVersion};

/// A chunk grid that repeats an inner chunk grid tile.
#[derive(Debug, Clone)]
pub(crate) struct RepeatChunkGrid {
    shape: ArrayShape,
    repeats: ArrayShape,
    array_shape: ArrayShape,
    grid_shape: ArrayShape,
    inner_chunk_grid: ChunkGrid,
}

/// A [`RepeatChunkGrid`] creation error.
#[derive(Clone, Debug, Error)]
pub(crate) enum RepeatChunkGridCreateError {
    #[error("incompatible dimensionality {got}, expected {expected}")]
    IncompatibleDimensionality { got: usize, expected: usize },
    #[error("repeat count is zero in dimension {0}")]
    ZeroRepeat(usize),
    #[error("repeat chunk grid shape overflow")]
    ShapeOverflow,
    #[error(
        "inner chunk edge lengths in dimension {dimension} have count {edge_count} and sum {edge_sum}, expected count {expected_edge_count} and sum {expected_edge_sum}"
    )]
    InnerChunkEdgeLengthsMismatch {
        dimension: usize,
        edge_count: u64,
        edge_sum: u64,
        expected_edge_count: u64,
        expected_edge_sum: u64,
    },
    #[error("dimension {0} is out of bounds, expected less than {1}")]
    IncompatibleDimension(usize, usize),
}

impl RepeatChunkGrid {
    /// Create a new repeated chunk grid.
    pub(crate) fn new(
        repeats: ArrayShape,
        inner_chunk_grid: ChunkGrid,
    ) -> Result<Self, RepeatChunkGridCreateError> {
        let shape = inner_chunk_grid.array_shape().to_vec();
        if inner_chunk_grid.dimensionality() != shape.len() {
            return Err(RepeatChunkGridCreateError::IncompatibleDimensionality {
                got: inner_chunk_grid.dimensionality(),
                expected: shape.len(),
            });
        }

        if shape.len() != repeats.len() {
            return Err(RepeatChunkGridCreateError::IncompatibleDimensionality {
                got: repeats.len(),
                expected: shape.len(),
            });
        }

        if let Some(dimension) = repeats.iter().position(|repeat| *repeat == 0) {
            return Err(RepeatChunkGridCreateError::ZeroRepeat(dimension));
        }

        let array_shape = std::iter::zip(&shape, &repeats)
            .map(|(shape, repeats)| shape.checked_mul(*repeats))
            .collect::<Option<ArrayShape>>()
            .ok_or(RepeatChunkGridCreateError::ShapeOverflow)?;

        let grid_shape = std::iter::zip(inner_chunk_grid.grid_shape(), &repeats)
            .map(|(inner_grid_shape, repeats)| inner_grid_shape.checked_mul(*repeats))
            .collect::<Option<ArrayShape>>()
            .ok_or(RepeatChunkGridCreateError::ShapeOverflow)?;

        for dimension in 0..shape.len() {
            let inner_edge_lengths =
                inner_chunk_grid
                    .chunk_edge_lengths(dimension)
                    .map_err(|_| {
                        RepeatChunkGridCreateError::IncompatibleDimension(dimension, shape.len())
                    })?;
            let inner_edge_length_count = u64::try_from(inner_edge_lengths.len())
                .map_err(|_| RepeatChunkGridCreateError::ShapeOverflow)?;
            let inner_edge_length_sum = inner_edge_lengths
                .iter()
                .try_fold(0u64, |sum, edge_length| sum.checked_add(edge_length.get()))
                .ok_or(RepeatChunkGridCreateError::ShapeOverflow)?;
            if inner_edge_length_count != inner_chunk_grid.grid_shape()[dimension]
                || inner_edge_length_sum != shape[dimension]
            {
                return Err(RepeatChunkGridCreateError::InnerChunkEdgeLengthsMismatch {
                    dimension,
                    edge_count: inner_edge_length_count,
                    edge_sum: inner_edge_length_sum,
                    expected_edge_count: inner_chunk_grid.grid_shape()[dimension],
                    expected_edge_sum: shape[dimension],
                });
            }
        }

        Ok(Self {
            shape,
            repeats,
            array_shape,
            grid_shape,
            inner_chunk_grid,
        })
    }

    fn check_dimensionality(
        &self,
        dimensionality: usize,
    ) -> Result<(), IncompatibleDimensionalityError> {
        if dimensionality == self.dimensionality() {
            Ok(())
        } else {
            Err(IncompatibleDimensionalityError::new(
                dimensionality,
                self.dimensionality(),
            ))
        }
    }

    fn has_zero_sized_dimension(&self) -> bool {
        self.array_shape.contains(&0)
    }

    fn repeat_inner_chunk_indices(
        &self,
        chunk_indices: &[u64],
    ) -> Option<(ArrayIndices, ArrayIndices)> {
        if self.has_zero_sized_dimension() || !self.chunk_indices_inbounds(chunk_indices) {
            return None;
        }

        std::iter::zip(chunk_indices, self.inner_chunk_grid.grid_shape())
            .map(|(chunk_index, inner_grid_shape)| {
                if *inner_grid_shape == 0 {
                    None
                } else {
                    Some((
                        chunk_index / inner_grid_shape,
                        chunk_index % inner_grid_shape,
                    ))
                }
            })
            .collect::<Option<Vec<_>>>()
            .map(|repeat_inner_indices| repeat_inner_indices.into_iter().unzip())
    }

    fn local_array_indices(&self, array_indices: &[u64]) -> Option<(ArrayIndices, ArrayIndices)> {
        if self.has_zero_sized_dimension() || !self.array_indices_inbounds(array_indices) {
            return None;
        }

        std::iter::zip(array_indices, &self.shape)
            .map(|(array_index, shape)| {
                if *shape == 0 {
                    None
                } else {
                    Some((array_index / shape, array_index % shape))
                }
            })
            .collect::<Option<Vec<_>>>()
            .map(|repeat_local_indices| repeat_local_indices.into_iter().unzip())
    }

    fn offset_origin(&self, repeat_indices: &[u64], inner_origin: &[u64]) -> ArrayIndices {
        itertools::izip!(repeat_indices, &self.shape, inner_origin)
            .map(|(repeat_index, shape, inner_origin)| repeat_index * shape + inner_origin)
            .collect()
    }
}

impl ExtensionName for RepeatChunkGrid {
    fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

unsafe impl ChunkGridTraits for RepeatChunkGrid {
    fn create(
        _metadata: &MetadataV3,
        _array_shape: &ArrayShape,
    ) -> Result<ChunkGrid, PluginCreateError> {
        Err(PluginCreateError::Other(
            "repeat chunk grid cannot be created from metadata".to_string(),
        ))
    }

    fn configuration(&self) -> Configuration {
        Configuration::default()
    }

    fn dimensionality(&self) -> usize {
        self.shape.len()
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

        let inner_edge_lengths = self.inner_chunk_grid.chunk_edge_lengths(dimension)?;
        let mut edge_lengths = Vec::with_capacity(
            inner_edge_lengths.len() * usize::try_from(self.repeats[dimension]).unwrap(),
        );
        for _ in 0..self.repeats[dimension] {
            edge_lengths.extend_from_slice(&inner_edge_lengths);
        }
        Ok(edge_lengths)
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        self.check_dimensionality(chunk_indices.len())?;

        let Some((_repeat_indices, inner_chunk_indices)) =
            self.repeat_inner_chunk_indices(chunk_indices)
        else {
            return Ok(None);
        };
        self.inner_chunk_grid.chunk_shape(&inner_chunk_indices)
    }

    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError> {
        self.check_dimensionality(chunk_indices.len())?;

        let Some((_repeat_indices, inner_chunk_indices)) =
            self.repeat_inner_chunk_indices(chunk_indices)
        else {
            return Ok(None);
        };
        self.inner_chunk_grid.chunk_shape_u64(&inner_chunk_indices)
    }

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.check_dimensionality(chunk_indices.len())?;

        let Some((repeat_indices, inner_chunk_indices)) =
            self.repeat_inner_chunk_indices(chunk_indices)
        else {
            return Ok(None);
        };
        let inner_origin = self.inner_chunk_grid.chunk_origin(&inner_chunk_indices)?;
        Ok(inner_origin.map(|inner_origin| self.offset_origin(&repeat_indices, &inner_origin)))
    }

    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.check_dimensionality(array_indices.len())?;

        let Some((repeat_indices, local_array_indices)) = self.local_array_indices(array_indices)
        else {
            return Ok(None);
        };
        let Some(inner_chunk_indices) =
            self.inner_chunk_grid.chunk_indices(&local_array_indices)?
        else {
            return Ok(None);
        };
        Ok(Some(
            itertools::izip!(
                repeat_indices,
                inner_chunk_indices,
                self.inner_chunk_grid.grid_shape()
            )
            .map(|(repeat_index, inner_chunk_index, inner_grid_shape)| {
                repeat_index * inner_grid_shape + inner_chunk_index
            })
            .collect(),
        ))
    }

    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.check_dimensionality(array_indices.len())?;

        let Some((_repeat_indices, local_array_indices)) = self.local_array_indices(array_indices)
        else {
            return Ok(None);
        };
        self.inner_chunk_grid
            .chunk_element_indices(&local_array_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::chunk_grid::{RegularBoundedChunkGrid, RegularChunkGrid};
    use crate::array::{ArraySubset, ChunkGridTraits};
    use zarrs_plugin::ExtensionName;

    fn regular_grid(shape: ArrayShape, chunk_shape: ChunkShape) -> ChunkGrid {
        RegularChunkGrid::new(shape, chunk_shape).unwrap().into()
    }

    fn regular_bounded_grid(shape: ArrayShape, chunk_shape: ChunkShape) -> ChunkGrid {
        RegularBoundedChunkGrid::new(shape, chunk_shape)
            .unwrap()
            .into()
    }

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    #[test]
    fn repeat_chunk_grid_regular_inner_grid() {
        let grid =
            RepeatChunkGrid::new(vec![2, 3], regular_grid(vec![6, 4], vec![nz(3), nz(2)])).unwrap();

        assert_eq!(grid.array_shape(), &[12, 12]);
        assert_eq!(grid.grid_shape(), &[4, 6]);
        assert_eq!(grid.name_v3(), None);
        assert!(grid.configuration().is_empty());
        assert_eq!(
            grid.chunk_edge_lengths(0).unwrap(),
            vec![nz(3), nz(3), nz(3), nz(3)]
        );
        assert_eq!(
            grid.chunk_edge_lengths(1).unwrap(),
            vec![nz(2), nz(2), nz(2), nz(2), nz(2), nz(2)]
        );

        assert_eq!(
            grid.subset(&[2, 3]).unwrap(),
            Some(ArraySubset::new_with_ranges(&[6..9, 6..8]))
        );
        assert_eq!(
            grid.chunks_subset(&[1..3, 2..4]).unwrap(),
            Some(ArraySubset::new_with_ranges(&[3..9, 4..8]))
        );
        assert_eq!(grid.chunk_origin(&[2, 3]).unwrap(), Some(vec![6, 6]));
        assert_eq!(grid.chunk_shape(&[2, 3]).unwrap(), Some(vec![nz(3), nz(2)]));
        assert_eq!(grid.chunk_indices(&[7, 7]).unwrap(), Some(vec![2, 3]));
        assert_eq!(
            grid.chunk_element_indices(&[7, 7]).unwrap(),
            Some(vec![1, 1])
        );
        assert_eq!(grid.chunk_indices(&[12, 0]).unwrap(), None);
    }

    #[test]
    fn repeat_chunk_grid_rejects_invalid_inputs() {
        assert!(RepeatChunkGrid::new(vec![1, 1], regular_grid(vec![6], vec![nz(3)])).is_err());
        assert!(RepeatChunkGrid::new(vec![0], regular_grid(vec![6], vec![nz(3)])).is_err());
        assert!(RepeatChunkGrid::new(vec![2], regular_grid(vec![u64::MAX], vec![nz(1)])).is_err());
        assert!(matches!(
            RepeatChunkGrid::new(vec![2], regular_grid(vec![5], vec![nz(3)])),
            Err(RepeatChunkGridCreateError::InnerChunkEdgeLengthsMismatch {
                dimension: 0,
                edge_count: 2,
                edge_sum: 6,
                expected_edge_count: 2,
                expected_edge_sum: 5,
            })
        ));
    }

    #[test]
    #[expect(clippy::single_range_in_vec_init)]
    fn repeat_chunk_grid_regular_bounded_inner_grid() {
        let grid =
            RepeatChunkGrid::new(vec![2], regular_bounded_grid(vec![5], vec![nz(3)])).unwrap();

        assert_eq!(grid.array_shape(), &[10]);
        assert_eq!(grid.grid_shape(), &[4]);
        assert_eq!(
            grid.chunk_edge_lengths(0).unwrap(),
            vec![nz(3), nz(2), nz(3), nz(2)]
        );
        assert_eq!(grid.subset(&[1]).unwrap(), Some(ArraySubset::from([3..5])));
        assert_eq!(grid.subset(&[2]).unwrap(), Some(ArraySubset::from([5..8])));
        assert_eq!(grid.chunk_origin(&[3]).unwrap(), Some(vec![8]));
        assert_eq!(grid.chunk_shape(&[3]).unwrap(), Some(vec![nz(2)]));
        assert_eq!(grid.chunk_indices(&[8]).unwrap(), Some(vec![3]));
        assert_eq!(grid.chunk_element_indices(&[8]).unwrap(), Some(vec![0]));
    }

    #[test]
    fn repeat_chunk_grid_zero_sized_tile_dimension() {
        let grid =
            RepeatChunkGrid::new(vec![2, 1], regular_grid(vec![0, 4], vec![nz(3), nz(2)])).unwrap();

        assert_eq!(grid.array_shape(), &[0, 4]);
        assert_eq!(grid.grid_shape(), &[0, 2]);
        assert_eq!(
            grid.chunk_edge_lengths(0).unwrap(),
            Vec::<NonZeroU64>::new()
        );
        assert_eq!(grid.chunk_origin(&[0, 0]).unwrap(), None);
        assert_eq!(grid.chunk_shape(&[0, 0]).unwrap(), None);
        assert_eq!(grid.subset(&[0, 0]).unwrap(), None);
        assert_eq!(grid.chunk_indices(&[0, 0]).unwrap(), None);
        assert_eq!(grid.chunk_element_indices(&[0, 0]).unwrap(), None);
    }
}

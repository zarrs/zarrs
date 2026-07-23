//! The `zarrs.unstructured_cartesian` chunk grid.
//!
//! # Compatible Implementations
//! None
//!
//! # Chunk Grid `name` Aliases (Zarr V3)
//! - `zarrs.unstructured_cartesian`

use std::collections::HashMap;
use std::num::NonZeroU64;

use thiserror::Error;

use crate::array::{
    ArrayIndices, ArrayIndicesTinyVec, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkShape,
    IncompatibleDimensionError, IncompatibleDimensionalityError,
};
use zarrs_chunk_grid::{ChunkGrid, ChunkGridCreateError, ChunkGridPlugin, ChunkGridTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata::v3::MetadataV3;
pub use zarrs_metadata_ext::chunk_grid::unstructured_cartesian::{
    UnstructuredCartesianChunk, UnstructuredCartesianChunkGridConfiguration,
};

zarrs_plugin::impl_extension_aliases!(UnstructuredCartesianChunkGrid, v3: "zarrs.unstructured_cartesian");

inventory::submit! {
    ChunkGridPlugin::new::<UnstructuredCartesianChunkGrid>()
}

/// A `zarrs.unstructured_cartesian` chunk grid.
#[derive(Debug, Clone)]
pub struct UnstructuredCartesianChunkGrid {
    array_shape: ArrayShape,
    chunks: Vec<UnstructuredCartesianRuntimeChunk>,
    origin_to_chunk: HashMap<ArrayIndices, usize>,
    grid_shape: ArrayShape,
}

#[derive(Debug, Clone)]
struct UnstructuredCartesianRuntimeChunk {
    origin: ArrayIndices,
    shape: ChunkShape,
    shape_non_zero: Vec<NonZeroU64>,
    subset: ArraySubset,
}

/// A [`UnstructuredCartesianChunkGrid`] creation error.
#[derive(Clone, Debug, Error)]
#[error(
    "unstructured cartesian chunk grid configuration: {_1:?} not compatible with array shape {_0:?}: {_2}"
)]
pub struct UnstructuredCartesianChunkGridCreateError(
    ArrayShape,
    Vec<UnstructuredCartesianChunk>,
    String,
);

impl From<UnstructuredCartesianChunkGridCreateError> for ChunkGridCreateError {
    fn from(value: UnstructuredCartesianChunkGridCreateError) -> Self {
        Self::Other(value.to_string())
    }
}

impl UnstructuredCartesianChunkGrid {
    /// Create a new `zarrs.unstructured_cartesian` chunk grid with inline `chunks`.
    ///
    /// # Errors
    /// Returns a [`UnstructuredCartesianChunkGridCreateError`] if `chunks` are not a complete,
    /// non-overlapping partition of `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunks: Vec<UnstructuredCartesianChunk>,
    ) -> Result<Self, UnstructuredCartesianChunkGridCreateError> {
        let dimensionality = array_shape.len();
        let mut origin_to_chunk = HashMap::with_capacity(chunks.len());
        let mut runtime_chunks = Vec::with_capacity(chunks.len());
        let mut volume_sum = 0u64;

        for chunk in &chunks {
            if chunk.origin.len() != dimensionality || chunk.shape.len() != dimensionality {
                return Err(Self::create_error(
                    array_shape,
                    chunks,
                    "chunk dimensionality does not match array dimensionality",
                ));
            }
            if origin_to_chunk.contains_key(&chunk.origin) {
                return Err(Self::create_error(
                    array_shape,
                    chunks,
                    "duplicate chunk origin",
                ));
            }

            let shape = chunk
                .shape
                .iter()
                .map(|edge_length| edge_length.get())
                .collect::<Vec<_>>();
            if std::iter::zip(std::iter::zip(&chunk.origin, &shape), &array_shape).any(
                |((&origin, &shape), &array_size)| {
                    origin.checked_add(shape).is_none_or(|end| end > array_size)
                },
            ) {
                return Err(Self::create_error(
                    array_shape,
                    chunks,
                    "chunk extends beyond the array shape",
                ));
            }

            let subset = ArraySubset::new_with_start_shape(chunk.origin.clone(), shape.clone())
                .expect("origin and shape dimensionality already matched");
            if runtime_chunks
                .iter()
                .any(|other: &UnstructuredCartesianRuntimeChunk| {
                    !subset
                        .overlap(&other.subset)
                        .expect("matching dimensionality")
                        .is_empty()
                })
            {
                return Err(Self::create_error(array_shape, chunks, "chunks overlap"));
            }

            let volume = shape
                .iter()
                .try_fold(1u64, |product, edge_length| {
                    product.checked_mul(*edge_length)
                })
                .ok_or_else(|| {
                    Self::create_error(array_shape.clone(), chunks.clone(), "chunk volume overflow")
                })?;
            volume_sum = volume_sum.checked_add(volume).ok_or_else(|| {
                Self::create_error(array_shape.clone(), chunks.clone(), "chunk volume overflow")
            })?;

            let chunk_index = runtime_chunks.len();
            origin_to_chunk.insert(chunk.origin.clone(), chunk_index);
            runtime_chunks.push(UnstructuredCartesianRuntimeChunk {
                origin: chunk.origin.clone(),
                shape,
                shape_non_zero: chunk.shape.clone(),
                subset,
            });
        }

        let array_volume = array_shape
            .iter()
            .try_fold(1u64, |product, edge_length| {
                product.checked_mul(*edge_length)
            })
            .ok_or_else(|| {
                Self::create_error(array_shape.clone(), chunks.clone(), "array volume overflow")
            })?;
        if volume_sum != array_volume {
            return Err(Self::create_error(
                array_shape,
                chunks,
                "chunks do not completely cover the array",
            ));
        }

        Ok(Self {
            grid_shape: array_shape.clone(),
            array_shape,
            chunks: runtime_chunks,
            origin_to_chunk,
        })
    }

    /// Create a new `zarrs.unstructured_cartesian` chunk grid from exact chunk subsets.
    ///
    /// # Errors
    /// Returns a [`UnstructuredCartesianChunkGridCreateError`] if `chunks` are not a complete,
    /// non-overlapping partition of `array_shape`, or if any chunk has a zero edge length.
    pub fn new_from_subsets(
        array_shape: ArrayShape,
        chunks: Vec<ArraySubset>,
    ) -> Result<Self, UnstructuredCartesianChunkGridCreateError> {
        let chunks = chunks
            .into_iter()
            .map(|chunk| {
                let shape = chunk
                    .shape()
                    .iter()
                    .copied()
                    .map(NonZeroU64::new)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        Self::create_error(
                            array_shape.clone(),
                            Vec::new(),
                            "chunk shape contains a zero edge length",
                        )
                    })?;
                Ok(UnstructuredCartesianChunk {
                    origin: chunk.start().to_vec(),
                    shape,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(array_shape, chunks)
    }

    fn create_error(
        array_shape: ArrayShape,
        chunks: Vec<UnstructuredCartesianChunk>,
        reason: impl Into<String>,
    ) -> UnstructuredCartesianChunkGridCreateError {
        UnstructuredCartesianChunkGridCreateError(array_shape, chunks, reason.into())
    }

    fn chunk_at_origin(&self, chunk_indices: &[u64]) -> Option<&UnstructuredCartesianRuntimeChunk> {
        self.origin_to_chunk
            .get(chunk_indices)
            .map(|&index| &self.chunks[index])
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
}

unsafe impl ChunkGridTraits for UnstructuredCartesianChunkGrid {
    fn create(
        metadata: &MetadataV3,
        array_shape: &ArrayShape,
    ) -> Result<ChunkGrid, ChunkGridCreateError> {
        let configuration: UnstructuredCartesianChunkGridConfiguration =
            metadata.to_typed_configuration()?;
        let UnstructuredCartesianChunkGridConfiguration::Inline { chunks } = configuration;
        let chunk_grid = UnstructuredCartesianChunkGrid::new(array_shape.clone(), chunks)?;
        Ok(ChunkGrid::new(chunk_grid))
    }

    fn configuration(&self) -> Configuration {
        UnstructuredCartesianChunkGridConfiguration::Inline {
            chunks: self
                .chunks
                .iter()
                .map(|chunk| UnstructuredCartesianChunk {
                    origin: chunk.origin.clone(),
                    shape: chunk.shape_non_zero.clone(),
                })
                .collect(),
        }
        .into()
    }

    fn dimensionality(&self) -> usize {
        self.array_shape.len()
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
    ) -> Result<Option<Vec<NonZeroU64>>, IncompatibleDimensionError> {
        if dimension >= self.dimensionality() {
            return Err(IncompatibleDimensionError::new(
                dimension,
                self.dimensionality(),
            ));
        }
        Ok(None)
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        self.check_dimensionality(chunk_indices.len())?;
        if let Some(chunk) = self.chunk_at_origin(chunk_indices) {
            Ok(Some(chunk.shape.clone()))
        } else if self.array_shape.contains(&0) {
            Ok(None)
        } else {
            Ok(Some(vec![0; self.dimensionality()]))
        }
    }

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.check_dimensionality(chunk_indices.len())?;
        if self.array_shape.contains(&0) {
            Ok(None)
        } else if self.chunk_at_origin(chunk_indices).is_some() {
            Ok(Some(chunk_indices.to_vec()))
        } else {
            Ok(None)
        }
    }

    fn subset(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.check_dimensionality(chunk_indices.len())?;
        if let Some(chunk) = self.chunk_at_origin(chunk_indices) {
            Ok(Some(chunk.subset.clone()))
        } else if self.array_shape.contains(&0) {
            Ok(None)
        } else {
            Ok(Some(ArraySubset::new_empty(self.dimensionality())))
        }
    }

    fn chunks_subset(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.check_dimensionality(chunks.dimensionality())?;
        if chunks.is_empty() {
            return Ok(Some(ArraySubset::new_empty(self.dimensionality())));
        }

        let mut start = vec![u64::MAX; self.dimensionality()];
        let mut end = vec![0; self.dimensionality()];
        let mut any = false;
        for chunk in &self.chunks {
            if chunks.contains(&chunk.origin) {
                any = true;
                for (dimension, (&chunk_start, chunk_end)) in
                    std::iter::zip(chunk.subset.start(), chunk.subset.end_exc()).enumerate()
                {
                    start[dimension] = start[dimension].min(chunk_start);
                    end[dimension] = end[dimension].max(chunk_end);
                }
            }
        }

        if any {
            Ok(Some(
                ArraySubset::new_with_start_end_exc(start, end).expect("valid bounding subset"),
            ))
        } else {
            Ok(None)
        }
    }

    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.check_dimensionality(array_indices.len())?;
        if self.array_shape.contains(&0) || !self.array_indices_inbounds(array_indices) {
            return Ok(None);
        }
        Ok(self
            .chunks
            .iter()
            .find(|chunk| chunk.subset.contains(array_indices))
            .map(|chunk| chunk.origin.clone()))
    }

    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.check_dimensionality(array_indices.len())?;
        let Some(chunk_indices) = self.chunk_indices(array_indices)? else {
            return Ok(None);
        };
        Ok(Some(
            std::iter::zip(array_indices, chunk_indices)
                .map(|(&index, origin)| index - origin)
                .collect(),
        ))
    }

    fn chunk_indices_inbounds(&self, chunk_indices: &[u64]) -> bool {
        chunk_indices.len() == self.dimensionality()
            && self.origin_to_chunk.contains_key(chunk_indices)
    }

    fn chunks_in_array_subset(
        &self,
        region: &dyn ArraySubsetTraits,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.check_dimensionality(region.dimensionality())?;
        if region.is_empty() {
            return Ok(Some(ArraySubset::new_empty(self.dimensionality())));
        }

        let mut start = vec![u64::MAX; self.dimensionality()];
        let mut end = vec![0; self.dimensionality()];
        let mut any = false;
        for chunk in &self.chunks {
            if !chunk
                .subset
                .overlap(region)
                .expect("matching dimensionality")
                .is_empty()
            {
                any = true;
                for (dimension, &origin) in chunk.origin.iter().enumerate() {
                    start[dimension] = start[dimension].min(origin);
                    end[dimension] = end[dimension].max(origin + 1);
                }
            }
        }

        if any {
            Ok(Some(
                ArraySubset::new_with_start_end_exc(start, end).expect("valid bounding subset"),
            ))
        } else {
            Ok(None)
        }
    }

    fn iter_chunk_indices_and_subsets(
        &self,
    ) -> Box<dyn Iterator<Item = (ArrayIndicesTinyVec, ArraySubset)> + '_> {
        Box::new(self.chunks.iter().map(|chunk| {
            (
                ArrayIndicesTinyVec::Heap(chunk.origin.clone()),
                chunk.subset.clone(),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn chunk(origin: &[u64], shape: &[u64]) -> UnstructuredCartesianChunk {
        UnstructuredCartesianChunk::new(
            origin.to_vec(),
            shape.iter().copied().map(nz).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn unstructured_cartesian_valid_complete_partition() {
        let grid = UnstructuredCartesianChunkGrid::new(
            vec![3, 4],
            vec![
                chunk(&[0, 0], &[2, 3]),
                chunk(&[0, 3], &[2, 1]),
                chunk(&[2, 0], &[1, 1]),
                chunk(&[2, 1], &[1, 3]),
            ],
        )
        .unwrap();

        assert_eq!(grid.dimensionality(), 2);
        assert_eq!(grid.grid_shape(), &[3, 4]);
    }

    #[test]
    fn unstructured_cartesian_rejects_invalid_partitions() {
        assert!(
            UnstructuredCartesianChunkGrid::new(
                vec![3, 3],
                vec![chunk(&[0, 0], &[2, 2]), chunk(&[1, 1], &[2, 2])]
            )
            .is_err()
        );
        assert!(
            UnstructuredCartesianChunkGrid::new(vec![3, 3], vec![chunk(&[0, 0], &[2, 3])]).is_err()
        );
        assert!(
            UnstructuredCartesianChunkGrid::new(vec![3, 3], vec![chunk(&[0, 0], &[4, 3])]).is_err()
        );
        assert!(
            UnstructuredCartesianChunkGrid::new(
                vec![3, 3],
                vec![chunk(&[0, 0], &[1, 3]), chunk(&[0, 0], &[2, 3])]
            )
            .is_err()
        );
        assert!(UnstructuredCartesianChunkGrid::new(vec![3, 3], vec![chunk(&[0], &[3])]).is_err());
    }

    #[test]
    fn unstructured_cartesian_queries() {
        let grid = UnstructuredCartesianChunkGrid::new(
            vec![3, 4],
            vec![
                chunk(&[0, 0], &[2, 3]),
                chunk(&[0, 3], &[2, 1]),
                chunk(&[2, 0], &[1, 1]),
                chunk(&[2, 1], &[1, 3]),
            ],
        )
        .unwrap();

        assert_eq!(grid.chunk_indices(&[1, 2]).unwrap(), Some(vec![0, 0]));
        assert_eq!(grid.chunk_edge_lengths(0).unwrap(), None);
        assert_eq!(grid.chunk_indices(&[2, 2]).unwrap(), Some(vec![2, 1]));
        assert_eq!(grid.chunk_origin(&[2, 1]).unwrap(), Some(vec![2, 1]));
        assert_eq!(grid.chunk_shape(&[2, 1]).unwrap(), Some(vec![1, 3]));
        assert_eq!(
            grid.chunk_element_indices(&[2, 3]).unwrap(),
            Some(vec![0, 2])
        );

        assert_eq!(grid.chunk_origin(&[1, 1]).unwrap(), None);
        assert_eq!(grid.chunk_shape(&[1, 1]).unwrap(), Some(vec![0, 0]));
        assert_eq!(
            grid.subset(&[1, 1]).unwrap(),
            Some(ArraySubset::new_empty(2))
        );
        assert_eq!(
            grid.chunks_in_array_subset(&ArraySubset::new_with_ranges(&[1..3, 2..4]))
                .unwrap(),
            Some(ArraySubset::new_with_ranges(&[0..3, 0..4]))
        );
        assert_eq!(
            grid.chunks_subset(&ArraySubset::new_with_ranges(&[0..3, 0..4]))
                .unwrap(),
            Some(ArraySubset::new_with_ranges(&[0..3, 0..4]))
        );
    }

    #[test]
    fn unstructured_cartesian_chunk_indices_and_subsets_iterator_skips_pseudo_chunks() {
        let grid = UnstructuredCartesianChunkGrid::new(
            vec![3, 4],
            vec![
                chunk(&[0, 0], &[2, 3]),
                chunk(&[0, 3], &[2, 1]),
                chunk(&[2, 0], &[1, 1]),
                chunk(&[2, 1], &[1, 3]),
            ],
        )
        .unwrap();

        let chunks = grid
            .iter_chunk_indices_and_subsets()
            .map(|(indices, subset)| (indices.to_vec(), subset))
            .collect::<Vec<_>>();

        assert_eq!(
            chunks,
            vec![
                (vec![0, 0], ArraySubset::new_with_ranges(&[0..2, 0..3])),
                (vec![0, 3], ArraySubset::new_with_ranges(&[0..2, 3..4])),
                (vec![2, 0], ArraySubset::new_with_ranges(&[2..3, 0..1])),
                (vec![2, 1], ArraySubset::new_with_ranges(&[2..3, 1..4])),
            ]
        );
    }
}

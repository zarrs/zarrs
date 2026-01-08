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

use crate::array::chunk_grid::{ChunkGrid, ChunkGridPlugin, ChunkGridTraits};
use crate::array::{ArrayIndices, ArrayShape, ChunkShape};
use crate::array_subset::{ArraySubset, IncompatibleDimensionalityError};
use crate::metadata::v3::MetadataV3;
pub use crate::metadata_ext::chunk_grid::regular::RegularChunkGridConfiguration;
use crate::plugin::{PluginCreateError, PluginMetadataInvalidError};
use zarrs_plugin::ExtensionIdentifier;

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new(RegularChunkGrid::IDENTIFIER, RegularChunkGrid::matches_name, RegularChunkGrid::default_name, create_chunk_grid_regular)
}
zarrs_plugin::impl_extension_aliases!(RegularChunkGrid, "regular");

/// Create a `regular` chunk grid from metadata.
///
/// # Errors
/// Returns a [`PluginCreateError`] if the metadata is invalid for a regular chunk grid.
pub(crate) fn create_chunk_grid_regular(
    metadata: &MetadataV3,
    array_shape: &ArrayShape,
) -> Result<ChunkGrid, PluginCreateError> {
    let configuration: RegularChunkGridConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(
                RegularChunkGrid::IDENTIFIER,
                "chunk grid",
                metadata.to_string(),
            )
        })?;
    let chunk_grid = RegularChunkGrid::new(array_shape.clone(), configuration.chunk_shape)
        .map_err(|_| {
            PluginCreateError::from(
                "regular chunk shape and array shape have inconsistent dimensionality",
            )
        })?;
    Ok(ChunkGrid::new(chunk_grid))
}

/// A `regular` chunk grid.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct RegularChunkGrid {
    array_shape: ArrayShape,
    grid_shape: ArrayShape,
    chunk_shape: ChunkShape,
}

/// A [`RegularChunkGrid`] creation error.
#[derive(Clone, Debug, Error)]
#[error("regular chunk shape: {_1:?} not compatible with array shape {_0:?}")]
pub struct RegularChunkGridCreateError(ArrayShape, ChunkShape);

impl From<RegularChunkGridCreateError> for IncompatibleDimensionalityError {
    fn from(value: RegularChunkGridCreateError) -> Self {
        Self::new(value.1.len(), value.0.len())
    }
}

impl RegularChunkGrid {
    /// Create a new `regular` chunk grid with chunk shape `chunk_shape`.
    ///
    /// # Errors
    /// Returns a [`RegularChunkGridCreateError`] if `chunk_shape` is not compatible with the `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunk_shape: ChunkShape,
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

    /// Return the chunk shape as an [`ArrayShape`] ([`Vec<u64>`]).
    #[must_use]
    pub fn chunk_shape_u64(&self) -> ArrayShape {
        self.chunk_shape
            .iter()
            .copied()
            .map(NonZeroU64::get)
            .collect::<ArrayShape>()
    }

    /// Determinate version of [`ChunkGridTraits::chunk_origin`].
    pub(crate) fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ArrayIndices, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(std::iter::zip(chunk_indices, self.chunk_shape.as_slice())
                .map(|(i, s)| i * s.get())
                .collect())
        } else {
            Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    /// Determinate version of [`ChunkGridTraits::chunk_indices`].
    pub(crate) fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<ArrayIndices, IncompatibleDimensionalityError> {
        if array_indices.len() == self.dimensionality() {
            Ok(std::iter::zip(array_indices, self.chunk_shape.as_slice())
                .map(|(i, s)| i / s.get())
                .collect())
        } else {
            Err(IncompatibleDimensionalityError::new(
                array_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    /// Determinate version of [`ChunkGridTraits::subset`].
    pub(crate) fn subset(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ArraySubset, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            let ranges =
                std::iter::zip(chunk_indices, self.chunk_shape.as_slice()).map(|(i, s)| {
                    let start = i * s.get();
                    start..(start + s.get())
                });
            Ok(ArraySubset::from(ranges))
        } else {
            Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ))
        }
    }

    /// Determinate version of [`ChunkGridTraits::chunks_in_array_subset`].
    pub(crate) fn chunks_in_array_subset(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<ArraySubset, IncompatibleDimensionalityError> {
        match array_subset.end_inc() {
            Some(end) => {
                let chunks_start = self.chunk_indices(array_subset.start())?;
                let chunks_end = self.chunk_indices(&end)?;
                // .unwrap_or_else(|| self.grid_shape());

                let shape = std::iter::zip(&chunks_start, chunks_end)
                    .map(|(&s, e)| e.saturating_sub(s) + 1)
                    .collect();
                Ok(ArraySubset::new_with_start_shape(chunks_start, shape)?)
            }
            None => Ok(ArraySubset::new_empty(self.dimensionality())),
        }
    }
}

unsafe impl ChunkGridTraits for RegularChunkGrid {
    fn create_metadata(&self) -> MetadataV3 {
        let configuration = RegularChunkGridConfiguration {
            chunk_shape: self.chunk_shape.clone(),
        };
        MetadataV3::new_with_serializable_configuration(
            Self::IDENTIFIER.to_string(),
            &configuration,
        )
        .unwrap()
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

    fn subset(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.subset(chunk_indices).map(Option::Some)
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(Some(self.chunk_shape.clone()))
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
            Ok(Some(self.chunk_shape_u64()))
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
        self.chunk_origin(chunk_indices).map(Option::Some)
    }

    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.chunk_indices(array_indices).map(Option::Some)
    }

    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if array_indices.len() == self.dimensionality() {
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

    fn chunks_in_array_subset(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.chunks_in_array_subset(array_subset).map(Option::Some)
    }
}

#[cfg(test)]
mod tests {
    use rayon::iter::ParallelIterator;

    use super::*;
    use crate::array::ArrayIndicesTinyVec;
    use crate::array::chunk_grid::ChunkGridTraitsIterators;
    use crate::array_subset::ArraySubset;

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
        assert!(create_chunk_grid_regular(&metadata, &vec![3, 3, 3]).is_ok());
    }

    #[test]
    fn chunk_grid_regular_metadata_invalid() {
        let metadata: MetadataV3 =
            serde_json::from_str(r#"{"name":"regular","configuration":{"invalid":[1,2,3]}}"#)
                .unwrap();
        assert!(create_chunk_grid_regular(&metadata, &vec![3, 3, 3]).is_err());
        assert_eq!(
            create_chunk_grid_regular(&metadata, &vec![3, 3, 3])
                .unwrap_err()
                .to_string(),
            r#"chunk grid regular is unsupported with metadata: regular {"invalid":[1,2,3]}"#
        );
    }

    #[test]
    fn chunk_grid_regular() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShape = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];

        {
            let chunk_grid =
                RegularChunkGrid::new(array_shape.clone(), chunk_shape.clone()).unwrap();
            assert_eq!(chunk_grid.dimensionality(), 3);
            assert_eq!(chunk_grid.chunk_origin(&[1, 1, 1]).unwrap(), vec![1, 2, 3]);
            assert_eq!(chunk_grid.chunk_shape(), chunk_shape.as_slice());
            let chunk_grid_shape = chunk_grid.grid_shape();
            assert_eq!(chunk_grid_shape, &[5, 4, 18]);
            let array_index: ArrayIndices = vec![3, 5, 50];
            assert_eq!(
                chunk_grid.chunk_indices(&array_index).unwrap(),
                vec![3, 2, 16]
            );
            assert_eq!(
                chunk_grid.chunk_element_indices(&array_index).unwrap(),
                Some(vec![0, 1, 2])
            );

            assert_eq!(
                chunk_grid
                    .chunks_subset(&ArraySubset::new_with_ranges(&[1..3, 1..2, 5..8]),)
                    .unwrap(),
                Some(ArraySubset::new_with_ranges(&[1..3, 2..4, 15..24]))
            );

            assert!(
                chunk_grid
                    .chunks_subset(&ArraySubset::new_with_ranges(&[1..3]))
                    .is_err()
            );

            assert!(
                chunk_grid
                    .chunks_subset(&ArraySubset::new_with_ranges(&[0..0, 0..0, 0..0]),)
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
        let chunk_shape: ChunkShape = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularChunkGrid::new(array_shape, chunk_shape).unwrap();

        let array_indices: ArrayIndices = vec![3, 5, 53];
        assert_eq!(
            chunk_grid.chunk_indices(&array_indices).unwrap(),
            vec![3, 2, 17]
        );

        let chunk_indices: ArrayShape = vec![6, 1, 1];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert_eq!(
            chunk_grid.chunk_origin(&chunk_indices).unwrap(),
            vec![6, 2, 3]
        );
    }

    #[test]
    fn chunk_grid_regular_unlimited() {
        let array_shape: ArrayShape = vec![5, 7, 0];
        let chunk_shape: ChunkShape = vec![
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_grid = RegularChunkGrid::new(array_shape, chunk_shape).unwrap();

        let array_indices: ArrayIndices = vec![3, 5, 1000];
        assert_eq!(
            chunk_grid.chunk_indices(&array_indices).unwrap(),
            vec![3, 2, 333]
        );

        assert_eq!(chunk_grid.grid_shape(), &[5, 4, 0]);

        let chunk_indices: ArrayShape = vec![3, 1, 1000];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
    }

    #[test]
    fn chunk_grid_regular_iterators() {
        let array_shape: ArrayShape = vec![2, 2, 6];
        let chunk_shape: ChunkShape = vec![
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

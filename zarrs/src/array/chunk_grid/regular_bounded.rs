//! The `regular_bounded` chunk grid.
#![allow(unreachable_pub)]

use std::num::NonZeroU64;

use itertools::izip;
// use zarrs_registry::chunk_grid::REGULAR_BOUNDED;
/// Unique identifier for the `regular_bounded` chunk grid (extension).
pub const REGULAR_BOUNDED: &str = "regular_bounded"; // TODO: Move to zarrs_registry on stabilisation

use crate::{
    array::{chunk_grid::ChunkGridPlugin, ArrayIndices, ArrayShape, ChunkShape},
    array_subset::{ArraySubset, IncompatibleDimensionalityError},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

pub use super::RegularChunkGridConfiguration;
use super::{ChunkGrid, ChunkGridTraits};

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new(REGULAR_BOUNDED, is_name_regular_bounded, create_chunk_grid_regular_bounded)
}

fn is_name_regular_bounded(name: &str) -> bool {
    name.eq(REGULAR_BOUNDED)
}

/// Create a `regular_bounded` chunk grid from metadata.
///
/// # Errors
/// Returns a [`PluginCreateError`] if the metadata is invalid for a `regular_bounded` chunk grid.
pub(crate) fn create_chunk_grid_regular_bounded(
    metadata_and_array_shape: &(MetadataV3, ArrayShape),
) -> Result<ChunkGrid, PluginCreateError> {
    let (metadata, array_shape) = metadata_and_array_shape;
    let configuration: RegularChunkGridConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(REGULAR_BOUNDED, "chunk grid", metadata.to_string())
        })?;
    let chunk_grid = RegularBoundedChunkGrid::new(array_shape.clone(), configuration.chunk_shape)
        .map_err(|_| {
        PluginCreateError::from(
            "`regular_bounded` chunk shape and array shape have inconsistent dimensionality",
        )
    })?;
    Ok(ChunkGrid::new(chunk_grid))
}

/// A `regular_bounded` chunk grid.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct RegularBoundedChunkGrid {
    array_shape: ArrayShape,
    grid_shape: ArrayShape,
    chunk_shape: ChunkShape,
}

impl RegularBoundedChunkGrid {
    /// Create a new `regular_bounded` chunk grid with chunk shape `chunk_shape`.
    ///
    /// # Errors
    /// Returns a [`IncompatibleDimensionalityError`] if `chunk_shape` is not compatible with the `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunk_shape: ChunkShape,
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

    /// Return the shape of a chunk as an [`ArrayShape`] ([`Vec<u64>`]).
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_indices` do not match the dimensionality of the chunk grid.
    pub fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError> {
        Ok(self
            .chunk_shape(chunk_indices)?
            .map(|c| c.into_iter().map(NonZeroU64::get).collect::<ArrayShape>()))
    }
}

impl ChunkGridTraits for RegularBoundedChunkGrid {
    fn create_metadata(&self) -> MetadataV3 {
        let configuration = RegularChunkGridConfiguration {
            chunk_shape: self.chunk_shape.clone(),
        };
        MetadataV3::new_with_serializable_configuration(REGULAR_BOUNDED.to_string(), &configuration)
            .unwrap()
    }

    fn dimensionality(&self) -> usize {
        self.chunk_shape.len()
    }

    fn array_shape(&self) -> &ArrayShape {
        &self.array_shape
    }

    fn grid_shape(&self) -> &ArrayShape {
        &self.grid_shape
    }

    unsafe fn chunk_shape_unchecked(&self, chunk_indices: &[u64]) -> Option<ChunkShape> {
        debug_assert_eq!(self.dimensionality(), chunk_indices.len());
        izip!(
            self.chunk_shape.as_slice(),
            &self.array_shape,
            chunk_indices
        )
        .map(|(chunk_shape, &array_shape, chunk_indices)| {
            let start = (chunk_indices * chunk_shape.get()).min(array_shape);
            let end = (start + chunk_shape.get()).min(array_shape);
            NonZeroU64::new(end.saturating_sub(start))
        })
        .collect::<Option<Vec<_>>>()
        .map(ChunkShape::from)
    }

    unsafe fn chunk_shape_u64_unchecked(&self, chunk_indices: &[u64]) -> Option<ArrayShape> {
        debug_assert_eq!(self.dimensionality(), chunk_indices.len());
        self.chunk_shape_u64(chunk_indices).unwrap()
    }

    unsafe fn chunk_origin_unchecked(&self, chunk_indices: &[u64]) -> Option<ArrayIndices> {
        debug_assert_eq!(chunk_indices.len(), self.dimensionality());
        izip!(
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
        .collect::<Option<Vec<_>>>()
    }

    unsafe fn chunk_indices_unchecked(&self, array_indices: &[u64]) -> Option<ArrayIndices> {
        debug_assert_eq!(array_indices.len(), self.dimensionality());
        izip!(
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
        .collect::<Option<Vec<_>>>()
    }

    unsafe fn chunk_element_indices_unchecked(
        &self,
        array_indices: &[u64],
    ) -> Option<ArrayIndices> {
        izip!(
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
        .collect::<Option<Vec<_>>>()
    }

    unsafe fn subset_unchecked(&self, chunk_indices: &[u64]) -> Option<ArraySubset> {
        debug_assert_eq!(self.dimensionality(), chunk_indices.len());
        let ranges = izip!(
            self.chunk_shape.as_slice(),
            &self.array_shape,
            chunk_indices
        )
        .map(|(chunk_shape, &array_shape, chunk_indices)| {
            let start = (chunk_indices * chunk_shape.get()).min(array_shape);
            let end = (start + chunk_shape.get()).min(array_shape);
            if end > start {
                Some(start..end)
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
        let array_subset = ArraySubset::new_with_ranges(&ranges);
        Some(array_subset)
    }
}

#[cfg(test)]
mod tests {
    use rayon::iter::ParallelIterator;

    use crate::array_subset::ArraySubset;

    use super::*;

    #[test]
    fn chunk_grid_regular_bounded_metadata() {
        let metadata: MetadataV3 = serde_json::from_str(
            r#"{"name":"regular_bounded","configuration":{"chunk_shape":[1,2,3]}}"#,
        )
        .unwrap();
        assert!(create_chunk_grid_regular_bounded(&(metadata, vec![3, 3, 3])).is_ok());
    }

    #[test]
    fn chunk_grid_regular_bounded_metadata_invalid() {
        let metadata: MetadataV3 = serde_json::from_str(
            r#"{"name":"regular_bounded","configuration":{"invalid":[1,2,3]}}"#,
        )
        .unwrap();
        assert!(create_chunk_grid_regular_bounded(&(metadata.clone(), vec![3, 3, 3])).is_err());
        assert_eq!(
            create_chunk_grid_regular_bounded(&(metadata, vec![3, 3, 3]))
                .unwrap_err()
                .to_string(),
            r#"chunk grid regular_bounded is unsupported with metadata: regular_bounded {"invalid":[1,2,3]}"#
        );
    }

    #[test]
    fn chunk_grid_regular_bounded() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShape = vec![1, 2, 3].try_into().unwrap();

        {
            let chunk_grid =
                RegularBoundedChunkGrid::new(array_shape.clone(), chunk_shape.clone()).unwrap();
            assert_eq!(chunk_grid.dimensionality(), 3);
            assert_eq!(
                chunk_grid.chunk_origin(&[1, 1, 1]).unwrap(),
                Some(vec![1, 2, 3])
            );
            assert_eq!(
                chunk_grid.chunk_shape(&[0, 0, 0]).unwrap(),
                Some(chunk_shape.clone())
            );
            assert_eq!(
                chunk_grid.chunk_shape(&[0, 3, 17]).unwrap(),
                Some(vec![1, 1, 1].try_into().unwrap())
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
                chunk_grid
                    .chunks_subset(&ArraySubset::new_with_ranges(&[1..3, 1..2, 5..8]),)
                    .unwrap(),
                Some(ArraySubset::new_with_ranges(&[1..3, 2..4, 15..24]))
            );

            assert!(chunk_grid
                .chunks_subset(&ArraySubset::new_with_ranges(&[1..3]))
                .is_err());

            assert!(chunk_grid
                .chunks_subset(&ArraySubset::new_with_ranges(&[0..0, 0..0, 0..0]),)
                .unwrap()
                .unwrap()
                .is_empty());
        }

        assert!(RegularBoundedChunkGrid::new(vec![0; 1], chunk_shape.clone()).is_err());
    }

    #[test]
    fn chunk_grid_regular_bounded_out_of_bounds() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShape = vec![1, 2, 3].try_into().unwrap();
        let chunk_grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape).unwrap();

        let array_indices: ArrayIndices = vec![3, 5, 53];
        assert_eq!(chunk_grid.chunk_indices(&array_indices).unwrap(), None);

        let chunk_indices: ArrayShape = vec![6, 1, 1];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert_eq!(chunk_grid.chunk_origin(&chunk_indices).unwrap(), None);
    }

    #[test]
    fn chunk_grid_regular_bounded_unlimited() {
        let array_shape: ArrayShape = vec![5, 7, 0];
        let chunk_shape: ChunkShape = vec![1, 2, 3].try_into().unwrap();
        let chunk_grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape).unwrap();

        let array_indices: ArrayIndices = vec![3, 5, 1000];
        assert_eq!(chunk_grid.chunk_indices(&array_indices).unwrap(), None);

        assert_eq!(chunk_grid.grid_shape(), &[5, 4, 0]);

        let chunk_indices: ArrayShape = vec![3, 1, 1000];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
    }

    #[test]
    fn chunk_grid_regular_bounded_iterators() {
        let array_shape: ArrayShape = vec![2, 2, 6];
        let chunk_shape: ChunkShape = vec![1, 2, 3].try_into().unwrap();
        let chunk_grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape).unwrap();

        let iter = chunk_grid.iter_chunk_indices();
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![vec![0, 0, 0], vec![0, 0, 1], vec![1, 0, 0], vec![1, 0, 1]]
        );

        let iter = chunk_grid.par_iter_chunk_indices();
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![vec![0, 0, 0], vec![0, 0, 1], vec![1, 0, 0], vec![1, 0, 1]]
        );
    }
}

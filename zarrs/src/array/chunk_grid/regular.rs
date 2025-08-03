//! The `regular` chunk grid.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/chunk-grids/regular-grid/index.html>.

use std::num::NonZeroU64;

use thiserror::Error;
use zarrs_registry::chunk_grid::REGULAR;

use crate::{
    array::{chunk_grid::ChunkGridPlugin, ArrayIndices, ArrayShape, ChunkShape},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

pub use super::RegularChunkGridConfiguration;
use super::{ChunkGrid, ChunkGridTraits};

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new(REGULAR, is_name_regular, create_chunk_grid_regular)
}

fn is_name_regular(name: &str) -> bool {
    name.eq(REGULAR)
}

/// Create a `regular` chunk grid from metadata.
///
/// # Errors
/// Returns a [`PluginCreateError`] if the metadata is invalid for a regular chunk grid.
pub(crate) fn create_chunk_grid_regular(
    metadata_and_array_shape: &(MetadataV3, ArrayShape),
) -> Result<ChunkGrid, PluginCreateError> {
    let (metadata, array_shape) = metadata_and_array_shape;
    let configuration: RegularChunkGridConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(REGULAR, "chunk grid", metadata.to_string())
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
    pub fn chunk_shape_u64(&self) -> Vec<u64> {
        self.chunk_shape
            .iter()
            .copied()
            .map(NonZeroU64::get)
            .collect::<Vec<_>>()
    }
}

impl ChunkGridTraits for RegularChunkGrid {
    fn create_metadata(&self) -> MetadataV3 {
        let configuration = RegularChunkGridConfiguration {
            chunk_shape: self.chunk_shape.clone(),
        };
        MetadataV3::new_with_serializable_configuration(REGULAR.to_string(), &configuration)
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

    /// The chunk shape. Fixed for a `regular` grid.
    unsafe fn chunk_shape_unchecked(&self, chunk_indices: &[u64]) -> Option<ChunkShape> {
        debug_assert_eq!(self.dimensionality(), chunk_indices.len());
        Some(self.chunk_shape.clone())
    }

    /// The chunk shape as an [`ArrayShape`] ([`Vec<u64>`]). Fixed for a `regular` grid.
    unsafe fn chunk_shape_u64_unchecked(&self, chunk_indices: &[u64]) -> Option<ArrayShape> {
        debug_assert_eq!(self.dimensionality(), chunk_indices.len());
        Some(
            self.chunk_shape
                .iter()
                .copied()
                .map(NonZeroU64::get)
                .collect::<ArrayShape>(),
        )
    }

    unsafe fn chunk_origin_unchecked(&self, chunk_indices: &[u64]) -> Option<ArrayIndices> {
        debug_assert_eq!(self.dimensionality(), chunk_indices.len());
        Some(
            std::iter::zip(chunk_indices, self.chunk_shape.as_slice())
                .map(|(i, s)| i * s.get())
                .collect(),
        )
    }

    unsafe fn chunk_indices_unchecked(&self, array_indices: &[u64]) -> Option<ArrayIndices> {
        debug_assert_eq!(self.dimensionality(), array_indices.len());
        Some(
            std::iter::zip(array_indices, self.chunk_shape.as_slice())
                .map(|(i, s)| i / s.get())
                .collect(),
        )
    }

    unsafe fn chunk_element_indices_unchecked(
        &self,
        array_indices: &[u64],
    ) -> Option<ArrayIndices> {
        debug_assert_eq!(self.dimensionality(), array_indices.len());
        Some(
            std::iter::zip(array_indices, self.chunk_shape.as_slice())
                .map(|(i, s)| i % s.get())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::array_subset::ArraySubset;

    use super::*;

    #[test]
    fn chunk_grid_regular_configuration() {
        let configuration: RegularChunkGridConfiguration =
            serde_json::from_str(r#"{"chunk_shape":[1,2,3]}"#).unwrap();
        assert_eq!(configuration.chunk_shape, vec![1, 2, 3].try_into().unwrap());
        assert_eq!(
            configuration.to_string(),
            r#"regular chunk grid {"chunk_shape":[1,2,3]}"#
        )
    }

    #[test]
    fn chunk_grid_regular_metadata() {
        let metadata: MetadataV3 =
            serde_json::from_str(r#"{"name":"regular","configuration":{"chunk_shape":[1,2,3]}}"#)
                .unwrap();
        assert!(create_chunk_grid_regular(&(metadata, vec![3, 3, 3])).is_ok());
    }

    #[test]
    fn chunk_grid_regular_metadata_invalid() {
        let metadata: MetadataV3 =
            serde_json::from_str(r#"{"name":"regular","configuration":{"invalid":[1,2,3]}}"#)
                .unwrap();
        assert!(create_chunk_grid_regular(&(metadata.clone(), vec![3, 3, 3])).is_err());
        assert_eq!(
            create_chunk_grid_regular(&(metadata, vec![3, 3, 3]))
                .unwrap_err()
                .to_string(),
            r#"chunk grid regular is unsupported with metadata: regular {"invalid":[1,2,3]}"#
        );
    }

    #[test]
    fn chunk_grid_regular() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShape = vec![1, 2, 3].try_into().unwrap();

        {
            let chunk_grid =
                RegularChunkGrid::new(array_shape.clone(), chunk_shape.clone()).unwrap();
            assert_eq!(chunk_grid.dimensionality(), 3);
            assert_eq!(
                chunk_grid.chunk_origin(&[1, 1, 1]).unwrap(),
                Some(vec![1, 2, 3])
            );
            assert_eq!(chunk_grid.chunk_shape(), chunk_shape.as_slice());
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

        assert!(RegularChunkGrid::new(vec![0; 1], chunk_shape.clone()).is_err());
    }

    #[test]
    fn chunk_grid_regular_out_of_bounds() {
        let array_shape: ArrayShape = vec![5, 7, 52];
        let chunk_shape: ChunkShape = vec![1, 2, 3].try_into().unwrap();
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
    fn chunk_grid_regular_unlimited() {
        let array_shape: ArrayShape = vec![5, 7, 0];
        let chunk_shape: ChunkShape = vec![1, 2, 3].try_into().unwrap();
        let chunk_grid = RegularChunkGrid::new(array_shape, chunk_shape).unwrap();

        let array_indices: ArrayIndices = vec![3, 5, 1000];
        assert!(chunk_grid.chunk_indices(&array_indices).unwrap().is_some());

        assert_eq!(chunk_grid.grid_shape(), &[5, 4, 0]);

        let chunk_indices: ArrayShape = vec![3, 1, 1000];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert!(chunk_grid.chunk_origin(&chunk_indices).unwrap().is_some());
    }
}

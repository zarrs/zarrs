//! The `rectangular` chunk grid.
//!
//!
//! <div class="warning">
//! This chunk grid is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! # Compatible Implementations
//! None
//!
//! # Specification
//! - <https://zarr.dev/zeps/draft/ZEP0003.html>
//!
//! # Chunk Grid `name` Aliases (Zarr V3)
//! - `rectangular`
//!
//! ### Chunk grid `configuration` Example - [`RectangularChunkGridConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!   "chunk_shape": [[5, 5, 5, 15, 15, 20, 35], 10]
//! }
//! # "#;
//! # use zarrs::metadata_ext::chunk_grid::rectangular::RectangularChunkGridConfiguration;
//! # let configuration: RectangularChunkGridConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

use std::num::NonZeroU64;

use derive_more::From;
use itertools::Itertools;
use thiserror::Error;

use crate::array::chunk_grid::{ChunkGrid, ChunkGridPlugin, ChunkGridTraits};
use crate::array::{ArrayIndices, ArrayShape, ChunkShape, IncompatibleDimensionalityError};
use crate::metadata::v3::MetadataV3;
pub use crate::metadata_ext::chunk_grid::rectangular::{
    RectangularChunkGridConfiguration, RectangularChunkGridDimensionConfiguration,
};
use crate::plugin::{PluginCreateError, PluginMetadataInvalidError};
use zarrs_plugin::ExtensionIdentifier;

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new(RectangularChunkGrid::IDENTIFIER, RectangularChunkGrid::matches_name, RectangularChunkGrid::default_name, create_chunk_grid_rectangular)
}
zarrs_plugin::impl_extension_aliases!(RectangularChunkGrid, "rectangular");

/// Create a `rectangular` chunk grid from metadata.
///
/// # Errors
/// Returns a [`PluginCreateError`] if the metadata is invalid for a regular chunk grid.
pub(crate) fn create_chunk_grid_rectangular(
    metadata: &MetadataV3,
    array_shape: &ArrayShape,
) -> Result<ChunkGrid, PluginCreateError> {
    crate::warn_experimental_extension(metadata.name(), "chunk grid");
    let configuration: RectangularChunkGridConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(
                RectangularChunkGrid::IDENTIFIER,
                "chunk grid",
                metadata.to_string(),
            )
        })?;
    let chunk_grid = RectangularChunkGrid::new(array_shape.clone(), &configuration.chunk_shape)
        .map_err(|err| PluginCreateError::Other(err.to_string()))?;
    Ok(ChunkGrid::new(chunk_grid))
}

/// A `rectangular` chunk grid.
#[derive(Debug, Clone)]
pub struct RectangularChunkGrid {
    array_shape: ArrayShape,
    chunks: Vec<RectangularChunkGridDimension>,
    grid_shape: ArrayShape,
}

#[derive(Debug, Clone)]
struct OffsetSize {
    offset: u64,
    size: NonZeroU64,
}

#[derive(Debug, Clone, From)]
enum RectangularChunkGridDimension {
    Fixed(NonZeroU64),
    Varying(Vec<OffsetSize>),
}

/// A [`RectangularChunkGrid`] creation error.
#[derive(Clone, Debug, Error)]
#[error("rectangular chunk grid configuration: {_1:?} not compatible with array shape {_0:?}")]
pub struct RectangularChunkGridCreateError(
    ArrayShape,
    Vec<RectangularChunkGridDimensionConfiguration>,
);

impl RectangularChunkGrid {
    /// Create a new `rectangular` chunk grid with chunk shapes `chunk_shapes`.
    ///
    /// # Errors
    /// Returns a [`RectangularChunkGridCreateError`] if `chunk_shapes` are not compatible with the `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunk_shapes: &[RectangularChunkGridDimensionConfiguration],
    ) -> Result<Self, RectangularChunkGridCreateError> {
        if array_shape.len() != chunk_shapes.len() {
            return Err(RectangularChunkGridCreateError(
                array_shape.clone(),
                chunk_shapes.to_vec(),
            ));
        }

        let chunks: Vec<RectangularChunkGridDimension> = chunk_shapes
            .iter()
            .map(|s| match s {
                RectangularChunkGridDimensionConfiguration::Fixed(f) => {
                    RectangularChunkGridDimension::Fixed(*f)
                }
                RectangularChunkGridDimensionConfiguration::Varying(chunk_sizes) => {
                    RectangularChunkGridDimension::Varying(
                        chunk_sizes
                            .as_slice()
                            .iter()
                            .scan(0, |offset, &size| {
                                let last_offset = *offset;
                                *offset += size.get();
                                Some(OffsetSize {
                                    offset: last_offset,
                                    size,
                                })
                            })
                            .collect(),
                    )
                }
            })
            .collect();
        let grid_shape = std::iter::zip(&array_shape, chunks.iter())
            .map(|(array_shape, chunks)| match chunks {
                RectangularChunkGridDimension::Fixed(s) => {
                    let s = s.get();
                    Some(array_shape.div_ceil(s))
                }
                RectangularChunkGridDimension::Varying(s) => {
                    let last_default = OffsetSize {
                        offset: 0,
                        // SAFETY: 1 is non-zero
                        size: unsafe { NonZeroU64::new_unchecked(1) },
                    };
                    let last = s.last().unwrap_or(&last_default);
                    if *array_shape == last.offset + last.size.get() {
                        Some(s.len() as u64)
                    } else {
                        None
                    }
                }
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                RectangularChunkGridCreateError(array_shape.clone(), chunk_shapes.to_vec())
            })?;
        Ok(Self {
            array_shape,
            chunks,
            grid_shape,
        })
    }
}

unsafe impl ChunkGridTraits for RectangularChunkGrid {
    fn create_metadata(&self) -> MetadataV3 {
        let chunk_shape = self
            .chunks
            .iter()
            .map(|chunk_dim| match chunk_dim {
                RectangularChunkGridDimension::Fixed(size) => {
                    RectangularChunkGridDimensionConfiguration::Fixed(*size)
                }
                RectangularChunkGridDimension::Varying(offsets_sizes) => {
                    RectangularChunkGridDimensionConfiguration::Varying(
                        offsets_sizes
                            .iter()
                            .map(|offset_size| offset_size.size)
                            .collect_vec(),
                    )
                }
            })
            .collect();
        let configuration = RectangularChunkGridConfiguration { chunk_shape };
        MetadataV3::new_with_serializable_configuration(
            Self::IDENTIFIER.to_string(),
            &configuration,
        )
        .unwrap()
    }

    fn dimensionality(&self) -> usize {
        self.chunks.len()
    }

    fn array_shape(&self) -> &[u64] {
        &self.array_shape
    }

    fn grid_shape(&self) -> &[u64] {
        &self.grid_shape
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(std::iter::zip(chunk_indices, &self.chunks)
                .map(|(chunk_index, chunks)| match chunks {
                    RectangularChunkGridDimension::Fixed(chunk_size) => Some(*chunk_size),
                    RectangularChunkGridDimension::Varying(offsets_sizes) => {
                        let chunk_index = usize::try_from(*chunk_index).unwrap();
                        if chunk_index < offsets_sizes.len() {
                            Some(offsets_sizes[chunk_index].size)
                        } else {
                            None
                        }
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

    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(std::iter::zip(chunk_indices, &self.chunks)
                .map(|(chunk_index, chunks)| match chunks {
                    RectangularChunkGridDimension::Fixed(chunk_size) => Some(chunk_size.get()),
                    RectangularChunkGridDimension::Varying(offsets_sizes) => {
                        let chunk_index = usize::try_from(*chunk_index).unwrap();
                        if chunk_index < offsets_sizes.len() {
                            Some(offsets_sizes[chunk_index].size.get())
                        } else {
                            None
                        }
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

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(std::iter::zip(chunk_indices, &self.chunks)
                .map(|(chunk_index, chunks)| match chunks {
                    RectangularChunkGridDimension::Fixed(chunk_size) => {
                        Some(chunk_index * chunk_size.get())
                    }
                    RectangularChunkGridDimension::Varying(offsets_sizes) => {
                        let chunk_index = usize::try_from(*chunk_index).unwrap();
                        if chunk_index < offsets_sizes.len() {
                            Some(offsets_sizes[chunk_index].offset)
                        } else {
                            None
                        }
                    }
                })
                .collect())
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
            Ok(std::iter::zip(array_indices, &self.chunks)
                .map(|(index, chunks)| match chunks {
                    RectangularChunkGridDimension::Fixed(size) => Some(index / size.get()),
                    RectangularChunkGridDimension::Varying(offsets_sizes) => {
                        let last_default = OffsetSize {
                            offset: 0,
                            // SAFETY: 1 is non-zero
                            size: unsafe { NonZeroU64::new_unchecked(1) },
                        };
                        let last = offsets_sizes.last().unwrap_or(&last_default);
                        if *index < last.offset + last.size.get() {
                            let partition = offsets_sizes
                                .partition_point(|offset_size| *index >= offset_size.offset);
                            if partition <= offsets_sizes.len() {
                                let partition = partition as u64;
                                Some(std::cmp::max(partition, 1) - 1)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                })
                .collect())
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
        // TODO: optimise by doing everything in one iter?
        let chunk_indices = self.chunk_indices(array_indices)?;
        Ok(chunk_indices.and_then(|chunk_indices| {
            // SAFETY: The length of chunk_indices matches the dimensionality of the chunk grid
            self.chunk_origin(&chunk_indices)
                .expect("matching dimensionality")
                .map(|chunk_start| {
                    std::iter::zip(array_indices, &chunk_start)
                        .map(|(i, s)| i - s)
                        .collect()
                })
        }))
    }

    fn array_indices_inbounds(&self, array_indices: &[u64]) -> bool {
        array_indices.len() == self.dimensionality()
            && itertools::izip!(array_indices, self.array_shape(), &self.chunks).all(
                |(array_index, array_size, chunks)| {
                    (*array_size == 0 || array_index < array_size)
                        && match chunks {
                            RectangularChunkGridDimension::Fixed(_) => true,
                            RectangularChunkGridDimension::Varying(offsets_sizes) => offsets_sizes
                                .last()
                                .is_some_and(|last| *array_index < last.offset + last.size.get()),
                        }
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::ArraySubset;

    #[test]
    fn chunk_grid_rectangular() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<RectangularChunkGridDimensionConfiguration> = vec![
            vec![
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(15).unwrap(),
                NonZeroU64::new(15).unwrap(),
                NonZeroU64::new(20).unwrap(),
                NonZeroU64::new(35).unwrap(),
            ]
            .into(),
            NonZeroU64::new(10).unwrap().into(),
        ];
        let chunk_grid = RectangularChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.dimensionality(), 2);
        assert_eq!(chunk_grid.grid_shape(), &[7, 10]);
        assert_eq!(
            chunk_grid.chunk_indices(&[17, 17]).unwrap(),
            Some(vec![3, 1])
        );
        assert_eq!(
            chunk_grid.chunk_element_indices(&[17, 17]).unwrap(),
            Some(vec![2, 7])
        );

        assert_eq!(
            chunk_grid
                .chunks_subset(&ArraySubset::new_with_ranges(&[1..5, 2..6]))
                .unwrap(),
            Some(ArraySubset::new_with_ranges(&[5..45, 20..60]))
        );

        // assert_eq!(
        //     chunk_grid.chunk_indices(&array_index, &array_shape)?,
        //     &[3, 2, 16]
        // );
        // assert_eq!(
        //     chunk_grid.chunk_element_indices(&array_index, &array_shape)?,
        //     &[0, 1, 2]
        // );

        assert!(RectangularChunkGrid::new(vec![100; 3], &chunk_shapes).is_err()); // incompatible dimensionality
        assert!(RectangularChunkGrid::new(vec![123, 100], &chunk_shapes).is_err());
        // incompatible chunk shapes
        // incompatible dimensionality
    }

    #[test]
    fn chunk_grid_rectangular_out_of_bounds() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<RectangularChunkGridDimensionConfiguration> = vec![
            vec![
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(15).unwrap(),
                NonZeroU64::new(15).unwrap(),
                NonZeroU64::new(20).unwrap(),
                NonZeroU64::new(35).unwrap(),
            ]
            .into(),
            NonZeroU64::new(10).unwrap().into(),
        ];
        let chunk_grid = RectangularChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.grid_shape(), &[7, 10]);

        let array_indices: ArrayIndices = vec![99, 99];
        assert!(chunk_grid.chunk_indices(&array_indices).unwrap().is_some());

        let array_indices: ArrayIndices = vec![100, 100];
        assert!(chunk_grid.chunk_indices(&array_indices).unwrap().is_none());

        let chunk_indices: ArrayShape = vec![6, 9];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert!(chunk_grid.chunk_origin(&chunk_indices).unwrap().is_some());

        let chunk_indices: ArrayShape = vec![7, 9];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert!(chunk_grid.chunk_origin(&chunk_indices).unwrap().is_none());

        let chunk_indices: ArrayShape = vec![6, 10];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
    }

    #[test]
    fn chunk_grid_rectangular_unlimited() {
        let array_shape: ArrayShape = vec![100, 0];
        let chunk_shapes: Vec<RectangularChunkGridDimensionConfiguration> = vec![
            vec![
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(15).unwrap(),
                NonZeroU64::new(15).unwrap(),
                NonZeroU64::new(20).unwrap(),
                NonZeroU64::new(35).unwrap(),
            ]
            .into(),
            NonZeroU64::new(10).unwrap().into(),
        ];
        let chunk_grid = RectangularChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.grid_shape(), &[7, 0]);

        let array_indices: ArrayIndices = vec![101, 150];
        assert!(chunk_grid.chunk_indices(&array_indices).unwrap().is_none());

        let chunk_indices: ArrayShape = vec![6, 9];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert!(chunk_grid.chunk_origin(&chunk_indices).unwrap().is_some());

        let chunk_indices: ArrayShape = vec![7, 9];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert!(chunk_grid.chunk_origin(&chunk_indices).unwrap().is_none());

        let chunk_indices: ArrayShape = vec![6, 123];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
    }
}

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

use crate::array::{ArrayIndices, ArrayShape, ChunkShape, IncompatibleDimensionalityError};
use zarrs_chunk_grid::{ChunkGrid, ChunkGridPlugin, ChunkGridTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata::v3::MetadataV3;
pub use zarrs_metadata_ext::chunk_grid::rectangular::{
    RectangularChunkGridConfiguration, RectangularChunkGridDimensionConfiguration,
};
use zarrs_plugin::PluginCreateError;

zarrs_plugin::impl_extension_aliases!(RectangularChunkGrid, v3: "rectangular");

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new::<RectangularChunkGrid>()
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
    fn create(
        metadata: &MetadataV3,
        array_shape: &ArrayShape,
    ) -> Result<ChunkGrid, PluginCreateError> {
        crate::warn_experimental_extension(metadata.name(), "chunk grid");
        let configuration: RectangularChunkGridConfiguration = metadata.to_typed_configuration()?;
        let chunk_grid = RectangularChunkGrid::new(array_shape.clone(), &configuration.chunk_shape)
            .map_err(|err| PluginCreateError::Other(err.to_string()))?;
        Ok(ChunkGrid::new(chunk_grid))
    }

    fn configuration(&self) -> Configuration {
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
        RectangularChunkGridConfiguration { chunk_shape }.into()
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
            if self.array_shape.contains(&0) {
                Ok(None)
            } else {
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
            }
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
            if self.array_shape.contains(&0) {
                Ok(None)
            } else {
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
            }
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
            if self.array_shape.contains(&0) {
                Ok(None)
            } else {
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
            }
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
            if self.array_shape.contains(&0) {
                Ok(None)
            } else {
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
            }
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
                    array_index < array_size
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
            chunk_grid.chunks_subset(&[1..5, 2..6]).unwrap(),
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
    fn chunk_grid_rectangular_fully_and_partially_out_of_bounds() {
        // Array [100, 100], dim 0: Varying [5,5,5,15,15,20,35], dim 1: Fixed(10)
        // Grid shape: [7, 10]
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
        let chunk_grid = RectangularChunkGrid::new(array_shape.clone(), &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.grid_shape(), &[7, 10]);

        // Fully out-of-bounds: varying dim index past list, fixed dim in bounds
        assert_eq!(chunk_grid.chunk_origin(&[7, 5]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_shape(&[7, 5]).unwrap(), None);

        // Fully out-of-bounds: both dims past extent
        assert_eq!(chunk_grid.chunk_origin(&[99, 99]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_shape(&[99, 99]).unwrap(), None);

        // Fully out-of-bounds: fixed dim past grid, varying dim in bounds
        // Fixed dim index 10 is past grid shape [7, 10], but varying dim is checked first
        // Returns None because Varying dim index 6 is valid, but fixed dim 10 is not
        // The fixed dim returns Some(10*10=100), but the overall method checks if
        // chunk_index >= offsets_sizes.len() for varying dims only.
        // For fixed dims, chunk_origin always returns Some.
        // However, chunk_shape for fixed dim returns Some even for out-of-bounds.
        // The grid only checks varying dim bounds for None returns.
        // So [6, 10] returns Some for fixed-dim-based queries but the varying dim 6 is valid.
        // This demonstrates that fixed dims have no per-chunk bounds check in the impl.

        // Fully out-of-bounds array indices
        assert_eq!(chunk_grid.chunk_indices(&[100, 100]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_indices(&[999, 999]).unwrap(), None);
        assert_eq!(chunk_grid.chunk_element_indices(&[100, 100]).unwrap(), None);

        // Partially out-of-bounds (edge) last chunk: [6, 9]
        // origin [65, 90], chunk size [35, 10], extent [100, 100] = exactly at boundary
        assert_eq!(
            chunk_grid.chunk_origin(&[6, 9]).unwrap(),
            Some(vec![65, 90])
        );
        assert_eq!(
            chunk_grid.chunk_shape(&[6, 9]).unwrap(),
            Some(vec![
                NonZeroU64::new(35).unwrap(),
                NonZeroU64::new(10).unwrap()
            ])
        );
        assert_eq!(
            chunk_grid.chunk_shape_u64(&[6, 9]).unwrap(),
            Some(vec![35, 10])
        );
        assert_eq!(
            chunk_grid.subset(&[6, 9]).unwrap(),
            Some(ArraySubset::new_with_ranges(&[65..100, 90..100]))
        );

        // In-bounds array index at array boundary -> last chunk
        assert_eq!(
            chunk_grid.chunk_indices(&[99, 99]).unwrap(),
            Some(vec![6, 9])
        );
        assert_eq!(
            chunk_grid.chunk_element_indices(&[99, 99]).unwrap(),
            Some(vec![34, 9])
        );

        // In-bounds interior chunk
        assert_eq!(
            chunk_grid.chunk_origin(&[3, 1]).unwrap(),
            Some(vec![15, 10])
        );
        assert_eq!(
            chunk_grid.chunk_shape(&[3, 1]).unwrap(),
            Some(vec![
                NonZeroU64::new(15).unwrap(),
                NonZeroU64::new(10).unwrap()
            ])
        );
    }

    #[test]
    fn chunk_grid_rectangular_zero_dim() {
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

        // No indices are in-bounds for zero-dim
        let chunk_indices: ArrayShape = vec![6, 0];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));

        // Out of bounds in first dimension
        let chunk_indices: ArrayShape = vec![7, 0];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));

        // No array indices are in-bounds when second dimension is zero
        let array_indices: ArrayIndices = vec![50, 0];
        assert!(!chunk_grid.array_indices_inbounds(&array_indices));

        // Array indices beyond explicit chunks in the first dimension
        let array_indices: ArrayIndices = vec![101, 0];
        assert!(chunk_grid.chunk_indices(&array_indices).unwrap().is_none());
    }
}

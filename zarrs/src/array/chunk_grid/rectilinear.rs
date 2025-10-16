//! The `rectilinear` chunk grid.
//!
//! <div class="warning">
//! This chunk grid is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! # Compatible Implementations
//! None
//!
//! # Specification
//! - TODO
//!
//! # Chunk Grid `name` Aliases (Zarr V3)
//! - `rectilinear`
//!
//! ### Chunk grid `configuration` Example - [`RectilinearChunkGridConfiguration`]:
//!
//! Explicit chunk sizes:
//! ```rust
//! # let JSON = r#"
//! {
//!   "kind": "inline",
//!   "chunk_shapes": [[5, 5, 5, 15, 15, 20, 35], [10]]
//! }
//! # "#;
//! # use zarrs_metadata_ext::chunk_grid::rectilinear::RectilinearChunkGridConfiguration;
//! # let configuration: RectilinearChunkGridConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! Run-length encoded chunk sizes (equivalent to the above):
//! ```rust
//! # let JSON = r#"
//! {
//!   "kind": "inline",
//!   "chunk_shapes": [[[5, 3], [15, 2], 20, 35], [10]]
//! }
//! # "#;
//! # use zarrs_metadata_ext::chunk_grid::rectilinear::RectilinearChunkGridConfiguration;
//! # let configuration: RectilinearChunkGridConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! In run-length encoding, `[value, count]` represents `count` repetitions of `value`.

use itertools::Itertools;
use std::num::NonZeroU64;
use thiserror::Error;

pub use zarrs_metadata_ext::chunk_grid::rectilinear::{
    RectilinearChunkGridConfiguration, RunLengthElement, RunLengthElements,
};
use zarrs_registry::chunk_grid::RECTILINEAR;

use crate::{
    array::{
        chunk_grid::{ChunkGrid, ChunkGridPlugin, ChunkGridTraits},
        ArrayIndices, ArrayShape, ChunkShape,
    },
    array_subset::IncompatibleDimensionalityError,
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the chunk grid.
inventory::submit! {
    ChunkGridPlugin::new(RECTILINEAR, is_name_rectilinear, create_chunk_grid_rectilinear)
}

fn is_name_rectilinear(name: &str) -> bool {
    name.eq(RECTILINEAR)
}

/// Create a `rectilinear` chunk grid from metadata.
///
/// # Errors
/// Returns a [`PluginCreateError`] if the metadata is invalid for a regular chunk grid.
pub(crate) fn create_chunk_grid_rectilinear(
    metadata_and_array_shape: &(MetadataV3, ArrayShape),
) -> Result<ChunkGrid, PluginCreateError> {
    crate::warn_experimental_extension(metadata_and_array_shape.0.name(), "chunk grid");
    let (metadata, array_shape) = metadata_and_array_shape;
    let configuration: RectilinearChunkGridConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(RECTILINEAR, "chunk grid", metadata.to_string())
        })?;
    let chunk_shape = match &configuration {
        RectilinearChunkGridConfiguration::Inline { chunk_shapes } => chunk_shapes,
    };
    let chunk_grid = RectilinearChunkGrid::new(array_shape.clone(), chunk_shape)
        .map_err(|err| PluginCreateError::Other(err.to_string()))?;
    Ok(ChunkGrid::new(chunk_grid))
}

/// A `rectilinear` chunk grid.
#[derive(Debug, Clone)]
pub struct RectilinearChunkGrid {
    array_shape: ArrayShape,
    chunks: Vec<RectilinearChunkGridDimension>,
    grid_shape: ArrayShape,
}

#[derive(Debug, Clone)]
struct OffsetSize {
    offset: u64,
    size: NonZeroU64,
}

type RectilinearChunkGridDimension = Vec<OffsetSize>;

/// A [`RectilinearChunkGrid`] creation error.
#[derive(Clone, Debug, Error)]
#[error("rectilinear chunk grid configuration: {_1:?} not compatible with array shape {_0:?}")]
pub struct RectilinearChunkGridCreateError(ArrayShape, Vec<RunLengthElements>);

/// Extension trait for `RunLengthElements`.
trait RunLengthElementsExt {
    /// Convert to an explicit chunk shape by expanding run-length encoding.
    fn expand(&self) -> Vec<NonZeroU64>;
}

impl RunLengthElementsExt for RunLengthElements {
    fn expand(&self) -> Vec<NonZeroU64> {
        let mut result = Vec::new();
        for element in self {
            match element {
                RunLengthElement::Single(value) => result.push(*value),
                RunLengthElement::Repeated([value, count]) => {
                    let count = count.get();
                    result.extend(std::iter::repeat_n(*value, usize::try_from(count).unwrap()));
                }
            }
        }
        result
    }
}

impl RectilinearChunkGrid {
    /// Create a new `rectilinear` chunk grid with chunk shapes `chunk_shapes`.
    ///
    /// # Errors
    /// Returns a [`RectilinearChunkGridCreateError`] if `chunk_shapes` are not compatible with the `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunk_shapes: &[RunLengthElements],
    ) -> Result<Self, RectilinearChunkGridCreateError> {
        if array_shape.len() != chunk_shapes.len() {
            return Err(RectilinearChunkGridCreateError(
                array_shape.clone(),
                chunk_shapes.to_vec(),
            ));
        }

        let chunks: Vec<RectilinearChunkGridDimension> = chunk_shapes
            .iter()
            .map(|s| {
                let chunk_sizes = s.expand();
                chunk_sizes
                    .iter()
                    .scan(0, |offset, &size| {
                        let last_offset = *offset;
                        *offset += size.get();
                        Some(OffsetSize {
                            offset: last_offset,
                            size,
                        })
                    })
                    .collect()
            })
            .collect();
        let grid_shape = std::iter::zip(&array_shape, chunks.iter())
            .map(|(array_shape, chunks)| {
                let last_default = OffsetSize {
                    offset: 0,
                    // SAFETY: 1 is non-zero
                    size: unsafe { NonZeroU64::new_unchecked(1) },
                };
                let last = chunks.last().unwrap_or(&last_default);
                if *array_shape == last.offset + last.size.get() {
                    Some(chunks.len() as u64)
                } else {
                    None
                }
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                RectilinearChunkGridCreateError(array_shape.clone(), chunk_shapes.to_vec())
            })?;
        Ok(Self {
            array_shape,
            chunks,
            grid_shape,
        })
    }
}

/// Compress a sequence of chunk sizes into run-length encoded form.
///
/// Consecutive repeated values are combined into `RunLengthElement::Repeated([value, count])`.
/// Single values remain as `RunLengthElement::Single(value)`.
fn compress_run_length(sizes: &[NonZeroU64]) -> RunLengthElements {
    sizes
        .iter()
        .chunk_by(|&&size| size)
        .into_iter()
        .map(|(size, group)| {
            let count = group.count() as u64;
            if count == 1 {
                RunLengthElement::Single(size)
            } else {
                RunLengthElement::Repeated([size, NonZeroU64::new(count).unwrap()])
            }
        })
        .collect()
}

unsafe impl ChunkGridTraits for RectilinearChunkGrid {
    fn create_metadata(&self) -> MetadataV3 {
        let chunk_shapes = self
            .chunks
            .iter()
            .map(|offsets_sizes| {
                let sizes: Vec<NonZeroU64> = offsets_sizes.iter().map(|os| os.size).collect();
                compress_run_length(&sizes)
            })
            .collect();
        let configuration = RectilinearChunkGridConfiguration::Inline { chunk_shapes };
        MetadataV3::new_with_serializable_configuration(RECTILINEAR.to_string(), &configuration)
            .unwrap()
    }

    fn dimensionality(&self) -> usize {
        self.chunks.len()
    }

    fn array_shape(&self) -> &ArrayShape {
        &self.array_shape
    }

    fn grid_shape(&self) -> &ArrayShape {
        &self.grid_shape
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(std::iter::zip(chunk_indices, &self.chunks)
                .map(|(chunk_index, offsets_sizes)| {
                    let chunk_index = usize::try_from(*chunk_index).unwrap();
                    if chunk_index < offsets_sizes.len() {
                        Some(offsets_sizes[chunk_index].size)
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<_>>>()
                .map(std::convert::Into::into))
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
                .map(|(chunk_index, offsets_sizes)| {
                    let chunk_index = usize::try_from(*chunk_index).unwrap();
                    if chunk_index < offsets_sizes.len() {
                        Some(offsets_sizes[chunk_index].size.get())
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

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if chunk_indices.len() == self.dimensionality() {
            Ok(std::iter::zip(chunk_indices, &self.chunks)
                .map(|(chunk_index, offsets_sizes)| {
                    let chunk_index = usize::try_from(*chunk_index).unwrap();
                    if chunk_index < offsets_sizes.len() {
                        Some(offsets_sizes[chunk_index].offset)
                    } else {
                        None
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
                .map(|(index, offsets_sizes)| {
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
                |(array_index, array_size, offsets_sizes)| {
                    (*array_size == 0 || array_index < array_size)
                        && offsets_sizes
                            .last()
                            .is_some_and(|last| *array_index < last.offset + last.size.get())
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use crate::array_subset::ArraySubset;

    use super::*;

    fn from_slice_u64(values: &[u64]) -> Result<RunLengthElements, std::num::TryFromIntError> {
        values
            .iter()
            .map(|&v| NonZeroU64::try_from(v).map(RunLengthElement::Single))
            .collect()
    }

    #[test]
    fn chunk_grid_rectilinear() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<RunLengthElements> = vec![
            from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap(),
            from_slice_u64(&[10; 10]).unwrap(),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

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

        assert!(RectilinearChunkGrid::new(vec![100; 3], &chunk_shapes).is_err()); // incompatible dimensionality
        assert!(RectilinearChunkGrid::new(vec![123, 100], &chunk_shapes).is_err());
        // incompatible chunk shapes
        // incompatible dimensionality
    }

    #[test]
    fn chunk_grid_rectilinear_out_of_bounds() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<RunLengthElements> = vec![
            from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap(),
            from_slice_u64(&[10; 10]).unwrap(),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

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
    fn chunk_grid_rectilinear_run_length_encoded() {
        let array_shape: ArrayShape = vec![100, 100];

        // Create run-length encoded configuration: [[5, 3], [15, 2], 20, 35]
        // This expands to: [5, 5, 5, 15, 15, 20, 35]
        let chunk_shapes: Vec<RunLengthElements> = vec![
            vec![
                RunLengthElement::Repeated([
                    NonZeroU64::new(5).unwrap(),
                    NonZeroU64::new(3).unwrap(),
                ]),
                RunLengthElement::Repeated([
                    NonZeroU64::new(15).unwrap(),
                    NonZeroU64::new(2).unwrap(),
                ]),
                RunLengthElement::Single(NonZeroU64::new(20).unwrap()),
                RunLengthElement::Single(NonZeroU64::new(35).unwrap()),
            ],
            vec![RunLengthElement::Repeated([
                NonZeroU64::new(10).unwrap(),
                NonZeroU64::new(10).unwrap(),
            ])],
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        // Should behave exactly the same as the explicit version
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

        // Test chunk shapes
        assert_eq!(
            chunk_grid.chunk_shape(&[0, 0]).unwrap(),
            Some(vec![NonZeroU64::new(5).unwrap(), NonZeroU64::new(10).unwrap()].into())
        );
        assert_eq!(
            chunk_grid.chunk_shape(&[3, 0]).unwrap(),
            Some(vec![NonZeroU64::new(15).unwrap(), NonZeroU64::new(10).unwrap()].into())
        );
        assert_eq!(
            chunk_grid.chunk_shape(&[5, 0]).unwrap(),
            Some(vec![NonZeroU64::new(20).unwrap(), NonZeroU64::new(10).unwrap()].into())
        );
        assert_eq!(
            chunk_grid.chunk_shape(&[6, 0]).unwrap(),
            Some(vec![NonZeroU64::new(35).unwrap(), NonZeroU64::new(10).unwrap()].into())
        );
    }

    #[test]
    fn chunk_grid_rectilinear_run_length_encoded_serialization() {
        use zarrs_metadata_ext::chunk_grid::rectilinear::RectilinearChunkGridConfiguration;

        // Test that run-length encoded format can be deserialized
        let json = r#"
        {
            "kind": "inline",
            "chunk_shapes": [[[5, 3], [15, 2], 20, 35], [[10, 10]]]
        }
        "#;

        let config: RectilinearChunkGridConfiguration = serde_json::from_str(json).unwrap();

        // Verify the first dimension is run-length encoded
        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = &config;
        let elements = &chunk_shapes[0];
        assert_eq!(elements.len(), 4);
        assert!(
            matches!(&elements[0], RunLengthElement::Repeated([val, count]) if val.get() == 5 && count.get() == 3)
        );

        // Verify it can be used to create a chunk grid
        let array_shape: ArrayShape = vec![100, 100];
        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = &config;
        let chunk_grid = RectilinearChunkGrid::new(array_shape, chunk_shapes).unwrap();
        assert_eq!(chunk_grid.grid_shape(), &[7, 10]);
    }

    #[test]
    fn chunk_grid_rectilinear_explicit_vs_rle() {
        let array_shape: ArrayShape = vec![100, 100];

        // Create explicit version
        let explicit_chunks: Vec<RunLengthElements> = vec![
            from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap(),
            from_slice_u64(&[10; 10]).unwrap(),
        ];
        let explicit_grid =
            RectilinearChunkGrid::new(array_shape.clone(), &explicit_chunks).unwrap();

        // Create run-length encoded version
        let rle_chunks: Vec<RunLengthElements> = vec![
            vec![
                RunLengthElement::Repeated([
                    NonZeroU64::new(5).unwrap(),
                    NonZeroU64::new(3).unwrap(),
                ]),
                RunLengthElement::Repeated([
                    NonZeroU64::new(15).unwrap(),
                    NonZeroU64::new(2).unwrap(),
                ]),
                RunLengthElement::Single(NonZeroU64::new(20).unwrap()),
                RunLengthElement::Single(NonZeroU64::new(35).unwrap()),
            ],
            vec![RunLengthElement::Repeated([
                NonZeroU64::new(10).unwrap(),
                NonZeroU64::new(10).unwrap(),
            ])],
        ];
        let rle_grid = RectilinearChunkGrid::new(array_shape, &rle_chunks).unwrap();

        // Both should produce identical behavior
        assert_eq!(explicit_grid.grid_shape(), rle_grid.grid_shape());

        for i in 0..7 {
            for j in 0..10 {
                let indices = vec![i, j];
                assert_eq!(
                    explicit_grid.chunk_shape(&indices).unwrap(),
                    rle_grid.chunk_shape(&indices).unwrap(),
                );
                assert_eq!(
                    explicit_grid.chunk_origin(&indices).unwrap(),
                    rle_grid.chunk_origin(&indices).unwrap(),
                );
            }
        }
    }

    #[test]
    fn chunk_grid_rectilinear_metadata_compression() {
        let array_shape: ArrayShape = vec![100, 100];

        // Create a chunk grid with repeated chunk sizes
        let chunk_shapes: Vec<RunLengthElements> = vec![
            from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap(),
            from_slice_u64(&[10; 10]).unwrap(),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        // Get the metadata
        let metadata = chunk_grid.create_metadata();
        let config: RectilinearChunkGridConfiguration = metadata.to_configuration().unwrap();

        // Verify the metadata is run-length encoded
        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = config;

        // First dimension should be compressed: [[5, 3], [15, 2], 20, 35]
        assert_eq!(chunk_shapes[0].len(), 4);
        assert!(
            matches!(&chunk_shapes[0][0], RunLengthElement::Repeated([val, count]) if val.get() == 5 && count.get() == 3)
        );
        assert!(
            matches!(&chunk_shapes[0][1], RunLengthElement::Repeated([val, count]) if val.get() == 15 && count.get() == 2)
        );
        assert!(matches!(&chunk_shapes[0][2], RunLengthElement::Single(val) if val.get() == 20));
        assert!(matches!(&chunk_shapes[0][3], RunLengthElement::Single(val) if val.get() == 35));

        // Second dimension should be a single value: [10]
        assert_eq!(chunk_shapes[1].len(), 1);
        assert!(
            matches!(&chunk_shapes[1][0], RunLengthElement::Repeated([val, count]) if val.get() == 10 && count.get() == 10)
        );
    }
}

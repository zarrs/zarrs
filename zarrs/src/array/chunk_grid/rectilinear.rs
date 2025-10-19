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
//! - <https://github.com/d-v-b/zarr-extensions/tree/main/chunk-grids/rectilinear>
//!
//! # Chunk Grid `name` Aliases (Zarr V3)
//! - `rectilinear`
//!
//! ### Chunk grid `configuration` Example - [`RectilinearChunkGridConfiguration`]:
//!
//! Scalar chunk shapes (regular grid):
//! ```rust
//! # let JSON = r#"
//! {
//!   "kind": "inline",
//!   "chunk_shapes": [10, 20]
//! }
//! # "#;
//! # use zarrs_metadata_ext::chunk_grid::rectilinear::RectilinearChunkGridConfiguration;
//! # let configuration: RectilinearChunkGridConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
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
//! Mixed scalar and array specifications:
//! ```rust
//! # let JSON = r#"
//! {
//!   "kind": "inline",
//!   "chunk_shapes": [10, [5, 5, 5, 15, 15, 20, 35]]
//! }
//! # "#;
//! # use zarrs_metadata_ext::chunk_grid::rectilinear::RectilinearChunkGridConfiguration;
//! # let configuration: RectilinearChunkGridConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! In run-length encoding, `[value, count]` represents `count` repetitions of `value`.
//! Scalar values represent a regular grid with fixed-size chunks.

use itertools::Itertools;
use std::num::NonZeroU64;
use thiserror::Error;

pub use zarrs_metadata_ext::chunk_grid::rectilinear::{
    ChunkEdgeLengths, RectilinearChunkGridConfiguration, RunLengthElement,
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

#[derive(Debug, Clone)]
enum RectilinearChunkGridDimension {
    Fixed(NonZeroU64),
    Varying(Vec<OffsetSize>),
}

/// A [`RectilinearChunkGrid`] creation error.
#[derive(Clone, Debug, Error)]
#[error("rectilinear chunk grid configuration: {_1:?} not compatible with array shape {_0:?}")]
pub struct RectilinearChunkGridCreateError(ArrayShape, Vec<ChunkEdgeLengths>);

/// Expand run-length encoding to explicit chunk sizes.
///
/// Only applies to varying chunk edge lengths.
fn expand_varying_chunks(elements: &[RunLengthElement]) -> Vec<NonZeroU64> {
    let mut result = Vec::new();
    for element in elements {
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

impl RectilinearChunkGrid {
    /// Create a new `rectilinear` chunk grid with chunk shapes `chunk_shapes`.
    ///
    /// # Errors
    /// Returns a [`RectilinearChunkGridCreateError`] if `chunk_shapes` are not compatible with the `array_shape`.
    pub fn new(
        array_shape: ArrayShape,
        chunk_shapes: &[ChunkEdgeLengths],
    ) -> Result<Self, RectilinearChunkGridCreateError> {
        if array_shape.len() != chunk_shapes.len() {
            return Err(RectilinearChunkGridCreateError(
                array_shape.clone(),
                chunk_shapes.to_vec(),
            ));
        }

        let chunks: Vec<RectilinearChunkGridDimension> = chunk_shapes
            .iter()
            .map(|chunk_shape| match chunk_shape {
                ChunkEdgeLengths::Scalar(chunk_size) => {
                    RectilinearChunkGridDimension::Fixed(*chunk_size)
                }
                ChunkEdgeLengths::Varying(elements) => {
                    let chunk_sizes = expand_varying_chunks(elements);
                    RectilinearChunkGridDimension::Varying(
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
                            .collect(),
                    )
                }
            })
            .collect();

        let grid_shape = std::iter::zip(&array_shape, chunks.iter())
            .map(|(array_size, chunk_dim)| match chunk_dim {
                RectilinearChunkGridDimension::Fixed(chunk_size) => {
                    if *array_size == 0 {
                        // Unlimited dimension
                        Some(0)
                    } else {
                        Some(array_size.div_ceil(chunk_size.get()))
                    }
                }
                RectilinearChunkGridDimension::Varying(chunks) => {
                    let last = chunks.last()?;
                    if *array_size == last.offset + last.size.get() {
                        Some(chunks.len() as u64)
                    } else {
                        None
                    }
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
fn compress_run_length(sizes: &[NonZeroU64]) -> Vec<RunLengthElement> {
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
            .map(|chunk_dim| match chunk_dim {
                RectilinearChunkGridDimension::Fixed(size) => ChunkEdgeLengths::Scalar(*size),
                RectilinearChunkGridDimension::Varying(offsets_sizes) => {
                    let sizes: Vec<NonZeroU64> = offsets_sizes.iter().map(|os| os.size).collect();
                    ChunkEdgeLengths::Varying(compress_run_length(&sizes))
                }
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
        if chunk_indices.len() != self.dimensionality() {
            return Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ));
        }

        Ok(std::iter::zip(chunk_indices, &self.chunks)
            .map(|(chunk_index, chunk_dim)| match chunk_dim {
                RectilinearChunkGridDimension::Fixed(chunk_size) => Some(*chunk_size),
                RectilinearChunkGridDimension::Varying(offsets_sizes) => {
                    let chunk_index = usize::try_from(*chunk_index).unwrap();
                    if chunk_index < offsets_sizes.len() {
                        Some(offsets_sizes[chunk_index].size)
                    } else {
                        None
                    }
                }
            })
            .collect::<Option<Vec<_>>>()
            .map(std::convert::Into::into))
    }

    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ));
        }

        Ok(std::iter::zip(chunk_indices, &self.chunks)
            .map(|(chunk_index, chunk_dim)| match chunk_dim {
                RectilinearChunkGridDimension::Fixed(chunk_size) => Some(chunk_size.get()),
                RectilinearChunkGridDimension::Varying(offsets_sizes) => {
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

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(IncompatibleDimensionalityError::new(
                chunk_indices.len(),
                self.dimensionality(),
            ));
        }

        Ok(std::iter::zip(chunk_indices, &self.chunks)
            .map(|(chunk_index, chunk_dim)| match chunk_dim {
                RectilinearChunkGridDimension::Fixed(chunk_size) => {
                    Some(chunk_index * chunk_size.get())
                }
                RectilinearChunkGridDimension::Varying(offsets_sizes) => {
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

    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if array_indices.len() != self.dimensionality() {
            return Err(IncompatibleDimensionalityError::new(
                array_indices.len(),
                self.dimensionality(),
            ));
        }

        Ok(std::iter::zip(array_indices, &self.chunks)
            .map(|(index, chunk_dim)| match chunk_dim {
                RectilinearChunkGridDimension::Fixed(size) => Some(index / size.get()),
                RectilinearChunkGridDimension::Varying(offsets_sizes) => {
                    let last = offsets_sizes.last()?;
                    if *index < last.offset + last.size.get() {
                        let partition = offsets_sizes
                            .partition_point(|offset_size| *index >= offset_size.offset);
                        partition.checked_sub(1).map(|p| p as u64)
                    } else {
                        None
                    }
                }
            })
            .collect())
    }

    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        if array_indices.len() != self.dimensionality() {
            return Err(IncompatibleDimensionalityError::new(
                array_indices.len(),
                self.dimensionality(),
            ));
        }

        Ok(std::iter::zip(array_indices, &self.chunks)
            .map(|(index, chunk_dim)| match chunk_dim {
                RectilinearChunkGridDimension::Fixed(size) => {
                    let chunk_offset = (index / size.get()) * size.get();
                    Some(index - chunk_offset)
                }
                RectilinearChunkGridDimension::Varying(offsets_sizes) => {
                    let last = offsets_sizes.last()?;
                    if *index < last.offset + last.size.get() {
                        let partition = offsets_sizes
                            .partition_point(|offset_size| *index >= offset_size.offset);
                        let chunk_index = partition.checked_sub(1)?;
                        let chunk_offset = offsets_sizes[chunk_index].offset;
                        Some(index - chunk_offset)
                    } else {
                        None
                    }
                }
            })
            .collect())
    }

    fn array_indices_inbounds(&self, array_indices: &[u64]) -> bool {
        array_indices.len() == self.dimensionality()
            && itertools::izip!(array_indices, self.array_shape(), &self.chunks).all(
                |(array_index, array_size, chunk_dim)| {
                    (*array_size == 0 || array_index < array_size)
                        && match chunk_dim {
                            RectilinearChunkGridDimension::Fixed(_) => true,
                            RectilinearChunkGridDimension::Varying(offsets_sizes) => offsets_sizes
                                .last()
                                .is_some_and(|last| *array_index < last.offset + last.size.get()),
                        }
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use crate::array_subset::ArraySubset;

    use super::*;

    fn from_slice_u64(values: &[u64]) -> Result<Vec<RunLengthElement>, std::num::TryFromIntError> {
        values
            .iter()
            .map(|&v| NonZeroU64::try_from(v).map(RunLengthElement::Single))
            .collect()
    }

    #[test]
    fn chunk_grid_rectilinear() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<ChunkEdgeLengths> = vec![
            ChunkEdgeLengths::Varying(from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap()),
            ChunkEdgeLengths::Varying(from_slice_u64(&[10; 10]).unwrap()),
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
        assert!(RectilinearChunkGrid::new(vec![100; 3], &chunk_shapes).is_err()); // incompatible dimensionality
        assert!(RectilinearChunkGrid::new(vec![123, 100], &chunk_shapes).is_err());
        // incompatible chunk sizes
    }

    #[test]
    fn chunk_grid_rectilinear_out_of_bounds() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<ChunkEdgeLengths> = vec![
            ChunkEdgeLengths::Varying(from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap()),
            ChunkEdgeLengths::Varying(from_slice_u64(&[10; 10]).unwrap()),
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
    fn chunk_grid_rectilinear_deserialization() {
        use zarrs_metadata_ext::chunk_grid::rectilinear::RectilinearChunkGridConfiguration;

        // Test that JSON with run-length encoding, scalars, and arrays can be deserialized
        let json = r#"
        {
            "kind": "inline",
            "chunk_shapes": [[[5, 3], [15, 2], 20, 35], 10]
        }
        "#;

        let config: RectilinearChunkGridConfiguration = serde_json::from_str(json).unwrap();
        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = &config;

        // First dimension should be varying (array with RLE)
        assert!(matches!(&chunk_shapes[0], ChunkEdgeLengths::Varying(_)));

        // Second dimension should be scalar
        assert!(matches!(&chunk_shapes[1], ChunkEdgeLengths::Scalar(v) if v.get() == 10));

        // Verify that expand_varying_chunks correctly expands RunLengthElement::Repeated
        // The RLE [[5, 3], [15, 2], 20, 35] should expand to [5, 5, 5, 15, 15, 20, 35]
        // Total = 5+5+5+15+15+20+35 = 100
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_grid = RectilinearChunkGrid::new(array_shape.clone(), &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.grid_shape(), &[7, 10]);

        // Verify chunk origins to ensure expansion worked correctly
        assert_eq!(chunk_grid.chunk_origin(&[0, 0]).unwrap(), Some(vec![0, 0]));
        assert_eq!(chunk_grid.chunk_origin(&[1, 0]).unwrap(), Some(vec![5, 0]));
        assert_eq!(chunk_grid.chunk_origin(&[2, 0]).unwrap(), Some(vec![10, 0]));
        assert_eq!(chunk_grid.chunk_origin(&[3, 0]).unwrap(), Some(vec![15, 0])); // After 3 chunks of 5
        assert_eq!(chunk_grid.chunk_origin(&[4, 0]).unwrap(), Some(vec![30, 0])); // After 3×5 + 15
        assert_eq!(chunk_grid.chunk_origin(&[5, 0]).unwrap(), Some(vec![45, 0])); // After 3×5 + 2×15
        assert_eq!(chunk_grid.chunk_origin(&[6, 0]).unwrap(), Some(vec![65, 0]));
        // After 3×5 + 2×15 + 20
    }

    #[test]
    fn chunk_grid_rectilinear_metadata_compression() {
        let array_shape: ArrayShape = vec![100, 100];

        // Create a chunk grid with repeated chunk sizes
        let chunk_shapes: Vec<ChunkEdgeLengths> = vec![
            ChunkEdgeLengths::Varying(from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap()),
            ChunkEdgeLengths::Varying(from_slice_u64(&[10; 10]).unwrap()),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        // Get the metadata
        let metadata = chunk_grid.create_metadata();
        let config: RectilinearChunkGridConfiguration = metadata.to_configuration().unwrap();

        // Verify the metadata is run-length encoded
        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = config;

        // First dimension should be compressed: [[5, 3], [15, 2], 20, 35]
        let elements = match &chunk_shapes[0] {
            ChunkEdgeLengths::Varying(elements) => elements,
            _ => panic!("Expected Varying"),
        };
        assert_eq!(elements.len(), 4);
        assert!(
            matches!(&elements[0], RunLengthElement::Repeated([val, count]) if val.get() == 5 && count.get() == 3)
        );
        assert!(
            matches!(&elements[1], RunLengthElement::Repeated([val, count]) if val.get() == 15 && count.get() == 2)
        );
        assert!(matches!(&elements[2], RunLengthElement::Single(val) if val.get() == 20));
        assert!(matches!(&elements[3], RunLengthElement::Single(val) if val.get() == 35));

        // Second dimension should be compressed: [[10, 10]]
        let elements = match &chunk_shapes[1] {
            ChunkEdgeLengths::Varying(elements) => elements,
            _ => panic!("Expected Varying"),
        };
        assert_eq!(elements.len(), 1);
        assert!(
            matches!(&elements[0], RunLengthElement::Repeated([val, count]) if val.get() == 10 && count.get() == 10)
        );
    }

    #[test]
    fn chunk_grid_rectilinear_unlimited() {
        let array_shape: ArrayShape = vec![100, 0];
        let chunk_shapes: Vec<ChunkEdgeLengths> = vec![
            ChunkEdgeLengths::Varying(from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap()),
            ChunkEdgeLengths::Scalar(NonZeroU64::new(10).unwrap()),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.grid_shape(), &[7, 0]);

        // Array indices beyond explicit chunks in the first dimension should be out of bounds
        let array_indices: ArrayIndices = vec![101, 150];
        assert!(chunk_grid.chunk_indices(&array_indices).unwrap().is_none());

        // But chunk indices within bounds for the first dimension are valid
        let chunk_indices: ArrayShape = vec![6, 9];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert!(chunk_grid.chunk_origin(&chunk_indices).unwrap().is_some());

        // Out of bounds in first dimension
        let chunk_indices: ArrayShape = vec![7, 9];
        assert!(!chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert!(chunk_grid.chunk_origin(&chunk_indices).unwrap().is_none());

        // Any chunk index is valid for unlimited dimension (second dimension)
        let chunk_indices: ArrayShape = vec![6, 123];
        assert!(chunk_grid.chunk_indices_inbounds(&chunk_indices));
        assert_eq!(
            chunk_grid.chunk_origin(&chunk_indices).unwrap(),
            Some(vec![65, 1230]) // 65 = 5+5+5+15+15+20, 1230 = 123*10
        );

        // Test chunk shape for unlimited dimension
        assert_eq!(
            chunk_grid.chunk_shape(&[6, 100]).unwrap(),
            Some(vec![NonZeroU64::new(35).unwrap(), NonZeroU64::new(10).unwrap()].into())
        );
    }

    #[test]
    fn chunk_grid_rectilinear_scalar() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<ChunkEdgeLengths> = vec![
            ChunkEdgeLengths::Scalar(NonZeroU64::new(10).unwrap()),
            ChunkEdgeLengths::Scalar(NonZeroU64::new(20).unwrap()),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.dimensionality(), 2);
        assert_eq!(chunk_grid.grid_shape(), &[10, 5]);

        // Test chunk indices calculation for scalar chunks
        assert_eq!(
            chunk_grid.chunk_indices(&[25, 45]).unwrap(),
            Some(vec![2, 2])
        );

        // Test chunk origin for scalar chunks
        assert_eq!(
            chunk_grid.chunk_origin(&[2, 2]).unwrap(),
            Some(vec![20, 40])
        );

        // Test chunk shape for scalar chunks (all chunks same size)
        assert_eq!(
            chunk_grid.chunk_shape(&[0, 0]).unwrap(),
            Some(vec![NonZeroU64::new(10).unwrap(), NonZeroU64::new(20).unwrap()].into())
        );
        assert_eq!(
            chunk_grid.chunk_shape(&[9, 4]).unwrap(),
            Some(vec![NonZeroU64::new(10).unwrap(), NonZeroU64::new(20).unwrap()].into())
        );

        // Test chunk element indices
        assert_eq!(
            chunk_grid.chunk_element_indices(&[25, 45]).unwrap(),
            Some(vec![5, 5])
        );
    }

    #[test]
    fn chunk_grid_rectilinear_mixed_scalar_and_varying() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<ChunkEdgeLengths> = vec![
            ChunkEdgeLengths::Scalar(NonZeroU64::new(10).unwrap()),
            ChunkEdgeLengths::Varying(from_slice_u64(&[5, 5, 5, 15, 15, 20, 35]).unwrap()),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes).unwrap();

        assert_eq!(chunk_grid.dimensionality(), 2);
        assert_eq!(chunk_grid.grid_shape(), &[10, 7]);

        // Scalar dimension behaves like a regular grid
        assert_eq!(
            chunk_grid.chunk_indices(&[25, 17]).unwrap(),
            Some(vec![2, 3])
        );
        assert_eq!(
            chunk_grid.chunk_origin(&[2, 3]).unwrap(),
            Some(vec![20, 15])
        );
    }

    #[test]
    fn chunk_grid_rectilinear_scalar_round_trip() {
        let array_shape: ArrayShape = vec![100, 100];
        let chunk_shapes: Vec<ChunkEdgeLengths> =
            vec![ChunkEdgeLengths::Scalar(NonZeroU64::new(10).unwrap()); 2];
        let chunk_grid = RectilinearChunkGrid::new(array_shape.clone(), &chunk_shapes).unwrap();

        // Serialize and deserialize
        let metadata = chunk_grid.create_metadata();
        let config: RectilinearChunkGridConfiguration = metadata.to_configuration().unwrap();
        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = &config;

        // Round-trip: verify scalar is preserved and grid is identical
        let chunk_grid2 = RectilinearChunkGrid::new(array_shape, chunk_shapes).unwrap();
        assert_eq!(chunk_grid.grid_shape(), chunk_grid2.grid_shape());
        assert!(matches!(&chunk_shapes[0], ChunkEdgeLengths::Scalar(v) if v.get() == 10));
        assert!(matches!(&chunk_shapes[1], ChunkEdgeLengths::Scalar(v) if v.get() == 10));
    }

    #[test]
    fn chunk_grid_rectilinear_comprehensive() {
        // Test from_metadata, array_shape, chunk_shape_u64, array_indices_inbounds,
        // and error paths for chunk_indices, chunk_origin, chunk_shape, chunk_shape_u64

        // Setup a mixed scalar/varying chunk grid
        let array_shape: ArrayShape = vec![100, 100, 50];
        let chunk_shapes: Vec<ChunkEdgeLengths> = vec![
            ChunkEdgeLengths::Varying(from_slice_u64(&[10, 20, 30, 40]).unwrap()),
            ChunkEdgeLengths::Scalar(NonZeroU64::new(25).unwrap()),
            ChunkEdgeLengths::Varying(from_slice_u64(&[5, 15, 30]).unwrap()),
        ];
        let chunk_grid = RectilinearChunkGrid::new(array_shape.clone(), &chunk_shapes).unwrap();

        // Test from_metadata
        let metadata = chunk_grid.create_metadata();
        let reconstructed_grid = ChunkGrid::from_metadata(&metadata, &array_shape)
            .expect("Failed to create chunk grid from metadata");

        // Verify the reconstructed grid has the same properties
        assert_eq!(reconstructed_grid.dimensionality(), 3);
        assert_eq!(reconstructed_grid.array_shape(), &array_shape);
        assert_eq!(reconstructed_grid.grid_shape(), &[4, 4, 3]);

        // Test array_shape
        assert_eq!(chunk_grid.array_shape(), &array_shape);
        assert_eq!(chunk_grid.array_shape(), &vec![100, 100, 50]);

        // Test chunk_shape_u64 with valid indices
        assert_eq!(
            chunk_grid.chunk_shape_u64(&[0, 0, 0]).unwrap(),
            Some(vec![10, 25, 5])
        );
        assert_eq!(
            chunk_grid.chunk_shape_u64(&[1, 2, 1]).unwrap(),
            Some(vec![20, 25, 15])
        );
        assert_eq!(
            chunk_grid.chunk_shape_u64(&[3, 3, 2]).unwrap(),
            Some(vec![40, 25, 30])
        );

        // Test chunk_shape_u64 with out-of-bounds chunk indices (returns None)
        assert_eq!(chunk_grid.chunk_shape_u64(&[4, 0, 0]).unwrap(), None); // First dim out of bounds
        assert_eq!(chunk_grid.chunk_shape_u64(&[0, 0, 3]).unwrap(), None); // Third dim out of bounds

        // Test array_indices_inbounds with valid indices
        assert!(chunk_grid.array_indices_inbounds(&[0, 0, 0]));
        assert!(chunk_grid.array_indices_inbounds(&[50, 50, 25]));
        assert!(chunk_grid.array_indices_inbounds(&[99, 99, 49]));

        // Test array_indices_inbounds with out-of-bounds indices
        assert!(!chunk_grid.array_indices_inbounds(&[100, 50, 25])); // First dim out
        assert!(!chunk_grid.array_indices_inbounds(&[50, 100, 25])); // Second dim out
        assert!(!chunk_grid.array_indices_inbounds(&[50, 50, 50])); // Third dim out

        // Test array_indices_inbounds with wrong dimensionality
        assert!(!chunk_grid.array_indices_inbounds(&[50, 50])); // Too few dimensions
        assert!(!chunk_grid.array_indices_inbounds(&[50, 50, 25, 0])); // Too many dimensions

        // Error paths: Test chunk_indices with incompatible dimensionality
        let result = chunk_grid.chunk_indices(&[50, 50]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        let result = chunk_grid.chunk_indices(&[50, 50, 25, 0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        // Error paths: Test chunk_origin with incompatible dimensionality
        let result = chunk_grid.chunk_origin(&[1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        let result = chunk_grid.chunk_origin(&[1, 1, 1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        // Error paths: Test chunk_shape with incompatible dimensionality
        let result = chunk_grid.chunk_shape(&[1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        let result = chunk_grid.chunk_shape(&[1, 1, 1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        // Error paths: Test chunk_shape_u64 with incompatible dimensionality
        let result = chunk_grid.chunk_shape_u64(&[1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        let result = chunk_grid.chunk_shape_u64(&[1, 1, 1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        // Error paths: Test chunk_element_indices with incompatible dimensionality
        let result = chunk_grid.chunk_element_indices(&[50, 50]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        let result = chunk_grid.chunk_element_indices(&[50, 50, 25, 0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IncompatibleDimensionalityError { .. }
        ));

        // Verify correct dimensionality returns Ok (even if indices are out of bounds)
        let result = chunk_grid.chunk_indices(&[150, 150, 150]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None); // Out of bounds returns None

        let result = chunk_grid.chunk_origin(&[10, 10, 10]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None); // Out of bounds chunk indices return None

        let result = chunk_grid.chunk_shape(&[10, 10, 10]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None); // Out of bounds chunk indices return None

        let result = chunk_grid.chunk_shape_u64(&[10, 10, 10]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None); // Out of bounds chunk indices return None
    }
}

use std::ops::IndexMut;

use derive_more::derive::Display;
use itertools::Itertools;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_data_type::DataTypeFillValueError;

use super::{
    ArrayBytesFixedDisjointView, DataType, FillValue,
    codec::{ArrayBytesDecodeIntoTarget, CodecError, InvalidBytesLengthError},
    ravel_indices,
};
use crate::{
    array_subset::ArraySubset,
    indexer::{IncompatibleIndexerError, Indexer},
    metadata::DataTypeSize,
    storage::byte_range::extract_byte_ranges_concat,
};

/// Count the nesting depth of optional types.
/// Returns 0 for non-optional types, 1 for `Option<T>`, 2 for `Option<Option<T>>`, etc.
pub(super) fn optional_nesting_depth(data_type: &DataType) -> usize {
    if let DataType::Optional(inner) = data_type {
        1 + optional_nesting_depth(inner)
    } else {
        0
    }
}

/// Build a nested `ArrayBytesDecodeIntoTarget` for optional types.
/// The `mask_views` slice should be ordered from outermost to innermost mask.
pub(super) fn build_nested_optional_target<'a>(
    data_view: &'a mut ArrayBytesFixedDisjointView<'a>,
    mask_views: &'a mut [ArrayBytesFixedDisjointView<'a>],
) -> ArrayBytesDecodeIntoTarget<'a> {
    if mask_views.is_empty() {
        ArrayBytesDecodeIntoTarget::Fixed(data_view)
    } else {
        let (first_mask, rest_masks) = mask_views.split_first_mut().unwrap();
        ArrayBytesDecodeIntoTarget::Optional(
            Box::new(build_nested_optional_target(data_view, rest_masks)),
            first_mask,
        )
    }
}

mod array_bytes_offsets;
pub use array_bytes_offsets::{ArrayBytesOffsets, RawBytesOffsetsCreateError};

mod array_bytes_raw;
pub use array_bytes_raw::ArrayBytesRaw;

/// Deprecated alias for [`ArrayBytesRaw`].
#[deprecated(since = "0.23.0", note = "Renamed to ArrayBytesRaw")]
pub type RawBytes<'a> = ArrayBytesRaw<'a>;

/// Deprecated alias for [`ArrayBytesOffsets`].
#[deprecated(since = "0.23.0", note = "Renamed to ArrayBytesOffsets")]
pub type RawBytesOffsets<'a> = ArrayBytesOffsets<'a>;

mod array_bytes_variable_length;
pub use array_bytes_variable_length::ArrayBytesVariableLength;

mod array_bytes_optional;
pub use array_bytes_optional::ArrayBytesOptional;

/// Fixed or variable length array bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArrayBytes<'a> {
    /// Bytes for a fixed length array.
    ///
    /// These represent elements in C-contiguous order (i.e. row-major order) where the last dimension varies the fastest.
    Fixed(ArrayBytesRaw<'a>),
    /// Bytes and element byte offsets for a variable length array.
    Variable(ArrayBytesVariableLength<'a>),
    /// Bytes for an optional array (data with a validity mask).
    ///
    /// The data can be either Fixed or Variable length.
    /// The mask is validated at construction to have 1 byte per element.
    Optional(ArrayBytesOptional<'a>),
}

/// An error raised if variable length array bytes offsets are out of bounds.
#[derive(Clone, Debug, Display, Error)]
#[display("Offset {offset} is out of bounds for bytes of length {len}")]
pub struct RawBytesOffsetsOutOfBoundsError {
    offset: usize,
    len: usize,
}

/// Errors related to [`ArrayBytes<'_>`] and [`ArrayBytes`].
#[derive(Clone, Debug, Error)]
pub enum ArrayBytesError {
    /// Invalid use of a fixed length method.
    #[error("Used a fixed length (flen) method on a variable length (vlen) array")]
    UsedFixedLengthMethodOnVariableLengthArray,
}

impl<'a> ArrayBytes<'a> {
    /// Create a new fixed length array bytes from `bytes`.
    ///
    /// `bytes` must be C-contiguous.
    pub fn new_flen(bytes: impl Into<ArrayBytesRaw<'a>>) -> Self {
        Self::Fixed(bytes.into())
    }

    /// Create a new variable length array bytes from `bytes` and `offsets`.
    ///
    /// # Errors
    /// Returns a [`RawBytesOffsetsOutOfBoundsError`] if the last offset is out of bounds of the bytes.
    pub fn new_vlen(
        bytes: impl Into<ArrayBytesRaw<'a>>,
        offsets: ArrayBytesOffsets<'a>,
    ) -> Result<Self, RawBytesOffsetsOutOfBoundsError> {
        ArrayBytesVariableLength::new(bytes, offsets).map(Self::Variable)
    }

    /// Create a new variable length array bytes from `bytes` and `offsets` without checking the offsets.
    ///
    /// # Safety
    /// The last offset must be less than or equal to the length of the bytes.
    pub unsafe fn new_vlen_unchecked(
        bytes: impl Into<ArrayBytesRaw<'a>>,
        offsets: ArrayBytesOffsets<'a>,
    ) -> Self {
        Self::Variable(unsafe { ArrayBytesVariableLength::new_unchecked(bytes, offsets) })
    }

    /// Wrap the array bytes with an optional validity mask.
    ///
    /// This creates an `Optional` variant that contains the current array bytes and the provided mask.
    #[must_use]
    pub fn with_optional_mask(self, mask: impl Into<ArrayBytesRaw<'a>>) -> Self {
        Self::Optional(ArrayBytesOptional::new(self, mask))
    }

    /// Create a new [`ArrayBytes`] with `num_elements` composed entirely of the `fill_value`.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueError`] if the fill value is incompatible with the data type.
    ///
    /// # Panics
    /// Panics if `num_elements` exceeds [`usize::MAX`].
    pub fn new_fill_value(
        data_type: &DataType,
        num_elements: u64,
        fill_value: &FillValue,
    ) -> Result<Self, DataTypeFillValueError> {
        if let Some(opt) = data_type.as_optional() {
            let num_elements_usize = usize::try_from(num_elements).unwrap();
            if opt.is_fill_value_null(fill_value) {
                // Null fill value for optional type: create mask of all zeros
                let inner_fill_value = if opt.is_fixed() {
                    FillValue::from(vec![0u8; opt.fixed_size().unwrap()])
                } else {
                    FillValue::from(&[])
                };
                let mask = vec![0u8; num_elements_usize];
                return Ok(
                    ArrayBytes::new_fill_value(opt, num_elements, &inner_fill_value)?
                        .with_optional_mask(mask),
                );
            }
            // Non-null fill value for optional type: strip suffix and use inner bytes
            let inner_bytes = opt.fill_value_inner_bytes(fill_value);
            let inner_fill_value = FillValue::new(inner_bytes.to_vec());
            let mask = vec![1u8; num_elements_usize]; // all non-null
            return Ok(
                ArrayBytes::new_fill_value(opt, num_elements, &inner_fill_value)?
                    .with_optional_mask(mask),
            );
        }

        match data_type.size() {
            DataTypeSize::Fixed(data_type_size) => {
                let num_elements = usize::try_from(num_elements).unwrap();
                if fill_value.size() == data_type_size {
                    Ok(Self::new_flen(
                        fill_value.as_ne_bytes().repeat(num_elements),
                    ))
                } else {
                    Err(DataTypeFillValueError::new(
                        data_type.name(),
                        fill_value.clone(),
                    ))
                }
            }
            DataTypeSize::Variable => {
                let num_elements = usize::try_from(num_elements).unwrap();
                let offsets = unsafe {
                    // SAFETY: The offsets are monotonically increasing.
                    ArrayBytesOffsets::new_unchecked(
                        (0..=num_elements)
                            .map(|i| i * fill_value.size())
                            .collect::<Vec<_>>(),
                    )
                };
                Ok(unsafe {
                    // SAFETY: The last offset is equal to the length of the bytes
                    Self::new_vlen_unchecked(fill_value.as_ne_bytes().repeat(num_elements), offsets)
                })
            }
        }
    }

    /// Convert the array bytes into fixed size bytes.
    ///
    /// # Errors
    /// Returns a [`CodecError::ExpectedFixedLengthBytes`] if the bytes are variable length.
    pub fn into_fixed(self) -> Result<ArrayBytesRaw<'a>, CodecError> {
        match self {
            Self::Fixed(bytes) => Ok(bytes),
            Self::Variable(..) => Err(CodecError::ExpectedFixedLengthBytes),
            Self::Optional(..) => Err(CodecError::ExpectedNonOptionalBytes),
        }
    }

    /// Convert the array bytes into variable sized bytes and element byte offsets.
    ///
    /// # Errors
    /// Returns a [`CodecError::ExpectedVariableLengthBytes`] if the bytes are fixed length.
    pub fn into_variable(self) -> Result<ArrayBytesVariableLength<'a>, CodecError> {
        match self {
            Self::Fixed(..) => Err(CodecError::ExpectedVariableLengthBytes),
            Self::Variable(variable_length_bytes) => Ok(variable_length_bytes),
            Self::Optional(..) => Err(CodecError::ExpectedNonOptionalBytes),
        }
    }

    /// Convert the array bytes into optional data and validity mask.
    ///
    /// # Errors
    /// Returns a [`CodecError::ExpectedNonOptionalBytes`] if the bytes are not optional.
    pub fn into_optional(self) -> Result<ArrayBytesOptional<'a>, CodecError> {
        match self {
            Self::Optional(optional_bytes) => Ok(optional_bytes),
            Self::Fixed(..) | Self::Variable(..) => Err(CodecError::ExpectedNonOptionalBytes),
        }
    }

    /// Convert the array bytes into [`ArrayBytesOptional`].
    ///
    /// # Errors
    /// Returns a [`CodecError::ExpectedNonOptionalBytes`] if the bytes are not optional.
    pub fn into_optional_bytes(self) -> Result<ArrayBytesOptional<'a>, CodecError> {
        match self {
            Self::Optional(optional_bytes) => Ok(optional_bytes),
            Self::Fixed(..) | Self::Variable(..) => Err(CodecError::ExpectedNonOptionalBytes),
        }
    }

    /// Returns the size (in bytes) of the underlying element bytes.
    ///
    /// This only considers the size of the element bytes, and does not include the element offsets for a variable sized array or the mask for optional arrays.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Fixed(bytes) | Self::Variable(ArrayBytesVariableLength { bytes, offsets: _ }) => {
                bytes.len()
            }
            Self::Optional(optional_bytes) => optional_bytes.data().size(),
        }
    }

    /// Return the byte offsets for variable sized bytes. Returns [`None`] for fixed size bytes.
    #[must_use]
    pub fn offsets(&self) -> Option<&ArrayBytesOffsets<'a>> {
        match self {
            Self::Fixed(..) => None,
            Self::Variable(ArrayBytesVariableLength { offsets, .. }) => Some(offsets),
            Self::Optional(optional_bytes) => optional_bytes.data().offsets(),
        }
    }

    /// Convert into owned [`ArrayBytes<'static>`].
    #[must_use]
    pub fn into_owned(self) -> ArrayBytes<'static> {
        match self {
            Self::Fixed(bytes) => ArrayBytes::Fixed(bytes.into_owned().into()),
            Self::Variable(ArrayBytesVariableLength { bytes, offsets }) => {
                ArrayBytes::Variable(ArrayBytesVariableLength {
                    bytes: bytes.into_owned().into(),
                    offsets: offsets.into_owned(),
                })
            }
            Self::Optional(optional_bytes) => ArrayBytes::Optional(optional_bytes.into_owned()),
        }
    }

    /// Validate that the array has a valid encoding.
    ///
    /// For a fixed-length array, check it matches the expected size.
    /// For a variable-length array, check that the offsets are monotonically increasing and the largest offset is equal to the array length.
    ///
    /// # Errors
    /// Returns an error if the array is not valid.
    pub fn validate(&self, num_elements: u64, data_type: &DataType) -> Result<(), CodecError> {
        validate_bytes(self, num_elements, data_type)
    }

    /// Returns [`true`] if the array is empty for the given fill value.
    #[must_use]
    pub fn is_fill_value(&self, fill_value: &FillValue) -> bool {
        match self {
            Self::Fixed(bytes) => fill_value.equals_all(bytes),
            Self::Variable(ArrayBytesVariableLength { bytes, offsets: _ }) => {
                fill_value.equals_all(bytes)
            }
            Self::Optional(optional_bytes) => {
                let bytes = fill_value.as_ne_bytes();
                // For optional types, check the suffix byte (last byte)
                let is_null = bytes.last() == Some(&0);
                if is_null {
                    // For optional arrays with null fill value, check if mask is all zeros
                    optional_bytes.mask().iter().all(|&b| b == 0)
                } else {
                    // For optional arrays with non-null fill value, check if mask is all ones
                    // and data matches inner fill value (fill value without suffix)
                    let inner_fill_value = if bytes.is_empty() {
                        FillValue::new(vec![])
                    } else {
                        FillValue::new(bytes[..bytes.len() - 1].to_vec())
                    };
                    optional_bytes.mask().iter().all(|&b| b == 1)
                        && optional_bytes.data().is_fill_value(&inner_fill_value)
                }
            }
        }
    }

    /// Extract a subset of the array bytes.
    ///
    /// # Errors
    /// Returns a [`CodecError::IncompatibleIndexer`] if the `indexer` is incompatible with `subset`.
    ///
    /// # Panics
    /// Panics if indices in the subset exceed [`usize::MAX`].
    pub fn extract_array_subset(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        array_shape: &[u64],
        data_type: &DataType,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        match self {
            ArrayBytes::Variable(ArrayBytesVariableLength { bytes, offsets }) => {
                let num_elements = indexer.len();
                let indices: Vec<_> = indexer.iter_linearised_indices(array_shape)?.collect();
                let mut bytes_length = 0;
                for index in &indices {
                    let index = usize::try_from(*index).unwrap();
                    let curr = offsets[index];
                    let next = offsets[index + 1];
                    debug_assert!(next >= curr);
                    bytes_length += next - curr;
                }
                let mut ss_bytes = Vec::with_capacity(bytes_length);
                let mut ss_offsets = Vec::with_capacity(usize::try_from(1 + num_elements).unwrap());
                for index in &indices {
                    let index = usize::try_from(*index).unwrap();
                    let curr = offsets[index];
                    let next = offsets[index + 1];
                    ss_offsets.push(ss_bytes.len());
                    ss_bytes.extend_from_slice(&bytes[curr..next]);
                }
                ss_offsets.push(ss_bytes.len());
                let ss_offsets = unsafe {
                    // SAFETY: The offsets are monotonically increasing.
                    ArrayBytesOffsets::new_unchecked(ss_offsets)
                };
                let array_bytes = unsafe {
                    // SAFETY: The last offset is equal to the length of the bytes
                    ArrayBytes::new_vlen_unchecked(ss_bytes, ss_offsets)
                };
                Ok(array_bytes)
            }
            ArrayBytes::Fixed(bytes) => {
                let byte_ranges = indexer
                    .iter_contiguous_byte_ranges(array_shape, data_type.fixed_size().unwrap())?;
                let bytes = extract_byte_ranges_concat(bytes, byte_ranges)?;
                Ok(ArrayBytes::new_flen(bytes))
            }
            ArrayBytes::Optional(optional_bytes) => {
                // Extract the inner data type from the optional data type
                let DataType::Optional(opt) = data_type else {
                    return Err(CodecError::Other(
                        "Optional array bytes requires an optional data type".to_string(),
                    ));
                };

                // Extract subset of the inner data recursively
                let subset_data =
                    optional_bytes
                        .data()
                        .extract_array_subset(indexer, array_shape, opt)?;

                // Extract subset of the mask (1 byte per element)
                let byte_ranges = indexer.iter_contiguous_byte_ranges(array_shape, 1)?; // mask is 1 byte per element
                let subset_mask = extract_byte_ranges_concat(optional_bytes.mask(), byte_ranges)?;

                Ok(subset_data.with_optional_mask(subset_mask))
            }
        }
    }
}

/// Validate fixed length array bytes for a given array size.
fn validate_bytes_flen(
    bytes: &ArrayBytesRaw,
    array_size: usize,
) -> Result<(), InvalidBytesLengthError> {
    if bytes.len() == array_size {
        Ok(())
    } else {
        Err(InvalidBytesLengthError::new(bytes.len(), array_size))
    }
}

/// Validate variable length array bytes for an array with `num_elements`.
fn validate_bytes_vlen(
    bytes: &ArrayBytesRaw,
    offsets: &ArrayBytesOffsets,
    num_elements: u64,
) -> Result<(), CodecError> {
    if offsets.len() as u64 != num_elements + 1 {
        return Err(CodecError::InvalidVariableSizedArrayOffsets);
    }
    let len = bytes.len();
    let mut offset_last = 0;
    for offset in offsets.iter() {
        if *offset < offset_last || *offset > len {
            return Err(CodecError::InvalidVariableSizedArrayOffsets);
        }
        offset_last = *offset;
    }
    if offset_last == len {
        Ok(())
    } else {
        Err(CodecError::InvalidVariableSizedArrayOffsets)
    }
}

/// Validate array bytes.
fn validate_bytes(
    bytes: &ArrayBytes<'_>,
    num_elements: u64,
    data_type: &DataType,
) -> Result<(), CodecError> {
    match bytes {
        ArrayBytes::Fixed(bytes) => {
            if data_type.is_optional() {
                return Err(CodecError::Other(
                    "Used non-optional array bytes with an optional data type.".to_string(),
                ));
            }
            match data_type.size() {
                DataTypeSize::Fixed(data_type_size) => Ok(validate_bytes_flen(
                    bytes,
                    usize::try_from(num_elements * data_type_size as u64).unwrap(),
                )?),
                DataTypeSize::Variable => Err(CodecError::Other(
                    "Used fixed length array bytes with a variable sized data type.".to_string(),
                )),
            }
        }
        ArrayBytes::Variable(ArrayBytesVariableLength { bytes, offsets }) => {
            if data_type.is_optional() {
                return Err(CodecError::Other(
                    "Used non-optional array bytes with an optional data type.".to_string(),
                ));
            }
            match data_type.size() {
                DataTypeSize::Variable => validate_bytes_vlen(bytes, offsets, num_elements),
                DataTypeSize::Fixed(_) => Err(CodecError::Other(
                    "Used variable length array bytes with a fixed length data type.".to_string(),
                )),
            }
        }
        ArrayBytes::Optional(optional_bytes) => {
            let DataType::Optional(inner_type) = data_type else {
                return Err(CodecError::Other(
                    "Used optional array bytes with a non-optional data type.".to_string(),
                ));
            };
            // Mask validation is already done at construction time
            // Just validate the underlying data with the inner type
            validate_bytes(optional_bytes.data(), num_elements, inner_type)
        }
    }
}

fn update_bytes_vlen_array_subset<'a>(
    input_bytes: &ArrayBytesRaw,
    input_offsets: &ArrayBytesOffsets,
    input_shape: &[u64],
    update_bytes: &ArrayBytesRaw,
    update_offsets: &ArrayBytesOffsets,
    update_subset: &ArraySubset,
) -> Result<ArrayBytes<'a>, IncompatibleIndexerError> {
    if !update_subset.inbounds_shape(input_shape) {
        return Err(IncompatibleIndexerError::new_oob(
            update_subset.end_exc(),
            input_shape.to_vec(),
        ));
    }

    // Get the current and new length of the bytes in the chunk subset
    let size_subset_new = update_offsets
        .iter()
        .tuple_windows()
        .map(|(curr, next)| next - curr)
        .sum::<usize>();
    let size_subset_old = {
        let chunk_indices = update_subset.linearised_indices(input_shape).unwrap();
        chunk_indices
            .iter()
            .map(|index| {
                let index = usize::try_from(index).unwrap();
                input_offsets[index + 1] - input_offsets[index]
            })
            .sum::<usize>()
    };

    // Populate new offsets and bytes
    let mut offsets_new = Vec::with_capacity(input_offsets.len());
    let bytes_new_len = (input_bytes.len() + size_subset_new)
        .checked_sub(size_subset_old)
        .unwrap();
    let mut bytes_new = Vec::with_capacity(bytes_new_len);
    let indices = ArraySubset::new_with_shape(input_shape.to_vec()).indices();
    for (chunk_index, indices) in indices.iter().enumerate() {
        offsets_new.push(bytes_new.len());
        if update_subset.contains(&indices) {
            let subset_indices = indices
                .iter()
                .zip(update_subset.start())
                .map(|(i, s)| i - s)
                .collect::<Vec<_>>();
            let subset_index =
                ravel_indices(&subset_indices, update_subset.shape()).expect("inbounds indices");
            let subset_index = usize::try_from(subset_index).unwrap();
            let start = update_offsets[subset_index];
            let end = update_offsets[subset_index + 1];
            bytes_new.extend_from_slice(&update_bytes[start..end]);
        } else {
            let start = input_offsets[chunk_index];
            let end = input_offsets[chunk_index + 1];
            bytes_new.extend_from_slice(&input_bytes[start..end]);
        }
    }
    offsets_new.push(bytes_new.len());
    let offsets_new = unsafe {
        // SAFETY: The offsets are monotonically increasing.
        ArrayBytesOffsets::new_unchecked(offsets_new)
    };
    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytes::new_vlen_unchecked(bytes_new, offsets_new)
    };
    Ok(array_bytes)
}

fn update_bytes_vlen_indexer<'a>(
    input_bytes: &ArrayBytesRaw,
    input_offsets: &ArrayBytesOffsets,
    input_shape: &[u64],
    update_bytes: &ArrayBytesRaw,
    update_offsets: &ArrayBytesOffsets,
    update_indexer: &dyn Indexer,
) -> Result<ArrayBytes<'a>, IncompatibleIndexerError> {
    // Get the size of the new bytes
    let updated_size_new = update_bytes.len();
    debug_assert_eq!(
        updated_size_new,
        update_offsets
            .iter()
            .tuple_windows()
            .map(|(curr, next)| next - curr)
            .sum::<usize>()
    );

    // Get the indices of elements to update and the size of the old bytes being replaced
    let num_elements = usize::try_from(input_shape.iter().product::<u64>()).unwrap();
    let update_indices = update_indexer.iter_linearised_indices(input_shape)?;
    let mut element_indices_update: Vec<Option<usize>> = vec![None; num_elements];
    let mut updated_size_old = 0;
    for (update_index, input_index) in update_indices.enumerate() {
        let input_index = usize::try_from(input_index).unwrap();
        updated_size_old += input_offsets[input_index + 1] - input_offsets[input_index];
        element_indices_update[input_index] = Some(update_index);
    }

    // Populate new offsets and bytes
    let mut offsets_new = Vec::with_capacity(input_offsets.len());
    let bytes_new_len = (input_bytes.len() + updated_size_new)
        .checked_sub(updated_size_old)
        .unwrap();
    let mut bytes_new = Vec::with_capacity(bytes_new_len);
    for input_index in 0..num_elements {
        offsets_new.push(bytes_new.len());
        if let Some(update_index) = element_indices_update[input_index] {
            let start = update_offsets[update_index];
            let end = update_offsets[update_index + 1];
            bytes_new.extend_from_slice(&update_bytes[start..end]);
        } else {
            let start = input_offsets[input_index];
            let end = input_offsets[input_index + 1];
            bytes_new.extend_from_slice(&input_bytes[start..end]);
        }
    }
    offsets_new.push(bytes_new.len());
    let offsets_new = unsafe {
        // SAFETY: The offsets are monotonically increasing.
        ArrayBytesOffsets::new_unchecked(offsets_new)
    };
    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytes::new_vlen_unchecked(bytes_new, offsets_new)
    };
    Ok(array_bytes)
}

/// Update array bytes. Specialised for `ArraySubset`.
///
/// # Errors
/// Returns a [`CodecError`] if
/// - `bytes` are not compatible with the `shape` and `data_type_size`,
/// - `update_bytes` are not compatible with the `update_subset` and `data_type_size`,
/// - `update_subset` is not within the bounds of `shape`
fn update_array_bytes_array_subset<'a>(
    bytes: ArrayBytes,
    shape: &[u64],
    update_subset: &ArraySubset,
    update_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (bytes, update_bytes, data_type_size) {
        (
            ArrayBytes::Variable(ArrayBytesVariableLength { bytes, offsets }),
            ArrayBytes::Variable(ArrayBytesVariableLength {
                bytes: update_bytes,
                offsets: update_offsets,
            }),
            DataTypeSize::Variable,
        ) => Ok(update_bytes_vlen_array_subset(
            &bytes,
            &offsets,
            shape,
            update_bytes,
            update_offsets,
            update_subset,
        )?),
        (
            ArrayBytes::Fixed(bytes),
            ArrayBytes::Fixed(update_bytes),
            DataTypeSize::Fixed(data_type_size),
        ) => {
            let mut bytes = bytes.into_owned();
            let mut output_view: ArrayBytesFixedDisjointView<'_> = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    data_type_size,
                    shape,
                    update_subset.clone(),
                )
            }
            .map_err(CodecError::from)?;
            output_view.copy_from_slice(update_bytes)?;
            Ok(ArrayBytes::new_flen(bytes))
        }
        (
            ArrayBytes::Optional(optional_bytes),
            ArrayBytes::Optional(update_optional_bytes),
            data_type_size,
        ) => {
            // Update the data recursively
            let (data, mask) = optional_bytes.into_parts();
            let data_after_update = update_array_bytes_array_subset(
                *data,
                shape,
                update_subset,
                update_optional_bytes.data(),
                data_type_size,
            )?;

            // Update the mask (it's a fixed-size bool array, 1 byte per element)
            let mut mask = mask.into_owned();
            let mut mask_view: ArrayBytesFixedDisjointView<'_> = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut mask),
                    1, // bool is 1 byte per element
                    shape,
                    update_subset.clone(),
                )
            }
            .map_err(CodecError::from)?;
            mask_view.copy_from_slice(update_optional_bytes.mask())?;

            Ok(ArrayBytes::Optional(ArrayBytesOptional::new(
                data_after_update,
                mask,
            )))
        }
        (_, _, DataTypeSize::Variable) => Err(CodecError::ExpectedVariableLengthBytes),
        (_, _, DataTypeSize::Fixed(_)) => Err(CodecError::ExpectedFixedLengthBytes),
    }
}

/// Update array bytes. Specialised for `Indexer`.
///
/// # Errors
/// Returns a [`CodecError`] if
/// - `bytes` are not compatible with the `shape` and `data_type_size`,
/// - `update_bytes` are not compatible with the `update_indexer` and `data_type_size`,
/// - `update_indexer` is not within the bounds of `shape`
fn update_array_bytes_indexer<'a>(
    bytes: ArrayBytes,
    shape: &[u64],
    update_indexer: &dyn crate::indexer::Indexer,
    update_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (bytes, update_bytes, data_type_size) {
        (
            ArrayBytes::Variable(ArrayBytesVariableLength { bytes, offsets }),
            ArrayBytes::Variable(ArrayBytesVariableLength {
                bytes: update_bytes,
                offsets: update_offsets,
            }),
            DataTypeSize::Variable,
        ) => Ok(update_bytes_vlen_indexer(
            &bytes,
            &offsets,
            shape,
            update_bytes,
            update_offsets,
            update_indexer,
        )?),
        (
            ArrayBytes::Fixed(bytes),
            ArrayBytes::Fixed(update_bytes),
            DataTypeSize::Fixed(data_type_size),
        ) => {
            let mut bytes = bytes.into_owned();
            let byte_ranges = update_indexer.iter_contiguous_byte_ranges(shape, data_type_size)?;
            let mut offset: usize = 0;
            for byte_range in byte_ranges {
                let start = usize::try_from(byte_range.start).unwrap();
                let end = usize::try_from(byte_range.end).unwrap();
                let byte_range_len = end.saturating_sub(start);
                bytes
                    .index_mut(start..end)
                    .copy_from_slice(&update_bytes[offset..offset + byte_range_len]);
                offset += byte_range_len;
            }
            Ok(ArrayBytes::new_flen(bytes))
        }
        (
            ArrayBytes::Optional(optional_bytes),
            ArrayBytes::Optional(update_optional_bytes),
            data_type_size,
        ) => {
            // Update the data recursively
            let (data, mask) = optional_bytes.into_parts();
            let data_after_update = update_array_bytes(
                *data,
                shape,
                update_indexer,
                update_optional_bytes.data(),
                data_type_size,
            )?;

            // Update the mask (it's a fixed-size bool array, 1 byte per element)
            let mut mask = mask.into_owned();
            let byte_ranges = update_indexer.iter_contiguous_byte_ranges(shape, 1)?; // 1 byte per bool
            let mut offset: usize = 0;
            for byte_range in byte_ranges {
                let start = usize::try_from(byte_range.start).unwrap();
                let end = usize::try_from(byte_range.end).unwrap();
                let byte_range_len = end.saturating_sub(start);
                mask.index_mut(start..end).copy_from_slice(
                    &update_optional_bytes.mask()[offset..offset + byte_range_len],
                );
                offset += byte_range_len;
            }

            Ok(ArrayBytes::Optional(ArrayBytesOptional::new(
                data_after_update,
                mask,
            )))
        }
        (_, _, DataTypeSize::Variable) => Err(CodecError::ExpectedVariableLengthBytes),
        (_, _, DataTypeSize::Fixed(_)) => Err(CodecError::ExpectedFixedLengthBytes),
    }
}

/// Update array bytes.
///
/// This function is used internally by [`crate::array::Array::store_chunk_subset_opt`] and [`crate::array::Array::async_store_chunk_subset_opt`].
///
/// # Errors
/// Returns a [`CodecError`] if
/// - `bytes` are not compatible with the `shape` and `data_type_size`,
/// - `update_bytes` are not compatible with the `update_indexer` and `data_type_size`,
/// - `update_indexer` is not within the bounds of `shape`
///
/// # Panics
/// Panics if the indexer references bytes beyond [`usize::MAX`].
pub fn update_array_bytes<'a>(
    bytes: ArrayBytes,
    shape: &[u64],
    update_indexer: &dyn crate::indexer::Indexer,
    update_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(output_subset) = update_indexer.as_array_subset() {
        update_array_bytes_array_subset(bytes, shape, output_subset, update_bytes, data_type_size)
    } else {
        update_array_bytes_indexer(bytes, shape, update_indexer, update_bytes, data_type_size)
    }
}

/// Merge a set of chunks into an array subset.
///
/// This function is used internally by [`retrieve_array_subset_opt`] and [`async_retrieve_array_subset_opt`].
pub(crate) fn merge_chunks_vlen<'a>(
    chunk_bytes_and_subsets: Vec<(ArrayBytes<'_>, ArraySubset)>,
    array_shape: &[u64],
) -> Result<ArrayBytes<'a>, CodecError> {
    let num_elements = usize::try_from(array_shape.iter().product::<u64>()).unwrap();

    #[cfg(debug_assertions)]
    {
        // Validate the input
        let mut element_in_input = vec![0; num_elements];
        for (_, chunk_subset) in &chunk_bytes_and_subsets {
            // println!("{chunk_subset:?}");
            let indices = chunk_subset.linearised_indices(array_shape).unwrap();
            for idx in indices {
                let idx = usize::try_from(idx).unwrap();
                element_in_input[idx] += 1;
            }
        }
        assert!(element_in_input.iter().all(|v| *v == 1));
    }

    // Get the size of each element
    // TODO: Go parallel
    let mut element_sizes = vec![0; num_elements];
    for (chunk_bytes, chunk_subset) in &chunk_bytes_and_subsets {
        let chunk_offsets = chunk_bytes.offsets().unwrap();
        debug_assert_eq!(chunk_offsets.len() as u64, chunk_subset.num_elements() + 1);
        let indices = chunk_subset.linearised_indices(array_shape).unwrap();
        for (subset_idx, (curr, next)) in
            indices.iter().zip_eq(chunk_offsets.iter().tuple_windows())
        {
            debug_assert!(next >= curr);
            let subset_idx = usize::try_from(subset_idx).unwrap();
            element_sizes[subset_idx] = next - curr;
        }
    }

    // Convert to offsets with a cumulative sum
    // TODO: Parallel cum sum
    let mut offsets = Vec::with_capacity(element_sizes.len() + 1);
    offsets.push(0); // first offset is always zero
    offsets.extend(element_sizes.iter().scan(0, |acc, &sz| {
        *acc += sz;
        Some(*acc)
    }));
    let offsets = unsafe {
        // SAFETY: The offsets are monotonically increasing.
        ArrayBytesOffsets::new_unchecked(offsets)
    };

    // Write bytes
    // TODO: Go parallel
    let mut bytes = vec![0; offsets.last()];
    for (chunk_bytes, chunk_subset) in chunk_bytes_and_subsets {
        let (chunk_bytes, chunk_offsets) = chunk_bytes.into_variable()?.into_parts();
        let indices = chunk_subset.linearised_indices(array_shape).unwrap();
        for (subset_idx, (&chunk_curr, &chunk_next)) in
            indices.iter().zip_eq(chunk_offsets.iter().tuple_windows())
        {
            let subset_idx = usize::try_from(subset_idx).unwrap();
            let subset_curr = offsets[subset_idx];
            let subset_next = offsets[subset_idx + 1];
            bytes[subset_curr..subset_next].copy_from_slice(&chunk_bytes[chunk_curr..chunk_next]);
        }
    }

    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytes::new_vlen_unchecked(bytes, offsets)
    };

    Ok(array_bytes)
}

pub(crate) fn extract_decoded_regions_vlen<'a>(
    bytes: &[u8],
    offsets: &[usize],
    indexer: &dyn crate::indexer::Indexer,
    array_shape: &[u64],
) -> Result<ArrayBytes<'a>, CodecError> {
    let indices = indexer.iter_linearised_indices(array_shape)?;
    let indices: Vec<_> = indices.into_iter().collect();
    let mut region_bytes_len = 0;
    for index in &indices {
        let index = usize::try_from(*index).unwrap();
        let curr = offsets[index];
        let next = offsets[index + 1];
        debug_assert!(next >= curr);
        region_bytes_len += next - curr;
    }
    let mut region_offsets = Vec::with_capacity(usize::try_from(indexer.len() + 1).unwrap());
    let mut region_bytes = Vec::with_capacity(region_bytes_len);
    for index in &indices {
        region_offsets.push(region_bytes.len());
        let index = usize::try_from(*index).unwrap();
        let curr = offsets[index];
        let next = offsets[index + 1];
        region_bytes.extend_from_slice(&bytes[curr..next]);
    }
    region_offsets.push(region_bytes.len());
    let region_offsets = unsafe {
        // SAFETY: The offsets are monotonically increasing.
        ArrayBytesOffsets::new_unchecked(region_offsets)
    };
    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytes::new_vlen_unchecked(region_bytes, region_offsets)
    };
    Ok(array_bytes)
}

/// Decode the fill value into a subset of a preallocated output.
///
/// This method is intended for internal use by Array.
/// It currently only works for fixed length data types.
///
/// # Errors
/// Returns [`CodecError::ExpectedFixedLengthBytes`] for variable-sized data.
///
/// # Safety
/// The caller must ensure that:
///  - `data_type` and `fill_value` are compatible,
///  - `output` holds enough space for the preallocated bytes of an array with `output_shape` and `data_type`, and
///  - `output_subset` is within the bounds of `output_shape`.
pub fn copy_fill_value_into(
    data_type: &DataType,
    fill_value: &FillValue,
    output_target: ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), CodecError> {
    let num_elements = output_target.num_elements();
    let fill_value_bytes = ArrayBytes::new_fill_value(data_type, num_elements, fill_value)?;
    decode_into_array_bytes_target(&fill_value_bytes, output_target)
}

/// Helper function to decode `ArrayBytes` into a target, handling mask and data separately.
///
/// This function handles the common pattern of decoding `ArrayBytes` into an `ArrayBytesDecodeIntoTarget`,
/// properly handling optional data types.
pub(crate) fn decode_into_array_bytes_target(
    bytes: &ArrayBytes,
    target: ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), CodecError> {
    match (bytes, target) {
        // Fixed source → Fixed target
        (ArrayBytes::Fixed(data_bytes), ArrayBytesDecodeIntoTarget::Fixed(data)) => {
            data.copy_from_slice(data_bytes)?;
            Ok(())
        }

        // Optional source → Optional target (recursive)
        (
            ArrayBytes::Optional(optional_bytes),
            ArrayBytesDecodeIntoTarget::Optional(data_target, mask_target),
        ) => {
            // Recursively decode the inner data
            decode_into_array_bytes_target(optional_bytes.data(), *data_target)?;
            // Decode the mask
            mask_target.copy_from_slice(optional_bytes.mask())?;
            Ok(())
        }

        // Type mismatches
        (ArrayBytes::Variable(..), _) => Err(CodecError::ExpectedFixedLengthBytes),
        (ArrayBytes::Fixed(_), ArrayBytesDecodeIntoTarget::Optional(..)) => Err(CodecError::Other(
            "Cannot decode non-optional data into optional target".to_string(),
        )),
        (ArrayBytes::Optional(..), ArrayBytesDecodeIntoTarget::Fixed(_)) => Err(CodecError::Other(
            "Cannot decode optional data into non-optional target".to_string(),
        )),
    }
}

impl<'a> From<ArrayBytesRaw<'a>> for ArrayBytes<'a> {
    fn from(bytes: ArrayBytesRaw<'a>) -> Self {
        Self::new_flen(bytes)
    }
}

// impl<'a, 'b> From<&ArrayBytes<'a>> for ArrayBytes<'b> {
//     fn from(bytes: &ArrayBytes<'a>) -> Self {
//         match bytes {
//             Self::Fixed(bytes) => {
//                 let bytes = bytes.to_vec();
//                 ArrayBytes::<'b>::new_flen(bytes)
//             },
//             Self::Variable(bytes, offsets) => {
//                 let bytes: ArrayBytesRaw<'b> = bytes.to_vec().into();
//                 let offsets: ArrayBytesOffsets<'b> = offsets.to_vec().into();
//                 ArrayBytes::new_vlen(bytes, offsets)
//             }
//         }
//     }
// }

// impl<'a> From<ArrayBytes<'_>> for ArrayBytes<'a> {
//     fn from(bytes: ArrayBytes<'_>) -> Self {
//         match bytes {
//             ArrayBytes::Fixed(bytes) => ArrayBytes::new_flen(bytes)
//             ArrayBytes::Variable(bytes, offsets) => ArrayBytes::new_vlen(bytes, offsets)
//         }
//     }
// }

impl<'a> From<&'a [u8]> for ArrayBytes<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

impl From<Vec<u8>> for ArrayBytes<'_> {
    fn from(bytes: Vec<u8>) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for ArrayBytes<'a> {
    fn from(bytes: &'a [u8; N]) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::array::Element;

    #[test]
    fn array_bytes_flen() -> Result<(), Box<dyn Error>> {
        let data = vec![0u32, 1, 2, 3, 4];
        let n_elements = data.len();
        let bytes = Element::into_array_bytes(&DataType::UInt32, data)?;
        let ArrayBytes::Fixed(bytes) = bytes else {
            panic!()
        };
        assert_eq!(bytes.len(), size_of::<u32>() * n_elements);

        Ok(())
    }

    #[test]
    fn array_bytes_vlen() {
        let data = [0u8, 1, 2, 3, 4];
        assert!(ArrayBytes::new_vlen(&data, vec![0].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 5, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 5, 6].try_into().unwrap()).is_err());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 1, 3, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 1, 3, 6].try_into().unwrap()).is_err());
    }

    #[test]
    fn array_bytes_str() -> Result<(), Box<dyn Error>> {
        let data = vec!["a", "bb", "ccc"];
        let bytes = Element::into_array_bytes(&DataType::String, data)?;
        let (bytes, offsets) = bytes.into_variable().unwrap().into_parts();
        assert_eq!(bytes, "abbccc".as_bytes());
        assert_eq!(*offsets, [0, 1, 3, 6]);

        Ok(())
    }

    #[test]
    fn test_flen_update_subset() {
        let mut bytes_array = vec![0u8; 4 * 4];
        {
            let bytes_array = UnsafeCellSlice::new(&mut bytes_array);
            let mut output_non_overlapping_0 = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    bytes_array,
                    size_of::<u8>(),
                    &[4, 4],
                    ArraySubset::new_with_ranges(&[1..2, 1..3]),
                )
                .unwrap()
            };
            output_non_overlapping_0.copy_from_slice(&[1u8, 2]).unwrap();

            let mut output_non_overlapping_1 = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    bytes_array,
                    size_of::<u8>(),
                    &[4, 4],
                    ArraySubset::new_with_ranges(&[3..4, 0..2]),
                )
                .unwrap()
            };
            output_non_overlapping_1.copy_from_slice(&[3u8, 4]).unwrap();
        }

        debug_assert_eq!(
            bytes_array,
            vec![0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 3, 4, 0, 0]
        );
    }

    #[test]
    fn update_array_bytes_array_subset_optional() -> Result<(), Box<dyn std::error::Error>> {
        // Create initial 4x4 array with optional u8 data
        // Layout (row-major):
        // [1  2  3  4 ]
        // [5  6  N  8 ]
        // [9  N  11 12]
        // [N  14 15 16]
        // where N = None
        let initial_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let initial_mask = vec![1u8, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]; // 0 = None
        let initial_bytes = ArrayBytes::new_flen(initial_data).with_optional_mask(initial_mask);

        // Create 2x2 update for subset [1..3, 1..3]
        // This will update positions:
        // [1,1]=6, [1,2]=N, [2,1]=N, [2,2]=11
        // With new values:
        // [99 N ]
        // [97 96]
        let update_data = vec![99u8, 98, 97, 96];
        let update_mask = vec![1u8, 0, 1, 1]; // Second element is None
        let update_bytes = ArrayBytes::new_flen(update_data).with_optional_mask(update_mask);

        // Update using ArraySubset
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let result = update_array_bytes(
            initial_bytes,
            &[4, 4],
            &subset,
            &update_bytes,
            DataTypeSize::Fixed(1),
        )?;

        // Update using indexer
        let indexer = vec![vec![3, 2], vec![3, 3]];
        let update_bytes =
            ArrayBytes::new_flen(vec![0u8, 255u8]).with_optional_mask(vec![0u8, 1u8]);
        let result = update_array_bytes(
            result,
            &[4, 4],
            &indexer,
            &update_bytes,
            DataTypeSize::Fixed(1),
        )?;

        // Verify result
        let (result_data, result_mask) = result.into_optional()?.into_parts();
        let ArrayBytes::Fixed(result_data) = *result_data else {
            panic!("Expected fixed bytes")
        };

        // Expected layout after update:
        // [1  2  3  4 ]
        // [5  99 N  8 ]
        // [9  97 96 12]
        // [N  14 N 255]

        // Verify updated positions
        assert_eq!(result_data[5], 99); // [1,1]
        assert_eq!(result_mask[5], 1);

        assert_eq!(result_mask[6], 0); // [1,2] - should be None

        assert_eq!(result_data[9], 97); // [2,1]
        assert_eq!(result_mask[9], 1);

        assert_eq!(result_data[10], 96); // [2,2]
        assert_eq!(result_mask[10], 1);

        // Verify unchanged positions
        assert_eq!(result_data[0], 1); // [0,0]
        assert_eq!(result_mask[0], 1);

        assert_eq!(result_mask[14], 0); // [3,2]

        assert_eq!(result_data[15], 255); // [3,3]
        assert_eq!(result_mask[15], 1);

        assert_eq!(result_mask[12], 0); // [3,0] - still None

        Ok(())
    }

    #[test]
    fn update_array_bytes_array_subset_nested_optional_2_level()
    -> Result<(), Box<dyn std::error::Error>> {
        // Create initial 4x4 array with Option<Option<u8>> data
        // Layout (row-major, S=Some, N=None, outer/inner):
        // [SS(1)  SS(2)  SN     NN   ]
        // [SS(5)  SS(6)  SS(7)  SS(8)]
        // [SN     SS(10) SS(11) SN   ]
        // [NN     SS(14) SS(15) SS(16)]
        let initial_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let initial_inner_mask = vec![1u8, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]; // inner Some/None
        let initial_outer_mask = vec![1u8, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]; // outer Some/None

        let initial_bytes = ArrayBytes::new_flen(initial_data)
            .with_optional_mask(initial_inner_mask)
            .with_optional_mask(initial_outer_mask);

        // Create 2x2 update for subset [1..3, 1..3]
        // Update positions [1,1], [1,2], [2,1], [2,2]
        // New values:
        // [SS(99) SN   ]
        // [NN     SS(96)]
        let update_data = vec![99u8, 98, 97, 96];
        let update_inner_mask = vec![1u8, 0, 1, 1]; // [99 is valid, 98 is inner None, ...]
        let update_outer_mask = vec![1u8, 1, 0, 1]; // [outer valid, outer valid, outer None, outer valid]

        let update_bytes = ArrayBytes::new_flen(update_data)
            .with_optional_mask(update_inner_mask)
            .with_optional_mask(update_outer_mask);

        // Update using ArraySubset
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let result = update_array_bytes(
            initial_bytes,
            &[4, 4],
            &subset,
            &update_bytes,
            DataTypeSize::Fixed(1),
        )?;

        // Verify result - extract outer optional layer
        let (result_middle, result_outer_mask) = result.into_optional()?.into_parts();

        // Extract inner optional layer
        let (result_data, result_inner_mask) = result_middle.into_optional()?.into_parts();
        let ArrayBytes::Fixed(result_data) = *result_data else {
            panic!("Expected fixed bytes")
        };

        // Verify updated positions
        // [1,1] = Some(Some(99))
        assert_eq!(result_data[5], 99);
        assert_eq!(result_inner_mask[5], 1);
        assert_eq!(result_outer_mask[5], 1);

        // [1,2] = Some(None)
        assert_eq!(result_inner_mask[6], 0); // inner None
        assert_eq!(result_outer_mask[6], 1); // outer Some

        // [2,1] = None (outer)
        assert_eq!(result_outer_mask[9], 0);

        // [2,2] = Some(Some(96))
        assert_eq!(result_data[10], 96);
        assert_eq!(result_inner_mask[10], 1);
        assert_eq!(result_outer_mask[10], 1);

        // Verify unchanged positions
        // [0,0] = Some(Some(1))
        assert_eq!(result_data[0], 1);
        assert_eq!(result_inner_mask[0], 1);
        assert_eq!(result_outer_mask[0], 1);

        // [0,2] = Some(None) - unchanged
        assert_eq!(result_inner_mask[2], 0);
        assert_eq!(result_outer_mask[2], 1);

        // [3,0] = None - unchanged
        assert_eq!(result_outer_mask[12], 0);

        Ok(())
    }
}

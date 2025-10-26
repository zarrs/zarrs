use std::{borrow::Cow, ops::IndexMut};

use derive_more::derive::Display;
use itertools::Itertools;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_data_type::DataTypeFillValueError;

use crate::{
    array_subset::ArraySubset,
    indexer::{IncompatibleIndexerError, Indexer},
    metadata::DataTypeSize,
    storage::byte_range::extract_byte_ranges_concat,
};

use super::{
    codec::{CodecError, InvalidBytesLengthError},
    ravel_indices, ArrayBytesFixedDisjointView, DataType, FillValue,
};

mod raw_bytes_offsets;
pub use raw_bytes_offsets::{RawBytesOffsets, RawBytesOffsetsCreateError};

/// Array element bytes.
///
/// These can represent:
/// - [`ArrayBytes::Fixed`]: fixed length elements of an array in C-contiguous order,
/// - [`ArrayBytes::Variable`]: variable length elements of an array in C-contiguous order with padding permitted,
/// - Encoded array bytes after an array to bytes or bytes to bytes codecs.
pub type RawBytes<'a> = Cow<'a, [u8]>;

/// Fixed or variable length array bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayBytes<'a> {
    /// Bytes for a fixed length array.
    ///
    /// These represent elements in C-contiguous order (i.e. row-major order) where the last dimension varies the fastest.
    Fixed(RawBytes<'a>),
    /// Bytes and element byte offsets for a variable length array.
    ///
    /// The bytes and offsets are modeled on the [Apache Arrow Variable-size Binary Layout](https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout).
    /// - The offsets buffer contains length + 1 ~~signed integers (either 32-bit or 64-bit, depending on the data type)~~ usize integers.
    /// - Offsets must be monotonically increasing, that is `offsets[j+1] >= offsets[j]` for `0 <= j < length`, even for null slots. Thus, the bytes represent C-contiguous elements with padding permitted.
    /// - The final offset must be less than or equal to the length of the bytes buffer.
    Variable(RawBytes<'a>, RawBytesOffsets<'a>),
    /// Bytes for an optional array (data with a validity mask).
    ///
    /// The data can be either Fixed or Variable length.
    /// The mask is a bitmask indicating which elements are valid/present.
    Optional(Box<ArrayBytes<'a>>, RawBytes<'a>),
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
    pub fn new_flen(bytes: impl Into<RawBytes<'a>>) -> Self {
        Self::Fixed(bytes.into())
    }

    /// Create a new variable length array bytes from `bytes` and `offsets`.
    ///
    /// # Errors
    /// Returns a [`RawBytesOffsetsOutOfBoundsError`] if the last offset is out of bounds of the bytes.
    pub fn new_vlen(
        bytes: impl Into<RawBytes<'a>>,
        offsets: RawBytesOffsets<'a>,
    ) -> Result<Self, RawBytesOffsetsOutOfBoundsError> {
        let bytes = bytes.into();
        if offsets.last() <= bytes.len() {
            Ok(Self::Variable(bytes, offsets))
        } else {
            Err(RawBytesOffsetsOutOfBoundsError {
                offset: offsets.last(),
                len: bytes.len(),
            })
        }
    }

    /// Create a new variable length array bytes from `bytes` and `offsets` without checking the offsets.
    ///
    /// # Safety
    /// The last offset must be less than or equal to the length of the bytes.
    pub unsafe fn new_vlen_unchecked(
        bytes: impl Into<RawBytes<'a>>,
        offsets: RawBytesOffsets<'a>,
    ) -> Self {
        let bytes = bytes.into();
        debug_assert!(offsets.last() <= bytes.len());
        Self::Variable(bytes, offsets)
    }

    /// Wrap the array bytes with an optional validity mask.
    ///
    /// This creates an `Optional` variant that contains the current array bytes and the provided mask.
    #[must_use]
    pub fn with_optional_mask(self, mask: impl Into<RawBytes<'a>>) -> Self {
        Self::Optional(Box::new(self), mask.into())
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
        if data_type.is_optional() && fill_value.is_null() {
            // For optional data types with null fill value, create an optional array with empty data and a zeroed mask
            todo!("Do this properly, data needs to be dense");
            let num_elements = usize::try_from(num_elements).unwrap();
            let data = ArrayBytes::new_flen(Vec::new());
            let mask = vec![0u8; num_elements];
            return Ok(ArrayBytes::Optional(Box::new(data), mask.into()));
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
                    RawBytesOffsets::new_unchecked(
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
    pub fn into_fixed(self) -> Result<RawBytes<'a>, CodecError> {
        match self {
            Self::Fixed(bytes) => Ok(bytes),
            Self::Variable(_, _) => Err(CodecError::ExpectedFixedLengthBytes),
            Self::Optional(_, _) => Err(CodecError::ExpectedNonOptionalBytes),
        }
    }

    /// Convert the array bytes into variable sized bytes and element byte offsets.
    ///
    /// # Errors
    /// Returns a [`CodecError::ExpectedVariableLengthBytes`] if the bytes are fixed length.
    pub fn into_variable(self) -> Result<(RawBytes<'a>, RawBytesOffsets<'a>), CodecError> {
        match self {
            Self::Fixed(_) => Err(CodecError::ExpectedVariableLengthBytes),
            Self::Variable(bytes, offsets) => Ok((bytes, offsets)),
            Self::Optional(_, _) => Err(CodecError::ExpectedNonOptionalBytes),
        }
    }

    /// Convert the array bytes into optional data and validity mask.
    ///
    /// # Errors
    /// Returns a [`CodecError::ExpectedNonOptionalBytes`] if the bytes are not optional.
    pub fn into_optional(self) -> Result<(ArrayBytes<'a>, RawBytes<'a>), CodecError> {
        match self {
            Self::Optional(data, mask) => Ok((*data, mask)),
            Self::Fixed(_) | Self::Variable(_, _) => Err(CodecError::ExpectedNonOptionalBytes),
        }
    }

    /// Returns the size (in bytes) of the underlying element bytes.
    ///
    /// This only considers the size of the element bytes, and does not include the element offsets for a variable sized array or the mask for optional arrays.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Fixed(bytes) | Self::Variable(bytes, _) => bytes.len(),
            Self::Optional(data, _) => data.size(),
        }
    }

    /// Return the byte offsets for variable sized bytes. Returns [`None`] for fixed size bytes.
    #[must_use]
    pub fn offsets(&self) -> Option<&RawBytesOffsets<'a>> {
        match self {
            Self::Fixed(_) => None,
            Self::Variable(_, offsets) => Some(offsets),
            Self::Optional(data, _) => data.offsets(),
        }
    }

    /// Convert into owned [`ArrayBytes<'static>`].
    #[must_use]
    pub fn into_owned(self) -> ArrayBytes<'static> {
        match self {
            Self::Fixed(bytes) => ArrayBytes::Fixed(bytes.into_owned().into()),
            Self::Variable(bytes, offsets) => {
                ArrayBytes::Variable(bytes.into_owned().into(), offsets.into_owned())
            }
            Self::Optional(data, mask) => {
                ArrayBytes::Optional(Box::new(data.into_owned()), mask.into_owned().into())
            }
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
            Self::Variable(bytes, _offsets) => fill_value.equals_all(bytes),
            Self::Optional(_data, mask) => {
                // For optional arrays with null fill value, check if mask is all zeros
                if fill_value.is_null() {
                    mask.iter().all(|&b| b == 0)
                } else {
                    // TODO: For non-null fill values, need to check mask is all 1s and data matches fill value
                    todo!("Proper null fill value check")
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
            ArrayBytes::Variable(bytes, offsets) => {
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
                    RawBytesOffsets::new_unchecked(ss_offsets)
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
            ArrayBytes::Optional(data, mask) => {
                // Extract the inner data type from the optional data type
                let inner_data_type = if let DataType::Optional(inner) = data_type {
                    inner.as_ref()
                } else {
                    return Err(CodecError::Other(
                        "Optional array bytes requires an optional data type".to_string(),
                    ));
                };

                // Extract subset of the inner data recursively
                let subset_data =
                    data.extract_array_subset(indexer, array_shape, inner_data_type)?;

                // Extract subset of the mask (1 byte per element)
                let byte_ranges = indexer.iter_contiguous_byte_ranges(array_shape, 1)?; // mask is 1 byte per element
                let subset_mask = extract_byte_ranges_concat(mask, byte_ranges)?;

                Ok(subset_data.with_optional_mask(subset_mask))
            }
        }
    }
}

/// Validate fixed length array bytes for a given array size.
fn validate_bytes_flen(bytes: &RawBytes, array_size: usize) -> Result<(), InvalidBytesLengthError> {
    if bytes.len() == array_size {
        Ok(())
    } else {
        Err(InvalidBytesLengthError::new(bytes.len(), array_size))
    }
}

/// Validate variable length array bytes for an array with `num_elements`.
fn validate_bytes_vlen(
    bytes: &RawBytes,
    offsets: &RawBytesOffsets,
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
        ArrayBytes::Variable(bytes, offsets) => {
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
        ArrayBytes::Optional(data, mask) => {
            let DataType::Optional(inner_type) = data_type else {
                return Err(CodecError::Other(
                    "Used optional array bytes with a non-optional data type.".to_string(),
                ));
            };
            // Validate the underlying data with the inner type
            validate_bytes(data, num_elements, inner_type)?;
            // Validate mask size (should be num_elements bytes for a bitmask)
            let expected_mask_size = usize::try_from(num_elements).unwrap();
            if mask.len() == expected_mask_size {
                Ok(())
            } else {
                Err(CodecError::Other(format!(
                    "Optional array mask size {} does not match number of elements {}",
                    mask.len(),
                    num_elements
                )))
            }
        }
    }
}

fn update_bytes_vlen_array_subset<'a>(
    input_bytes: &RawBytes,
    input_offsets: &RawBytesOffsets,
    input_shape: &[u64],
    update_bytes: &RawBytes,
    update_offsets: &RawBytesOffsets,
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
        RawBytesOffsets::new_unchecked(offsets_new)
    };
    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytes::new_vlen_unchecked(bytes_new, offsets_new)
    };
    Ok(array_bytes)
}

fn update_bytes_vlen_indexer<'a>(
    input_bytes: &RawBytes,
    input_offsets: &RawBytesOffsets,
    input_shape: &[u64],
    update_bytes: &RawBytes,
    update_offsets: &RawBytesOffsets,
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
        RawBytesOffsets::new_unchecked(offsets_new)
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
/// - `output_subset_bytes` are not compatible with the `output_subset` and `data_type_size`,
/// - `output_subset` is not within the bounds of `shape`
fn update_array_bytes_array_subset<'a>(
    bytes: ArrayBytes,
    shape: &[u64],
    update_subset: &ArraySubset,
    update_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (bytes, update_bytes, data_type_size) {
        (
            ArrayBytes::Variable(bytes, offsets),
            ArrayBytes::Variable(update_bytes, update_offsets),
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
            ArrayBytes::Optional(data, mask),
            ArrayBytes::Optional(update_data, update_mask),
            data_type_size,
        ) => {
            // Update the data recursively
            let data_after_update = update_array_bytes_array_subset(
                *data,
                shape,
                update_subset,
                update_data,
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
            mask_view.copy_from_slice(update_mask)?;

            Ok(ArrayBytes::Optional(
                Box::new(data_after_update),
                mask.into(),
            ))
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
/// - `output_subset_bytes` are not compatible with the `output_subset` and `data_type_size`,
/// - `output_subset` is not within the bounds of `shape`
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
        return update_array_bytes_array_subset(
            bytes,
            shape,
            output_subset,
            update_bytes,
            data_type_size,
        );
    }

    match (bytes, update_bytes, data_type_size) {
        (
            ArrayBytes::Variable(bytes, offsets),
            ArrayBytes::Variable(update_bytes, update_offsets),
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
            ArrayBytes::Optional(data, mask),
            ArrayBytes::Optional(update_data, update_mask),
            data_type_size,
        ) => {
            // Update the data recursively
            let data_after_update =
                update_array_bytes(*data, shape, update_indexer, update_data, data_type_size)?;

            // Update the mask (it's a fixed-size bool array, 1 byte per element)
            let mut mask = mask.into_owned();
            let byte_ranges = update_indexer.iter_contiguous_byte_ranges(shape, 1)?; // 1 byte per bool
            let mut offset: usize = 0;
            for byte_range in byte_ranges {
                let start = usize::try_from(byte_range.start).unwrap();
                let end = usize::try_from(byte_range.end).unwrap();
                let byte_range_len = end.saturating_sub(start);
                mask.index_mut(start..end)
                    .copy_from_slice(&update_mask[offset..offset + byte_range_len]);
                offset += byte_range_len;
            }

            Ok(ArrayBytes::Optional(
                Box::new(data_after_update),
                mask.into(),
            ))
        }
        (_, _, DataTypeSize::Variable) => Err(CodecError::ExpectedVariableLengthBytes),
        (_, _, DataTypeSize::Fixed(_)) => Err(CodecError::ExpectedFixedLengthBytes),
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
        RawBytesOffsets::new_unchecked(offsets)
    };

    // Write bytes
    // TODO: Go parallel
    let mut bytes = vec![0; offsets.last()];
    for (chunk_bytes, chunk_subset) in chunk_bytes_and_subsets {
        let (chunk_bytes, chunk_offsets) = chunk_bytes.into_variable()?;
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
        RawBytesOffsets::new_unchecked(region_offsets)
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
    output_view: &mut ArrayBytesFixedDisjointView,
) -> Result<(), CodecError> {
    if let ArrayBytes::Fixed(fill_value_bytes) =
        ArrayBytes::new_fill_value(data_type, output_view.num_elements(), fill_value)?
    {
        output_view.copy_from_slice(&fill_value_bytes)?;
        Ok(())
    } else {
        // TODO: Variable length data type support?
        Err(CodecError::ExpectedFixedLengthBytes)
    }
}

/// Helper function to decode `ArrayBytes` into a target, handling mask and data separately.
///
/// This function handles the common pattern of decoding `ArrayBytes` into an `ArrayBytesDecodeIntoTarget`,
/// properly handling optional data types.
pub(crate) fn decode_into_array_bytes_target(
    bytes: &ArrayBytes,
    target: crate::array::codec::ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), CodecError> {
    // Handle data based on whether it's fixed or variable length
    match &bytes {
        ArrayBytes::Fixed(data_bytes) => {
            target.data.copy_from_slice(data_bytes)?;
            Ok(())
        }
        ArrayBytes::Variable(..) => Err(CodecError::ExpectedFixedLengthBytes),
        ArrayBytes::Optional(data, mask_bytes) => {
            // Decode the data into the target
            match data.as_ref() {
                ArrayBytes::Fixed(data_bytes) => {
                    target.data.copy_from_slice(data_bytes)?;
                }
                ArrayBytes::Variable(..) | ArrayBytes::Optional(..) => {
                    return Err(CodecError::ExpectedFixedLengthBytes);
                }
            }

            // If a mask target is provided, decode the mask
            if let Some(mask_target) = target.mask {
                mask_target.copy_from_slice(mask_bytes)?;
            }

            Ok(())
        }
    }
}

impl<'a> From<RawBytes<'a>> for ArrayBytes<'a> {
    fn from(bytes: RawBytes<'a>) -> Self {
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
//                 let bytes: RawBytes<'b> = bytes.to_vec().into();
//                 let offsets: RawBytesOffsets<'b> = offsets.to_vec().into();
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

    use crate::array::Element;

    use super::*;

    #[test]
    fn array_bytes_flen() -> Result<(), Box<dyn Error>> {
        let data = [0u32, 1, 2, 3, 4];
        let bytes = Element::into_array_bytes(&DataType::UInt32, &data)?;
        let ArrayBytes::Fixed(bytes) = bytes else {
            panic!()
        };
        assert_eq!(bytes.len(), size_of::<u32>() * data.len());

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
        let data = ["a", "bb", "ccc"];
        let bytes = Element::into_array_bytes(&DataType::String, &data)?;
        let ArrayBytes::Variable(bytes, offsets) = bytes else {
            panic!()
        };
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
}

use std::borrow::Cow;

use derive_more::derive::Display;
use itertools::Itertools;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;

use crate::{
    array_subset::{ArraySubset, IncompatibleArraySubsetAndShapeError},
    byte_range::extract_byte_ranges_concat_unchecked,
    metadata::DataTypeSize,
};

use super::{
    codec::{CodecError, InvalidBytesLengthError},
    ravel_indices, ArrayBytesFixedDisjointView, ArraySize, DataType, FillValue,
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

    /// Create a new [`ArrayBytes`] with `num_elements` composed entirely of the `fill_value`.
    ///
    /// # Panics
    /// Panics if the number of elements in `array_size` exceeds [`usize::MAX`].
    #[must_use]
    pub fn new_fill_value(array_size: ArraySize, fill_value: &FillValue) -> Self {
        match array_size {
            ArraySize::Fixed {
                num_elements,
                data_type_size: _,
            } => {
                let num_elements = usize::try_from(num_elements).unwrap();
                Self::new_flen(fill_value.as_ne_bytes().repeat(num_elements))
            }
            ArraySize::Variable { num_elements } => {
                let num_elements = usize::try_from(num_elements).unwrap();
                let offsets = unsafe {
                    // SAFETY: The offsets are monotonically increasing.
                    RawBytesOffsets::new_unchecked(
                        (0..=num_elements)
                            .map(|i| i * fill_value.size())
                            .collect::<Vec<_>>(),
                    )
                };
                unsafe {
                    // SAFETY: The last offset is equal to the length of the bytes
                    Self::new_vlen_unchecked(fill_value.as_ne_bytes().repeat(num_elements), offsets)
                }
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
        }
    }

    /// Returns the size (in bytes) of the underlying element bytes.
    ///
    /// This only considers the size of the element bytes, and does not include the element offsets for a variable sized array.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Fixed(bytes) | Self::Variable(bytes, _) => bytes.len(),
        }
    }

    /// Return the byte offsets for variable sized bytes. Returns [`None`] for fixed size bytes.
    #[must_use]
    pub fn offsets(&self) -> Option<&RawBytesOffsets<'a>> {
        match self {
            Self::Fixed(_) => None,
            Self::Variable(_, offsets) => Some(offsets),
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
        }
    }

    /// Validate that the array has a valid encoding.
    ///
    /// For a fixed-length array, check it matches the expected size.
    /// For a variable-length array, check that the offsets are monotonically increasing and the largest offset is equal to the array length.
    ///
    /// # Errors
    /// Returns an error if the array is not valid.
    pub fn validate(
        &self,
        num_elements: u64,
        data_type_size: DataTypeSize,
    ) -> Result<(), CodecError> {
        validate_bytes(self, num_elements, data_type_size)
    }

    /// Returns [`true`] if the array is empty for the given fill value.
    #[must_use]
    pub fn is_fill_value(&self, fill_value: &FillValue) -> bool {
        match self {
            Self::Fixed(bytes) => fill_value.equals_all(bytes),
            Self::Variable(bytes, _offsets) => fill_value.equals_all(bytes),
        }
    }

    /// Extract a subset of the array bytes.
    ///
    /// # Errors
    /// Returns a [`CodecError::InvalidArraySubsetError`] if the `array_shape` is incompatible with `subset`.
    ///
    /// # Panics
    /// Panics if indices in the subset exceed [`usize::MAX`].
    pub fn extract_array_subset(
        &self,
        indexer: &ArraySubset,
        array_shape: &[u64],
        data_type: &DataType,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        match self {
            ArrayBytes::Variable(bytes, offsets) => {
                let num_elements = indexer.num_elements();
                let indices: Vec<_> = indexer
                    .linearised_indices(array_shape)
                    .map_err(|_| {
                        IncompatibleArraySubsetAndShapeError::new(
                            indexer.clone(),
                            array_shape.to_vec(),
                        )
                    })?
                    .into_iter()
                    .collect();
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
                let byte_ranges =
                    indexer.byte_ranges(array_shape, data_type.fixed_size().unwrap())?;
                let bytes = unsafe { extract_byte_ranges_concat_unchecked(bytes, &byte_ranges) };
                Ok(ArrayBytes::new_flen(bytes))
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
    data_type_size: DataTypeSize,
) -> Result<(), CodecError> {
    match (bytes, data_type_size) {
        (ArrayBytes::Fixed(bytes), DataTypeSize::Fixed(data_type_size)) => Ok(validate_bytes_flen(
            bytes,
            usize::try_from(num_elements * data_type_size as u64).unwrap(),
        )?),
        (ArrayBytes::Variable(bytes, offsets), DataTypeSize::Variable) => {
            validate_bytes_vlen(bytes, offsets, num_elements)
        }
        (ArrayBytes::Fixed(_), DataTypeSize::Variable) => Err(CodecError::Other(
            "Used fixed length array bytes with a variable sized data type.".to_string(),
        )),
        (ArrayBytes::Variable(_, _), DataTypeSize::Fixed(_)) => Err(CodecError::Other(
            "Used variable length array bytes with a fixed length data type.".to_string(),
        )),
    }
}

pub(crate) fn update_bytes_vlen<'a>(
    input_bytes: &RawBytes,
    input_offsets: &RawBytesOffsets,
    input_shape: &[u64],
    update_bytes: &RawBytes,
    update_offsets: &RawBytesOffsets,
    update_subset: &ArraySubset,
) -> Result<ArrayBytes<'a>, IncompatibleArraySubsetAndShapeError> {
    if !update_subset.inbounds_shape(input_shape) {
        return Err(IncompatibleArraySubsetAndShapeError::new(
            update_subset.clone(),
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
                usize::try_from(ravel_indices(&subset_indices, update_subset.shape())).unwrap();
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

/// Update a subset of an array.
///
/// This function is used internally by [`crate::array::Array::store_chunk_subset_opt`] and [`crate::array::Array::async_store_chunk_subset_opt`].
///
/// # Errors
/// Returns a [`CodecError`] if
/// - `output_bytes` are not compatible with the `output_shape` and `data_type_size`,
/// - `output_subset_bytes` are not compatible with the `output_subset` and `data_type_size`,
/// - `output_subset` is not within the bounds of `output_shape`
pub fn update_array_bytes<'a>(
    output_bytes: ArrayBytes,
    output_shape: &[u64],
    output_subset: &ArraySubset,
    output_subset_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (output_bytes, output_subset_bytes, data_type_size) {
        (
            ArrayBytes::Variable(chunk_bytes, chunk_offsets),
            ArrayBytes::Variable(chunk_subset_bytes, chunk_subset_offsets),
            DataTypeSize::Variable,
        ) => Ok(update_bytes_vlen(
            &chunk_bytes,
            &chunk_offsets,
            output_shape,
            chunk_subset_bytes,
            chunk_subset_offsets,
            output_subset,
        )?),
        (
            ArrayBytes::Fixed(chunk_bytes),
            ArrayBytes::Fixed(chunk_subset_bytes),
            DataTypeSize::Fixed(data_type_size),
        ) => {
            let mut chunk_bytes = chunk_bytes.into_owned();
            let mut output_view = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut chunk_bytes),
                    data_type_size,
                    output_shape,
                    output_subset.clone(),
                )
            }
            .map_err(CodecError::from)?;
            output_view.copy_from_slice(chunk_subset_bytes)?;
            Ok(ArrayBytes::new_flen(chunk_bytes))
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
            for idx in &indices {
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
    indexer: &ArraySubset,
    array_shape: &[u64],
) -> Result<ArrayBytes<'a>, CodecError> {
    let indices = indexer.linearised_indices(array_shape)?;
    let indices: Vec<_> = indices.into_iter().collect();
    let mut region_bytes_len = 0;
    for index in &indices {
        let index = usize::try_from(*index).unwrap();
        let curr = offsets[index];
        let next = offsets[index + 1];
        debug_assert!(next >= curr);
        region_bytes_len += next - curr;
    }
    let mut region_offsets =
        Vec::with_capacity(usize::try_from(indexer.num_elements() + 1).unwrap());
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
    let array_size = ArraySize::new(data_type.size(), output_view.num_elements());
    if let ArrayBytes::Fixed(fill_value_bytes) = ArrayBytes::new_fill_value(array_size, fill_value)
    {
        output_view.copy_from_slice(&fill_value_bytes)?;
        Ok(())
    } else {
        // TODO: Variable length data type support?
        Err(CodecError::ExpectedFixedLengthBytes)
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

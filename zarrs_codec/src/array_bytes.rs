//! `ArrayBytes` extension methods and free functions for codec operations.
//!
//! Core `ArrayBytes` types are defined in `zarrs_data_type` and re-exported from the crate root.

use std::ops::IndexMut;

use itertools::Itertools;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::{ArraySubsetTraits, ravel_indices};
use zarrs_data_type::FillValue;
use zarrs_data_type::array_bytes::{
    ArrayBytes, ArrayBytesOffsets, ArrayBytesOptional, ArrayBytesRaw, ArrayBytesVariableLength,
    ExpectedFixedLengthBytesError, ExpectedOptionalBytesError, ExpectedVariableLengthBytesError,
};
use zarrs_metadata::DataTypeSize;
use zarrs_storage::byte_range::extract_byte_ranges_concat;

use crate::{
    ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, CodecError, DataType,
    Indexer, IndexerError,
};

/// Extension trait for `ArrayBytes` with methods that require codec types.
pub trait ArrayBytesExt<'a> {
    /// Extract a subset of the array bytes.
    ///
    /// # Errors
    /// Returns a [`CodecError::IncompatibleIndexer`] if the `indexer` is incompatible.
    fn extract_array_subset(
        &self,
        indexer: &dyn Indexer,
        array_shape: &[u64],
        data_type: &DataType,
    ) -> Result<ArrayBytes<'static>, CodecError>;
}

impl<'a> ArrayBytesExt<'a> for ArrayBytes<'a> {
    fn extract_array_subset(
        &self,
        indexer: &dyn Indexer,
        array_shape: &[u64],
        data_type: &DataType,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        match self {
            ArrayBytes::Variable(vlen) => {
                let bytes = vlen.bytes();
                let offsets = vlen.offsets();
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
                let Some(opt) = data_type.as_optional() else {
                    return Err(CodecError::Other(
                        "Optional array bytes requires an optional data type".to_string(),
                    ));
                };

                // Extract subset of the inner data recursively
                let subset_data = optional_bytes.data().extract_array_subset(
                    indexer,
                    array_shape,
                    opt.data_type(),
                )?;

                // Extract subset of the mask (1 byte per element)
                let byte_ranges = indexer.iter_contiguous_byte_ranges(array_shape, 1)?; // mask is 1 byte per element
                let subset_mask = extract_byte_ranges_concat(optional_bytes.mask(), byte_ranges)?;

                Ok(subset_data.with_optional_mask(subset_mask))
            }
        }
    }
}

fn update_bytes_vlen_array_subset<'a>(
    input: &ArrayBytesVariableLength<'_>,
    input_shape: &[u64],
    update: &ArrayBytesVariableLength<'_>,
    update_subset: &dyn ArraySubsetTraits,
) -> Result<ArrayBytesVariableLength<'a>, IndexerError> {
    let input_offsets = input.offsets();
    let input_bytes = input.bytes();
    let update_offsets = update.offsets();
    let update_bytes = update.bytes();

    if !update_subset.inbounds_shape(input_shape) {
        return Err(IndexerError::new_oob(
            update_subset.end_exc(),
            input_shape.to_vec(),
        ));
    }

    // Get the current and new length of the bytes in the chunk subset
    let size_subset_new = update
        .offsets()
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
    let update_subset_start = update_subset.start();
    let update_subset_shape = update_subset.shape();
    for (chunk_index, indices) in indices.iter().enumerate() {
        offsets_new.push(bytes_new.len());
        if update_subset.contains(&indices) {
            let subset_indices = indices
                .iter()
                .zip(update_subset_start.iter())
                .map(|(i, s)| i - s)
                .collect::<Vec<_>>();
            let subset_index =
                ravel_indices(&subset_indices, &update_subset_shape).expect("inbounds indices");
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
        ArrayBytesVariableLength::new_unchecked(bytes_new, offsets_new)
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
) -> Result<ArrayBytes<'a>, IndexerError> {
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
    update_subset: &dyn ArraySubsetTraits,
    update_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (bytes, update_bytes, data_type_size) {
        (ArrayBytes::Variable(bytes), ArrayBytes::Variable(update), DataTypeSize::Variable) => {
            Ok(ArrayBytes::Variable(update_bytes_vlen_array_subset(
                &bytes,
                shape,
                update,
                update_subset,
            )?))
        }
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
                    update_subset.to_array_subset(),
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
                    update_subset.to_array_subset(),
                )
            }
            .map_err(CodecError::from)?;
            mask_view.copy_from_slice(update_optional_bytes.mask())?;

            Ok(ArrayBytes::Optional(ArrayBytesOptional::new(
                data_after_update,
                mask,
            )))
        }
        (_, _, DataTypeSize::Variable) => Err(ExpectedVariableLengthBytesError.into()),
        (_, _, DataTypeSize::Fixed(_)) => Err(ExpectedFixedLengthBytesError.into()),
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
    update_indexer: &dyn Indexer,
    update_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (bytes, update_bytes, data_type_size) {
        (ArrayBytes::Variable(vlen), ArrayBytes::Variable(update_vlen), DataTypeSize::Variable) => {
            Ok(update_bytes_vlen_indexer(
                vlen.bytes(),
                vlen.offsets(),
                shape,
                update_vlen.bytes(),
                update_vlen.offsets(),
                update_indexer,
            )?)
        }
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
        (_, _, DataTypeSize::Variable) => Err(ExpectedVariableLengthBytesError.into()),
        (_, _, DataTypeSize::Fixed(_)) => Err(ExpectedFixedLengthBytesError.into()),
    }
}

/// Update array bytes.
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
    update_indexer: &dyn Indexer,
    update_bytes: &ArrayBytes,
    data_type_size: DataTypeSize,
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(output_subset) = update_indexer.as_array_subset() {
        update_array_bytes_array_subset(bytes, shape, output_subset, update_bytes, data_type_size)
    } else {
        update_array_bytes_indexer(bytes, shape, update_indexer, update_bytes, data_type_size)
    }
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
///
/// # Errors
/// Returns a [`CodecError`] if the bytes are incompatible with the target.
pub fn decode_into_array_bytes_target(
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
        (ArrayBytes::Variable(..), _) => Err(ExpectedFixedLengthBytesError.into()),
        (ArrayBytes::Fixed(_), ArrayBytesDecodeIntoTarget::Optional(..)) => {
            Err(ExpectedOptionalBytesError.into())
        }
        (ArrayBytes::Optional(..), ArrayBytesDecodeIntoTarget::Fixed(_)) => {
            Err(ExpectedFixedLengthBytesError.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

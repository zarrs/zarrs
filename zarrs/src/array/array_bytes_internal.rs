//! Internal utilities for handling array bytes.

use std::sync::Arc;

use itertools::Itertools;
use unsafe_cell_slice::UnsafeCellSlice;

use super::{ArraySubset, DataType};
use zarrs_codec::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesOffsets, ArrayBytesOffsetsCreateError,
    ArrayBytesVariableLength, CodecError,
};
pub(crate) use zarrs_codec::{build_nested_optional_target, extract_target_views};

pub(crate) fn offsets_from_usize(
    offsets: Vec<usize>,
) -> Result<ArrayBytesOffsets, ArrayBytesOffsetsCreateError> {
    if u32::try_from(offsets.iter().copied().max().unwrap_or(0)).is_ok() {
        #[cfg(target_pointer_width = "32")]
        let offsets = bytemuck::allocation::cast_vec::<usize, u32>(offsets);
        #[cfg(not(target_pointer_width = "32"))]
        let offsets = offsets
            .into_iter()
            .map(|offset| u32::try_from(offset).unwrap())
            .collect::<Vec<_>>();
        ArrayBytesOffsets::new(offsets)
    } else {
        #[cfg(target_pointer_width = "64")]
        let offsets = bytemuck::allocation::cast_vec::<usize, u64>(offsets);
        #[cfg(not(target_pointer_width = "64"))]
        let offsets = offsets
            .into_iter()
            .map(|offset| u64::try_from(offset).unwrap())
            .collect::<Vec<_>>();
        ArrayBytesOffsets::new(offsets)
    }
}

/// The innermost data type of (possibly nested) optional types.
/// Returns `data_type` for non-optional types.
pub(crate) fn optional_innermost(data_type: &DataType) -> &DataType {
    let mut data_type = data_type;
    while let Some(inner) = data_type.optional_inner() {
        data_type = inner;
    }
    data_type
}

/// Split keyed optional array bytes into keyed inner data and keyed validity masks.
///
/// # Errors
/// Returns a [`CodecError`] if any array bytes are not optional.
#[expect(clippy::type_complexity)]
pub(crate) fn split_optional<K: Clone>(
    bytes_and_keys: Vec<(ArrayBytes<'_>, K)>,
) -> Result<(Vec<(ArrayBytes<'_>, K)>, Vec<(ArrayBytes<'_>, K)>), CodecError> {
    let mut data_and_keys = Vec::with_capacity(bytes_and_keys.len());
    let mut masks_and_keys = Vec::with_capacity(bytes_and_keys.len());
    for (bytes, key) in bytes_and_keys {
        let (data, mask) = bytes.into_optional()?.into_parts();
        data_and_keys.push((*data, key.clone()));
        masks_and_keys.push((ArrayBytes::new_flen(mask), key));
    }
    Ok((data_and_keys, masks_and_keys))
}

/// Merge a set of chunks of any data type (fixed, variable, or optional) into an array subset.
///
/// Optional data is merged by independently merging its inner data and its validity mask.
///
/// # Errors
/// Returns a [`CodecError`] if the chunk bytes are incompatible with `data_type` or their subsets are out of bounds.
///
/// # Panics
/// Panics if the `array_shape` exceeds `usize::MAX` elements.
pub(crate) fn merge_chunks<'a>(
    chunk_bytes_and_subsets: Vec<(ArrayBytes<'_>, ArraySubset)>,
    array_shape: &[u64],
    data_type: &DataType,
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(inner_data_type) = data_type.optional_inner() {
        let (data_and_subsets, masks_and_subsets) = split_optional(chunk_bytes_and_subsets)?;
        let data = merge_chunks(data_and_subsets, array_shape, inner_data_type)?;
        let mask = merge_chunks(masks_and_subsets, array_shape, &super::data_type::uint8())?;
        Ok(data.with_optional_mask(mask.into_fixed()?))
    } else if let Some(data_type_size) = data_type.fixed_size() {
        let num_elements = usize::try_from(array_shape.iter().product::<u64>()).unwrap();
        let mut output = vec![0; num_elements * data_type_size];
        let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
        for (chunk_bytes, chunk_subset) in chunk_bytes_and_subsets {
            let mut output_view = unsafe {
                // SAFETY: chunks represent disjoint array subsets
                ArrayBytesFixedDisjointView::new(
                    output_slice,
                    data_type_size,
                    array_shape,
                    chunk_subset,
                )?
            };
            output_view.copy_from_slice(&chunk_bytes.into_fixed()?)?;
        }
        Ok(ArrayBytes::new_flen(output))
    } else {
        let chunk_bytes_and_subsets = chunk_bytes_and_subsets
            .into_iter()
            .map(|(chunk_bytes, chunk_subset)| Ok((chunk_bytes.into_variable()?, chunk_subset)))
            .collect::<Result<Vec<_>, CodecError>>()?;
        Ok(ArrayBytes::Variable(merge_chunks_vlen(
            chunk_bytes_and_subsets,
            array_shape,
        )))
    }
}

/// Merge a set of variable length chunks into an array subset.
///
/// # Panics
/// Panics if the `array_shape` exceeds `usize::MAX` elements.
pub(crate) fn merge_chunks_vlen<'a>(
    chunk_bytes_and_subsets: Vec<(ArrayBytesVariableLength<'_>, ArraySubset)>,
    array_shape: &[u64],
) -> ArrayBytesVariableLength<'a> {
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
        let chunk_offsets = chunk_bytes.offsets();
        debug_assert_eq!(chunk_offsets.len() as u64, chunk_subset.num_elements() + 1);
        let indices = chunk_subset.linearised_indices(array_shape).unwrap();
        for (subset_idx, range) in indices.iter().zip_eq(chunk_offsets.element_ranges()) {
            let subset_idx = usize::try_from(subset_idx).unwrap();
            element_sizes[subset_idx] = range.len();
        }
    }

    // Convert to offsets with a cumulative sum
    // TODO: Parallel cum sum
    let mut offsets = Vec::with_capacity(element_sizes.len() + 1);
    offsets.push(0); // first offset is always zero
    let mut offset = 0;
    for size in element_sizes {
        offset += size;
        offsets.push(offset);
    }
    let offsets = offsets_from_usize(offsets).unwrap();

    // Write bytes
    // TODO: Go parallel
    let mut bytes = vec![0; offsets.last()];
    for (chunk_bytes, chunk_subset) in chunk_bytes_and_subsets {
        let (chunk_bytes, chunk_offsets) = chunk_bytes.into_parts();
        let indices = chunk_subset.linearised_indices(array_shape).unwrap();
        for (subset_idx, chunk_range) in indices.iter().zip_eq(chunk_offsets.element_ranges()) {
            let subset_idx = usize::try_from(subset_idx).unwrap();
            bytes[offsets.element_range(subset_idx)].copy_from_slice(&chunk_bytes[chunk_range]);
        }
    }

    unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytesVariableLength::new_unchecked(bytes, offsets)
    }
}

/// Merge cached chunk bytes into an array subset.
///
/// # Errors
/// Returns a [`CodecError`] if the chunk bytes are incompatible with `data_type`.
///
/// # Panics
/// Panics if `array_shape` exceeds `usize::MAX` elements.
pub(crate) fn merge_cached_chunks(
    chunk_bytes_and_subsets: Vec<(Arc<ArrayBytes<'static>>, ArraySubset)>,
    array_shape: &[u64],
    data_type: &DataType,
) -> Result<ArrayBytes<'static>, CodecError> {
    let chunk_bytes_and_subsets = chunk_bytes_and_subsets
        .into_iter()
        .map(|(bytes, subset)| (Arc::unwrap_or_clone(bytes), subset))
        .collect();
    merge_chunks(chunk_bytes_and_subsets, array_shape, data_type)
}

#[cfg(test)]
mod tests {
    use super::offsets_from_usize;

    #[test]
    fn offsets_from_usize_reuses_matching_allocation() {
        #[cfg(target_pointer_width = "64")]
        {
            let input = vec![0, u32::MAX as usize + 1];
            let input_ptr = input.as_ptr().cast::<u64>();
            let offsets = offsets_from_usize(input).unwrap();
            assert_eq!(offsets.as_u64().unwrap().as_ptr(), input_ptr);
        }

        #[cfg(target_pointer_width = "32")]
        {
            let input = vec![0, 1];
            let input_ptr = input.as_ptr().cast::<u32>();
            let offsets = offsets_from_usize(input).unwrap();
            assert_eq!(offsets.as_u32().unwrap().as_ptr(), input_ptr);
        }
    }
}

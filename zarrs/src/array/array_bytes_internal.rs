//! Internal utilities for handling array bytes.

use std::num::NonZeroU64;

use itertools::Itertools;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_codec::decode_into_array_bytes_target;

use super::{ArraySubset, DataType, FillValue, Indexer};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayBytesOffsets,
    ArrayBytesOptional, ArrayBytesVariableLength, CodecError,
};

/// Count the nesting depth of optional types.
/// Returns 0 for non-optional types, 1 for `Option<T>`, 2 for `Option<Option<T>>`, etc.
pub(crate) fn optional_nesting_depth(data_type: &DataType) -> usize {
    if let Some(inner) = data_type.optional_inner() {
        1 + optional_nesting_depth(inner)
    } else {
        0
    }
}

/// Build a nested `ArrayBytesDecodeIntoTarget` for optional types.
/// The `mask_views` slice should be ordered from outermost to innermost mask.
pub(crate) fn build_nested_optional_target<'a>(
    data_view: &'a mut ArrayBytesFixedDisjointView<'a>,
    mask_views: &'a mut [ArrayBytesFixedDisjointView<'a>],
) -> ArrayBytesDecodeIntoTarget<'a> {
    if let Some((first_mask, rest_masks)) = mask_views.split_first_mut() {
        ArrayBytesDecodeIntoTarget::Optional(
            Box::new(build_nested_optional_target(data_view, rest_masks)),
            first_mask,
        )
    } else {
        ArrayBytesDecodeIntoTarget::Fixed(data_view)
    }
}

/// Wrap `bytes` in one [`ArrayBytes::Optional`] layer per mask.
///
/// The `masks` slice should be ordered from outermost to innermost mask, matching the convention of
/// [`build_nested_optional_target`].
pub(crate) fn wrap_optional_masks<'a>(
    bytes: ArrayBytes<'a>,
    masks: Vec<Vec<u8>>,
) -> ArrayBytes<'a> {
    masks
        .into_iter()
        .rev()
        .fold(bytes, ArrayBytes::with_optional_mask)
}

/// Extract shared references to the data view and mask views from an [`ArrayBytesDecodeIntoTarget`].
///
/// Mask views are returned outer-to-inner, matching the convention of [`build_nested_optional_target`].
pub(crate) fn extract_target_views<'a, 'b>(
    target: &'b ArrayBytesDecodeIntoTarget<'a>,
) -> (
    &'b ArrayBytesFixedDisjointView<'a>,
    Vec<&'b ArrayBytesFixedDisjointView<'a>>,
) {
    match target {
        ArrayBytesDecodeIntoTarget::Fixed(view) => (view, vec![]),
        ArrayBytesDecodeIntoTarget::Optional(inner, mask_view) => {
            let (data_view, mut mask_views) = extract_target_views(inner);
            mask_views.insert(0, mask_view);
            (data_view, mask_views)
        }
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
        let (chunk_bytes, chunk_offsets) = chunk_bytes.into_parts();
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

    unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytesVariableLength::new_unchecked(bytes, offsets)
    }
}

/// Merge multiple chunks with optional variable-length data types.
///
/// This handles optional wrappers (including nested optionals like `Option<Option<String>>`)
/// around variable-length data. Each chunk should contain an `ArrayBytes::Optional` with
/// variable-length inner data.
///
/// # Arguments
/// * `chunk_bytes_and_subsets` - Pairs of `(ArrayBytes, ArraySubset)` for each chunk
/// * `array_shape` - The shape of the output array
/// * `nesting_depth` - The number of nested `Option` layers (e.g., 1 for `Option<String>`, 2 for `Option<Option<String>>`)
///
/// # Errors
/// Returns an error if the chunks don't have the expected optional structure.
///
/// # Panics
/// Panics if the `array_shape` exceeds `usize::MAX` elements.
pub(crate) fn merge_chunks_vlen_optional<'a>(
    chunk_bytes_and_subsets: Vec<(ArrayBytesOptional<'_>, ArraySubset)>,
    array_shape: &[u64],
    nesting_depth: usize,
) -> Result<ArrayBytesOptional<'a>, CodecError> {
    debug_assert!(nesting_depth > 0);

    let num_elements = usize::try_from(array_shape.iter().product::<u64>()).unwrap();

    // Allocate mask buffers for each nesting level (1 byte per element per level)
    let mut merged_masks: Vec<Vec<u8>> = (0..nesting_depth)
        .map(|_| vec![0u8; num_elements])
        .collect();

    // Unwrap optionals and collect inner variable-length data
    let mut inner_bytes_and_subsets = Vec::with_capacity(chunk_bytes_and_subsets.len());

    for (chunk_bytes, chunk_subset) in chunk_bytes_and_subsets {
        // Unwrap nesting_depth levels of Optional, collecting masks
        let mut current = ArrayBytes::Optional(chunk_bytes);
        let mut chunk_masks = Vec::with_capacity(nesting_depth);

        for _ in 0..nesting_depth {
            let optional = current.into_optional()?;
            let (data, mask) = optional.into_parts();
            chunk_masks.push(mask);
            current = *data;
        }

        // Copy chunk masks to merged masks at correct positions
        let indices: Vec<_> = chunk_subset
            .linearised_indices(array_shape)
            .unwrap()
            .into_iter()
            .collect();
        for (level, chunk_mask) in chunk_masks.iter().enumerate() {
            for (chunk_idx, &array_idx) in indices.iter().enumerate() {
                let array_idx = usize::try_from(array_idx).unwrap();
                merged_masks[level][array_idx] = chunk_mask[chunk_idx];
            }
        }

        inner_bytes_and_subsets.push((current.into_variable()?, chunk_subset));
    }

    // Merge the inner variable-length data using the existing function
    let merged_vlen = merge_chunks_vlen(inner_bytes_and_subsets, array_shape);

    // Wrap with masks in reverse order (innermost first)
    let mut result = ArrayBytes::Variable(merged_vlen);
    for mask in merged_masks.into_iter().rev() {
        result = result.with_optional_mask(mask);
    }

    Ok(result.into_optional()?)
}

/// Extract decoded variable-length regions from bytes and offsets using an indexer.
///
/// # Errors
/// Returns a [`CodecError`] if the indexer is incompatible with the array shape.
///
/// # Panics
/// Panics if indices in the indexer exceed [`usize::MAX`].
pub(crate) fn extract_decoded_regions_vlen<'a>(
    bytes: &[u8],
    offsets: &[usize],
    indexer: &dyn Indexer,
    array_shape: &[NonZeroU64],
) -> Result<ArrayBytesVariableLength<'a>, CodecError> {
    let indices = indexer.iter_linearised_indices(bytemuck::must_cast_slice(array_shape))?;
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
        ArrayBytesVariableLength::new_unchecked(region_bytes, region_offsets)
    };
    Ok(array_bytes)
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

/// Output buffers for decoding fixed length data (including optional data with fixed length inner data) into views.
pub(crate) struct FixedDecodeBuffers {
    data: Vec<u8>,
    masks: Vec<Vec<u8>>,
    data_type_size: usize,
    shape: Vec<u64>,
}

impl FixedDecodeBuffers {
    /// Allocate uninitialised buffers for an array of `shape` and `data_type`.
    ///
    /// Returns [`None`] if the innermost data type is not fixed length.
    ///
    /// # Panics
    /// Panics if the number of elements in `shape` exceeds [`usize::MAX`].
    pub(crate) fn new(data_type: &DataType, shape: &[u64]) -> Option<Self> {
        let data_type_size = optional_innermost(data_type).fixed_size()?;
        let num_elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
        Some(Self {
            data: Vec::with_capacity(num_elements * data_type_size),
            masks: (0..optional_nesting_depth(data_type))
                .map(|_| Vec::with_capacity(num_elements))
                .collect(),
            data_type_size,
            shape: shape.to_vec(),
        })
    }

    /// Return views of the data and each validity mask covering the entire array.
    ///
    /// Mask views are returned outer-to-inner, matching the convention of [`build_nested_optional_target`].
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the views cannot be created.
    pub(crate) fn views(
        &mut self,
    ) -> Result<
        (
            ArrayBytesFixedDisjointView<'_>,
            Vec<ArrayBytesFixedDisjointView<'_>>,
        ),
        CodecError,
    > {
        let subset = ArraySubset::new_with_shape(self.shape.clone());
        let data_view = unsafe {
            // SAFETY: the view is the only view of the data
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut self.data),
                self.data_type_size,
                &self.shape,
                subset.clone(),
            )?
        };
        let mask_views = self
            .masks
            .iter_mut()
            .map(|mask| unsafe {
                // SAFETY: the view is the only view of the mask
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new_from_vec_with_spare_capacity(mask),
                    1,
                    &self.shape,
                    subset.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((data_view, mask_views))
    }

    /// Convert the buffers into array bytes.
    ///
    /// # Safety
    /// Every element of the views returned by [`views`](Self::views) must have been written.
    pub(crate) unsafe fn into_array_bytes(mut self) -> ArrayBytes<'static> {
        let num_elements = usize::try_from(self.shape.iter().product::<u64>()).unwrap();
        unsafe { self.data.set_len(num_elements * self.data_type_size) };
        for mask in &mut self.masks {
            unsafe { mask.set_len(num_elements) };
        }
        wrap_optional_masks(ArrayBytes::new_flen(self.data), self.masks)
    }
}

/// Fill every element of `target` with `fill_value`.
///
/// # Errors
/// Returns a [`CodecError`] if `fill_value` is incompatible with `data_type` or `target`.
pub(crate) fn fill_target(
    target: ArrayBytesDecodeIntoTarget<'_>,
    data_type: &DataType,
    fill_value: &FillValue,
) -> Result<(), CodecError> {
    match target {
        ArrayBytesDecodeIntoTarget::Fixed(view) => view
            .fill(fill_value.as_ne_bytes())
            .map_err(CodecError::from),
        target @ ArrayBytesDecodeIntoTarget::Optional(..) => {
            let fill_bytes =
                ArrayBytes::new_fill_value(data_type, target.num_elements(), fill_value)?;
            decode_into_array_bytes_target(&fill_bytes, target)
        }
    }
}

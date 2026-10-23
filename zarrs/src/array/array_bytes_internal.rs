//! Internal utilities for handling array bytes.

use std::num::NonZeroU64;
use std::sync::Arc;

use itertools::Itertools;
use unsafe_cell_slice::UnsafeCellSlice;

use super::{ArraySubset, DataType, Indexer};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayBytesOffsets,
    ArrayBytesOffsetsCreateError, ArrayBytesVariableLength, CodecError,
};

pub(crate) fn offsets_from_usize(
    offsets: Vec<usize>,
) -> Result<ArrayBytesOffsets, ArrayBytesOffsetsCreateError> {
    if offsets.iter().copied().max().unwrap_or(0) <= u32::MAX as usize {
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

/// Count the nesting depth of optional types.
/// Returns 0 for non-optional types, 1 for `Option<T>`, 2 for `Option<Option<T>>`, etc.
pub(crate) fn optional_nesting_depth(data_type: &DataType) -> usize {
    if let Some(inner) = data_type.optional_inner() {
        1 + optional_nesting_depth(inner)
    } else {
        0
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
        let mut mask_views = Vec::with_capacity(self.masks.len());
        for mask in &mut self.masks {
            mask_views.push(unsafe {
                // SAFETY: the view is the only view of the mask
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new_from_vec_with_spare_capacity(mask),
                    1,
                    &self.shape,
                    subset.clone(),
                )?
            });
        }
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

/// Extract decoded variable-length regions from bytes and offsets using an indexer.
///
/// # Errors
/// Returns a [`CodecError`] if the indexer is incompatible with the array shape.
///
/// # Panics
/// Panics if indices in the indexer exceed [`usize::MAX`].
pub(crate) fn extract_decoded_regions_vlen<'a>(
    bytes: &[u8],
    offsets: &ArrayBytesOffsets,
    indexer: &dyn Indexer,
    array_shape: &[NonZeroU64],
) -> Result<ArrayBytesVariableLength<'a>, CodecError> {
    let indices = indexer.iter_linearised_indices(bytemuck::must_cast_slice(array_shape))?;
    let indices: Vec<_> = indices.into_iter().collect();
    let mut region_bytes_len = 0;
    for index in &indices {
        let index = usize::try_from(*index).unwrap();
        region_bytes_len += offsets.element_range(index).len();
    }
    let mut region_offsets = Vec::with_capacity(usize::try_from(indexer.len() + 1).unwrap());
    let mut region_bytes = Vec::with_capacity(region_bytes_len);
    for index in &indices {
        region_offsets.push(region_bytes.len());
        let index = usize::try_from(*index).unwrap();
        region_bytes.extend_from_slice(&bytes[offsets.element_range(index)]);
    }
    region_offsets.push(region_bytes.len());
    let region_offsets = offsets_from_usize(region_offsets).unwrap();
    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytesVariableLength::new_unchecked(region_bytes, region_offsets)
    };
    Ok(array_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{FillValue, data_type};
    use zarrs_codec::copy_fill_value_into;

    #[test]
    fn copy_fill_value_into_matches_materialised_fill() {
        let cases = [
            (data_type::uint16(), FillValue::from(0x1234u16)),
            (data_type::raw_bits(0), FillValue::from(())),
            (
                data_type::uint16().to_optional(),
                FillValue::from(None::<u16>),
            ),
            (
                data_type::uint16().to_optional(),
                FillValue::from(Some(0x1234u16)),
            ),
            (
                data_type::uint16().to_optional().to_optional(),
                FillValue::from(None::<Option<u16>>),
            ),
            (
                data_type::uint16().to_optional().to_optional(),
                FillValue::from(Some(None::<u16>)),
            ),
            (
                data_type::uint16().to_optional().to_optional(),
                FillValue::from(Some(Some(0x1234u16))),
            ),
            (
                data_type::uint16()
                    .to_optional()
                    .to_optional()
                    .to_optional(),
                FillValue::from(Some(Some(None::<u16>))),
            ),
        ];
        for (data_type, fill_value) in cases {
            for subset in [
                ArraySubset::new_with_shape(vec![3, 5]),
                ArraySubset::new_with_ranges(&[1..3, 1..4]),
                ArraySubset::new_with_start_shape(vec![99, 99], vec![0, 0]).unwrap(),
            ] {
                let mut outputs = Vec::new();
                for direct in [false, true] {
                    let mut buffers = FixedDecodeBuffers::new(&data_type, &[3, 5]).unwrap();
                    buffers.data.resize(15 * buffers.data_type_size, 0xa5);
                    for mask in &mut buffers.masks {
                        mask.resize(15, 0xa5);
                    }
                    {
                        let shape = [3, 5];
                        // SAFETY: each view is the only view of its backing buffer.
                        let mut data_view = unsafe {
                            ArrayBytesFixedDisjointView::new(
                                UnsafeCellSlice::new(&mut buffers.data),
                                buffers.data_type_size,
                                &shape,
                                subset.clone(),
                            )
                            .unwrap()
                        };
                        let mut mask_views: Vec<_> = buffers
                            .masks
                            .iter_mut()
                            .map(|mask| unsafe {
                                ArrayBytesFixedDisjointView::new(
                                    UnsafeCellSlice::new(mask),
                                    1,
                                    &shape,
                                    subset.clone(),
                                )
                                .unwrap()
                            })
                            .collect();
                        let target = build_nested_optional_target(&mut data_view, &mut mask_views);
                        if direct {
                            copy_fill_value_into(&data_type, &fill_value, target).unwrap();
                        } else {
                            let bytes = ArrayBytes::new_fill_value(
                                &data_type,
                                subset.num_elements(),
                                &fill_value,
                            )
                            .unwrap();
                            zarrs_codec::decode_into_array_bytes_target(&bytes, target).unwrap();
                        }
                    }
                    // Compare raw data too, including placeholder bytes beneath null masks.
                    outputs.push((buffers.data, buffers.masks));
                }
                assert_eq!(
                    outputs[0], outputs[1],
                    "{data_type:?} {fill_value} {subset:?}"
                );
            }
        }
    }

    #[test]
    fn optional_fill_value_representations() {
        let cases = [
            (
                data_type::uint16(),
                FillValue::from(None::<Option<u16>>),
                vec![0, 0],
                vec![0, 0],
            ),
            (
                data_type::uint16(),
                FillValue::from(Some(None::<u16>)),
                vec![0, 0],
                vec![1, 0],
            ),
            (
                data_type::uint16(),
                FillValue::from(Some(Some(0x1234u16))),
                0x1234u16.to_ne_bytes().to_vec(),
                vec![1, 1],
            ),
            (
                data_type::raw_bits(0),
                FillValue::from(None::<Option<()>>),
                vec![],
                vec![0, 1],
            ),
            (
                data_type::string(),
                FillValue::from(None::<Option<&str>>),
                vec![],
                vec![0, 1],
            ),
            (
                data_type::string(),
                FillValue::from(Some(None::<&str>)),
                vec![],
                vec![1, 0],
            ),
            (
                data_type::string(),
                FillValue::from(Some(Some("abc"))),
                b"abc".to_vec(),
                vec![1, 1],
            ),
        ];
        for (inner_type, fill_value, element_bytes, masks) in cases {
            let data_type = inner_type.to_optional().to_optional();
            for count in [0usize, 1, 3] {
                let bytes =
                    ArrayBytes::new_fill_value(&data_type, count as u64, &fill_value).unwrap();
                let mut data = &bytes;
                for &mask in &masks {
                    let ArrayBytes::Optional(optional) = data else {
                        panic!("expected optional bytes")
                    };
                    assert_eq!(optional.mask().as_ref(), vec![mask; count]);
                    data = optional.data();
                }
                let expected = element_bytes.repeat(count);
                match data {
                    ArrayBytes::Fixed(bytes) => assert_eq!(bytes.as_ref(), expected),
                    ArrayBytes::Variable(bytes) => {
                        assert_eq!(bytes.bytes().as_ref(), expected);
                        assert_eq!(
                            bytes.offsets().iter().collect::<Vec<_>>(),
                            (0..=count)
                                .map(|i| i * element_bytes.len())
                                .collect::<Vec<_>>()
                        );
                    }
                    ArrayBytes::Optional(_) => panic!("unexpected optional layer"),
                }
            }
        }
    }

    #[test]
    fn copy_fill_value_into_rejects_incompatible_values_and_targets() {
        for (data_type, fill_value) in [
            (data_type::uint16(), FillValue::from(1u8)),
            (data_type::uint8(), FillValue::from(1u8)),
            (
                data_type::uint16().to_optional(),
                FillValue::from(Some(1u16)),
            ),
            (data_type::string(), FillValue::from("")),
        ] {
            let mut buffers = FixedDecodeBuffers::new(&data_type::uint16(), &[3]).unwrap();
            let (mut data_view, _) = buffers.views().unwrap();
            assert!(
                copy_fill_value_into(&data_type, &fill_value, (&mut data_view).into()).is_err()
            );
        }

        let mut buffers =
            FixedDecodeBuffers::new(&data_type::uint16().to_optional(), &[3]).unwrap();
        let (mut data_view, mut mask_views) = buffers.views().unwrap();
        let target = build_nested_optional_target(&mut data_view, &mut mask_views);
        assert!(
            copy_fill_value_into(&data_type::uint16(), &FillValue::from(1u16), target).is_err()
        );
    }

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

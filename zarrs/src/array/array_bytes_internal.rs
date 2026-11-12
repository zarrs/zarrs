//! Internal utilities for handling array bytes.

use std::num::NonZeroU64;
use std::sync::Arc;

use itertools::Itertools;
use unsafe_cell_slice::UnsafeCellSlice;

use super::iterators::ContiguousLinearisedIndices;
use super::{ArrayIndices, ArraySubset, DataType, ElementOrder, Indexer};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayBytesOffsets,
    ArrayBytesOffsetsCreateError, ArrayBytesVariableLength, CodecError,
    ExpectedFixedLengthBytesError, ExpectedOptionalBytesError, InvalidBytesLengthError,
};

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

    /// Return a decode target covering the entire array.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the views cannot be created.
    pub(crate) fn target(&mut self) -> Result<ArrayBytesDecodeIntoTarget<'_>, CodecError> {
        self.target_in_order(ElementOrder::C)
    }

    /// Return a decode target covering the entire array, with elements in `order`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the views cannot be created.
    pub(crate) fn target_in_order(
        &mut self,
        order: ElementOrder,
    ) -> Result<ArrayBytesDecodeIntoTarget<'_>, CodecError> {
        let strides = order.strides(&self.shape);
        let data_view = unsafe {
            // SAFETY: the view is the only view of the data, and distinct indices of a compact layout address distinct elements
            ArrayBytesFixedDisjointView::new_strided(
                UnsafeCellSlice::new(self.data.spare_capacity_mut()),
                self.data_type_size,
                0,
                &self.shape,
                &strides,
            )?
        };
        let mask_views = self
            .masks
            .iter_mut()
            .map(|mask| unsafe {
                // SAFETY: the view is the only view of the mask, with the same layout as the data
                ArrayBytesFixedDisjointView::new_strided(
                    UnsafeCellSlice::new(mask.spare_capacity_mut()),
                    1,
                    0,
                    &self.shape,
                    &strides,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ArrayBytesDecodeIntoTarget::Fixed(data_view).with_optional_masks(mask_views)?)
    }

    /// Convert the buffers into array bytes.
    ///
    /// # Safety
    /// Every element of the target returned by [`target`](Self::target) or [`target_in_order`](Self::target_in_order) must have been written.
    pub(crate) unsafe fn into_array_bytes(mut self) -> ArrayBytes<'static> {
        let num_elements = usize::try_from(self.shape.iter().product::<u64>()).unwrap();
        unsafe { self.data.set_len(num_elements * self.data_type_size) };
        for mask in &mut self.masks {
            unsafe { mask.set_len(num_elements) };
        }
        ArrayBytes::new_flen(self.data).with_optional_masks(self.masks)
    }
}

/// The elements of `bytes` at a strided grid of element indices, in C order of the grid.
///
/// The grid starts at element `offset` and has `shape` and `strides` (in elements).
///
/// # Errors
/// Returns a [`CodecError`] if `bytes` are incompatible with `data_type` or the grid addresses elements beyond them.
pub(crate) fn extract_strided(
    bytes: &ArrayBytes<'_>,
    data_type: &DataType,
    offset: u64,
    shape: &[u64],
    strides: &[u64],
) -> Result<ArrayBytes<'static>, CodecError> {
    let empty = shape.contains(&0);
    let runs = ContiguousLinearisedIndices::new_strided(
        if empty { 0 } else { offset },
        std::iter::zip(shape.iter().copied(), strides.iter().copied()),
    );
    extract_runs(bytes, data_type, &runs)
}

/// Evaluate `$body` with `$len` bound to `$length`, specialised for common small lengths.
macro_rules! with_small_len {
    ($length:expr, |$len:ident| $body:expr) => {
        match $length {
            1 => {
                let $len: usize = 1;
                $body
            }
            2 => {
                let $len: usize = 2;
                $body
            }
            4 => {
                let $len: usize = 4;
                $body
            }
            8 => {
                let $len: usize = 8;
                $body
            }
            16 => {
                let $len: usize = 16;
                $body
            }
            $len => $body,
        }
    };
}

/// The elements of `bytes` in the contiguous `runs` of element indices, in order.
fn extract_runs(
    bytes: &ArrayBytes<'_>,
    data_type: &DataType,
    runs: &ContiguousLinearisedIndices,
) -> Result<ArrayBytes<'static>, CodecError> {
    match bytes {
        ArrayBytes::Fixed(bytes) => {
            let size = data_type
                .fixed_size()
                .ok_or(ExpectedFixedLengthBytesError)?;
            Ok(ArrayBytes::new_flen(extract_runs_fixed(bytes, size, runs)?))
        }
        ArrayBytes::Variable(variable) => {
            let (bytes, offsets) = (variable.bytes(), variable.offsets());
            let num_elements = offsets.len() - 1;
            let run_elements = runs.contiguous_elements_usize();
            if let Some((index, _)) = runs.iter().next_back()
                && usize::try_from(index)
                    .ok()
                    .and_then(|start| start.checked_add(run_elements))
                    .is_none_or(|end| end > num_elements)
            {
                return Err(CodecError::InvalidVariableSizedArrayOffsets);
            }
            let bytes_len: usize = if runs.len() * run_elements == num_elements {
                // The runs cover as many elements as the bytes, e.g. reordering all elements
                offsets.get(num_elements).unwrap() - offsets.get(0).unwrap()
            } else {
                runs.iter()
                    .map(|(index, _)| {
                        let start = usize::try_from(index).unwrap();
                        offsets.get(start + run_elements).unwrap() - offsets.get(start).unwrap()
                    })
                    .sum()
            };
            let mut extracted = Vec::with_capacity(bytes_len);
            let mut extracted_offsets = Vec::with_capacity(runs.len() * run_elements + 1);
            for (index, _) in runs {
                let start = usize::try_from(index).unwrap();
                for element in start..start + run_elements {
                    extracted_offsets.push(extracted.len());
                    extracted.extend_from_slice(&bytes[offsets.element_range(element)]);
                }
            }
            extracted_offsets.push(extracted.len());
            let offsets = offsets_from_usize(extracted_offsets)?;
            // SAFETY: the offsets are monotonically increasing and the last is the length of the bytes
            Ok(unsafe { ArrayBytes::new_vlen_unchecked(extracted, offsets) })
        }
        ArrayBytes::Optional(optional) => {
            let inner_data_type = data_type
                .optional_inner()
                .ok_or(ExpectedOptionalBytesError)?;
            let data = extract_runs(optional.data(), inner_data_type, runs)?;
            let mask = extract_runs_fixed(optional.mask(), 1, runs)?;
            Ok(data.with_optional_mask(mask))
        }
    }
}

/// The bytes of the elements of `size` bytes in the contiguous `runs` of element indices of `bytes`, in order.
fn extract_runs_fixed(
    bytes: &[u8],
    size: usize,
    runs: &ContiguousLinearisedIndices,
) -> Result<Vec<u8>, InvalidBytesLengthError> {
    let run_len = runs.contiguous_elements_usize() * size;
    if let Some((index, _)) = runs.iter().next_back() {
        let end = usize::try_from(index)
            .ok()
            .and_then(|start| start.checked_mul(size))
            .and_then(|start| start.checked_add(run_len));
        if end.is_none_or(|end| end > bytes.len()) {
            return Err(InvalidBytesLengthError::new(
                bytes.len(),
                end.unwrap_or(usize::MAX),
            ));
        }
    }
    let mut extracted = Vec::with_capacity(runs.len() * run_len);
    with_small_len!(run_len, |run_len| {
        for (index, _) in runs {
            let start = usize::try_from(index).unwrap() * size;
            extracted.extend_from_slice(&bytes[start..start + run_len]);
        }
    });
    Ok(extracted)
}

/// Reorder the elements of `bytes`, an array of `shape` with elements in `from` order, to `to` order.
///
/// # Errors
/// Returns a [`CodecError`] if `bytes` are incompatible with `data_type` or `shape`.
pub(crate) fn reorder<'a>(
    bytes: ArrayBytes<'a>,
    data_type: &DataType,
    shape: &[u64],
    from: ElementOrder,
    to: ElementOrder,
) -> Result<ArrayBytes<'a>, CodecError> {
    if from == to || !ElementOrder::F.is_reordered(shape) {
        return Ok(bytes);
    }
    reordered(&bytes, data_type, shape, from, to)
}

/// Copy the elements of `bytes`, an array of `shape` with elements in `from` order, in `to` order.
///
/// # Errors
/// Returns a [`CodecError`] if `bytes` are incompatible with `data_type` or `shape`.
pub(crate) fn reordered(
    bytes: &ArrayBytes<'_>,
    data_type: &DataType,
    shape: &[u64],
    from: ElementOrder,
    to: ElementOrder,
) -> Result<ArrayBytes<'static>, CodecError> {
    bytes.validate(shape.iter().product(), data_type)?;
    // Gather the elements of the source in the C order of the destination grid
    match to {
        ElementOrder::C => extract_strided(bytes, data_type, 0, shape, &from.strides(shape)),
        ElementOrder::F => {
            // The C order of the reversed shape is the F order of `shape`
            let reversed_shape: ArrayIndices = shape.iter().rev().copied().collect();
            let mut reversed_strides = from.strides(shape);
            reversed_strides.reverse();
            extract_strided(bytes, data_type, 0, &reversed_shape, &reversed_strides)
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
    fn reorder_between_element_orders() {
        // A 2x3 array with elements 0..6 in C order
        let shape = [2, 3];
        let c_order = [0u8, 1, 2, 3, 4, 5];
        let f_order = [0u8, 3, 1, 4, 2, 5];
        let to_f = |bytes: ArrayBytes<'static>, data_type: &DataType| {
            reorder(bytes, data_type, &shape, ElementOrder::C, ElementOrder::F).unwrap()
        };
        let to_c = |bytes: ArrayBytes<'static>, data_type: &DataType| {
            reorder(bytes, data_type, &shape, ElementOrder::F, ElementOrder::C).unwrap()
        };

        // Fixed length
        let uint8 = data_type::uint8();
        let f = to_f(ArrayBytes::new_flen(c_order.to_vec()), &uint8);
        assert_eq!(f, ArrayBytes::new_flen(f_order.to_vec()));
        assert_eq!(to_c(f, &uint8), ArrayBytes::new_flen(c_order.to_vec()));

        // Fixed length elements of more than one byte
        let uint16 = data_type::uint16();
        let bytes = |order: &[u8]| -> Vec<u8> {
            order
                .iter()
                .flat_map(|&e| (u16::from(e) * 257).to_ne_bytes())
                .collect()
        };
        assert_eq!(
            to_f(ArrayBytes::new_flen(bytes(&c_order)), &uint16),
            ArrayBytes::new_flen(bytes(&f_order))
        );

        // Variable length
        let string = data_type::string();
        let vlen = |order: &[u8]| {
            let elements: Vec<String> = order.iter().map(|e| "x".repeat(usize::from(*e))).collect();
            <String as crate::array::Element>::into_array_bytes(&string, elements).unwrap()
        };
        let f = to_f(vlen(&c_order), &string);
        assert_eq!(f, vlen(&f_order));
        assert_eq!(to_c(f, &string), vlen(&c_order));

        // Optional
        let optional = data_type::uint8().to_optional();
        let mask = |order: &[u8]| {
            order
                .iter()
                .map(|e| u8::from(e % 2 == 0))
                .collect::<Vec<u8>>()
        };
        let opt =
            |order: &[u8]| ArrayBytes::new_flen(order.to_vec()).with_optional_mask(mask(order));
        let f = to_f(opt(&c_order), &optional);
        assert_eq!(f, opt(&f_order));
        assert_eq!(to_c(f, &optional), opt(&c_order));

        // Arrays with at most one dimension of more than one element are not reordered
        for shape in [vec![6], vec![1, 6], vec![6, 1], vec![]] {
            let bytes = ArrayBytes::new_flen(vec![
                0u8;
                usize::try_from(shape.iter().product::<u64>())
                    .unwrap()
            ]);
            let reordered = reorder(
                bytes.clone(),
                &uint8,
                &shape,
                ElementOrder::C,
                ElementOrder::F,
            )
            .unwrap();
            assert_eq!(reordered, bytes);
        }

        // Bytes of the wrong length are rejected
        assert!(
            reorder(
                ArrayBytes::new_flen(vec![0u8; 5]),
                &uint8,
                &shape,
                ElementOrder::C,
                ElementOrder::F
            )
            .is_err()
        );
    }

    #[test]
    fn extract_strided_elements() {
        // The column 1 of a 3x4 array, and a 2x2 block in F order of a 3x4 F-order array
        let bytes = ArrayBytes::new_flen((0u8..12).collect::<Vec<u8>>());
        let uint8 = data_type::uint8();
        assert_eq!(
            extract_strided(&bytes, &uint8, 1, &[3], &[4]).unwrap(),
            ArrayBytes::new_flen(vec![1u8, 5, 9])
        );
        assert_eq!(
            extract_strided(&bytes, &uint8, 4, &[2, 2], &[1, 3]).unwrap(),
            ArrayBytes::new_flen(vec![4u8, 7, 5, 8])
        );
        // Out of bounds grids are rejected
        assert!(extract_strided(&bytes, &uint8, 4, &[3], &[4]).is_err());
        let string = data_type::string();
        let vlen = <String as crate::array::Element>::into_array_bytes(
            &string,
            vec!["a".to_string(), "b".to_string()],
        )
        .unwrap();
        assert!(extract_strided(&vlen, &string, 1, &[2], &[1]).is_err());
        // Empty grids extract nothing
        assert_eq!(
            extract_strided(&bytes, &uint8, 99, &[0, 2], &[1, 3]).unwrap(),
            ArrayBytes::new_flen(Vec::<u8>::new())
        );
    }

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
                        let data_view = unsafe {
                            ArrayBytesFixedDisjointView::new(
                                UnsafeCellSlice::new(&mut buffers.data),
                                buffers.data_type_size,
                                &shape,
                                subset.clone(),
                            )
                            .unwrap()
                        };
                        let mask_views: Vec<_> = buffers
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
                        let mut target = ArrayBytesDecodeIntoTarget::from(data_view)
                            .with_optional_masks(mask_views)
                            .unwrap();
                        if direct {
                            copy_fill_value_into(&data_type, &fill_value, &mut target).unwrap();
                        } else {
                            let bytes = ArrayBytes::new_fill_value(
                                &data_type,
                                subset.num_elements(),
                                &fill_value,
                            )
                            .unwrap();
                            zarrs_codec::decode_into_array_bytes_target(&bytes, &mut target)
                                .unwrap();
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
            let mut target = buffers.target().unwrap();
            assert!(copy_fill_value_into(&data_type, &fill_value, &mut target).is_err());
        }

        let mut buffers =
            FixedDecodeBuffers::new(&data_type::uint16().to_optional(), &[3]).unwrap();
        let mut target = buffers.target().unwrap();
        assert!(
            copy_fill_value_into(&data_type::uint16(), &FillValue::from(1u16), &mut target)
                .is_err()
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

    #[test]
    fn validate_optional_masks() {
        use crate::array::ArrayBytesOffsets;

        let fixed = || ArrayBytes::new_flen(vec![7u8; 3]);
        let variable = || {
            ArrayBytes::new_vlen(b"abc", ArrayBytesOffsets::new(vec![0u32, 1, 2, 3]).unwrap())
                .unwrap()
        };
        let payloads: [(fn() -> ArrayBytes<'static>, DataType); 2] =
            [(fixed, data_type::uint8()), (variable, data_type::string())];
        for (payload, data_type) in payloads {
            let depth1 = data_type.to_optional();
            let depth2 = depth1.to_optional();
            assert!(
                payload()
                    .with_optional_mask(vec![1, 0, 1])
                    .validate(3, &depth1)
                    .is_ok()
            );
            assert!(
                payload()
                    .with_optional_mask(vec![1, 0, 1])
                    .with_optional_mask(vec![0, 1, 1])
                    .validate(3, &depth2)
                    .is_ok()
            );
            // Short, long, empty and nonbinary masks
            for mask in [vec![0u8; 2], vec![0; 4], vec![], vec![1, 2, 1]] {
                assert!(
                    payload()
                        .with_optional_mask(mask.clone())
                        .validate(3, &depth1)
                        .is_err(),
                    "{mask:?}"
                );
                // At the outer layer
                assert!(
                    payload()
                        .with_optional_mask(vec![1, 0, 1])
                        .with_optional_mask(mask.clone())
                        .validate(3, &depth2)
                        .is_err()
                );
                // At the inner layer, including beneath null outer elements
                for outer_mask in [[1, 1, 1], [1, 0, 1], [0, 0, 0]] {
                    assert!(
                        payload()
                            .with_optional_mask(mask.clone())
                            .with_optional_mask(outer_mask.to_vec())
                            .validate(3, &depth2)
                            .is_err()
                    );
                }
            }
        }
        let empty = ArrayBytes::new_flen(vec![]).with_optional_mask(vec![]);
        assert!(empty.validate(0, &data_type::uint8().to_optional()).is_ok());
    }
}

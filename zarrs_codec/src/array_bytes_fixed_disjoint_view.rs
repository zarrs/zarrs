use derive_more::derive::Display;
use std::sync::OnceLock;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::iterators::ContiguousLinearisedIndices;
use zarrs_chunk_grid::{ArraySubset, IndexerError};

use crate::{CodecError, InvalidBytesLengthError, SubsetOutOfBoundsError};

/// A disjoint view of the bytes in an array with a fixed-length data type.
///
/// The `subset` represented by this view must not overlap with the `subset` of any other created views that reference the same array bytes.
pub struct ArrayBytesFixedDisjointView<'a> {
    bytes: UnsafeCellSlice<'a, u8>,
    data_type_size: usize,
    shape: &'a [u64],
    subset: ArraySubset,
    bytes_in_subset_len: usize,
    /// The contiguous regions of the view, computed on first use.
    contiguous_linearised_indices: OnceLock<ContiguousLinearisedIndices>,
}

/// Errors that can occur when creating a [`ArrayBytesFixedDisjointView`].
#[derive(Clone, Debug, Display, Error)]
pub enum ArrayBytesFixedDisjointViewCreateError {
    /// The subset is out-of-bounds of, or has an incompatible dimensionality with, the array shape.
    SubsetOutOfBounds(#[from] SubsetOutOfBoundsError),
    /// The length of the bytes is not the correct length.
    InvalidBytesLength(#[from] InvalidBytesLengthError),
    /// The indexer and shape are incompatible.
    IndexerError(#[from] IndexerError),
}

impl From<ArrayBytesFixedDisjointViewCreateError> for CodecError {
    fn from(value: ArrayBytesFixedDisjointViewCreateError) -> Self {
        match value {
            ArrayBytesFixedDisjointViewCreateError::SubsetOutOfBounds(e) => e.into(),
            ArrayBytesFixedDisjointViewCreateError::InvalidBytesLength(e) => e.into(),
            ArrayBytesFixedDisjointViewCreateError::IndexerError(e) => e.into(),
        }
    }
}

impl<'a> ArrayBytesFixedDisjointView<'a> {
    /// Create a new non-overlapping view of the bytes in an array.
    ///
    /// An empty `subset` references no elements, so it is accepted irrespective of its start and
    /// yields a view that reads and writes nothing.
    ///
    /// # Errors
    /// Returns [`ArrayBytesFixedDisjointViewCreateError`] if
    /// - `subset` has an incompatible dimensionality with, or is out-of-bounds of, `shape`, or
    /// - the length of `bytes` is not the product of the elements in `shape` multiplied by `data_type_size`.
    ///
    /// # Safety
    /// The `subset` represented by this view must not overlap with the `subset` of any other created views that reference the same array bytes.
    ///
    /// # Panics
    /// Panics if the product of the elements in `shape` multiplied by `data_type_size` exceeds [`usize::MAX`].
    pub unsafe fn new(
        bytes: UnsafeCellSlice<'a, u8>,
        data_type_size: usize,
        shape: &'a [u64],
        subset: ArraySubset,
    ) -> Result<Self, ArrayBytesFixedDisjointViewCreateError> {
        if !subset.inbounds_shape(shape) {
            let bounding_subset = ArraySubset::new_with_shape(shape);
            return Err(SubsetOutOfBoundsError::new(subset, bounding_subset).into());
        }
        let bytes_in_array_len =
            usize::try_from(shape.iter().product::<u64>()).unwrap() * data_type_size;
        if bytes.len() != bytes_in_array_len {
            return Err(InvalidBytesLengthError::new(bytes.len(), bytes_in_array_len).into());
        }

        let bytes_in_subset_len = subset.num_elements_usize() * data_type_size;
        Ok(Self {
            bytes,
            data_type_size,
            shape,
            subset,
            bytes_in_subset_len,
            contiguous_linearised_indices: OnceLock::new(),
        })
    }

    /// The contiguous regions of the view.
    fn contiguous_linearised_indices(&self) -> &ContiguousLinearisedIndices {
        self.contiguous_linearised_indices.get_or_init(|| {
            self.subset
                .contiguous_linearised_indices(self.shape)
                .expect("the subset is within the shape")
        })
    }

    /// Create a new non-overlapping view of the bytes in an array that is a subset of the current view.
    ///
    /// An empty `subset` references no elements, so it is accepted irrespective of its start.
    ///
    /// # Errors
    /// Returns [`SubsetOutOfBoundsError`] if `subset` has an incompatible dimensionality with, or is out-of-bounds of, the parent subset.
    ///
    /// # Safety
    /// The `subset` represented by this view must not overlap with the `subset` of any other created views that reference the same array bytes.
    pub unsafe fn subdivide(
        &self,
        subset: ArraySubset,
    ) -> Result<ArrayBytesFixedDisjointView<'a>, ArrayBytesFixedDisjointViewCreateError> {
        if !subset.inbounds(&self.subset) {
            return Err(SubsetOutOfBoundsError::new(subset, self.subset.clone()).into());
        }
        unsafe { Self::new(self.bytes, self.data_type_size, self.shape, subset) }
    }

    /// Return the shape of the bytes this view is created from.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.shape
    }

    /// Return the subset of the bytes this view is created from.
    #[must_use]
    pub fn subset(&self) -> &ArraySubset {
        &self.subset
    }

    /// Return the number of elements in the view.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.subset.num_elements()
    }

    /// Return the contiguous element length of the view.
    ///
    /// This is the number of elements that are accessed in a single contiguous block.
    #[must_use]
    pub fn num_contiguous_elements(&self) -> usize {
        self.contiguous_linearised_indices()
            .contiguous_elements_usize()
    }

    /// Return the size in bytes of contiguous elements in the view.
    ///
    /// This is the number of elements that are accessed in a single contiguous block.
    #[must_use]
    pub fn contiguous_bytes_len(&self) -> usize {
        self.contiguous_linearised_indices()
            .contiguous_elements_usize()
            * self.data_type_size
    }

    /// Fill the view with the fill value.
    ///
    /// # Errors
    /// Returns [`InvalidBytesLengthError`] if the length of the `fill_value` does not match the data type size.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    pub fn fill(&mut self, fill_value: &[u8]) -> Result<(), InvalidBytesLengthError> {
        if fill_value.len() != self.data_type_size {
            return Err(InvalidBytesLengthError::new(
                fill_value.len(),
                self.data_type_size,
            ));
        }

        let length = self.contiguous_bytes_len();
        if length == 0 {
            return Ok(());
        }

        let uniform = fill_value.iter().all(|&byte| byte == fill_value[0]);
        let mut regions = self.contiguous_linearised_indices().iter();
        let (first_index, _) = regions.next().expect("a nonempty view has a region");
        let first_offset = usize::try_from(first_index * self.data_type_size as u64).unwrap();
        // SAFETY: this region belongs to the disjoint view and is only accessed here.
        let first = unsafe { self.bytes.index_mut(first_offset..first_offset + length) };
        if uniform {
            first.fill(fill_value[0]);
        } else {
            // Seed one element, then double the initialised prefix in place.
            first[..fill_value.len()].copy_from_slice(fill_value);
            let mut written = fill_value.len();
            while written < length {
                let count = written.min(length - written);
                let (prefix, remaining) = first.split_at_mut(written);
                remaining[..count].copy_from_slice(&prefix[..count]);
                written += count;
            }
        }

        // Reuse the first filled region rather than constructing a temporary row.
        for (index, _) in regions {
            let offset = usize::try_from(index * self.data_type_size as u64).unwrap();
            // SAFETY: the regions of an array subset are disjoint, including from `first`.
            let target = unsafe { self.bytes.index_mut(offset..offset + length) };
            if uniform {
                target.fill(fill_value[0]);
            } else {
                target.copy_from_slice(first);
            }
        }
        Ok(())
    }

    /// Copy bytes into the view.
    ///
    /// The `subset_bytes` must be the same length as the byte length of the elements in the view.
    ///
    /// # Errors
    /// Returns an [`InvalidBytesLengthError`] if the length of `subset_bytes` is not the same as the byte length of the elements in the view.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    pub fn copy_from_slice(&mut self, subset_bytes: &[u8]) -> Result<(), InvalidBytesLengthError> {
        if subset_bytes.len() != self.bytes_in_subset_len {
            return Err(InvalidBytesLengthError::new(
                subset_bytes.len(),
                self.bytes_in_subset_len,
            ));
        }
        let data_type_size = self.data_type_size as u64;
        let bytes_copied = self.contiguous_linearised_indices().iter().fold(
            0,
            |subset_offset: usize,
             (array_subset_element_index, contiguous_elements): (u64, u64)| {
                let output_offset =
                    usize::try_from(array_subset_element_index * data_type_size).unwrap();
                let length = usize::try_from(contiguous_elements * data_type_size).unwrap();
                debug_assert!((output_offset + length) <= self.bytes.len());
                debug_assert!((subset_offset + length) <= subset_bytes.len());
                let subset_offset_end = subset_offset + length;
                unsafe {
                    self.bytes
                        .index_mut(output_offset..output_offset + length)
                        .copy_from_slice(&subset_bytes[subset_offset..subset_offset_end]);
                }
                subset_offset_end
            },
        );
        debug_assert_eq!(bytes_copied, subset_bytes.len());

        Ok(())
    }

    /// Copy elements into the view at `positions`.
    ///
    /// Element `i` of `bytes` is written to element `positions[i]` of the view, where positions index the elements of the view in C order.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if
    /// - the length of `bytes` is not the length of `positions` multiplied by the data type size, or
    /// - a position is out-of-bounds of the view.
    pub fn copy_elements_from_slice(
        &mut self,
        positions: &[usize],
        bytes: &[u8],
    ) -> Result<(), CodecError> {
        let size = self.data_type_size;
        if bytes.len() != positions.len() * size {
            return Err(InvalidBytesLengthError::new(bytes.len(), positions.len() * size).into());
        }
        for (&position, element) in std::iter::zip(positions, bytes.chunks_exact(size)) {
            let offset = self.element_offset(position)?;
            unsafe {
                self.bytes
                    .index_mut(offset..offset + size)
                    .copy_from_slice(element);
            }
        }
        Ok(())
    }

    /// Fill the elements of the view at `positions` with the fill value.
    ///
    /// Positions index the elements of the view in C order.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if
    /// - the length of the `fill_value` does not match the data type size, or
    /// - a position is out-of-bounds of the view.
    pub fn fill_elements(
        &mut self,
        positions: &[usize],
        fill_value: &[u8],
    ) -> Result<(), CodecError> {
        let size = self.data_type_size;
        if fill_value.len() != size {
            return Err(InvalidBytesLengthError::new(fill_value.len(), size).into());
        }
        for &position in positions {
            let offset = self.element_offset(position)?;
            unsafe {
                self.bytes
                    .index_mut(offset..offset + size)
                    .copy_from_slice(fill_value);
            }
        }
        Ok(())
    }

    /// Return the byte offset in the underlying bytes of the element at `position` in the view.
    fn element_offset(&self, position: usize) -> Result<usize, IndexerError> {
        let position = position as u64;
        if position >= self.num_elements() {
            return Err(IndexerError::new_oob(
                [position].as_slice().into(),
                vec![self.num_elements()],
            ));
        }
        let index = if self.contiguous_linearised_indices().len() == 1 {
            // The view is a single contiguous block
            let (start, _) = self
                .contiguous_linearised_indices()
                .iter()
                .next()
                .expect("one block");
            start + position
        } else {
            let mut remainder = position;
            let mut stride = 1;
            let mut index = 0;
            for ((&start, &len), &dim) in std::iter::zip(
                std::iter::zip(self.subset.start(), self.subset.shape()),
                self.shape,
            )
            .rev()
            {
                index += (start + remainder % len) * stride;
                remainder /= len;
                stride *= dim;
            }
            index
        };
        Ok(usize::try_from(index).unwrap() * self.data_type_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_view() {
        let mut bytes = (0..9).collect::<Vec<u8>>();
        let shape = vec![3, 3];
        {
            let bytes = UnsafeCellSlice::new(&mut bytes);

            assert!(
                unsafe {
                    ArrayBytesFixedDisjointView::new(
                        bytes,
                        1,
                        &[10, 10],
                        ArraySubset::new_with_ranges(&[0..2, 1..3]),
                    )
                }
                .is_err()
            ); // incompatible shape
            assert!(
                unsafe {
                    ArrayBytesFixedDisjointView::new(
                        bytes,
                        2,
                        &shape,
                        ArraySubset::new_with_ranges(&[0..2, 1..3]),
                    )
                }
                .is_err()
            ); // invalid bytes length
            assert!(
                unsafe {
                    ArrayBytesFixedDisjointView::new(
                        bytes,
                        1,
                        &shape,
                        ArraySubset::new_with_ranges(&[0..2, 1..10]),
                    )
                }
                .is_err()
            ); // OOB

            let mut view0 = unsafe {
                ArrayBytesFixedDisjointView::new(
                    bytes,
                    1,
                    &shape,
                    ArraySubset::new_with_ranges(&[0..2, 1..3]),
                )
            }
            .unwrap();
            assert_eq!(view0.shape(), shape);

            view0.copy_from_slice(&[11, 12, 14, 15]).unwrap();
            assert!(view0.copy_from_slice(&[11, 12, 14, 15, 255]).is_err()); // wrong length

            let mut view0a =
                unsafe { view0.subdivide(ArraySubset::new_with_ranges(&[1..2, 1..3])) }.unwrap();
            view0a.copy_from_slice(&[24, 25]).unwrap();
            assert!(view0a.copy_from_slice(&[]).is_err()); // wrong length

            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[1..2, 1..3])) }.is_ok()
            );
            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[1..2, 2..3])) }.is_ok()
            );
            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[0..2, 1..3])) }.is_err()
            ); // OOB
            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[1..2, 1..4])) }.is_err()
            ); // OOB

            let mut view1 = unsafe {
                ArrayBytesFixedDisjointView::new(
                    bytes,
                    1,
                    &shape,
                    ArraySubset::new_with_ranges(&[2..3, 1..3]),
                )
            }
            .unwrap();
            view1.fill(&[255]).unwrap();
            assert!(view1.fill(&[255, 255]).is_err()); // invalid fill value
        }
        assert_eq!(&bytes, &[0, 11, 12, 3, 24, 25, 6, 255, 255]);
    }

    #[test]
    fn fill_patterns_and_strides() {
        for size in [0usize, 1, 2, 3, 4, 8, 16, 17] {
            for width in [0usize, 1, 2, 3, 7, 8, 9, 33, 1024] {
                for uniform in [false, true] {
                    let pattern: Vec<u8> = (0..size)
                        .map(|i| {
                            if uniform {
                                0x7b
                            } else {
                                u8::try_from(i).unwrap()
                            }
                        })
                        .collect();
                    let shape = [3, u64::try_from(width + 2).unwrap()];
                    for rows in [0..3, 1..2] {
                        let subset = ArraySubset::new_with_ranges(&[
                            rows.clone(),
                            1..u64::try_from(width + 1).unwrap(),
                        ]);
                        let mut bytes = vec![0xa5; 3 * (width + 2) * size];
                        let mut expected = bytes.clone();
                        for row in rows {
                            for col in 1..=width {
                                let offset =
                                    (usize::try_from(row).unwrap() * (width + 2) + col) * size;
                                expected[offset..offset + size].copy_from_slice(&pattern);
                            }
                        }
                        {
                            // SAFETY: this is the only view of bytes.
                            let mut view = unsafe {
                                ArrayBytesFixedDisjointView::new(
                                    UnsafeCellSlice::new(&mut bytes),
                                    size,
                                    &shape,
                                    subset,
                                )
                                .unwrap()
                            };
                            assert!(view.fill(&vec![0; size + 1]).is_err());
                            view.fill(&pattern).unwrap();
                        }
                        assert_eq!(bytes, expected, "{size} {width} {uniform}");
                    }
                }
            }
        }
    }

    #[test]
    fn empty_disjoint_view() {
        let mut bytes = vec![7u8; 18];
        let shape = [3, 3];
        {
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    2,
                    &shape,
                    ArraySubset::new_with_start_shape(vec![99, 99], vec![0, 0]).unwrap(),
                )
            }
            .unwrap();
            assert_eq!(view.num_elements(), 0);
            view.copy_from_slice(&[]).unwrap();
            view.fill(&[1, 2]).unwrap();
        }
        assert_eq!(bytes, vec![7u8; 18]);
    }

    #[test]
    fn disjoint_view_elements() {
        let shape = vec![3, 3];

        // Contiguous view
        let mut bytes = vec![0u8; 9];
        {
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    1,
                    &shape,
                    ArraySubset::new_with_ranges(&[1..3, 0..3]),
                )
            }
            .unwrap();
            view.copy_elements_from_slice(&[5, 0, 2], &[15, 10, 12])
                .unwrap();
            view.fill_elements(&[1, 3], &[255]).unwrap();
        }
        assert_eq!(&bytes, &[0, 0, 0, 10, 255, 12, 255, 0, 15]);

        // Non-contiguous view
        let mut bytes = vec![0u8; 9];
        {
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    1,
                    &shape,
                    ArraySubset::new_with_ranges(&[0..2, 1..3]),
                )
            }
            .unwrap();
            view.copy_elements_from_slice(&[3, 0], &[14, 11]).unwrap();
            view.fill_elements(&[2, 1], &[255]).unwrap();
            assert!(view.copy_elements_from_slice(&[4], &[1]).is_err()); // OOB
            assert!(view.copy_elements_from_slice(&[0], &[1, 2]).is_err()); // wrong length
            assert!(view.fill_elements(&[4], &[1]).is_err()); // OOB
            assert!(view.fill_elements(&[0], &[1, 2]).is_err()); // invalid fill value
        }
        assert_eq!(&bytes, &[0, 11, 255, 0, 255, 14, 0, 0, 0]);
    }
}

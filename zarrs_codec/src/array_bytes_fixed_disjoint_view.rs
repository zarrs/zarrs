use derive_more::derive::Display;
use std::sync::OnceLock;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;

use zarrs_chunk_grid::iterators::ContiguousLinearisedIndices;
use zarrs_chunk_grid::{ArraySubset, ArraySubsetTraits, IndexerError};

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
            let bounding_subset = ArraySubset::new_with_shape(shape.to_vec());
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
        self.num_contiguous_elements() * self.data_type_size
    }

    /// Return the complete view as a mutable slice if it occupies one contiguous region.
    ///
    /// Returns [`None`] if the view is empty or occupies more than one contiguous region of the
    /// underlying bytes.
    ///
    /// The slice covers only the bytes of this view. It does not alias the bytes of any other
    /// view of the same array bytes, since [`new`](Self::new) requires views to be disjoint.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    #[must_use]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        let (index, _) = self.single_contiguous_block()?;
        let offset = usize::try_from(index * self.data_type_size as u64).unwrap();
        Some(unsafe {
            self.bytes
                .index_mut(offset..offset + self.bytes_in_subset_len)
        })
    }

    /// Return the start element index and number of elements of the view if it is a single contiguous block.
    fn single_contiguous_block(&self) -> Option<(u64, u64)> {
        let contiguous_linearised_indices = self.contiguous_linearised_indices();
        if contiguous_linearised_indices.len() == 1 {
            contiguous_linearised_indices.iter().next()
        } else {
            None
        }
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

        let fill_value_contiguous = fill_value.repeat(self.num_contiguous_elements());
        let length = self.contiguous_bytes_len();
        debug_assert_eq!(fill_value_contiguous.len(), length);
        self.contiguous_linearised_indices()
            .iter()
            .for_each(|(index, _contiguous_elements)| {
                let offset = usize::try_from(index * self.data_type_size as u64).unwrap();
                unsafe {
                    self.bytes
                        .index_mut(offset..offset + length)
                        .copy_from_slice(&fill_value_contiguous);
                }
            });
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
        self.try_copy_from_slice_with(subset_bytes, |source, destination| {
            destination.copy_from_slice(source);
            Ok(())
        })
    }

    /// Copy bytes into the view with a custom copy for each contiguous region.
    ///
    /// The `subset_bytes` must be the same length as the byte length of the elements in the view.
    /// `copy` is called with each contiguous region of `subset_bytes` and the region of the view it is destined for, which have the same length.
    /// This can transform bytes as they are copied, e.g. to change their endianness.
    ///
    /// # Errors
    /// Returns an [`InvalidBytesLengthError`] if the length of `subset_bytes` is not the same as the byte length of the elements in the view, or the first error returned by `copy`.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    pub fn try_copy_from_slice_with<E: From<InvalidBytesLengthError>>(
        &mut self,
        subset_bytes: &[u8],
        mut copy: impl FnMut(&[u8], &mut [u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        if subset_bytes.len() != self.bytes_in_subset_len {
            return Err(
                InvalidBytesLengthError::new(subset_bytes.len(), self.bytes_in_subset_len).into(),
            );
        }
        let data_type_size = self.data_type_size as u64;
        let mut subset_offset = 0;
        for (array_subset_element_index, contiguous_elements) in
            self.contiguous_linearised_indices()
        {
            let output_offset =
                usize::try_from(array_subset_element_index * data_type_size).unwrap();
            let length = usize::try_from(contiguous_elements * data_type_size).unwrap();
            debug_assert!((output_offset + length) <= self.bytes.len());
            debug_assert!((subset_offset + length) <= subset_bytes.len());
            let subset_offset_end = subset_offset + length;
            copy(&subset_bytes[subset_offset..subset_offset_end], unsafe {
                self.bytes.index_mut(output_offset..output_offset + length)
            })?;
            subset_offset = subset_offset_end;
        }
        debug_assert_eq!(subset_offset, subset_bytes.len());

        Ok(())
    }

    /// Copy the elements of `subset` of `source`, the bytes of an array of `source_shape`, into the view.
    ///
    /// The subset must have as many elements as the view. This is equivalent to extracting the subset and copying it
    /// with [`copy_from_slice`](Self::copy_from_slice), without the intermediate copy.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the length of `source` is not that of an array of `source_shape`, `subset` is
    /// out-of-bounds of `source_shape`, or `subset` does not have as many elements as the view.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    pub fn copy_from_subset(
        &mut self,
        source: &[u8],
        source_shape: &[u64],
        subset: &dyn ArraySubsetTraits,
    ) -> Result<(), CodecError> {
        self.try_copy_from_subset_with(source, source_shape, subset, |source, destination| {
            destination.copy_from_slice(source);
            Ok(())
        })
    }

    /// Copy the elements of `subset` of `source` into the view with a custom copy for each contiguous region.
    ///
    /// This is [`copy_from_subset`](Self::copy_from_subset) where `copy` is called with each contiguous region of
    /// `source` and the region of the view it is destined for, which have the same length and hold whole elements.
    /// This can transform bytes as they are copied, e.g. to change their endianness.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the length of `source` is not that of an array of `source_shape`, `subset` is
    /// out-of-bounds of `source_shape`, `subset` does not have as many elements as the view, or `copy` fails.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    pub fn try_copy_from_subset_with(
        &mut self,
        source: &[u8],
        source_shape: &[u64],
        subset: &dyn ArraySubsetTraits,
        mut copy: impl FnMut(&[u8], &mut [u8]) -> Result<(), CodecError>,
    ) -> Result<(), CodecError> {
        let size = self.data_type_size;
        let source_len = usize::try_from(source_shape.iter().product::<u64>()).unwrap() * size;
        if source.len() != source_len {
            return Err(InvalidBytesLengthError::new(source.len(), source_len).into());
        }
        let subset_len = subset.num_elements_usize() * size;
        if subset_len != self.bytes_in_subset_len {
            return Err(InvalidBytesLengthError::new(subset_len, self.bytes_in_subset_len).into());
        }
        let to_bytes = |elements: u64| usize::try_from(elements * size as u64).unwrap();
        // Copy the contiguous runs of the subset into the contiguous runs of the view, which have the same elements in
        // C order but may be split differently
        let mut destinations = self.contiguous_linearised_indices().iter();
        let (mut destination, mut destination_len) = (0, 0);
        for (index, elements) in &subset.contiguous_linearised_indices(source_shape)? {
            let (mut offset, mut len) = (to_bytes(index), to_bytes(elements));
            while len > 0 {
                if destination_len == 0 {
                    let (index, elements) = destinations
                        .next()
                        .expect("the view has as many elements as the subset");
                    (destination, destination_len) = (to_bytes(index), to_bytes(elements));
                }
                let n = len.min(destination_len);
                copy(&source[offset..offset + n], unsafe {
                    self.bytes.index_mut(destination..destination + n)
                })?;
                (offset, len) = (offset + n, len - n);
                (destination, destination_len) = (destination + n, destination_len - n);
            }
        }
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
                vec![position],
                vec![self.num_elements()],
            ));
        }
        let index = if let Some((start, _)) = self.single_contiguous_block() {
            // The view is a single contiguous block
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
            assert!(view0.as_mut_slice().is_none());

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
            assert_eq!(view1.as_mut_slice().unwrap(), &[7, 8]);
            view1.fill(&[255]).unwrap();
            assert!(view1.fill(&[255, 255]).is_err()); // invalid fill value
        }
        assert_eq!(&bytes, &[0, 11, 12, 3, 24, 25, 6, 255, 255]);
    }

    #[test]
    fn disjoint_view_try_copy_from_slice_with() {
        // A 2x2 view of a 3x3 array of 2 byte elements is two contiguous regions
        let shape = vec![3, 3];
        let mut bytes = vec![0u8; 18];
        {
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    2,
                    &shape,
                    ArraySubset::new_with_ranges(&[1..3, 0..2]),
                )
            }
            .unwrap();
            let mut regions = 0;
            // Swap the bytes of each element as they are copied
            view.try_copy_from_slice_with::<InvalidBytesLengthError>(
                &[1, 2, 3, 4, 5, 6, 7, 8],
                |source, destination| {
                    regions += 1;
                    assert_eq!(source.len(), 4);
                    assert_eq!(destination.len(), 4);
                    destination.copy_from_slice(source);
                    for element in destination.as_chunks_mut::<2>().0 {
                        element.reverse();
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(regions, 2);

            // Wrong length
            assert!(
                view.try_copy_from_slice_with::<InvalidBytesLengthError>(&[0; 7], |_, _| Ok(()))
                    .is_err()
            );
            // Errors from the copy are returned
            assert!(
                view.try_copy_from_slice_with(&[0; 8], |_, _| {
                    Err::<(), _>(InvalidBytesLengthError::new(0, 1))
                })
                .is_err()
            );
        }
        assert_eq!(
            bytes,
            [0, 0, 0, 0, 0, 0, 2, 1, 4, 3, 0, 0, 6, 5, 8, 7, 0, 0]
        );
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

    #[test]
    fn copy_from_subset_matches_extract_and_copy() {
        let size = 2;
        let source_shape = [4u64, 5, 6];
        let source: Vec<u8> = (0..source_shape.iter().product::<u64>() * size as u64)
            .map(|i| (i % 251) as u8)
            .collect();
        // Subsets whose contiguous runs split differently from the runs of the views
        for (subset, output_shape, output_subset) in [
            (
                ArraySubset::new_with_ranges(&[1..3, 0..5, 0..6]),
                vec![2u64, 5, 6],
                ArraySubset::new_with_shape(vec![2, 5, 6]),
            ),
            (
                ArraySubset::new_with_ranges(&[1..3, 2..4, 1..5]),
                vec![3, 4, 6],
                ArraySubset::new_with_ranges(&[1..3, 1..3, 2..6]),
            ),
            (
                ArraySubset::new_with_ranges(&[0..4, 0..5, 0..6]),
                vec![5, 5, 7],
                ArraySubset::new_with_ranges(&[1..5, 0..5, 1..7]),
            ),
            (
                ArraySubset::new_with_ranges(&[2..2, 0..5, 0..6]),
                vec![1, 5, 6],
                ArraySubset::new_with_ranges(&[0..0, 0..5, 0..6]),
            ),
        ] {
            let len = usize::try_from(output_shape.iter().product::<u64>()).unwrap() * size;
            let mut expected = vec![0u8; len];
            let mut fused = vec![0u8; len];
            let extracted: Vec<u8> = subset
                .contiguous_linearised_indices(&source_shape)
                .unwrap()
                .iter()
                .flat_map(|(index, elements)| {
                    let start = usize::try_from(index).unwrap() * size;
                    source[start..start + usize::try_from(elements).unwrap() * size].to_vec()
                })
                .collect();
            unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut expected),
                    size,
                    &output_shape,
                    output_subset.clone(),
                )
            }
            .unwrap()
            .copy_from_slice(&extracted)
            .unwrap();
            unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut fused),
                    size,
                    &output_shape,
                    output_subset,
                )
            }
            .unwrap()
            .copy_from_subset(&source, &source_shape, &subset)
            .unwrap();
            assert_eq!(fused, expected);
        }

        // The subset must have as many elements as the view
        let mut output = vec![0u8; 8];
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut output),
                size,
                &[4],
                ArraySubset::new_with_shape(vec![4]),
            )
        }
        .unwrap();
        let subset = ArraySubset::new_with_ranges(&[0..1, 0..1, 0..3]);
        assert!(
            view.copy_from_subset(&source, &source_shape, &subset)
                .is_err()
        );
        assert!(
            view.copy_from_subset(&source[1..], &source_shape, &subset)
                .is_err()
        );
    }
}

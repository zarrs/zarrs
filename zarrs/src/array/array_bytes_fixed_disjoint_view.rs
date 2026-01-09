use derive_more::derive::Display;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;

use super::codec::{CodecError, InvalidBytesLengthError, SubsetOutOfBoundsError};
use crate::array::iterators::{ContiguousIndices, ContiguousLinearisedIndices};
use crate::array::{ArraySubset, IndexerError};

/// A disjoint view of the bytes in an array with a fixed-length data type.
///
/// The `subset` represented by this view must not overlap with the `subset` of any other created views that reference the same array bytes.
pub struct ArrayBytesFixedDisjointView<'a> {
    bytes: UnsafeCellSlice<'a, u8>,
    data_type_size: usize,
    shape: &'a [u64],
    subset: ArraySubset,
    bytes_in_subset_len: usize,
    contiguous_indices: ContiguousIndices,
    contiguous_linearised_indices: ContiguousLinearisedIndices,
}

/// Errors that can occur when creating a [`ArrayBytesFixedDisjointView`].
#[derive(Clone, Debug, Display, Error)]
pub enum ArrayBytesFixedDisjointViewCreateError {
    /// The subset is out-of-bounds of the array shape.
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
    /// # Errors
    /// Returns [`ArrayBytesFixedDisjointViewCreateError`] if
    /// - `subset` is out-of-bounds of `shape`, or
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
        let contiguous_indices = subset.contiguous_indices(shape)?;
        let contiguous_linearised_indices = subset.contiguous_linearised_indices(shape)?;
        Ok(Self {
            bytes,
            data_type_size,
            shape,
            subset,
            bytes_in_subset_len,
            contiguous_indices,
            contiguous_linearised_indices,
        })
    }

    /// Create a new non-overlapping view of the bytes in an array that is a subset of the current view.
    ///
    /// # Errors
    /// Returns [`SubsetOutOfBoundsError`] if `subset` is out-of-bounds of the parent subset.
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
        self.contiguous_indices.contiguous_elements_usize()
    }

    /// Return the size in bytes of contiguous elements in the view.
    ///
    /// This is the number of elements that are accessed in a single contiguous block.
    #[must_use]
    pub fn contiguous_bytes_len(&self) -> usize {
        self.contiguous_indices.contiguous_elements_usize() * self.data_type_size
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
        self.contiguous_linearised_indices
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
        if subset_bytes.len() != self.bytes_in_subset_len {
            return Err(InvalidBytesLengthError::new(
                subset_bytes.len(),
                self.bytes_in_subset_len,
            ));
        }
        let data_type_size = self.data_type_size as u64;
        let bytes_copied = self.contiguous_linearised_indices.iter().fold(
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
}

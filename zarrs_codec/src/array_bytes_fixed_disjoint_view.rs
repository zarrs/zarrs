use derive_more::derive::Display;
use std::mem::MaybeUninit;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::iterators::{
    ContiguousLinearisedIndices, ContiguousLinearisedIndicesIterator,
};
use zarrs_chunk_grid::{ArraySubset, IndexerError};

use crate::small_copy::with_small_len;
use crate::{CodecError, InvalidBytesLengthError, SubsetOutOfBoundsError, write_copy_of_slice};

/// A disjoint view of the bytes in an array with a fixed-length data type.
///
/// The `subset` represented by this view must not overlap with the `subset` of any other created views that reference the same array bytes.
///
/// The bytes may be uninitialised, such as the spare capacity of a [`Vec`], so the view only writes them and never exposes them for reading.
pub struct ArrayBytesFixedDisjointView<'a> {
    bytes: UnsafeCellSlice<'a, MaybeUninit<u8>>,
    data_type_size: usize,
    shape: &'a [u64],
    subset: ArraySubset,
    bytes_in_subset_len: usize,
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
    /// `bytes` may be initialised (`UnsafeCellSlice<u8>`) or uninitialised (`UnsafeCellSlice<MaybeUninit<u8>>`, e.g. from [`Vec::spare_capacity_mut`]).
    /// The view never writes uninitialised bytes, so initialised `bytes` remain initialised.
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
        bytes: impl Into<UnsafeCellSlice<'a, MaybeUninit<u8>>>,
        data_type_size: usize,
        shape: &'a [u64],
        subset: ArraySubset,
    ) -> Result<Self, ArrayBytesFixedDisjointViewCreateError> {
        let bytes = bytes.into();
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
        })
    }

    /// The contiguous regions of the view.
    fn contiguous_linearised_indices(&self) -> ContiguousLinearisedIndices {
        self.subset
            .contiguous_linearised_indices(self.shape)
            .expect("the subset is within the shape")
    }

    /// The `length` bytes starting at element `element_index` of the underlying array.
    ///
    /// # Safety
    /// The region must belong to this view and must not be accessed through any other reference while the returned slice is live.
    ///
    /// # Panics
    /// Panics if the byte offset of `element_index` exceeds [`usize::MAX`].
    #[expect(
        clippy::mut_from_ref,
        reason = "regions of a disjoint view are exclusive"
    )]
    unsafe fn region_mut(&self, element_index: u64, length: usize) -> &mut [MaybeUninit<u8>] {
        let offset = usize::try_from(element_index * self.data_type_size as u64).unwrap();
        unsafe { self.bytes.index_mut(offset..offset + length) }
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

    /// Return the size in bytes of each element.
    #[must_use]
    pub fn data_type_size(&self) -> usize {
        self.data_type_size
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

    /// Return the complete view as a mutable slice if it occupies one contiguous region.
    ///
    /// Returns [`None`] if the view is empty or occupies more than one contiguous region of the
    /// underlying bytes.
    ///
    /// The slice covers only the bytes of this view. It does not alias the bytes of any other
    /// view of the same array bytes, since [`new`](Self::new) requires views to be disjoint.
    ///
    /// # Safety
    /// The bytes may be uninitialised, so they must be written before they are read.
    /// The caller must never write uninitialised values to the slice, and must initialise all of it before reporting success to the owner of the view's bytes, which may then assume that they are initialised.
    /// For example, an [`ArrayToBytesCodecTraits::decode_into`](crate::ArrayToBytesCodecTraits::decode_into) implementation must initialise the slice before returning `Ok`.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    #[must_use]
    pub unsafe fn as_mut_slice(&mut self) -> Option<&mut [MaybeUninit<u8>]> {
        let indices = self.contiguous_linearised_indices();
        if indices.len() != 1 {
            return None;
        }
        let (index, _) = indices.iter().next()?;
        // SAFETY: this mutable borrow exclusively accesses the single region of this disjoint view.
        Some(unsafe { self.region_mut(index, self.bytes_in_subset_len) })
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

        let indices = self.contiguous_linearised_indices();
        let length = indices.contiguous_elements_usize() * self.data_type_size;
        if length == 0 {
            return Ok(());
        }

        let uniform = fill_value.iter().all(|&byte| byte == fill_value[0]);
        let mut regions = indices.iter();
        let (first_index, _) = regions.next().expect("a nonempty view has a region");
        // SAFETY: this region belongs to the disjoint view and is only accessed here.
        let first = unsafe { self.region_mut(first_index, length) };
        if uniform {
            first.fill(MaybeUninit::new(fill_value[0]));
        } else {
            // Seed one element, then double the initialised prefix in place.
            write_copy_of_slice(&mut first[..fill_value.len()], fill_value);
            let mut written = fill_value.len();
            while written < length {
                let count = written.min(length - written);
                let (prefix, remaining) = first.split_at_mut(written);
                remaining[..count].copy_from_slice(&prefix[..count]);
                written += count;
            }
        }

        // Reuse the first filled region rather than constructing a temporary row.
        with_small_len!(length, |length| {
            let first = &first[..length];
            for (index, _) in regions {
                // SAFETY: the regions of an array subset are disjoint, including from `first`.
                let target = unsafe { self.region_mut(index, length) };
                if uniform {
                    target.fill(MaybeUninit::new(fill_value[0]));
                } else {
                    target.copy_from_slice(first);
                }
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
        let indices = self.contiguous_linearised_indices();
        let length = indices.contiguous_elements_usize() * self.data_type_size;
        if length == 0 {
            return Ok(());
        }
        with_small_len!(length, |length| {
            for ((index, _), source) in indices.iter().zip(subset_bytes.chunks_exact(length)) {
                // SAFETY: the regions of the view are disjoint from each other and from other views.
                write_copy_of_slice(unsafe { self.region_mut(index, length) }, source);
            }
        });
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
        let indices = self.contiguous_linearised_indices();
        let mut element_indices = self.element_indices(&indices);
        if size == 0 {
            // Nothing to copy, but positions are still validated (`chunks_exact` rejects a size of 0)
            for &position in positions {
                element_indices.get(position)?;
            }
            return Ok(());
        }
        with_small_len!(size, |size| {
            for (&position, element) in std::iter::zip(positions, bytes.chunks_exact(size)) {
                let index = element_indices.get(position)?;
                // SAFETY: `index` is an element of the view, which is disjoint from other views.
                write_copy_of_slice(unsafe { self.region_mut(index, size) }, element);
            }
        });
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
        let indices = self.contiguous_linearised_indices();
        let mut element_indices = self.element_indices(&indices);
        with_small_len!(size, |size| {
            let fill_value = &fill_value[..size];
            for &position in positions {
                let index = element_indices.get(position)?;
                // SAFETY: `index` is an element of the view, which is disjoint from other views.
                write_copy_of_slice(unsafe { self.region_mut(index, size) }, fill_value);
            }
        });
        Ok(())
    }

    /// Map the positions of elements in the view (in C order) to their element indices in the underlying bytes.
    fn element_indices<'i>(&self, indices: &'i ContiguousLinearisedIndices) -> ElementIndices<'i> {
        let mut regions = indices.iter();
        let region_index = regions.next().map_or(0, |(index, _)| index);
        ElementIndices {
            indices,
            regions,
            num_elements: self.num_elements(),
            contiguous_elements: indices.contiguous_elements(),
            region: 0,
            region_index,
        }
    }
}

/// Maps the positions of elements in a view (in C order) to their element indices in the underlying bytes.
///
/// Positions in the current or next contiguous region of the view are mapped without division, so ascending positions are cheap.
struct ElementIndices<'a> {
    indices: &'a ContiguousLinearisedIndices,
    /// The regions after the current region.
    regions: ContiguousLinearisedIndicesIterator<'a>,
    num_elements: u64,
    contiguous_elements: u64,
    /// The current region.
    region: u64,
    /// The element index of the first element of the current region.
    region_index: u64,
}

impl ElementIndices<'_> {
    /// Return the element index of the element at `position` in the view.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if `position` is out-of-bounds of the view.
    fn get(&mut self, position: usize) -> Result<u64, IndexerError> {
        let position = position as u64;
        if position >= self.num_elements {
            return Err(IndexerError::new_oob(
                vec![position],
                vec![self.num_elements],
            ));
        }
        let mut offset = position.wrapping_sub(self.region * self.contiguous_elements);
        if offset >= self.contiguous_elements {
            if offset - self.contiguous_elements < self.contiguous_elements {
                // The next region
                self.region += 1;
                offset -= self.contiguous_elements;
                self.region_index = self.next_region(None);
            } else {
                let region = position / self.contiguous_elements;
                let skip = if region > self.region {
                    region - self.region - 1
                } else {
                    self.regions = self.indices.iter();
                    region
                };
                self.region = region;
                offset = position - region * self.contiguous_elements;
                self.region_index = self.next_region(Some(usize::try_from(skip).unwrap()));
            }
        }
        Ok(self.region_index + offset)
    }

    /// Skip `n` regions (if any), and return the element index of the first element of the region after them.
    fn next_region(&mut self, n: Option<usize>) -> u64 {
        // Stepping to the adjacent region is the common case, so it avoids the skip in `nth`
        let region = match n {
            None => self.regions.next(),
            Some(n) => self.regions.nth(n),
        };
        region
            .expect("a position within the view is within a region")
            .0
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
            // SAFETY: the slice is not used.
            assert!(unsafe { view0.as_mut_slice() }.is_none());

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
            // SAFETY: the slice is only measured, then the view is filled.
            assert_eq!(unsafe { view1.as_mut_slice() }.unwrap().len(), 2);
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
    fn copy_patterns_and_strides() {
        for size in [0usize, 1, 2, 3, 4, 8, 16, 17] {
            for width in [0usize, 1, 2, 9] {
                let shape = [3, u64::try_from(width + 2).unwrap()];
                for rows in [0..3, 1..2] {
                    let subset = ArraySubset::new_with_ranges(&[
                        rows.clone(),
                        1..u64::try_from(width + 1).unwrap(),
                    ]);
                    let run = width * size;
                    let count = rows.clone().count() * run;
                    let input: Vec<u8> = (0..=255u8).cycle().take(count).collect();
                    let mut bytes = vec![0xa5; 3 * (width + 2) * size];
                    let mut expected = bytes.clone();
                    for (k, row) in rows.clone().enumerate() {
                        let destination = (usize::try_from(row).unwrap() * (width + 2) + 1) * size;
                        expected[destination..destination + run]
                            .copy_from_slice(&input[k * run..][..run]);
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
                        assert!(view.copy_from_slice(&vec![0; count + 1]).is_err());
                        view.copy_from_slice(&input).unwrap();
                    }
                    assert_eq!(bytes, expected, "{size} {width} {rows:?}");
                }
            }
        }
    }

    #[test]
    fn element_positions() {
        let shape = [3, 4, 5];
        let subset = ArraySubset::new_with_ranges(&[1..3, 0..3, 1..4]);
        // The element index in `shape` of each position in `subset`, in C order
        let element_index = |position: usize| {
            let (i, j, k) = (position / 9, position / 3 % 3, position % 3);
            ((1 + i) * 4 + j) * 5 + 1 + k
        };
        let dense: Vec<usize> = (0..18).collect();
        for positions in [
            dense.clone(),
            dense.iter().copied().step_by(4).collect(),
            dense.iter().copied().rev().collect(),
            vec![17, 0, 5, 5, 6, 12, 1, 2, 3, 15],
        ] {
            let values: Vec<u8> = (0..=255u8).cycle().take(positions.len() * 2).collect();
            let mut expected_copy = vec![0xa5; 60 * 2];
            let mut expected_fill = expected_copy.clone();
            for (i, &position) in positions.iter().enumerate() {
                let offset = element_index(position) * 2;
                expected_copy[offset..offset + 2].copy_from_slice(&values[i * 2..i * 2 + 2]);
                expected_fill[offset..offset + 2].copy_from_slice(&[7, 9]);
            }
            let mut copied = vec![0xa5; 60 * 2];
            let mut filled = copied.clone();
            // SAFETY: these are the only views of each of the bytes.
            unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut copied),
                    2,
                    &shape,
                    subset.clone(),
                )
                .unwrap()
                .copy_elements_from_slice(&positions, &values)
                .unwrap();
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut filled),
                    2,
                    &shape,
                    subset.clone(),
                )
                .unwrap()
                .fill_elements(&positions, &[7, 9])
                .unwrap();
            }
            assert_eq!(copied, expected_copy, "{positions:?}");
            assert_eq!(filled, expected_fill, "{positions:?}");
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
            // SAFETY: the slice is not used.
            assert!(unsafe { view.as_mut_slice() }.is_none());
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

        // Zero-sized elements
        let mut bytes: Vec<u8> = vec![];
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut bytes),
                0,
                &shape,
                ArraySubset::new_with_ranges(&[0..2, 1..3]),
            )
        }
        .unwrap();
        view.copy_elements_from_slice(&[3, 0], &[]).unwrap();
        view.fill_elements(&[2, 1], &[]).unwrap();
        assert!(view.copy_elements_from_slice(&[4], &[]).is_err()); // OOB
        assert!(view.fill_elements(&[4], &[]).is_err()); // OOB
    }
}

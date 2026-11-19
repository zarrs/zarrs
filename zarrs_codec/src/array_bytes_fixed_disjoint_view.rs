use derive_more::derive::Display;
use std::mem::MaybeUninit;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::iterators::{
    ContiguousLinearisedIndices, ContiguousLinearisedIndicesIterator,
};
use zarrs_chunk_grid::{ArrayIndices, ArraySubset, IndexerError};

use crate::array_bytes::{extract_runs_fixed, validate_runs_within};
use crate::small_copy::with_small_len;
use crate::{
    CodecError, InvalidBytesLengthError, InvalidNumberOfElementsError, SubsetOutOfBoundsError,
    write_copy_of_slice,
};
use zarrs_storage::byte_range::{ByteRange, InvalidByteRangeError};

/// A disjoint view of the bytes in an array with a fixed-length data type.
///
/// A view is a strided grid of elements: it has a [`shape`](Self::shape), and its elements are addressed in C order of
/// that shape, at a per-dimension stride (in elements) from the first element in the underlying bytes.
/// A view of an array subset has the C-order strides of the array, but views can also address elements in other orders,
/// such as a column-major (F-order) array or an array with permuted axes (see [`new_strided`](Self::new_strided)).
///
/// The elements of a view must not overlap with the elements of any other created views that reference the same array bytes.
///
/// The bytes may be uninitialised, such as the spare capacity of a [`Vec`], so the view only writes them and never exposes them for reading.
pub struct ArrayBytesFixedDisjointView<'a> {
    bytes: UnsafeCellSlice<'a, MaybeUninit<u8>>,
    data_type_size: usize,
    /// The element index of the first element in the underlying bytes, or zero if the view is empty.
    offset: u64,
    /// The number of elements along each dimension.
    shape: ArrayIndices,
    /// The distance in elements between adjacent elements along each dimension.
    strides: ArrayIndices,
    /// The contiguous runs of elements, in C order of the view.
    runs: ContiguousLinearisedIndices,
    num_elements: u64,
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

/// The C-order strides (in elements) of an array with `shape`.
fn c_order_strides(shape: &[u64]) -> ArrayIndices {
    let mut strides: ArrayIndices = shape.iter().map(|_| 0).collect();
    let mut stride = 1u64;
    for (s, &size) in std::iter::zip(strides.iter_mut(), shape).rev() {
        *s = stride;
        stride = stride.wrapping_mul(size);
    }
    strides
}

/// The number of elements in `shape`, short-circuiting on zero dimensions so large non-zero dimensions do not overflow.
fn checked_num_elements(shape: &[u64]) -> Option<u64> {
    if shape.contains(&0) {
        Some(0)
    } else {
        shape
            .iter()
            .try_fold(1u64, |num_elements, &size| num_elements.checked_mul(size))
    }
}

/// The element offset of `start` in a grid with `strides` from `offset`.
fn offset_of(offset: u64, start: &[u64], strides: &[u64]) -> u64 {
    std::iter::zip(start, strides).fold(offset, |offset, (&start, &stride)| offset + start * stride)
}

impl<'a> ArrayBytesFixedDisjointView<'a> {
    /// Create a new non-overlapping view of a `subset` of the bytes in a C-order array with `shape`.
    ///
    /// `bytes` may be initialised (`UnsafeCellSlice<u8>`) or uninitialised (`UnsafeCellSlice<MaybeUninit<u8>>`, e.g. from [`Vec::spare_capacity_mut`]).
    /// The view never writes uninitialised bytes, so initialised `bytes` remain initialised.
    ///
    /// The view has the shape of `subset`.
    /// An empty `subset` references no elements, so it is accepted irrespective of its start and
    /// yields a view that reads and writes nothing.
    ///
    /// # Errors
    /// Returns [`ArrayBytesFixedDisjointViewCreateError`] if
    /// - `subset` has an incompatible dimensionality with, or is out-of-bounds of, `shape`, or
    /// - the length of `bytes` is not the product of the elements in `shape` multiplied by `data_type_size`.
    ///
    /// # Safety
    /// The `subset` represented by this view must not overlap with the elements of any other created views that reference the same array bytes.
    ///
    /// # Panics
    /// Panics if the product of the elements in `shape` multiplied by `data_type_size` exceeds [`usize::MAX`].
    pub unsafe fn new(
        bytes: impl Into<UnsafeCellSlice<'a, MaybeUninit<u8>>>,
        data_type_size: usize,
        shape: &[u64],
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

        let runs = ContiguousLinearisedIndices::new(&subset, shape)
            .expect("the subset is within the shape");
        let strides = c_order_strides(shape);
        let num_elements = subset.num_elements();
        let offset = if num_elements == 0 {
            0
        } else {
            offset_of(0, subset.start(), &strides)
        };
        Ok(Self {
            bytes,
            data_type_size,
            offset,
            shape: subset.shape().iter().copied().collect(),
            strides,
            runs,
            num_elements,
            bytes_in_subset_len: usize::try_from(num_elements).unwrap() * data_type_size,
        })
    }

    /// Create a new non-overlapping view of a strided grid of elements in `bytes`.
    ///
    /// The element at indices `i` of the view is element `offset + Σ i[d] * strides[d]` of `bytes`.
    ///
    /// # Errors
    /// Returns [`ArrayBytesFixedDisjointViewCreateError`] if
    /// - `shape` and `strides` have different lengths, or
    /// - an element of a nonempty view is beyond the end of `bytes`.
    ///
    /// # Safety
    /// Distinct indices of the view must address distinct elements, and the elements must not overlap with the elements of any other created views that reference the same array bytes.
    pub unsafe fn new_strided(
        bytes: impl Into<UnsafeCellSlice<'a, MaybeUninit<u8>>>,
        data_type_size: usize,
        offset: u64,
        shape: &[u64],
        strides: &[u64],
    ) -> Result<Self, ArrayBytesFixedDisjointViewCreateError> {
        let bytes = bytes.into();
        if shape.len() != strides.len() {
            return Err(
                IndexerError::new_incompatible_dimensionality(strides.len(), shape.len()).into(),
            );
        }
        let num_elements = if shape.contains(&0) {
            0
        } else {
            let num_elements = shape
                .iter()
                .try_fold(1u64, |num_elements, &size| num_elements.checked_mul(size));
            // The byte length required to hold the last element
            let required_len = std::iter::zip(shape, strides)
                .try_fold(offset, |last, (&size, &stride)| {
                    last.checked_add((size - 1).checked_mul(stride)?)
                })
                .and_then(|last| last.checked_add(1))
                .and_then(|len| usize::try_from(len).ok())
                .and_then(|len| len.checked_mul(data_type_size));
            match (num_elements, required_len) {
                (Some(num_elements), Some(required_len)) if required_len <= bytes.len() => {
                    num_elements
                }
                (_, required_len) => {
                    return Err(InvalidBytesLengthError::new(
                        bytes.len(),
                        required_len.unwrap_or(usize::MAX),
                    )
                    .into());
                }
            }
        };
        Self::from_layout(
            bytes,
            data_type_size,
            offset,
            shape.iter().copied().collect(),
            strides.iter().copied().collect(),
            num_elements,
        )
    }

    /// Construct a view from validated layout parts.
    fn from_layout(
        bytes: UnsafeCellSlice<'a, MaybeUninit<u8>>,
        data_type_size: usize,
        offset: u64,
        shape: ArrayIndices,
        strides: ArrayIndices,
        num_elements: u64,
    ) -> Result<Self, ArrayBytesFixedDisjointViewCreateError> {
        let offset = if num_elements == 0 { 0 } else { offset };
        let runs = ContiguousLinearisedIndices::new_strided(
            offset,
            std::iter::zip(shape.iter().copied(), strides.iter().copied()),
        )
        .map_err(|_| InvalidBytesLengthError::new(bytes.len(), usize::MAX))?;
        Ok(Self {
            bytes,
            data_type_size,
            offset,
            shape,
            strides,
            runs,
            num_elements,
            bytes_in_subset_len: usize::try_from(num_elements).unwrap() * data_type_size,
        })
    }

    /// A view of the same bytes with a layout derived from this view.
    ///
    /// # Safety
    /// The elements of the returned view must not be accessed through any other live view (including this one) while the returned view is live.
    unsafe fn with_layout<'b>(
        &self,
        offset: u64,
        shape: ArrayIndices,
        strides: ArrayIndices,
    ) -> Result<ArrayBytesFixedDisjointView<'b>, ArrayBytesFixedDisjointViewCreateError>
    where
        'a: 'b,
    {
        let num_elements = checked_num_elements(&shape)
            .ok_or_else(|| InvalidBytesLengthError::new(self.bytes.len(), usize::MAX))?;
        ArrayBytesFixedDisjointView::from_layout(
            self.bytes,
            self.data_type_size,
            offset,
            shape,
            strides,
            num_elements,
        )
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

    /// Create a new non-overlapping view of a `subset` of this view.
    ///
    /// The `subset` is relative to this view (its origin is the first element of the view), and the new view has the shape of `subset`.
    /// An empty `subset` references no elements, so it is accepted irrespective of its start.
    ///
    /// # Errors
    /// Returns [`SubsetOutOfBoundsError`] if `subset` has an incompatible dimensionality with, or is out-of-bounds of, the [`shape`](Self::shape) of this view.
    ///
    /// # Safety
    /// The `subset` represented by this view must not overlap with the elements of any other created views that reference the same array bytes.
    pub unsafe fn subdivide(
        &self,
        subset: ArraySubset,
    ) -> Result<ArrayBytesFixedDisjointView<'a>, ArrayBytesFixedDisjointViewCreateError> {
        if !subset.inbounds_shape(&self.shape) {
            let bounding_subset = ArraySubset::new_with_shape(&self.shape);
            return Err(SubsetOutOfBoundsError::new(subset, bounding_subset).into());
        }
        let offset = if subset.is_empty() {
            0
        } else {
            offset_of(self.offset, subset.start(), &self.strides)
        };
        // SAFETY: the caller guarantees `subset` does not overlap the elements of any other view.
        unsafe {
            self.with_layout(
                offset,
                subset.shape().iter().copied().collect(),
                self.strides.clone(),
            )
        }
    }

    /// Return the size in bytes of each element.
    #[must_use]
    pub fn data_type_size(&self) -> usize {
        self.data_type_size
    }

    /// Return the shape of the view.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Return the index in the underlying bytes of the first element of the view, in elements.
    ///
    /// The offset of an empty view is zero.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Return the distance in elements between adjacent elements along each dimension of the view.
    #[must_use]
    pub fn strides(&self) -> &[u64] {
        &self.strides
    }

    /// Return true if this view addresses the same elements as `other` (of possibly different bytes), in the same order.
    pub(crate) fn same_layout(&self, other: &Self) -> bool {
        self.shape == other.shape
            && (self.num_elements == 0
                || (self.offset == other.offset
                    && itertools::izip!(&self.shape, &self.strides, &other.strides)
                        .all(|(&size, a, b)| size == 1 || a == b)))
    }

    /// Return the number of elements in the view.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.num_elements
    }

    /// Return the contiguous element length of the view.
    ///
    /// This is the number of elements that are accessed in a single contiguous block.
    #[must_use]
    pub fn num_contiguous_elements(&self) -> usize {
        self.runs.contiguous_elements_usize()
    }

    /// Return the size in bytes of contiguous elements in the view.
    ///
    /// This is the number of elements that are accessed in a single contiguous block.
    #[must_use]
    pub fn contiguous_bytes_len(&self) -> usize {
        self.runs.contiguous_elements_usize() * self.data_type_size
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
        if self.runs.len() != 1 {
            return None;
        }
        let (index, _) = self.runs.iter().next()?;
        // SAFETY: this mutable borrow exclusively accesses the single region of this disjoint view.
        Some(unsafe { self.region_mut(index, self.bytes_in_subset_len) })
    }

    /// Fill every byte in the view with `byte`.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    pub(crate) fn fill_byte(&mut self, byte: u8) {
        let indices = &self.runs;
        let length = indices.contiguous_elements_usize() * self.data_type_size;
        if length == 0 {
            return;
        }

        let fill = MaybeUninit::new(byte);
        with_small_len!(length, |length| {
            for (index, _) in indices {
                // SAFETY: every region belongs to the disjoint view and is only accessed here.
                let target = unsafe { self.region_mut(index, length) };
                target.fill(fill);
            }
        });
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
        let Some(&first_byte) = fill_value.first() else {
            return Ok(());
        };
        if fill_value.iter().all(|&byte| byte == first_byte) {
            self.fill_byte(first_byte);
            return Ok(());
        }

        let indices = &self.runs;
        let length = indices.contiguous_elements_usize() * self.data_type_size;
        if length == 0 {
            return Ok(());
        }

        let mut regions = indices.iter();
        let (first_index, _) = regions.next().expect("a nonempty view has a region");
        // SAFETY: this region belongs to the disjoint view and is only accessed here.
        let first = unsafe { self.region_mut(first_index, length) };
        // Seed one element, then double the initialised prefix in place.
        write_copy_of_slice(&mut first[..fill_value.len()], fill_value);
        let mut written = fill_value.len();
        while written < length {
            let count = written.min(length - written);
            let (prefix, remaining) = first.split_at_mut(written);
            remaining[..count].copy_from_slice(&prefix[..count]);
            written += count;
        }

        // Reuse the first filled region rather than constructing a temporary row.
        with_small_len!(length, |length| {
            let first = &first[..length];
            for (index, _) in regions {
                // SAFETY: the regions of an array subset are disjoint, including from `first`.
                let target = unsafe { self.region_mut(index, length) };
                target.copy_from_slice(first);
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
        self.copy_from_slice_with(subset_bytes, |_| ())
    }

    /// Copy bytes into the view, then call `f` on each contiguous region of the view once it is written.
    ///
    /// The regions are passed in C order, so `f` can transform the copied bytes while they are in cache.
    /// Each region is a whole number of elements, with the length of [`contiguous_bytes_len`](Self::contiguous_bytes_len).
    ///
    /// The `subset_bytes` must be the same length as the byte length of the elements in the view.
    ///
    /// # Errors
    /// Returns an [`InvalidBytesLengthError`] if the length of `subset_bytes` is not the same as the byte length of the elements in the view.
    ///
    /// # Panics
    /// Panics if an offset into the internal bytes reference exceeds [`usize::MAX`].
    pub fn copy_from_slice_with(
        &mut self,
        subset_bytes: &[u8],
        mut f: impl FnMut(&mut [u8]),
    ) -> Result<(), InvalidBytesLengthError> {
        if subset_bytes.len() != self.bytes_in_subset_len {
            return Err(InvalidBytesLengthError::new(
                subset_bytes.len(),
                self.bytes_in_subset_len,
            ));
        }
        let indices = &self.runs;
        let length = indices.contiguous_elements_usize() * self.data_type_size;
        if length == 0 {
            return Ok(());
        }
        with_small_len!(length, |length| {
            for ((index, _), source) in indices.iter().zip(subset_bytes.chunks_exact(length)) {
                // SAFETY: the regions of the view are disjoint from each other and from other views.
                f(write_copy_of_slice(
                    unsafe { self.region_mut(index, length) },
                    source,
                ));
            }
        });
        Ok(())
    }

    /// Copy the elements of `bytes` in the contiguous `runs` of element indices into the view, in order.
    ///
    /// The elements of `bytes` have the data type size of the view.
    /// This is equivalent to copying the elements of the runs into a new buffer, then into the view with [`copy_from_slice`](Self::copy_from_slice), without the buffer.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `runs` address elements beyond `bytes`, or do not have as many elements as the view.
    ///
    /// # Panics
    /// Panics if an offset into `bytes` or the internal bytes reference exceeds [`usize::MAX`].
    pub(crate) fn copy_runs_from_slice(
        &mut self,
        bytes: &[u8],
        runs: &ContiguousLinearisedIndices,
    ) -> Result<(), CodecError> {
        let num_elements = self.num_elements();
        let run_elements = runs.contiguous_elements();
        let runs_num_elements = (runs.len() as u64).checked_mul(run_elements);
        if runs_num_elements != Some(num_elements) {
            return Err(InvalidNumberOfElementsError::new(
                runs_num_elements.unwrap_or(u64::MAX),
                num_elements,
            )
            .into());
        }
        let size = self.data_type_size;
        validate_runs_within(bytes, runs, size)?;
        if num_elements == 0 || size == 0 {
            return Ok(());
        }
        // Match extraction's limit even when repeated source runs can be copied directly.
        if self.bytes_in_subset_len > bytes.len() {
            return Err(IndexerError::IncompatibleLength(
                num_elements,
                (bytes.len() / size) as u64,
            )
            .into());
        }
        let regions = &self.runs;
        let region_elements = regions.contiguous_elements();
        // A piece with the length of the shorter of the runs and regions is contiguous in both if it divides the longer.
        // It does if the runs and regions are of the same subset shape in C-order arrays, since each length is a product of trailing dimensions of the subset.
        // It also does if either has runs of one element, such as a view in F order.
        let piece_elements = run_elements.min(region_elements);
        let pieces = run_elements.max(region_elements) / piece_elements;
        if pieces * piece_elements != run_elements.max(region_elements) {
            return Ok(self.copy_from_slice(&extract_runs_fixed(bytes, runs, size)?)?);
        }
        // A piece is within a run, which is within `bytes`, so its length does not overflow
        let piece_len = usize::try_from(piece_elements).unwrap() * size;
        with_small_len!(piece_len, |piece_len| {
            let copy = |run_index: u64, region_index: u64| {
                let start = run_index.checked_mul(size as u64);
                let source = start
                    .and_then(|start| usize::try_from(start).ok())
                    .and_then(|start| bytes.get(start..)?.get(..piece_len))
                    .ok_or_else(|| {
                        let start = start.unwrap_or(u64::MAX);
                        let range = start..start.saturating_add(piece_len as u64);
                        InvalidByteRangeError::new(ByteRange::new(range), bytes.len() as u64)
                    })?;
                // SAFETY: each piece is within a region of the view, and the regions of the view are disjoint from each other and from other views.
                write_copy_of_slice(unsafe { self.region_mut(region_index, piece_len) }, source);
                Ok::<_, CodecError>(())
            };
            // Split runs or regions into pieces, so that both have the same number of pieces
            let split = move |(index, _): (u64, u64)| {
                (0..pieces).map(move |piece| index + piece * piece_elements)
            };
            if run_elements == region_elements {
                std::iter::zip(runs, regions).try_for_each(|((run_index, _), (region_index, _))| {
                    copy(run_index, region_index)
                })
            } else if run_elements > region_elements {
                std::iter::zip(runs.iter().flat_map(split), regions)
                    .try_for_each(|(run_index, (region_index, _))| copy(run_index, region_index))
            } else {
                std::iter::zip(runs, regions.iter().flat_map(split))
                    .try_for_each(|((run_index, _), region_index)| copy(run_index, region_index))
            }
        })
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
        let mut element_indices = self.element_indices();
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
        let mut element_indices = self.element_indices();
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
    fn element_indices(&self) -> ElementIndices<'_> {
        let indices = &self.runs;
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
            assert_eq!(view0.shape(), [2, 2]);
            assert_eq!(view0.strides(), [3, 1]);
            // SAFETY: the slice is not used.
            assert!(unsafe { view0.as_mut_slice() }.is_none());

            view0.copy_from_slice(&[11, 12, 14, 15]).unwrap();
            assert!(view0.copy_from_slice(&[11, 12, 14, 15, 255]).is_err()); // wrong length

            // Subsets are relative to the view
            let mut view0a =
                unsafe { view0.subdivide(ArraySubset::new_with_ranges(&[1..2, 0..2])) }.unwrap();
            view0a.copy_from_slice(&[24, 25]).unwrap();
            assert!(view0a.copy_from_slice(&[]).is_err()); // wrong length

            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[0..1, 0..2])) }.is_ok()
            );
            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[0..1, 1..2])) }.is_ok()
            );
            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[0..2, 0..2])) }.is_err()
            ); // OOB
            assert!(
                unsafe { view0a.subdivide(ArraySubset::new_with_ranges(&[0..1, 0..3])) }.is_err()
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
                                subset.clone(),
                            )
                            .unwrap()
                        };
                        assert!(view.copy_from_slice(&vec![0; count + 1]).is_err());
                        view.copy_from_slice(&input).unwrap();
                    }
                    assert_eq!(bytes, expected, "{size} {width} {rows:?}");

                    // Each region is passed to `f` once it is written, in order
                    let mut transformed = vec![0xa5; bytes.len()];
                    let mut regions = Vec::new();
                    {
                        // SAFETY: this is the only view of transformed.
                        let mut view = unsafe {
                            ArrayBytesFixedDisjointView::new(
                                UnsafeCellSlice::new(&mut transformed),
                                size,
                                &shape,
                                subset.clone(),
                            )
                            .unwrap()
                        };
                        assert!(
                            view.copy_from_slice_with(&vec![0; count + 1], |_| unreachable!())
                                .is_err()
                        );
                        view.copy_from_slice_with(&input, |region| {
                            regions.push(region.to_vec());
                            for byte in region.iter_mut() {
                                *byte = !*byte;
                            }
                        })
                        .unwrap();
                    }
                    let expected_regions: Vec<Vec<u8>> = if run == 0 {
                        Vec::new()
                    } else {
                        input.chunks_exact(run).map(<[u8]>::to_vec).collect()
                    };
                    assert_eq!(regions, expected_regions, "{size} {width} {rows:?}");
                    let mut expected_transformed = vec![0xa5; bytes.len()];
                    for (k, row) in rows.clone().enumerate() {
                        let destination = (usize::try_from(row).unwrap() * (width + 2) + 1) * size;
                        for (dst, src) in expected_transformed[destination..destination + run]
                            .iter_mut()
                            .zip(&input[k * run..][..run])
                        {
                            *dst = !*src;
                        }
                    }
                    assert_eq!(transformed, expected_transformed, "{size} {width} {rows:?}");
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

    /// The C-order indices of the elements of an array with `shape`.
    fn c_order_indices(shape: &[u64]) -> Vec<Vec<u64>> {
        let num_elements = shape.iter().product::<u64>();
        (0..num_elements)
            .map(|mut element| {
                let mut indices = vec![0; shape.len()];
                for (index, &size) in std::iter::zip(&mut indices, shape).rev() {
                    *index = element % size;
                    element /= size;
                }
                indices
            })
            .collect()
    }

    /// Check every operation of a view of `bytes` against the element indices of its positions, `expected`.
    fn check_view_operations(
        make_view: MakeView<'_>,
        len: usize,
        size: usize,
        expected: &[u64],
        random: &mut dyn FnMut(u64) -> u64,
    ) {
        let element = |bytes: &[u8], index: u64| {
            let index = usize::try_from(index).unwrap() * size;
            bytes[index..index + size].to_vec()
        };
        let untouched = |bytes: &[u8], written: &[u64]| {
            (0..len as u64)
                .filter(|index| !written.contains(index))
                .all(|index| element(bytes, index) == vec![0xa5; size])
        };
        let source: Vec<u8> = (0..expected.len() * size)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();

        // copy_from_slice and copy_from_slice_with
        let mut bytes = vec![0xa5; len * size];
        let mut regions = 0;
        {
            let mut view = make_view(UnsafeCellSlice::new(&mut bytes));
            assert_eq!(view.num_elements(), expected.len() as u64);
            let run = view.contiguous_bytes_len();
            view.copy_from_slice_with(&source, |region| {
                assert_eq!(region.len(), run);
                regions += 1;
            })
            .unwrap();
        }
        assert_eq!(regions * contiguous_run(expected), expected.len());
        for (position, &index) in expected.iter().enumerate() {
            assert_eq!(element(&bytes, index), source[position * size..][..size]);
        }
        assert!(untouched(&bytes, expected));

        // as_mut_slice is the whole view if it is one run
        let mut bytes = vec![0xa5; len * size];
        {
            let mut view = make_view(UnsafeCellSlice::new(&mut bytes));
            // SAFETY: the slice is initialised before the view is dropped.
            let slice = unsafe { view.as_mut_slice() };
            let one_run = !expected.is_empty() && contiguous_run(expected) == expected.len();
            assert_eq!(slice.is_some(), one_run);
            if let Some(slice) = slice {
                write_copy_of_slice(slice, &source);
            }
        }
        if !expected.is_empty() && contiguous_run(expected) == expected.len() {
            for (position, &index) in expected.iter().enumerate() {
                assert_eq!(element(&bytes, index), source[position * size..][..size]);
            }
        }

        // fill
        let fill_value: Vec<u8> = (0..size).map(|i| u8::try_from(i + 1).unwrap()).collect();
        let mut bytes = vec![0xa5; len * size];
        make_view(UnsafeCellSlice::new(&mut bytes))
            .fill(&fill_value)
            .unwrap();
        for &index in expected {
            assert_eq!(element(&bytes, index), fill_value);
        }
        assert!(untouched(&bytes, expected));

        // copy_elements_from_slice and fill_elements at random positions
        if !expected.is_empty() {
            let positions: Vec<usize> = (0..expected.len() * 2)
                .map(|_| usize::try_from(random(expected.len() as u64)).unwrap())
                .collect();
            let values: Vec<u8> = (0..positions.len() * size)
                .map(|i| u8::try_from(i % 251).unwrap())
                .collect();
            let mut copied = vec![0xa5; len * size];
            let mut filled = vec![0xa5; len * size];
            make_view(UnsafeCellSlice::new(&mut copied))
                .copy_elements_from_slice(&positions, &values)
                .unwrap();
            make_view(UnsafeCellSlice::new(&mut filled))
                .fill_elements(&positions, &fill_value)
                .unwrap();
            let mut expected_copied = vec![0xa5; len * size];
            let mut expected_filled = vec![0xa5; len * size];
            for (k, &position) in positions.iter().enumerate() {
                let index = usize::try_from(expected[position]).unwrap() * size;
                expected_copied[index..index + size].copy_from_slice(&values[k * size..][..size]);
                expected_filled[index..index + size].copy_from_slice(&fill_value);
            }
            assert_eq!(copied, expected_copied);
            assert_eq!(filled, expected_filled);
            let mut bytes = vec![0xa5; len * size];
            let mut view = make_view(UnsafeCellSlice::new(&mut bytes));
            assert!(
                view.copy_elements_from_slice(&[expected.len()], &fill_value)
                    .is_err()
            );
            assert!(view.fill_elements(&[expected.len()], &fill_value).is_err());
        }
    }

    /// The number of elements in each run of consecutive element indices of `expected`, if they have a common length.
    fn contiguous_run(expected: &[u64]) -> usize {
        if expected.is_empty() {
            return 0;
        }
        let mut run = 1;
        while run < expected.len() && expected[run] == expected[run - 1] + 1 {
            run += 1;
        }
        run
    }

    /// A constructor of a view of some bytes.
    type MakeView<'f> =
        &'f dyn for<'b> Fn(UnsafeCellSlice<'b, u8>) -> ArrayBytesFixedDisjointView<'b>;

    /// Coerce a closure to a higher-ranked view constructor.
    fn view_fn<F>(f: F) -> F
    where
        F: for<'b> Fn(UnsafeCellSlice<'b, u8>) -> ArrayBytesFixedDisjointView<'b>,
    {
        f
    }

    /// Views with C, F, and subdivided layouts address the elements given by index arithmetic.
    #[test]
    fn strided_layouts() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = move |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        for case in 0..300 {
            let size = [1, 2, 3][case % 3];
            let dimensionality = 1 + usize::try_from(random(3)).unwrap();
            let shape: Vec<u64> = (0..dimensionality).map(|_| 1 + random(4)).collect();
            let len = usize::try_from(shape.iter().product::<u64>()).unwrap();
            let ranges: Vec<_> = shape
                .iter()
                .map(|&size| {
                    let start = random(size + 1);
                    start..start + random(size - start + 1)
                })
                .collect();
            let subset = ArraySubset::new_with_ranges(&ranges);
            let positions = c_order_indices(subset.shape());
            let in_array = |position: &[u64]| -> Vec<u64> {
                std::iter::zip(position, subset.start())
                    .map(|(a, b)| a + b)
                    .collect()
            };
            let c_index = |indices: &[u64]| {
                std::iter::zip(indices, &shape)
                    .fold(0, |linear, (&index, &size)| linear * size + index)
            };
            let f_index = |indices: &[u64]| {
                std::iter::zip(indices, &shape)
                    .rev()
                    .fold(0, |linear, (&index, &size)| linear * size + index)
            };
            let c_strides = c_order_strides(&shape);
            let f_strides: Vec<u64> = (0..dimensionality)
                .map(|d| shape[..d].iter().product())
                .collect();

            // C order
            let expected_c: Vec<u64> = positions.iter().map(|p| c_index(&in_array(p))).collect();
            let c_view = view_fn(|bytes| unsafe {
                ArrayBytesFixedDisjointView::new(bytes, size, &shape, subset.clone()).unwrap()
            });
            check_view_operations(&c_view, len, size, &expected_c, &mut random);
            let strided_c_view = view_fn(|bytes| unsafe {
                ArrayBytesFixedDisjointView::new_strided(
                    bytes,
                    size,
                    offset_of(0, subset.start(), &c_strides),
                    subset.shape(),
                    &c_strides,
                )
                .unwrap()
            });
            if !subset.is_empty() {
                let mut bytes = vec![0; len * size];
                let bytes = UnsafeCellSlice::new(&mut bytes);
                assert_eq!(strided_c_view(bytes).runs, c_view(bytes).runs);
            }

            // F order
            let expected_f: Vec<u64> = positions.iter().map(|p| f_index(&in_array(p))).collect();
            let f_view = view_fn(|bytes| unsafe {
                ArrayBytesFixedDisjointView::new_strided(
                    bytes,
                    size,
                    if subset.is_empty() {
                        0
                    } else {
                        offset_of(0, subset.start(), &f_strides)
                    },
                    subset.shape(),
                    &f_strides,
                )
                .unwrap()
            });
            check_view_operations(&f_view, len, size, &expected_f, &mut random);

            // Subdivisions are relative to the view
            let sub_ranges: Vec<_> = subset
                .shape()
                .iter()
                .map(|&size| {
                    let start = random(size + 1);
                    start..start + random(size - start + 1)
                })
                .collect();
            let sub_subset = ArraySubset::new_with_ranges(&sub_ranges);
            let sub_positions = c_order_indices(sub_subset.shape());
            let views: [(MakeView<'_>, _); 2] = [(&c_view, &expected_c), (&f_view, &expected_f)];
            for (view, expected) in views {
                let expected_sub: Vec<u64> = sub_positions
                    .iter()
                    .map(|q| {
                        let p: Vec<u64> = std::iter::zip(q, sub_subset.start())
                            .map(|(a, b)| a + b)
                            .collect();
                        expected[usize::try_from(c_index_of(&p, subset.shape())).unwrap()]
                    })
                    .collect();
                let sub_view =
                    view_fn(|bytes| unsafe { view(bytes).subdivide(sub_subset.clone()).unwrap() });
                check_view_operations(&sub_view, len, size, &expected_sub, &mut random);
            }
        }
    }

    /// The C-order linear index of `indices` in `shape`.
    fn c_index_of(indices: &[u64], shape: &[u64]) -> u64 {
        std::iter::zip(indices, shape).fold(0, |linear, (&index, &size)| linear * size + index)
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
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new_strided(
                    UnsafeCellSlice::new(&mut bytes),
                    2,
                    u64::MAX,
                    &[u64::MAX, 2, 0],
                    &[u64::MAX, u64::MAX, 1],
                )
            }
            .unwrap();
            assert_eq!(view.num_elements(), 0);
            view.copy_from_slice(&[]).unwrap();
            view.fill(&[1, 2]).unwrap();
            let sub = unsafe {
                view.subdivide(
                    ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![u64::MAX, 2, 0]).unwrap(),
                )
            }
            .unwrap();
            assert_eq!(sub.num_elements(), 0);
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

    #[test]
    fn copy_repeated_runs_matches_extraction_for_all_target_layouts() {
        let runs =
            ContiguousLinearisedIndices::new_strided(0, [(3, 0), (2, 1)].into_iter()).unwrap();
        let cases = [
            (vec![6], vec![0..6]),
            (vec![3, 3], vec![0..3, 0..2]),
            (vec![6, 2], vec![0..6, 0..1]),
            (vec![2, 4], vec![0..2, 0..3]),
        ];
        for size in [0usize, 1, 2, 3, 8] {
            for source_elements in [2usize, 6] {
                let source: Vec<u8> = (0..=255).cycle().take(source_elements * size).collect();
                for (shape, ranges) in &cases {
                    let subset = ArraySubset::new_with_ranges(ranges);
                    let len = usize::try_from(shape.iter().product::<u64>()).unwrap() * size;
                    let mut actual = vec![0xa5; len];
                    let mut expected = actual.clone();
                    // SAFETY: each view is the only view of its bytes.
                    unsafe {
                        let mut target = crate::ArrayBytesDecodeIntoTarget::Fixed(
                            ArrayBytesFixedDisjointView::new(
                                UnsafeCellSlice::new(&mut actual),
                                size,
                                shape,
                                subset.clone(),
                            )
                            .unwrap(),
                        );
                        let result = crate::ArrayBytes::new_flen(source.as_slice())
                            .extract_runs_into(&runs, &mut target);
                        let extracted = extract_runs_fixed(&source, &runs, size);
                        if size > 0 && source_elements == 2 {
                            assert!(matches!(
                                result,
                                Err(CodecError::IncompatibleIndexer(
                                    IndexerError::IncompatibleLength(6, 2)
                                ))
                            ));
                            assert!(extracted.is_err());
                        } else {
                            result.unwrap();
                            ArrayBytesFixedDisjointView::new(
                                UnsafeCellSlice::new(&mut expected),
                                size,
                                shape,
                                subset,
                            )
                            .unwrap()
                            .copy_from_slice(&extracted.unwrap())
                            .unwrap();
                        }
                    }
                    assert_eq!(actual, expected, "{size} {source_elements} {shape:?}");
                }
            }
        }
    }

    #[test]
    fn copy_runs_from_slice_matches_extracted_copy() {
        // (source shape, source subset, target shape, target subset)
        let cases = [
            // Runs and regions of equal length
            (vec![4, 6], vec![1..3, 2..5], vec![4, 5], vec![1..3, 1..4]),
            // Runs shorter than regions
            (vec![4, 6], vec![1..3, 2..5], vec![2, 3], vec![0..2, 0..3]),
            // Runs longer than regions
            (vec![3, 4], vec![0..2, 0..4], vec![4, 6], vec![1..3, 2..6]),
            (
                vec![2, 3, 4],
                vec![0..2, 0..3, 0..4],
                vec![2, 3, 5],
                vec![0..2, 0..3, 1..5],
            ),
            // Runs of 3 and regions of 2 elements, which do not divide each other
            (vec![2, 4], vec![0..2, 0..3], vec![3, 3], vec![0..3, 0..2]),
            // Empty
            (vec![2, 4], vec![0..0, 0..3], vec![3, 3], vec![0..3, 0..0]),
        ];
        for size in [0usize, 1, 2, 3, 8] {
            for (source_shape, source_subset, target_shape, target_subset) in &cases {
                let source_subset = ArraySubset::new_with_ranges(source_subset);
                let target_subset = ArraySubset::new_with_ranges(target_subset);
                let source_len = usize::try_from(source_shape.iter().product::<u64>()).unwrap();
                let source: Vec<u8> = (0..=255u8).cycle().take(source_len * size).collect();
                let runs = source_subset
                    .contiguous_linearised_indices(source_shape)
                    .unwrap();
                let mut extracted = Vec::new();
                for (index, elements) in &runs {
                    let start = usize::try_from(index).unwrap() * size;
                    let end = start + usize::try_from(elements).unwrap() * size;
                    extracted.extend_from_slice(&source[start..end]);
                }
                let target_len = usize::try_from(target_shape.iter().product::<u64>()).unwrap();
                let mut expected = vec![0xa5; target_len * size];
                let mut actual = expected.clone();
                // SAFETY: each view is the only view of its bytes.
                unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(&mut expected),
                        size,
                        target_shape,
                        target_subset.clone(),
                    )
                    .unwrap()
                    .copy_from_slice(&extracted)
                    .unwrap();
                    let mut view = ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(&mut actual),
                        size,
                        target_shape,
                        target_subset.clone(),
                    )
                    .unwrap();
                    view.copy_runs_from_slice(&source, &runs).unwrap();
                    // The last run is truncated
                    if size > 0 && !extracted.is_empty() {
                        let (index, elements) = runs.iter().next_back().unwrap();
                        let end = usize::try_from(index + elements).unwrap() * size;
                        assert!(matches!(
                            view.copy_runs_from_slice(&source[..end - 1], &runs),
                            Err(CodecError::InvalidByteRangeError(_))
                        ));
                    }
                }
                assert_eq!(
                    actual, expected,
                    "{size} {source_subset:?} {target_subset:?}"
                );
            }
        }

        // A target in F order has runs of one element
        let source: Vec<u8> = (0..24).collect();
        for ranges in [[1..3, 2..5], [0..2, 0..6]] {
            let runs = ArraySubset::new_with_ranges(&ranges)
                .contiguous_linearised_indices(&[4, 6])
                .unwrap();
            let num_elements = u64::try_from(runs.len()).unwrap() * runs.contiguous_elements();
            let mut expected = vec![0xa5; usize::try_from(num_elements).unwrap()];
            let mut actual = expected.clone();
            let shape = [2, num_elements / 2];
            let strides = [1, 2];
            // SAFETY: each view is the only view of its bytes.
            unsafe {
                let extracted = runs
                    .iter()
                    .flat_map(|(index, elements)| {
                        let start = usize::try_from(index).unwrap();
                        source[start..start + usize::try_from(elements).unwrap()].to_vec()
                    })
                    .collect::<Vec<_>>();
                ArrayBytesFixedDisjointView::new_strided(
                    UnsafeCellSlice::new(&mut expected),
                    1,
                    0,
                    &shape,
                    &strides,
                )
                .unwrap()
                .copy_from_slice(&extracted)
                .unwrap();
                ArrayBytesFixedDisjointView::new_strided(
                    UnsafeCellSlice::new(&mut actual),
                    1,
                    0,
                    &shape,
                    &strides,
                )
                .unwrap()
                .copy_runs_from_slice(&source, &runs)
                .unwrap();
            }
            assert_eq!(actual, expected, "{ranges:?}");
        }

        // The runs must have as many elements as the view
        let mut bytes = vec![0u8; 9];
        // SAFETY: this is the only view of bytes.
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut bytes),
                1,
                &[3, 3],
                ArraySubset::new_with_ranges(&[0..2, 0..2]),
            )
            .unwrap()
        };
        let runs = ArraySubset::new_with_ranges(&[0..1, 0..3])
            .contiguous_linearised_indices(&[3, 3])
            .unwrap();
        assert!(matches!(
            view.copy_runs_from_slice(&[0; 9], &runs),
            Err(CodecError::InvalidNumberOfElements(_))
        ));
    }
}

//! Array subsets.
//!
//! An [`ArraySubset`] represents a subset of an array or chunk.
//!
//! [`iterators`] includes various types of [`ArraySubset`] iterators.
//!
//! This module also provides convenience functions for:
//!  - computing the byte ranges of array subsets within an array with a fixed element size.

use std::borrow::Cow;
use std::ops::Range;

use itertools::izip;

use crate::iterators::{
    ContiguousIndices, ContiguousLinearisedIndices, Indices, LinearisedIndices,
};
use crate::{ArrayIndices, ArraySubset, ArraySubsetError, Indexer, IndexerError};

mod private {
    pub trait Sealed {}
}

impl private::Sealed for ArraySubset {}
impl<const N: usize> private::Sealed for [Range<u64>; N] {}
impl private::Sealed for [Range<u64>] {}
impl private::Sealed for Vec<Range<u64>> {}

/// Trait for types that represent an array region (start and shape).
///
/// This trait enables ergonomic APIs that accept various region representations,
/// including `ArraySubset`, arrays of ranges like `[0..5, 0..10]`, and slices of ranges.
pub trait ArraySubsetTraits: Indexer + private::Sealed {
    /// Returns the start indices of the subset.
    fn start(&self) -> Cow<'_, [u64]>;

    /// Returns the shape (size along each dimension) of the subset.
    fn shape(&self) -> Cow<'_, [u64]>;

    /// Returns the total number of elements.
    fn num_elements(&self) -> u64 {
        self.shape().iter().product()
    }

    /// Returns the number of elements as usize.
    ///
    /// # Panics
    /// Panics if [`num_elements()`](Self::num_elements) is greater than [`usize::MAX`].
    fn num_elements_usize(&self) -> usize {
        usize::try_from(self.num_elements()).unwrap()
    }

    // /// Returns true if any dimension has size 0.
    // fn is_empty(&self) -> bool {
    //     self.shape().contains(&0)
    // }

    /// Returns exclusive end indices.
    fn end_exc(&self) -> ArrayIndices {
        std::iter::zip(self.start().iter(), self.shape().iter())
            .map(|(&s, &l)| s + l)
            .collect()
    }

    /// Returns inclusive end indices, or None if empty.
    fn end_inc(&self) -> Option<ArrayIndices> {
        if self.shape().contains(&0) {
            None
        } else {
            Some(
                std::iter::zip(self.start().iter(), self.shape().iter())
                    .map(|(&s, &l)| s + l - 1)
                    .collect(),
            )
        }
    }

    /// Converts to ranges.
    fn to_ranges(&self) -> Vec<Range<u64>> {
        std::iter::zip(self.start().iter(), self.shape().iter())
            .map(|(&start, &size)| start..start + size)
            .collect()
    }

    /// Returns true if the region contains the given indices.
    fn contains(&self, indices: &[u64]) -> bool {
        let start = self.start();
        let shape = self.shape();
        izip!(indices, start.iter(), shape.iter()).all(|(&i, &o, &s)| i >= o && i < o + s)
    }

    /// Returns true if the region is within the bounds of an array with the given shape.
    fn inbounds_shape(&self, array_shape: &[u64]) -> bool {
        if self.start().len() != array_shape.len() || self.shape().len() != array_shape.len() {
            return false;
        }
        let start = self.start();
        let shape = self.shape();
        izip!(start.iter(), shape.iter(), array_shape)
            .all(|(&start, &size, &bound)| start + size <= bound)
    }

    /// Returns true if the region is within another region.
    fn inbounds(&self, other: &dyn ArraySubsetTraits) -> bool {
        if self.start().len() != other.start().len() || self.shape().len() != other.shape().len() {
            return false;
        }
        let self_start = self.start();
        let self_shape = self.shape();
        let other_start = other.start();
        let other_shape = other.shape();
        izip!(
            self_start.iter(),
            self_shape.iter(),
            other_start.iter(),
            other_shape.iter()
        )
        .all(|(&s, &ss, &o, &os)| s >= o && s + ss <= o + os)
    }

    /// Return the overlapping subset between this region and `other`.
    ///
    /// # Errors
    /// Returns [`ArraySubsetError`] if the dimensionality of `other` does not match.
    fn overlap(&self, other: &dyn ArraySubsetTraits) -> Result<ArraySubset, ArraySubsetError> {
        if other.start().len() == self.start().len() && other.shape().len() == self.shape().len() {
            let self_start = self.start();
            let self_shape = self.shape();
            let other_start = other.start();
            let other_shape = other.shape();
            let ranges = izip!(
                self_start.iter(),
                self_shape.iter(),
                other_start.iter(),
                other_shape.iter(),
            )
            .map(|(&start, &size, &other_start, &other_size)| {
                let overlap_start = std::cmp::max(start, other_start);
                let overlap_end = std::cmp::min(start + size, other_start + other_size);
                overlap_start..overlap_end
            });
            Ok(ArraySubset::from(ranges))
        } else {
            Err(ArraySubsetError::IncompatibleDimensionality {
                got: other.start().len(),
                expected: self.start().len(),
            })
        }
    }

    /// Return the region relative to `offset`.
    ///
    /// Creates an array subset starting at `self.start()` - `offset`.
    ///
    /// # Errors
    /// Returns [`ArraySubset`] if the length of `offset` does not match the dimensionality,
    /// or if `offset` is greater than `start` in any dimension.
    fn relative_to(&self, offset: &[u64]) -> Result<ArraySubset, ArraySubsetError> {
        let self_start = self.start();
        if offset.len() != self_start.len()
            || std::iter::zip(self_start.iter(), offset.iter())
                .any(|(&start, &offset)| start < offset)
        {
            Err(ArraySubsetError::IncompatibleOffset {
                start: self_start.to_vec(),
                offset: offset.to_vec(),
            })
        } else {
            ArraySubset::new_with_start_shape(
                std::iter::zip(self_start.iter(), offset)
                    .map(|(&start, offset)| start - offset)
                    .collect::<Vec<_>>(),
                self.shape().into_owned(),
            )
        }
    }

    /// Converts to an owned `ArraySubset`.
    fn to_array_subset(&self) -> ArraySubset {
        ArraySubset::new_with_start_shape(self.start().into_owned(), self.shape().into_owned())
            .expect("start and shape have the same dimensionality") // true for all sealed impls
    }

    /// Returns an iterator over the indices of elements within the subset.
    #[must_use]
    fn indices(&self) -> Indices {
        Indices::new(self.to_array_subset())
    }

    /// Returns an iterator over the linearised indices of elements within the subset.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate this array subset.
    fn linearised_indices(&self, array_shape: &[u64]) -> Result<LinearisedIndices, IndexerError> {
        LinearisedIndices::new(self.to_array_subset(), array_shape.to_vec())
    }

    /// Returns an iterator over the indices of contiguous elements within the subset.
    ///
    /// # Errors
    ///
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate this array subset.
    fn contiguous_indices(&self, array_shape: &[u64]) -> Result<ContiguousIndices, IndexerError> {
        ContiguousIndices::new(self.to_array_subset(), array_shape)
    }

    /// Returns an iterator over the linearised indices of contiguous elements within the subset.
    ///
    /// # Errors
    ///
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate this array subset.
    fn contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousLinearisedIndices, IndexerError> {
        ContiguousLinearisedIndices::new(self.to_array_subset(), array_shape.to_vec())
    }
}

impl PartialEq<ArraySubset> for &dyn ArraySubsetTraits {
    fn eq(&self, other: &ArraySubset) -> bool {
        self.start() == other.start() && self.shape() == other.shape()
    }
}

impl PartialEq<&dyn ArraySubsetTraits> for ArraySubset {
    fn eq(&self, other: &&dyn ArraySubsetTraits) -> bool {
        self.start() == other.start().as_ref() && self.shape() == other.shape().as_ref()
    }
}

impl ArraySubsetTraits for ArraySubset {
    fn start(&self) -> Cow<'_, [u64]> {
        Cow::Borrowed(&self.start)
    }

    fn shape(&self) -> Cow<'_, [u64]> {
        Cow::Borrowed(&self.shape)
    }

    fn to_array_subset(&self) -> ArraySubset {
        self.clone()
    }
}

impl<const N: usize> ArraySubsetTraits for [Range<u64>; N] {
    fn start(&self) -> Cow<'_, [u64]> {
        Cow::Owned(self.iter().map(|r| r.start).collect())
    }

    fn shape(&self) -> Cow<'_, [u64]> {
        Cow::Owned(self.iter().map(|r| r.end.saturating_sub(r.start)).collect())
    }
}

impl ArraySubsetTraits for [Range<u64>] {
    fn start(&self) -> Cow<'_, [u64]> {
        Cow::Owned(self.iter().map(|r| r.start).collect())
    }

    fn shape(&self) -> Cow<'_, [u64]> {
        Cow::Owned(self.iter().map(|r| r.end.saturating_sub(r.start)).collect())
    }
}

impl ArraySubsetTraits for Vec<Range<u64>> {
    fn start(&self) -> Cow<'_, [u64]> {
        Cow::Owned(self.iter().map(|r| r.start).collect())
    }

    fn shape(&self) -> Cow<'_, [u64]> {
        Cow::Owned(self.iter().map(|r| r.end.saturating_sub(r.start)).collect())
    }
}

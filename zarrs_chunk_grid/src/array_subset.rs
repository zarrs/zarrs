//! Array subsets.
//!
//! An [`ArraySubset`] represents a subset of an array or chunk.
//!
//! [`iterators`] includes various types of [`ArraySubset`] iterators.
//!
//! This module also provides convenience functions for:
//!  - computing the byte ranges of array subsets within an array with a fixed element size.

use std::fmt::{Debug, Display};
use std::num::NonZeroU64;
use std::ops::Range;

use crate::iterators::{
    ContiguousIndices, ContiguousLinearisedIndices, Indices, LinearisedIndices,
};
use thiserror::Error;

use crate::indexer::{Indexer, IndexerError, IndexerIterator};
use crate::{ArrayIndices, ArrayIndicesTinyVec, ArrayShape, ChunkShape};

/// An incompatible start/end indices error.
#[derive(Clone, Debug, Error)]
#[error("incompatible start {0:?} with end {1:?}")]
#[allow(missing_docs)]
pub enum ArraySubsetError {
    /// Incompatible dimensionality.
    #[error("incompatible dimensionality {got}, expected {expected}")]
    IncompatibleDimensionality { got: usize, expected: usize },
    /// Incompatible start and shape.
    #[error("incompatible start {start:?} with shape {shape:?}")]
    IncompatibleStartShape {
        start: ArrayIndices,
        shape: ArrayShape,
    },
    /// Incompatible start and end indices.
    #[error("incompatible start {start:?} with end {end:?} (inclusive: {inclusive})")]
    IncompatibleStartEnd {
        start: ArrayIndices,
        end: ArrayIndices,
        inclusive: bool,
    },
    /// Incompatible offset.
    #[error("incompatible offset {offset:?} for region with start {start:?}")]
    IncompatibleOffset { start: Vec<u64>, offset: Vec<u64> },
}

/// An array subset.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct ArraySubset {
    /// The start of the array subset.
    pub(crate) start: ArrayIndices,
    /// The shape of the array subset.
    pub(crate) shape: ArrayShape,
}

impl Display for ArraySubset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::ArraySubsetTraits;
        self.to_ranges().fmt(f)
    }
}

impl<T: IntoIterator<Item = Range<u64>>> From<T> for ArraySubset {
    fn from(ranges: T) -> Self {
        let (start, shape) = ranges
            .into_iter()
            .map(|range| (range.start, range.end.saturating_sub(range.start)))
            .unzip();
        Self { start, shape }
    }
}

impl ArraySubset {
    /// Create a new empty array subset.
    #[must_use]
    pub fn new_empty(dimensionality: usize) -> Self {
        Self {
            start: vec![0; dimensionality],
            shape: vec![0; dimensionality],
        }
    }

    /// Create a new array subset from a list of [`Range`]s.
    #[must_use]
    pub fn new_with_ranges(ranges: &[Range<u64>]) -> Self {
        let (start, shape) = ranges
            .iter()
            .map(|range| (range.start, range.end.saturating_sub(range.start)))
            .unzip();
        Self { start, shape }
    }

    /// Create a new array subset with `size` starting at the origin.
    #[must_use]
    pub fn new_with_shape(shape: ArrayShape) -> Self {
        Self {
            start: vec![0; shape.len()],
            shape,
        }
    }

    /// Create a new array subset.
    ///
    /// # Errors
    ///
    /// Returns [`ArraySubsetError`] if the size of `start` and `size` do not match.
    pub fn new_with_start_shape(
        start: ArrayIndices,
        shape: ArrayShape,
    ) -> Result<Self, ArraySubsetError> {
        if start.len() == shape.len() {
            Ok(Self { start, shape })
        } else {
            Err(ArraySubsetError::IncompatibleStartShape { start, shape })
        }
    }

    /// Create a new array subset from a start and end (inclusive).
    ///
    /// # Errors
    /// Returns [`ArraySubsetError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_inc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, ArraySubsetError> {
        if start.len() != end.len() || std::iter::zip(&start, &end).any(|(start, end)| end < start)
        {
            Err(ArraySubsetError::IncompatibleStartEnd {
                start,
                end,
                inclusive: true,
            })
        } else {
            let shape = std::iter::zip(&start, end)
                .map(|(&start, end)| end.saturating_sub(start) + 1)
                .collect();
            Ok(Self { start, shape })
        }
    }

    /// Create a new array subset from a start and end (exclusive).
    ///
    /// # Errors
    /// Returns [`ArraySubsetError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_exc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, ArraySubsetError> {
        if start.len() != end.len() || std::iter::zip(&start, &end).any(|(start, end)| end < start)
        {
            Err(ArraySubsetError::IncompatibleStartEnd {
                start,
                end,
                inclusive: false,
            })
        } else {
            let shape = std::iter::zip(&start, end)
                .map(|(&start, end)| end.saturating_sub(start))
                .collect();
            Ok(Self { start, shape })
        }
    }

    /// Bound the array subset to the domain within `end` (exclusive).
    ///
    /// # Errors
    /// Returns an error if `end` does not match the array subset dimensionality.
    pub fn bound(&self, end: &[u64]) -> Result<Self, ArraySubsetError> {
        if end.len() == self.start.len() {
            let start = std::iter::zip(&self.start, end)
                .map(|(&a, &b)| std::cmp::min(a, b))
                .collect();
            let end_exc = std::iter::zip(&self.start, &self.shape).map(|(&s, &l)| s + l);
            let end = std::iter::zip(end_exc, end)
                .map(|(a, &b)| std::cmp::min(a, b))
                .collect();
            Ok(Self::new_with_start_end_exc(start, end)?)
        } else {
            Err(ArraySubsetError::IncompatibleStartEnd {
                start: self.start.clone(),
                end: end.to_vec(),
                inclusive: false,
            })
        }
    }

    /// Return the start of the array subset.
    #[must_use]
    pub fn start(&self) -> &[u64] {
        &self.start
    }

    /// Return the shape of the array subset.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Return the shape of the array as a chunk shape.
    ///
    /// Returns [`None`] if the shape is not a chunk shape (i.e. it has zero dimensions).
    #[must_use]
    pub fn chunk_shape(&self) -> Option<ChunkShape> {
        self.shape.iter().map(|s| NonZeroU64::new(*s)).collect()
    }

    /// Return the shape of the array subset.
    ///
    /// # Panics
    /// Panics if a dimension exceeds [`usize::MAX`].
    #[must_use]
    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape
            .iter()
            .map(|d| usize::try_from(*d).unwrap())
            .collect()
    }

    /// Returns if the array subset is empty (i.e. has a zero element in its shape).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.iter().any(|i| i == &0)
    }

    /// Return the dimensionality of the array subset.
    #[must_use]
    pub fn dimensionality(&self) -> usize {
        self.start.len()
    }

    /// Return the number of elements of the array subset.
    ///
    /// Equal to the product of the components of its shape.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Returns an iterator over the indices of elements within the subset.
    #[must_use]
    pub fn indices(&self) -> Indices {
        Indices::new(self.clone())
    }

    /// Returns an iterator over the linearised indices of elements within the subset.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate this array subset.
    pub fn linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<LinearisedIndices, IndexerError> {
        LinearisedIndices::new(self.clone(), array_shape.to_vec())
    }

    /// Returns an iterator over the indices of contiguous elements within the subset.
    ///
    /// # Errors
    ///
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate this array subset.
    pub fn contiguous_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousIndices, IndexerError> {
        ContiguousIndices::new(self.clone(), array_shape)
    }

    /// Returns an iterator over the linearised indices of contiguous elements within the subset.
    ///
    /// # Errors
    ///
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate this array subset.
    pub fn contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousLinearisedIndices, IndexerError> {
        ContiguousLinearisedIndices::new(self.clone(), array_shape.to_vec())
    }
}

impl Indexer for ArraySubset {
    fn dimensionality(&self) -> usize {
        self.start.len()
    }

    fn len(&self) -> u64 {
        self.shape.iter().product()
    }

    fn output_shape(&self) -> Vec<u64> {
        self.shape.clone()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        Box::new(self.indices().into_iter())
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        Ok(Box::new(self.linearised_indices(array_shape)?.into_iter()))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        Ok(Box::new(
            self.contiguous_linearised_indices(array_shape)?.into_iter(),
        ))
    }

    fn as_array_subset(&self) -> Option<&ArraySubset> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_subset() {
        use crate::ArraySubsetTraits;

        assert!(ArraySubset::new_with_start_shape(vec![0, 0], vec![10, 10]).is_ok());
        assert!(ArraySubset::new_with_start_shape(vec![0, 0], vec![10]).is_err());
        assert!(ArraySubset::new_with_start_end_inc(vec![0, 0], vec![10, 10]).is_ok());
        assert!(ArraySubset::new_with_start_end_inc(vec![0, 0], vec![10]).is_err());
        assert!(ArraySubset::new_with_start_end_inc(vec![5, 5], vec![0, 0]).is_err());
        assert!(ArraySubset::new_with_start_end_exc(vec![0, 0], vec![10, 10]).is_ok());
        assert!(ArraySubset::new_with_start_end_exc(vec![0, 0], vec![10]).is_err());
        assert!(ArraySubset::new_with_start_end_exc(vec![5, 5], vec![0, 0]).is_err());
        let array_subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![10, 10])
            .unwrap()
            .bound(&[5, 5])
            .unwrap();
        assert_eq!(array_subset.shape().as_ref(), &[5, 5]);
        assert!(ArraySubset::new_with_start_shape(vec![0, 0], vec![10, 10])
            .unwrap()
            .bound(&[5, 5, 5])
            .is_err());

        let array_subset0 = ArraySubset::new_with_ranges(&[1..5, 2..6]);
        let array_subset1 = ArraySubset::new_with_ranges(&[3..6, 4..7]);
        assert_eq!(
            array_subset0.overlap(&array_subset1).unwrap(),
            ArraySubset::new_with_ranges(&[3..5, 4..6])
        );
        assert_eq!(
            array_subset0.relative_to(&[1, 1]).unwrap(),
            ArraySubset::new_with_ranges(&[0..4, 1..5])
        );
        assert!(array_subset0.relative_to(&[1, 1, 1]).is_err());
        assert!(array_subset0.inbounds_shape(&[10, 10]));
        assert!(!array_subset0.inbounds_shape(&[2, 2]));
        assert!(!array_subset0.inbounds_shape(&[10, 10, 10]));
        assert!(array_subset0.inbounds(&ArraySubset::new_with_ranges(&[0..6, 1..7])));
        assert!(array_subset0.inbounds(&ArraySubset::new_with_ranges(&[1..5, 2..6])));
        assert!(!array_subset0.inbounds(&ArraySubset::new_with_ranges(&[2..5, 2..6])));
        assert!(!array_subset0.inbounds(&ArraySubset::new_with_ranges(&[1..5, 2..5])));
        assert!(!array_subset0.inbounds(&ArraySubset::new_with_ranges(&[2..5])));
        assert_eq!(array_subset0.to_ranges(), vec![1..5, 2..6]);

        let array_subset2 = ArraySubset::new_with_ranges(&[3..6, 4..7, 0..1]);
        assert!(array_subset0.overlap(&array_subset2).is_err());
        assert_eq!(
            array_subset2
                .linearised_indices(&[6, 7, 1])
                .unwrap()
                .into_iter()
                .next(),
            Some(4 + (3 * 7))
        );
    }

    #[test]
    fn array_subset_bytes() {
        let array_subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);

        assert!(array_subset
            .iter_contiguous_byte_ranges(&[1, 1], 1)
            .is_err());
        let ranges = array_subset
            .iter_contiguous_byte_ranges(&[4, 4], 1)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(ranges, vec![5..7, 9..11]);
    }
}

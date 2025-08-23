//! Array subsets.
//!
//! An [`ArraySubset`] represents a subset of an array or chunk.
//!
//! Many [`Array`](crate::array::Array) store and retrieve methods have an [`ArraySubset`] parameter.
//! [`iterators`] includes various types of [`ArraySubset`] iterators.
//!
//! This module also provides convenience functions for:
//!  - computing the byte ranges of array subsets within an array with a fixed element size.

pub mod iterators;
use thiserror::Error;
use zarrs_metadata::ChunkShape;

use std::{
    fmt::{Debug, Display},
    num::NonZeroU64,
    ops::Range,
};

use iterators::{ContiguousIndices, ContiguousLinearisedIndices, Indices, LinearisedIndices};

use derive_more::From;
use itertools::izip;

use crate::{
    array::{ArrayError, ArrayIndices, ArrayShape},
    indexer::{IncompatibleIndexerError, Indexer},
};

/// An array subset.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct ArraySubset {
    /// The start of the array subset.
    start: ArrayIndices,
    /// The shape of the array subset.
    shape: ArrayShape,
}

impl Display for ArraySubset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    /// Returns [`IncompatibleDimensionalityError`] if the size of `start` and `size` do not match.
    pub fn new_with_start_shape(
        start: ArrayIndices,
        shape: ArrayShape,
    ) -> Result<Self, IncompatibleDimensionalityError> {
        if start.len() == shape.len() {
            Ok(Self { start, shape })
        } else {
            Err(IncompatibleDimensionalityError::new(
                start.len(),
                shape.len(),
            ))
        }
    }

    /// Create a new array subset from a start and end (inclusive).
    ///
    /// # Errors
    /// Returns [`IncompatibleStartEndIndicesError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_inc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, IncompatibleStartEndIndicesError> {
        if start.len() != end.len() || std::iter::zip(&start, &end).any(|(start, end)| end < start)
        {
            Err(IncompatibleStartEndIndicesError::from((start, end)))
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
    /// Returns [`IncompatibleStartEndIndicesError`] if `start` and `end` are incompatible, such as if any element of `end` is less than `start` or they differ in length.
    pub fn new_with_start_end_exc(
        start: ArrayIndices,
        end: ArrayIndices,
    ) -> Result<Self, IncompatibleStartEndIndicesError> {
        if start.len() != end.len() || std::iter::zip(&start, &end).any(|(start, end)| end < start)
        {
            Err(IncompatibleStartEndIndicesError::from((start, end)))
        } else {
            let shape = std::iter::zip(&start, end)
                .map(|(&start, end)| end.saturating_sub(start))
                .collect();
            Ok(Self { start, shape })
        }
    }

    /// Return the array subset as a vec of ranges.
    #[must_use]
    pub fn to_ranges(&self) -> Vec<Range<u64>> {
        std::iter::zip(&self.start, &self.shape)
            .map(|(&start, &size)| start..start + size)
            .collect()
    }

    /// Bound the array subset to the domain within `end` (exclusive).
    ///
    /// # Errors
    /// Returns an error if `end` does not match the array subset dimensionality.
    pub fn bound(&self, end: &[u64]) -> Result<Self, ArraySubsetError> {
        if end.len() == self.dimensionality() {
            let start = std::iter::zip(self.start(), end)
                .map(|(&a, &b)| std::cmp::min(a, b))
                .collect();
            let end = std::iter::zip(self.end_exc(), end)
                .map(|(a, &b)| std::cmp::min(a, b))
                .collect();
            Ok(Self::new_with_start_end_exc(start, end)?)
        } else {
            Err(IncompatibleDimensionalityError(end.len(), self.dimensionality()).into())
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
        self.shape
            .iter()
            .map(|s| NonZeroU64::new(*s))
            .collect::<Option<Vec<_>>>()
            .map(ChunkShape::from)
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

    /// Return the end (inclusive) of the array subset.
    ///
    /// Returns [`None`] if the array subset is empty.
    #[must_use]
    pub fn end_inc(&self) -> Option<ArrayIndices> {
        if self.is_empty() {
            None
        } else {
            Some(
                std::iter::zip(&self.start, &self.shape)
                    .map(|(start, size)| start + size - 1)
                    .collect(),
            )
        }
    }

    /// Return the end (exclusive) of the array subset.
    #[must_use]
    pub fn end_exc(&self) -> ArrayIndices {
        std::iter::zip(&self.start, &self.shape)
            .map(|(start, size)| start + size)
            .collect()
    }

    /// Return the number of elements of the array subset.
    ///
    /// Equal to the product of the components of its shape.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Return the number of elements of the array subset as a `usize`.
    ///
    /// # Panics
    ///
    /// Panics if [`num_elements()`](Self::num_elements()) is greater than [`usize::MAX`].
    #[must_use]
    pub fn num_elements_usize(&self) -> usize {
        usize::try_from(self.num_elements()).unwrap()
    }

    /// Returns [`true`] if the array subset contains `indices`.
    #[must_use]
    pub fn contains(&self, indices: &[u64]) -> bool {
        izip!(indices, &self.start, &self.shape).all(|(&i, &o, &s)| i >= o && i < o + s)
    }

    /// Return the elements in this array subset from an array with shape `array_shape`.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerError`] if the length of `array_shape` does not match the array subset dimensionality or the array subset is outside of the bounds of `array_shape`.
    ///
    /// # Panics
    /// Panics if attempting to access a byte index beyond [`usize::MAX`].
    pub fn extract_elements<T: std::marker::Copy>(
        &self,
        elements: &[T],
        array_shape: &[u64],
    ) -> Result<Vec<T>, IncompatibleIndexerError> {
        if self.dimensionality() != array_shape.len() {
            return Err(IncompatibleDimensionalityError::new(
                self.dimensionality(),
                array_shape.len(),
            )
            .into());
        }
        if elements.len() as u64 != array_shape.iter().product::<u64>() {
            return Err(IncompatibleIndexerError::new_incompatible_length(
                elements.len() as u64,
                array_shape.iter().product::<u64>(),
            ));
        }
        if self
            .end_exc()
            .iter()
            .zip(array_shape)
            .any(|(end, shape)| end > shape)
        {
            return Err(IncompatibleIndexerError::new_oob(
                self.end_exc(),
                array_shape.to_vec(),
            ));
        }

        let num_elements = usize::try_from(self.num_elements()).unwrap();
        let mut elements_subset = Vec::with_capacity(num_elements);
        let elements_subset_slice = crate::vec_spare_capacity_to_mut_slice(&mut elements_subset);
        let mut subset_offset = 0;
        // SAFETY: `array_shape` is encapsulated by an array with `array_shape`.
        for (array_index, contiguous_elements) in
            &self.contiguous_linearised_indices(array_shape)?
        {
            let element_offset = usize::try_from(array_index).unwrap();
            let element_length =
                usize::try_from(contiguous_elements * size_of::<T>() as u64).unwrap();
            debug_assert!(element_offset + element_length <= elements.len());
            debug_assert!(subset_offset + element_length <= num_elements);
            elements_subset_slice[subset_offset..subset_offset + element_length]
                .copy_from_slice(&elements[element_offset..element_offset + element_length]);
            subset_offset += element_length;
        }
        unsafe { elements_subset.set_len(num_elements) };
        Ok(elements_subset)
    }

    /// Returns an iterator over the indices of elements within the subset.
    #[must_use]
    pub fn indices(&self) -> Indices {
        Indices::new(self.clone())
    }

    /// Returns an iterator over the linearised indices of elements within the subset.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleIndexerError`] if the `array_shape` does not encapsulate this array subset.
    pub fn linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<LinearisedIndices, IncompatibleIndexerError> {
        LinearisedIndices::new(self.clone(), array_shape.to_vec())
    }

    /// Returns an iterator over the indices of contiguous elements within the subset.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleIndexerError`] if the `array_shape` does not encapsulate this array subset.
    pub fn contiguous_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousIndices, IncompatibleIndexerError> {
        ContiguousIndices::new(self, array_shape)
    }

    /// Returns an iterator over the linearised indices of contiguous elements within the subset.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleIndexerError`] if the `array_shape` does not encapsulate this array subset.
    pub fn contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousLinearisedIndices, IncompatibleIndexerError> {
        ContiguousLinearisedIndices::new(self, array_shape.to_vec())
    }

    /// Return the overlapping subset between this array subset and `subset_other`.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleDimensionalityError`] if the dimensionality of `subset_other` does not match the dimensionality of this array subset.
    pub fn overlap(&self, subset_other: &Self) -> Result<Self, IncompatibleDimensionalityError> {
        if subset_other.dimensionality() == self.dimensionality() {
            let ranges = izip!(
                &self.start,
                &self.shape,
                subset_other.start(),
                subset_other.shape(),
            )
            .map(|(start, size, other_start, other_size)| {
                let overlap_start = *std::cmp::max(start, other_start);
                let overlap_end = std::cmp::min(start + size, other_start + other_size);
                overlap_start..overlap_end
            });
            Ok(Self::from(ranges))
        } else {
            Err(IncompatibleDimensionalityError::new(
                subset_other.dimensionality(),
                self.dimensionality(),
            ))
        }
    }

    /// Return the subset relative to `offset`.
    ///
    /// Creates an array subset starting at [`ArraySubset::start()`] - `offset`.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the length of `start` does not match the dimensionality of this array subset.
    pub fn relative_to(&self, offset: &[u64]) -> Result<Self, ArraySubsetError> {
        if offset.len() != self.dimensionality() {
            Err(IncompatibleDimensionalityError::new(offset.len(), self.dimensionality()).into())
        } else if std::iter::zip(self.start(), offset.iter()).any(|(start, offset)| start < offset)
        {
            Err(IncompatibleOffsetError {
                offset: offset.to_vec(),
                start: self.start.clone(),
            }
            .into())
        } else {
            Ok(Self {
                start: std::iter::zip(self.start(), offset)
                    .map(|(start, offset)| start - offset)
                    .collect::<Vec<_>>(),
                shape: self.shape().to_vec(),
            })
        }
    }

    /// Returns true if this array subset is within the bounds of `subset`.
    #[must_use]
    pub fn inbounds(&self, subset: &ArraySubset) -> bool {
        if self.dimensionality() != subset.dimensionality() {
            return false;
        }

        for (self_start, self_shape, other_start, other_shape) in
            izip!(self.start(), self.shape(), subset.start(), subset.shape())
        {
            if self_start < other_start || self_start + self_shape > other_start + other_shape {
                return false;
            }
        }
        true
    }

    /// Returns true if the array subset is within the bounds of an `ArraySubset` with zero origin and a shape of `array_shape`.
    #[must_use]
    pub fn inbounds_shape(&self, array_shape: &[u64]) -> bool {
        if self.dimensionality() != array_shape.len() {
            return false;
        }

        for (subset_start, subset_shape, shape) in izip!(self.start(), self.shape(), array_shape) {
            if subset_start + subset_shape > *shape {
                return false;
            }
        }
        true
    }
}

impl Indexer for ArraySubset {
    fn dimensionality(&self) -> usize {
        self.start.len()
    }

    fn len(&self) -> u64 {
        self.num_elements()
    }

    fn output_shape(&self) -> Vec<u64> {
        self.shape().to_vec()
    }

    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync> {
        Box::new(self.indices().into_iter())
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerError> {
        Ok(Box::new(self.linearised_indices(array_shape)?.into_iter()))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerError> {
        Ok(Box::new(
            self.contiguous_linearised_indices(array_shape)?.into_iter(),
        ))
    }

    fn as_array_subset(&self) -> Option<&ArraySubset> {
        Some(self)
    }
}

/// An incompatible dimensionality error.
#[derive(Copy, Clone, Debug, Error)]
#[error("incompatible dimensionality {0}, expected {1}")]
pub struct IncompatibleDimensionalityError(usize, usize);

impl IncompatibleDimensionalityError {
    /// Create a new incompatible dimensionality error.
    #[must_use]
    pub const fn new(got: usize, expected: usize) -> Self {
        Self(got, expected)
    }
}

/// An incompatible start/end indices error.
#[derive(Clone, Debug, Error, From)]
#[error("incompatible start {0:?} with end {1:?}")]
pub struct IncompatibleStartEndIndicesError(ArrayIndices, ArrayIndices);

/// An incompatible offset error.
#[derive(Clone, Debug, Error, From)]
#[error("incompatible offset {offset:?} for start {start:?}")]
pub struct IncompatibleOffsetError {
    offset: ArrayIndices,
    start: ArrayIndices,
}

/// Array errors.
#[derive(Clone, Debug, Error)]
pub enum ArraySubsetError {
    /// Incompatible dimensionality.
    #[error(transparent)]
    IncompatibleDimensionalityError(#[from] IncompatibleDimensionalityError),
    /// Start and end are not compatible.
    #[error(transparent)]
    IncompatibleStartEndIndicesError(#[from] IncompatibleStartEndIndicesError),
    /// An incompatible offset.
    #[error(transparent)]
    IncompatibleOffset(#[from] IncompatibleOffsetError),
}

impl From<ArraySubsetError> for ArrayError {
    fn from(arr_subset_err: ArraySubsetError) -> Self {
        match arr_subset_err {
            ArraySubsetError::IncompatibleDimensionalityError(v) => v.into(),
            ArraySubsetError::IncompatibleStartEndIndicesError(v) => v.into(),
            ArraySubsetError::IncompatibleOffset(v) => v.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::byte_range::ByteRange;

    #[test]
    fn array_subset() {
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
        assert_eq!(array_subset.shape(), &[5, 5]);
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
            Some(4 * 1 + 3 * 7 * 1)
        )
    }

    #[test]
    fn array_subset_bytes() {
        let array_subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);

        assert!(array_subset.byte_ranges(&[1, 1], 1).is_err());
        let ranges = array_subset
            .byte_ranges(&[4, 4], 1)
            .unwrap()
            .collect::<Vec<ByteRange>>();

        assert_eq!(
            ranges,
            vec![
                ByteRange::FromStart(5, Some(2)),
                ByteRange::FromStart(9, Some(2))
            ]
        );
    }
}


use thiserror::Error;

use std::{
    fmt::{Debug, Display},
    num::NonZeroU64,
    ops::Range,
};

use crate::array_subset::{iterators::{
    Chunks, ContiguousIndices, ContiguousLinearisedIndices, Indices, LinearisedIndices,
}, ArraySubset, ArraySubsetError, IncompatibleStartEndIndicesError};

use derive_more::From;
use itertools::izip;

use crate::{
    array::{unravel_index, ArrayError, ArrayIndices, ArrayShape}, array_subset::{IncompatibleDimensionalityError,IncompatibleArraySubsetAndShapeError}, storage::byte_range::ByteRange
};

use crate::array_subset::indexers::{Indexer, IndexerEnum};

/// An array subset.
///
/// The unsafe `_unchecked methods` are mostly intended for internal use to avoid redundant input validation.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct RangeSubset {
    /// The start of the array subset.
    start: ArrayIndices,
    /// The shape of the array subset.
    shape: ArrayShape,
}

impl Display for RangeSubset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_ranges().fmt(f)
    }
}

impl<T: IntoIterator<Item = Range<u64>>> From<T> for RangeSubset {
    fn from(ranges: T) -> Self {
        let (start, shape) = ranges
        .into_iter()
        .map(|range| (range.start, range.end.saturating_sub(range.start)))
        .unzip();
        Self { start, shape }
    }
}

impl Indexer for RangeSubset {

    fn num_elements(&self) -> u64 {
        todo!()
    }

    fn is_compatible_shape(&self,array_shape: &[u64]) -> bool {
        todo!()
    }

    fn find_linearised_index(&self,index:usize) -> ArrayIndices {
        todo!()
    }

 
    fn shape(&self) ->  &[u64] {
        todo!()
    }


    fn start(&self) ->  &[u64] {
        todo!()
    }


    fn dimensionality(&self) -> usize {
        todo!()
    }

    fn end_exc(&self) -> ArrayIndices {
        todo!()
    }

    fn byte_ranges(&self,array_shape: &[u64],element_size:usize,) -> Result<Vec<ByteRange>,IncompatibleArraySubsetAndShapeError> {
        todo!()
    }

    fn to_enum(&self) -> IndexerEnum {
        todo!()
    }

    fn contains(&self,indices: &[u64]) -> bool {
        todo!()
    }

    fn overlap(&self,subset_other: &ArraySubset) -> Result<ArraySubset,IncompatibleDimensionalityError> {
        todo!()
    }

    fn relative_to(&self,start: &[u64]) -> Result<ArraySubset,IncompatibleDimensionalityError> {
        todo!()
    }

    fn contiguous_indices(&self,array_shape: &[u64],) -> Result<ContiguousIndices,IncompatibleArraySubsetAndShapeError> {
        todo!()
    }

    fn contiguous_linearised_indices(&self,array_shape: &[u64],) -> Result<ContiguousLinearisedIndices,IncompatibleArraySubsetAndShapeError> {
        todo!()
    }

    fn end_inc(&self) -> Option<ArrayIndices> {
        todo!()
    }
}

impl RangeSubset {
 
    /// Create a new array subset from a list of [`Range`]s.
    #[must_use]
    pub fn new_with_ranges(ranges: &[Range<u64>]) -> Self {
        let start = ranges.iter().map(|range| range.start).collect();
        let shape = ranges.iter().map(|range| range.end - range.start).collect();
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
            Err(IncompatibleDimensionalityError::new(end.len(), self.dimensionality()).into())
        }
    }

    /// Return the elements in this array subset from an array with shape `array_shape`.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleArraySubsetAndShapeError`] if the length of `array_shape` does not match the array subset dimensionality or the array subset is outside of the bounds of `array_shape`.
    ///
    /// # Panics
    /// Panics if attempting to access a byte index beyond [`usize::MAX`].
    pub fn extract_elements<T: std::marker::Copy>(
        &self,
        elements: &[T],
        array_shape: &[u64],
    ) -> Result<Vec<T>, IncompatibleArraySubsetAndShapeError> {
        let is_same_shape = elements.len() as u64 == array_shape.iter().product::<u64>();
        let is_correct_dimensionality = array_shape.len() == self.dimensionality();
        let is_in_bounds = self
            .end_exc()
            .iter()
            .zip(array_shape)
            .all(|(end, shape)| end <= shape);
        if !(is_correct_dimensionality && is_in_bounds && is_same_shape) {
            return Err(IncompatibleArraySubsetAndShapeError::new(
self.to_enum().into(),
                array_shape.to_vec(),
            ));
        }
        let num_elements = usize::try_from(self.num_elements()).unwrap();
        let mut elements_subset = Vec::with_capacity(num_elements);
        let elements_subset_slice = crate::vec_spare_capacity_to_mut_slice(&mut elements_subset);
        let mut subset_offset = 0;
        // SAFETY: `array_shape` is encapsulated by an array with `array_shape`.
        let contiguous_elements = self.contiguous_linearised_indices(array_shape)?;
        let element_length = contiguous_elements.contiguous_elements_usize();
        for array_index in &contiguous_elements {
            let element_offset = usize::try_from(array_index).unwrap();
            debug_assert!(element_offset + element_length <= elements.len());
            debug_assert!(subset_offset + element_length <= num_elements);
            elements_subset_slice[subset_offset..subset_offset + element_length]
                .copy_from_slice(&elements[element_offset..element_offset + element_length]);
            subset_offset += element_length;
        }
        unsafe { elements_subset.set_len(num_elements) };
        Ok(elements_subset)
    }

    fn find_on_axis(&self, index: &u64, axis: usize) -> u64 {
        let shape = self.start();
        shape[axis] + index
    }
    /// Create a new empty array subset.
    #[must_use]
    pub fn new_empty(dimensionality: usize) -> Self {
        Self {
            start: vec![0; dimensionality],
            shape: vec![0; dimensionality],
        }
    }
}
//! Indexer trait with common functionality
use std::num::NonZeroU64;

use derive_more::{Display, From};
use enum_dispatch::enum_dispatch;
use itertools::izip;
use serde_json::value::Index;
use thiserror::Error;
use zarrs_metadata::ArrayShape;
use zarrs_storage::byte_range::ByteRange;

use crate::{
    array::ArrayIndices,
    array_subset::{
        indexers::{RangeSubset, VIndex},
        iterators::{Chunks, ContiguousIndices, ContiguousLinearisedIndices, LinearisedIndices},
        ArraySubset, IncompatibleArraySubsetAndShapeError, IncompatibleDimensionalityError,
    },
};

#[enum_dispatch]
pub trait Indexer: Send + Sync + Clone {
    /// Return the number of elements of the array subset.
    ///
    /// Equal to the product of the components of its shape.
    fn num_elements(&self) -> u64;
    /// Return the number of elements of the array subset as a `usize`.
    ///
    /// # Panics
    ///
    /// Panics if [`num_elements()`](Self::num_elements()) is greater than [`usize::MAX`].
    fn num_elements_usize(&self) -> usize {
        usize::try_from(self.num_elements()).unwrap()
    }
    /// Determines if the given shape is compatible with the current indexer's shape
    /// i.e., it's shape is less than or equal [`shape()`](Self::shape())
    /// and fulfills other any constraints e.g., equal axis lengths for v-indexing.
    /// This function answers the question: given a parent array's shape, is this
    /// subset compatible?
    fn is_compatible_shape(&self, array_shape: &[u64]) -> bool;
    /// Returns true if the [`Indexer`] is within the bounds of an `ArraySubset` with zero origin and a shape of `array_shape`.
    fn inbounds_shape(&self, array_shape: &[u64]) -> bool {
        if self.dimensionality() != array_shape.len() {
            return false;
        }

        for (end, shape) in self.end_exc().iter().zip(array_shape) {
            if end > shape {
                return false;
            }
        }
        true
    }
    /// For a linearised index, unravel it and return the resulting [`ArrayIndices`] that represents
    /// the `index`-th value of this [`Indexer`] i.e., for a range subset, `index` offset by [`start()`](Self::start())
    fn find_linearised_index(&self, index: usize) -> ArrayIndices;
    /// Shape of the [`Indexer`]
    #[must_use]
    fn shape(&self) -> &[u64];
    /// Get the `index`-th value along an `axis` i.e., for a range subset, `index` offset by the `axis` of [`start()`](Self::start())
    /// Returns true if this array subset is within the bounds of `subset`.
    #[must_use]
    fn inbounds(&self, subset: &ArraySubset) -> bool {
        if self.dimensionality() != subset.dimensionality() {
            return false;
        }

        for (self_start, self_end, other_start, other_end) in izip!(
            self.start(),
            self.end_exc(),
            subset.start(),
            subset.end_exc()
        ) {
            if self_start < other_start || self_end > other_end {
                return false;
            }
        }
        true
    }

    /// Return the start of the array subset.
    #[must_use]
    fn start(&self) -> &[u64];
    /// Return the dimensionality of the array subset.
    #[must_use]
    fn dimensionality(&self) -> usize;
    /// Return the end (exclusive) of the array subset.
    #[must_use]
    fn end_exc(&self) -> ArrayIndices;

    /// Return the byte ranges of an array subset in an array with `array_shape` and `element_size`.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleArraySubsetAndShapeError`] if the `array_shape` does not encapsulate this array subset.
    fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Vec<ByteRange>, IncompatibleArraySubsetAndShapeError>;

    /// Returns [`true`] if the array subset contains `indices`.
    #[must_use]
    fn contains(&self, indices: &[u64]) -> bool;

    /// Return the overlapping subset between this array subset and `subset_other`.
    ///
    /// # Errors
    ///
    /// Returns [`IncompatibleDimensionalityError`] if the dimensionality of `subset_other` does not match the dimensionality of this array subset.
    fn overlap(
        &self,
        subset_other: &ArraySubset,
    ) -> Result<ArraySubset, IncompatibleDimensionalityError>;

    /// Return the subset relative to `start`.
    ///
    /// Creates an array subset starting at [`ArraySubset::start()`] - `start`.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the length of `start` does not match the dimensionality of this array subset.
    fn relative_to(&self, start: &[u64]) -> Result<ArraySubset, IncompatibleDimensionalityError>;

    /// Return the shape of the array subset.
    ///
    /// # Panics
    /// Panics if a dimension exceeds [`usize::MAX`].
    #[must_use]
    fn shape_usize(&self) -> Vec<usize> {
        self.shape()
            .iter()
            .map(|d| usize::try_from(*d).unwrap())
            .collect()
    }

    /// Returns if the array subset is empty (i.e. has a zero element in its shape).
    #[must_use]
    fn is_empty(&self) -> bool {
        self.shape().iter().any(|i| i == &0)
    }

    fn end_inc(&self) -> Option<ArrayIndices>;

    fn contiguous_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<ContiguousIndices, IncompatibleArraySubsetAndShapeError>;

    fn to_enum(&self) -> IndexerEnum;

    fn chunks(
        &self,
        chunk_shape: &[NonZeroU64],
    ) -> Result<Chunks, IncompatibleDimensionalityError>;
}

#[enum_dispatch(Indexer)]
#[derive(Clone, Error, Debug, Eq, PartialEq, Display, PartialOrd, Ord, Hash)]
pub enum IndexerEnum {
    RangeSubset,
    VIndex,
}

impl Default for IndexerEnum {
    fn default() -> Self {
        IndexerEnum::RangeSubset(Default::default())
    }
}


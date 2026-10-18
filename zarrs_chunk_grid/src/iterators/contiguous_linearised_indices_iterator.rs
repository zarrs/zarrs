use std::iter::FusedIterator;

use super::contiguous_indices_iterator::shrink_to_run_starts;
use super::strided_indices::{StridedIndices, StridedOdometer};
use crate::indexer::validate_start_shape;
use crate::{ArrayIndices, ArraySubsetTraits, IndexerError};

/// Iterates over contiguous linearised element indices in an array subset.
///
/// The iterator item is a tuple: (linearised index, # contiguous elements).
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// For example, consider a 4x3 array with linearised element indices
/// ```text
/// 0   1   2
/// 3   4   5
/// 6   7   8
/// 9  10  11
/// ```
/// An iterator with an array subset covering the entire array will produce
/// ```rust,ignore
/// [(0, 9)]
/// ```
/// An iterator with an array subset corresponding to the lower right 2x2 region will produce
/// ```rust,ignore
/// [(7, 2), (10, 2)]
/// ```
#[derive(Clone)]
pub struct ContiguousLinearisedIndices {
    /// The linearised indices of the starts of the contiguous runs.
    runs: StridedIndices,
    contiguous_elements: u64,
}

impl ContiguousLinearisedIndices {
    /// Return a new contiguous linearised indices iterator.
    ///
    /// # Errors
    ///
    /// Returns [`IndexerError`] if `array_shape` has an incompatible dimensionality or does not encapsulate `subset`.
    ///
    /// # Panics
    /// Panics if the linearised indices of `subset` exceed [`u64::MAX`].
    pub fn new(
        subset: &(impl ArraySubsetTraits + ?Sized),
        array_shape: &[u64],
    ) -> Result<Self, IndexerError> {
        let start = subset.start();
        let mut shape: ArrayIndices = subset.shape().iter().copied().collect();
        validate_start_shape(&start, &shape, array_shape)?;
        // The shape of the starts of contiguous runs, in place of the shape of the subset
        let contiguous_elements = shrink_to_run_starts(&start, &mut shape, array_shape);
        Ok(Self {
            runs: StridedIndices::new(&start, &shape, array_shape),
            contiguous_elements,
        })
    }

    /// Return the number of starting indices (i.e. the length of the iterator).
    ///
    /// # Panics
    /// Panics if the number of starting indices exceeds [`usize::MAX`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.runs.len()
    }

    /// Returns true if the number of starting indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.num_elements() == 0
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    #[must_use]
    pub fn contiguous_elements(&self) -> u64 {
        self.contiguous_elements
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    ///
    /// # Panics
    /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
    #[must_use]
    pub fn contiguous_elements_usize(&self) -> usize {
        usize::try_from(self.contiguous_elements).unwrap()
    }

    /// Create a new serial iterator.
    ///
    /// # Panics
    /// Panics if the number of starting indices exceeds [`usize::MAX`].
    #[must_use]
    pub fn iter(&self) -> ContiguousLinearisedIndicesIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a> IntoIterator for &'a ContiguousLinearisedIndices {
    type Item = (u64, u64);
    type IntoIter = ContiguousLinearisedIndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ContiguousLinearisedIndicesIterator {
            odometer: StridedOdometer::new(&self.runs),
            indices: self,
        }
    }
}

impl IntoIterator for ContiguousLinearisedIndices {
    type Item = (u64, u64);
    type IntoIter = ContiguousLinearisedIndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        ContiguousLinearisedIndicesIntoIterator {
            odometer: StridedOdometer::new(&self.runs),
            indices: self,
        }
    }
}

/// Serial contiguous linearised indices iterator.
///
/// See [`ContiguousLinearisedIndices`].
#[derive(Clone)]
pub struct ContiguousLinearisedIndicesIterator<'a> {
    indices: &'a ContiguousLinearisedIndices,
    odometer: StridedOdometer,
}

/// Serial contiguous linearised indices iterator.
///
/// See [`ContiguousLinearisedIndices`].
#[derive(Clone)]
pub struct ContiguousLinearisedIndicesIntoIterator {
    indices: ContiguousLinearisedIndices,
    odometer: StridedOdometer,
}

macro_rules! impl_contiguous_linearised_indices_iterator {
    ($iterator_type:ty) => {
        impl $iterator_type {
            /// Return the number of contiguous elements (fixed on each iteration).
            #[must_use]
            pub fn contiguous_elements(&self) -> u64 {
                self.indices.contiguous_elements()
            }

            /// Return the number of contiguous elements (fixed on each iteration).
            ///
            /// # Panics
            /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
            #[must_use]
            pub fn contiguous_elements_usize(&self) -> usize {
                self.indices.contiguous_elements_usize()
            }
        }

        impl Iterator for $iterator_type {
            type Item = (u64, u64);

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let run = self.odometer.next(&self.indices.runs)?;
                Some((run, self.indices.contiguous_elements))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.odometer.size_hint()
            }
        }

        impl DoubleEndedIterator for $iterator_type {
            fn next_back(&mut self) -> Option<Self::Item> {
                let run = self.odometer.next_back(&self.indices.runs)?;
                Some((run, self.indices.contiguous_elements))
            }
        }

        impl ExactSizeIterator for $iterator_type {}

        impl FusedIterator for $iterator_type {}
    };
}

impl_contiguous_linearised_indices_iterator!(ContiguousLinearisedIndicesIterator<'_>);
impl_contiguous_linearised_indices_iterator!(ContiguousLinearisedIndicesIntoIterator);

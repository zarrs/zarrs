use std::iter::FusedIterator;

use super::contiguous_indices_iterator::ContiguousIndices;
use crate::{ArrayIndicesTinyVec, ArraySubset, IndexerError};

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
    /// The number of contiguous runs.
    len: usize,
    contiguous_elements: u64,
    /// The linearised index of the first run.
    first: u64,
    /// The number of runs along each dimension in which there is more than one, outermost first.
    counts: ArrayIndicesTinyVec,
    /// The distance between the linearised indices of adjacent runs along each dimension of `counts`.
    strides: ArrayIndicesTinyVec,
}

impl ContiguousLinearisedIndices {
    /// Return a new contiguous linearised indices iterator.
    ///
    /// # Errors
    ///
    /// Returns [`IndexerError`] if `array_shape` has an incompatible dimensionality or does not encapsulate `subset`.
    pub fn new(subset: ArraySubset, array_shape: &[u64]) -> Result<Self, IndexerError> {
        let contiguous = ContiguousIndices::new(subset, array_shape)?;
        let len = contiguous.len();
        let runs = contiguous.run_starts();
        let mut counts = ArrayIndicesTinyVec::new();
        let mut strides = ArrayIndicesTinyVec::new();
        let mut first = 0;
        if len > 0 {
            // Iterating runs only steps through the dimensions with more than one run
            let mut stride = 1;
            for ((&start, &count), &array_size) in std::iter::zip(runs.start(), runs.shape())
                .zip(array_shape)
                .rev()
            {
                first += start * stride;
                if count > 1 {
                    counts.push(count);
                    strides.push(stride);
                }
                stride *= array_size;
            }
            counts.reverse();
            strides.reverse();
        }
        Ok(Self {
            len,
            contiguous_elements: contiguous.contiguous_elements(),
            first,
            counts,
            strides,
        })
    }

    /// Return the number of starting indices (i.e. the length of the iterator).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the number of starting indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
    #[must_use]
    pub fn iter(&self) -> ContiguousLinearisedIndicesIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }

    /// The linearised index of the run at `index`.
    fn run_at(&self, mut index: u64) -> u64 {
        let mut run = self.first;
        for (&count, &stride) in std::iter::zip(&self.counts, &self.strides).rev() {
            run += (index % count) * stride;
            index /= count;
        }
        run
    }
}

/// The position of an iterator over the runs of [`ContiguousLinearisedIndices`].
///
/// Runs are iterated forwards like an odometer, by adding the stride of the innermost dimension and carrying into
/// outer dimensions, so that no indices are unravelled or ravelled.
#[derive(Clone)]
struct Odometer {
    /// The position of the next run from the front along each dimension of the counts.
    positions: ArrayIndicesTinyVec,
    /// The linearised index of the next run from the front.
    front_run: u64,
    /// The indices of the remaining runs.
    range: std::ops::Range<usize>,
}

impl Odometer {
    fn new(indices: &ContiguousLinearisedIndices) -> Self {
        Self {
            positions: std::iter::repeat_n(0, indices.counts.len()).collect(),
            front_run: indices.first,
            range: 0..indices.len,
        }
    }

    fn next(&mut self, indices: &ContiguousLinearisedIndices) -> Option<(u64, u64)> {
        if self.range.is_empty() {
            return None;
        }
        let run = self.front_run;
        self.range.start += 1;
        if !self.range.is_empty() {
            for ((position, &count), &stride) in
                std::iter::zip(&mut self.positions, &indices.counts)
                    .zip(&indices.strides)
                    .rev()
            {
                *position += 1;
                self.front_run += stride;
                if *position < count {
                    break;
                }
                *position = 0;
                self.front_run -= count * stride;
            }
        }
        Some((run, indices.contiguous_elements))
    }

    fn next_back(&mut self, indices: &ContiguousLinearisedIndices) -> Option<(u64, u64)> {
        let index = self.range.next_back()?;
        Some((indices.run_at(index as u64), indices.contiguous_elements))
    }
}

impl<'a> IntoIterator for &'a ContiguousLinearisedIndices {
    type Item = (u64, u64);
    type IntoIter = ContiguousLinearisedIndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ContiguousLinearisedIndicesIterator {
            odometer: Odometer::new(self),
            indices: self,
        }
    }
}

impl IntoIterator for ContiguousLinearisedIndices {
    type Item = (u64, u64);
    type IntoIter = ContiguousLinearisedIndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        ContiguousLinearisedIndicesIntoIterator {
            odometer: Odometer::new(&self),
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
    odometer: Odometer,
}

/// Serial contiguous linearised indices iterator.
///
/// See [`ContiguousLinearisedIndices`].
#[derive(Clone)]
pub struct ContiguousLinearisedIndicesIntoIterator {
    indices: ContiguousLinearisedIndices,
    odometer: Odometer,
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
                self.odometer.next(&self.indices)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.odometer.range.size_hint()
            }
        }

        impl DoubleEndedIterator for $iterator_type {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.odometer.next_back(&self.indices)
            }
        }

        impl ExactSizeIterator for $iterator_type {}

        impl FusedIterator for $iterator_type {}
    };
}

impl_contiguous_linearised_indices_iterator!(ContiguousLinearisedIndicesIterator<'_>);
impl_contiguous_linearised_indices_iterator!(ContiguousLinearisedIndicesIntoIterator);

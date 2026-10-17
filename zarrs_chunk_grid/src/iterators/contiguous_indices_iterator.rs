use std::iter::FusedIterator;

use itertools::izip;

use super::IndicesIterator;
use crate::iterators::indices_iterator::IndicesIntoIterator;
use crate::{ArrayIndicesTinyVec, ArraySubset, IndexerError};

/// Iterates over contiguous element indices in an array subset.
///
/// The iterator item is a tuple: (indices, # contiguous elements).
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// For example, consider a 4x3 array with element indices
/// ```text
/// (0, 0)  (0, 1)  (0, 2)
/// (1, 0)  (1, 1)  (1, 2)
/// (2, 0)  (2, 1)  (2, 2)
/// (3, 0)  (3, 1)  (3, 2)
/// ```
/// An iterator with an array subset covering the entire array will produce
/// ```rust,ignore
/// [((0, 0), 9)]
/// ```
/// An iterator with an array subset corresponding to the lower right 2x2 region will produce
/// ```rust,ignore
/// [((2, 1), 2), ((3, 1), 2)]
/// ```
#[derive(Clone)]
pub struct ContiguousIndices {
    subset_contiguous_start: ArraySubset,
    contiguous_elements: u64,
}

impl ContiguousIndices {
    /// Create a new contiguous indices iterator.
    ///
    /// An empty `subset` is always in-bounds unless it has an incompatible dimensionality.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if `array_shape` has an incompatible dimensionality or does not encapsulate `subset`.
    pub fn new(mut subset: ArraySubset, array_shape: &[u64]) -> Result<Self, IndexerError> {
        if subset.dimensionality() != array_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                subset.dimensionality(),
                array_shape.len(),
            ));
        }
        if subset.is_empty() {
            // An empty subset references no elements, so it is always in-bounds and yields
            // no contiguous indices.
            return Ok(Self {
                subset_contiguous_start: subset,
                contiguous_elements: 0,
            });
        }
        if izip!(&subset.start, &subset.shape, array_shape)
            .any(|(&start, &size, &bound)| start.checked_add(size).is_none_or(|end| end > bound))
        {
            return Err(IndexerError::new_oob(
                subset.end_exc(),
                array_shape.to_vec(),
            ));
        }

        // The subset of the starts of contiguous runs, in place of the subset: the trailing dimensions of a run have a
        // length of one
        let mut contiguous = true;
        let mut contiguous_elements = 1;
        for (&start, size, &array_size) in izip!(
            subset.start.iter().rev(),
            subset.shape.iter_mut().rev(),
            array_shape.iter().rev(),
        ) {
            if !contiguous {
                break;
            }
            contiguous_elements *= *size;
            contiguous = start == 0 && *size == array_size;
            *size = 1;
        }
        Ok(Self {
            subset_contiguous_start: subset,
            contiguous_elements,
        })
    }

    /// Return the number of starting indices (i.e. the length of the iterator).
    #[must_use]
    pub fn len(&self) -> usize {
        self.subset_contiguous_start.num_elements_usize()
    }

    /// The subset of the starts of the contiguous runs.
    pub(crate) fn run_starts(&self) -> &ArraySubset {
        &self.subset_contiguous_start
    }

    /// Returns true if the number of starting indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    pub fn iter(&self) -> ContiguousIndicesIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a> IntoIterator for &'a ContiguousIndices {
    type Item = (ArrayIndicesTinyVec, u64);
    type IntoIter = ContiguousIndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let n_elements = self.subset_contiguous_start.num_elements_usize();
        ContiguousIndicesIterator {
            inner: IndicesIterator {
                subset: &self.subset_contiguous_start,
                range: 0..n_elements,
            },
            contiguous_elements: self.contiguous_elements,
        }
    }
}

impl IntoIterator for ContiguousIndices {
    type Item = (ArrayIndicesTinyVec, u64);
    type IntoIter = ContiguousIndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let n_elements = self.subset_contiguous_start.num_elements_usize();
        ContiguousIndicesIntoIterator {
            inner: IndicesIntoIterator {
                subset: self.subset_contiguous_start,
                range: 0..n_elements,
            },
            contiguous_elements: self.contiguous_elements,
        }
    }
}

/// Serial contiguous indices iterator.
///
/// See [`ContiguousIndices`].
pub struct ContiguousIndicesIterator<'a> {
    inner: IndicesIterator<'a>,
    contiguous_elements: u64,
}

/// Serial contiguous indices iterator.
///
/// See [`ContiguousIndices`].
pub struct ContiguousIndicesIntoIterator {
    inner: IndicesIntoIterator,
    contiguous_elements: u64,
}

macro_rules! impl_contiguous_indices_iterator {
    ($iterator_type:ty) => {
        impl $iterator_type {
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
        }

        impl Iterator for $iterator_type {
            type Item = (ArrayIndicesTinyVec, u64);

            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next().map(|i| (i, self.contiguous_elements()))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        impl DoubleEndedIterator for $iterator_type {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.inner
                    .next_back()
                    .map(|i| (i, self.contiguous_elements()))
            }
        }

        impl ExactSizeIterator for $iterator_type {}

        impl FusedIterator for $iterator_type {}
    };
}

impl_contiguous_indices_iterator!(ContiguousIndicesIterator<'_>);
impl_contiguous_indices_iterator!(ContiguousIndicesIntoIterator);

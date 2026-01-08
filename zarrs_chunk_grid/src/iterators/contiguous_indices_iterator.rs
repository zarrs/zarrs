use std::iter::FusedIterator;

use itertools::izip;

use super::IndicesIterator;
use crate::iterators::indices_iterator::IndicesIntoIterator;
use crate::{ArrayIndicesTinyVec, ArraySubset, ArraySubsetTraits, IndexerError};

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
    /// # Errors
    /// Returns [`IndexerError`] if `array_shape` does not encapsulate `subset`.
    pub fn new(subset: ArraySubset, array_shape: &[u64]) -> Result<Self, IndexerError> {
        if subset.dimensionality() != array_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                subset.dimensionality(),
                array_shape.len(),
            ));
        }
        if std::iter::zip(subset.end_exc(), array_shape).any(|(end, shape)| end > *shape) {
            return Err(IndexerError::new_oob(
                subset.end_exc(),
                array_shape.to_vec(),
            ));
        }

        if subset.is_empty() {
            if std::iter::zip(subset.start().iter(), array_shape)
                .any(|(start, shape)| start >= shape)
            {
                // The empty subset is out-of-bounds.
                return Err(IndexerError::new_oob(
                    subset.start().to_vec(),
                    array_shape.to_vec(),
                ));
            }

            // The empty subset is in-bounds, not an error.
            return Ok(Self {
                subset_contiguous_start: subset,
                contiguous_elements: 0,
            });
        }

        let mut contiguous = true;
        let mut contiguous_elements = 1;
        let mut shape_out: Vec<u64> = Vec::with_capacity(array_shape.len());
        let subset_start = subset.start();
        let subset_shape = subset.shape();
        for (&subset_start, &subset_size, &array_size, shape_out_i) in izip!(
            subset_start.iter().rev(),
            subset_shape.iter().rev(),
            array_shape.iter().rev(),
            shape_out.spare_capacity_mut().iter_mut().rev(),
        ) {
            if contiguous {
                contiguous_elements *= subset_size;
                shape_out_i.write(1);
                contiguous = subset_start == 0 && subset_size == array_size;
            } else {
                shape_out_i.write(subset_size);
            }
        }
        // SAFETY: each element is initialised
        unsafe { shape_out.set_len(array_shape.len()) };
        let ranges = subset_start
            .iter()
            .zip(shape_out)
            .map(|(&st, sh)| st..(st + sh));
        let subset_contiguous_start = ArraySubset::from(ranges);
        // let inner = subset_contiguous_start.iter_indices();
        Ok(Self {
            subset_contiguous_start,
            contiguous_elements,
        })
    }

    /// Return the number of starting indices (i.e. the length of the iterator).
    #[must_use]
    pub fn len(&self) -> usize {
        self.subset_contiguous_start.num_elements_usize()
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

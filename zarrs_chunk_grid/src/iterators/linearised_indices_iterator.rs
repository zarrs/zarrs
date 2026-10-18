use std::iter::FusedIterator;

use super::strided_indices::{StridedIndices, StridedOdometer};
use crate::indexer::validate_start_shape;
use crate::{ArraySubsetTraits, IndexerError};

/// An iterator over the linearised (lexicographic) indices in an array subset.
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// For example, consider a 4x3 array with linearised element indices
/// ```text
/// 0   1   2
/// 3   4   5
/// 6   7   8
/// 9  10  11
/// ```
/// An iterator with an array subset corresponding to the lower right 2x2 region will produce `[7, 8, 10, 11]`.
#[derive(Clone)]
pub struct LinearisedIndices {
    indices: StridedIndices,
}

impl LinearisedIndices {
    /// Create a new linearised indices iterator.
    ///
    /// An empty `subset` is always in-bounds.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if `array_shape` has an incompatible dimensionality or does not encapsulate `subset`.
    pub fn new(
        subset: &(impl ArraySubsetTraits + ?Sized),
        array_shape: &[u64],
    ) -> Result<Self, IndexerError> {
        let (start, shape) = (subset.start(), subset.shape());
        validate_start_shape(&start, &shape, array_shape)?;
        Ok(Self {
            indices: StridedIndices::new(&start, &shape, array_shape),
        })
    }

    /// Create a new linearised indices iterator.
    ///
    /// # Safety
    /// `array_shape` must encapsulate `subset`.
    #[must_use]
    pub unsafe fn new_unchecked(
        subset: &(impl ArraySubsetTraits + ?Sized),
        array_shape: &[u64],
    ) -> Self {
        let (start, shape) = (subset.start(), subset.shape());
        debug_assert!(validate_start_shape(&start, &shape, array_shape).is_ok());
        Self {
            indices: StridedIndices::new(&start, &shape, array_shape),
        }
    }

    /// Return the number of indices.
    ///
    /// # Panics
    /// Panics if the number of indices exceeds [`usize::MAX`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if the number of indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.num_elements() == 0
    }

    /// Create a new serial iterator.
    ///
    /// # Panics
    /// Panics if the number of indices exceeds [`usize::MAX`].
    #[must_use]
    pub fn iter(&self) -> LinearisedIndicesIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a> IntoIterator for &'a LinearisedIndices {
    type Item = u64;
    type IntoIter = LinearisedIndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        LinearisedIndicesIterator {
            odometer: StridedOdometer::new(&self.indices),
            indices: &self.indices,
        }
    }
}

impl IntoIterator for LinearisedIndices {
    type Item = u64;
    type IntoIter = LinearisedIndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        LinearisedIndicesIntoIterator {
            odometer: StridedOdometer::new(&self.indices),
            indices: self.indices,
        }
    }
}

/// Serial linearised indices iterator.
///
/// See [`LinearisedIndices`].
#[derive(Clone)]
pub struct LinearisedIndicesIterator<'a> {
    indices: &'a StridedIndices,
    odometer: StridedOdometer,
}

/// Serial linearised indices iterator.
///
/// See [`LinearisedIndices`].
#[derive(Clone)]
pub struct LinearisedIndicesIntoIterator {
    indices: StridedIndices,
    odometer: StridedOdometer,
}

macro_rules! impl_linearised_indices_iterator {
    ($iterator_type:ty) => {
        impl Iterator for $iterator_type {
            type Item = u64;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.odometer.next(&self.indices)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.odometer.size_hint()
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

impl_linearised_indices_iterator!(LinearisedIndicesIterator<'_>);
impl_linearised_indices_iterator!(LinearisedIndicesIntoIterator);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArraySubset;
    use crate::iterators::ContiguousLinearisedIndices;

    #[test]
    fn linearised_indices_iterator_partial() {
        let indices =
            LinearisedIndices::new(&ArraySubset::new_with_ranges(&[1..3, 5..7]), &[8, 8]).unwrap();
        assert_eq!(indices.len(), 4);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(13)); // [1,5]
        assert_eq!(iter.next(), Some(14)); // [1,6]
        assert_eq!(iter.next_back(), Some(22)); // [2,6]
        assert_eq!(iter.next(), Some(21)); // [2,5]
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn linearised_indices_iterator_oob() {
        assert!(
            LinearisedIndices::new(&ArraySubset::new_with_ranges(&[1..3, 5..7]), &[1, 1]).is_err()
        );
    }

    #[test]
    fn linearised_indices_iterator_oob_overflow() {
        let subset = ArraySubset::new_with_start_shape(vec![u64::MAX], vec![2]).unwrap();
        assert!(LinearisedIndices::new(&subset, &[10]).is_err());
        let subset = ArraySubset::new_with_start_shape(vec![u64::MAX, 0], vec![2, 1]).unwrap();
        assert!(ContiguousLinearisedIndices::new(&subset, &[10, 1]).is_err());
    }

    #[test]
    fn linearised_indices_iterator_empty() {
        let indices =
            LinearisedIndices::new(&ArraySubset::new_with_ranges(&[1..1, 5..5]), &[5, 5]).unwrap();
        assert_eq!(indices.len(), 0);
        assert!(indices.is_empty());
    }
}

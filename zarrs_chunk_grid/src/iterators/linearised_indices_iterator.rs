use std::iter::FusedIterator;

use super::strided_indices::{StridedIndices, StridedOdometer};
use crate::{ArraySubset, IndexerError};

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
    pub fn new(subset: &ArraySubset, array_shape: &[u64]) -> Result<Self, IndexerError> {
        if subset.dimensionality() != array_shape.len() {
            Err(IndexerError::new_incompatible_dimensionality(
                subset.dimensionality(),
                array_shape.len(),
            ))
        } else if !subset.is_empty()
            && std::iter::zip(subset.end_exc(), array_shape).any(|(end, shape)| end > *shape)
        {
            // An empty subset references no elements, so it is always in-bounds.
            Err(IndexerError::new_oob(
                subset.end_exc(),
                array_shape.to_vec(),
            ))
        } else {
            // SAFETY: `array_shape` encapsulates `subset`
            Ok(unsafe { Self::new_unchecked(subset, array_shape) })
        }
    }

    /// Create a new linearised indices iterator.
    ///
    /// # Safety
    /// `array_shape` must encapsulate `subset`.
    #[must_use]
    pub unsafe fn new_unchecked(subset: &ArraySubset, array_shape: &[u64]) -> Self {
        debug_assert_eq!(subset.dimensionality(), array_shape.len());
        debug_assert!(
            subset.is_empty()
                || std::iter::zip(subset.end_exc(), array_shape).all(|(end, shape)| end <= *shape)
        );
        Self {
            indices: StridedIndices::new(subset.start(), subset.shape(), array_shape),
        }
    }

    /// Return the number of indices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if the number of indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a new serial iterator.
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
    fn linearised_indices_iterator_empty() {
        let indices =
            LinearisedIndices::new(&ArraySubset::new_with_ranges(&[1..1, 5..5]), &[5, 5]).unwrap();
        assert_eq!(indices.len(), 0);
        assert!(indices.is_empty());
    }
}

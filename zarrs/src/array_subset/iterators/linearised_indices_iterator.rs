use std::iter::FusedIterator;

use crate::{
    array::{ravel_indices, ArrayShape},
    array_subset::{
        iterators::indices_iterator::IndicesIntoIterator, ArraySubset,
        IncompatibleArraySubsetAndShapeError,
    },
};

use super::IndicesIterator;

/// An iterator over the linearised indices in an array subset.
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
pub struct LinearisedIndices {
    subset: ArraySubset,
    array_shape: ArrayShape,
}

impl LinearisedIndices {
    /// Create a new linearised indices iterator.
    ///
    /// # Errors
    /// Returns [`IncompatibleArraySubsetAndShapeError`] if `array_shape` does not encapsulate `subset`.
    pub fn new(
        subset: ArraySubset,
        array_shape: ArrayShape,
    ) -> Result<Self, IncompatibleArraySubsetAndShapeError> {
        if subset.dimensionality() == array_shape.len()
            && std::iter::zip(subset.end_exc(), &array_shape).all(|(end, shape)| end <= *shape)
        {
            Ok(Self {
                subset,
                array_shape,
            })
        } else {
            Err(IncompatibleArraySubsetAndShapeError(subset, array_shape))
        }
    }

    /// Create a new linearised indices iterator.
    ///
    /// # Safety
    /// `array_shape` must encapsulate `subset`.
    #[must_use]
    pub unsafe fn new_unchecked(subset: ArraySubset, array_shape: ArrayShape) -> Self {
        debug_assert_eq!(subset.dimensionality(), array_shape.len());
        debug_assert!(
            std::iter::zip(subset.end_exc(), &array_shape).all(|(end, shape)| end <= *shape)
        );
        Self {
            subset,
            array_shape,
        }
    }

    /// Return the number of indices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.subset.num_elements_usize()
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
            inner: IndicesIterator {
                subset: &self.subset,
                range: 0..self.subset.num_elements_usize(),
            },
            array_shape: &self.array_shape,
        }
    }
}

impl IntoIterator for LinearisedIndices {
    type Item = u64;
    type IntoIter = LinearisedIndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let num_elements = self.subset.num_elements_usize();
        LinearisedIndicesIntoIterator {
            inner: IndicesIntoIterator {
                subset: self.subset,
                range: 0..num_elements,
            },
            array_shape: self.array_shape,
        }
    }
}

/// Serial linearised indices iterator.
///
/// See [`LinearisedIndices`].
pub struct LinearisedIndicesIterator<'a> {
    inner: IndicesIterator<'a>,
    array_shape: &'a [u64],
}

impl Iterator for LinearisedIndicesIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|indices| ravel_indices(&indices, self.array_shape))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for LinearisedIndicesIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner
            .next_back()
            .map(|indices| ravel_indices(&indices, self.array_shape))
    }
}

impl ExactSizeIterator for LinearisedIndicesIterator<'_> {}

impl FusedIterator for LinearisedIndicesIterator<'_> {}

/// Serial linearised indices iterator.
///
/// See [`LinearisedIndices`].
pub struct LinearisedIndicesIntoIterator {
    inner: IndicesIntoIterator,
    array_shape: ArrayShape,
}

impl Iterator for LinearisedIndicesIntoIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|indices| ravel_indices(&indices, &self.array_shape))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for LinearisedIndicesIntoIterator {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner
            .next_back()
            .map(|indices| ravel_indices(&indices, &self.array_shape))
    }
}

impl ExactSizeIterator for LinearisedIndicesIntoIterator {}

impl FusedIterator for LinearisedIndicesIntoIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linearised_indices_iterator_partial() {
        let indices =
            LinearisedIndices::new(ArraySubset::new_with_ranges(&[1..3, 5..7]), vec![8, 8])
                .unwrap();
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
            LinearisedIndices::new(ArraySubset::new_with_ranges(&[1..3, 5..7]), vec![1, 1])
                .is_err()
        );
    }

    #[test]
    fn linearised_indices_iterator_empty() {
        let indices =
            LinearisedIndices::new(ArraySubset::new_with_ranges(&[1..1, 5..5]), vec![5, 5])
                .unwrap();
        assert_eq!(indices.len(), 0);
        assert!(indices.is_empty());
    }
}

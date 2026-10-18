use std::iter::FusedIterator;

use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use super::odometer::{GridDimension, Odometer, unravel};
use crate::{ArrayIndicesTinyVec, ArraySubset};

/// An iterator over the indices in an array subset.
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// For example, consider a 4x3 array with element indices
/// ```text
/// (0, 0)  (0, 1)  (0, 2)
/// (1, 0)  (1, 1)  (1, 2)
/// (2, 0)  (2, 1)  (2, 2)
/// (3, 0)  (3, 1)  (3, 2)
/// ```
/// An iterator with an array subset corresponding to the lower right 2x2 region will produce `[(2, 1), (2, 2), (3, 1), (3, 2)]`.
#[derive(Clone)]
pub struct Indices {
    pub(crate) subset: ArraySubset,
    pub(crate) range: std::ops::Range<usize>,
}

impl Indices {
    /// Create a new indices struct.
    #[must_use]
    pub fn new(subset: ArraySubset) -> Self {
        let length = subset.num_elements_usize();
        Self {
            subset,
            range: 0..length,
        }
    }

    /// Create a new indices struct spanning `range`.
    #[must_use]
    pub fn new_with_start_end(
        subset: ArraySubset,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Self {
        let length = subset.num_elements_usize();
        let start = match range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start.saturating_add(1),
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Excluded(end) => (*end).min(length),
            std::ops::Bound::Included(end) => end.saturating_add(1).min(length),
            std::ops::Bound::Unbounded => length,
        };
        Self {
            subset,
            range: start..end,
        }
    }

    /// Return the number of indices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.range.end.saturating_sub(self.range.start)
    }

    /// Returns true if the number of indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a new serial iterator.
    #[must_use]
    pub fn iter(&self) -> IndicesIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a> IntoIterator for &'a Indices {
    type Item = ArrayIndicesTinyVec;
    type IntoIter = IndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIterator::new(&self.subset, self.range.clone())
    }
}

impl<'a> IntoParallelRefIterator<'a> for &'a Indices {
    type Item = ArrayIndicesTinyVec;
    type Iter = ParIndicesIterator<'a>;

    fn par_iter(&self) -> Self::Iter {
        ParIndicesIterator {
            subset: &self.subset,
            range: self.range.clone(),
        }
    }
}

impl<'a> IntoParallelIterator for &'a Indices {
    type Item = ArrayIndicesTinyVec;
    type Iter = ParIndicesIterator<'a>;

    fn into_par_iter(self) -> Self::Iter {
        ParIndicesIterator {
            subset: &self.subset,
            range: self.range.clone(),
        }
    }
}

impl IntoIterator for Indices {
    type Item = ArrayIndicesTinyVec;
    type IntoIter = IndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIntoIterator::new(self.subset, self.range)
    }
}

impl IntoParallelIterator for Indices {
    type Item = ArrayIndicesTinyVec;
    type Iter = ParIndicesIntoIterator;

    fn into_par_iter(self) -> Self::Iter {
        ParIndicesIntoIterator {
            subset: self.subset,
            range: self.range,
        }
    }
}

/// Serial indices iterator.
///
/// See [`Indices`].
#[derive(Clone)]
pub struct IndicesIterator<'a> {
    subset: &'a ArraySubset,
    odometer: IndicesOdometer,
}

/// Serial indices iterator.
///
/// See [`Indices`].
#[derive(Clone)]
pub struct IndicesIntoIterator {
    subset: ArraySubset,
    odometer: IndicesOdometer,
}

impl<'a> IndicesIterator<'a> {
    pub(crate) fn new(subset: &'a ArraySubset, range: std::ops::Range<usize>) -> Self {
        Self {
            odometer: IndicesOdometer::new(subset, range),
            subset,
        }
    }
}

impl IndicesIntoIterator {
    pub(crate) fn new(subset: ArraySubset, range: std::ops::Range<usize>) -> Self {
        Self {
            odometer: IndicesOdometer::new(&subset, range),
            subset,
        }
    }
}

/// An iterator position over the indices of the elements in a range of an array subset.
///
/// Only the dimensions of the subset with more than one element are stepped through, with an [`Odometer`].
#[derive(Clone)]
struct IndicesOdometer {
    /// The dimensions of the subset with more than one element.
    dimensions: tinyvec::TinyVec<[SubsetDimension; 4]>,
    odometer: Odometer,
    /// The indices of the next element from the front, if any.
    front: ArrayIndicesTinyVec,
}

/// A dimension of an array subset with more than one element.
#[derive(Clone, Copy, Default)]
struct SubsetDimension {
    /// The dimension of the subset.
    axis: usize,
    /// The length of the subset along the dimension.
    count: u64,
}

impl GridDimension for SubsetDimension {
    fn count(&self) -> u64 {
        self.count
    }
}

impl IndicesOdometer {
    fn new(subset: &ArraySubset, range: std::ops::Range<usize>) -> Self {
        let dimensions: tinyvec::TinyVec<[SubsetDimension; 4]> =
            (subset.shape().iter().enumerate())
                .filter(|&(_, &count)| count > 1)
                .map(|(axis, &count)| SubsetDimension { axis, count })
                .collect();
        let odometer = Odometer::new(&dimensions, range);
        let mut front: ArrayIndicesTinyVec = subset.start().iter().copied().collect();
        for (dimension, &position) in std::iter::zip(&dimensions, odometer.positions()) {
            front[dimension.axis] += position;
        }
        Self {
            dimensions,
            odometer,
            front,
        }
    }

    #[inline]
    fn next(&mut self) -> Option<ArrayIndicesTinyVec> {
        let indices = self.front.clone();
        let front = &mut self.front;
        self.odometer
            .pop_front(&self.dimensions, |dimension, position| {
                let index = &mut front[dimension.axis];
                if position == 0 {
                    *index -= dimension.count - 1;
                } else {
                    *index += 1;
                }
            })
            .then_some(indices)
    }

    /// The next indices from the back of a subset with `start`.
    fn next_back(&mut self, start: &[u64]) -> Option<ArrayIndicesTinyVec> {
        let index = self.odometer.pop_back()?;
        let mut indices: ArrayIndicesTinyVec = start.iter().copied().collect();
        unravel(index, &self.dimensions, |dimension, position| {
            indices[dimension.axis] += position;
        });
        Some(indices)
    }

    fn len(&self) -> usize {
        self.odometer.len()
    }
}

macro_rules! impl_indices_iterator {
    ($iterator_type:ty) => {
        impl Iterator for $iterator_type {
            type Item = ArrayIndicesTinyVec;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.odometer.next()
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let length = self.odometer.len();
                (length, Some(length))
            }
        }

        impl DoubleEndedIterator for $iterator_type {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.odometer.next_back(self.subset.start())
            }
        }

        impl ExactSizeIterator for $iterator_type {}

        impl FusedIterator for $iterator_type {}
    };
}

impl_indices_iterator!(IndicesIterator<'_>);
impl_indices_iterator!(IndicesIntoIterator);

/// Parallel indices iterator.
///
/// See [`Indices`].
pub struct ParIndicesIterator<'a> {
    pub(crate) subset: &'a ArraySubset,
    pub(crate) range: std::ops::Range<usize>,
}

/// Parallel indices iterator.
///
/// See [`Indices`].
pub struct ParIndicesIntoIterator {
    pub(crate) subset: ArraySubset,
    pub(crate) range: std::ops::Range<usize>,
}

macro_rules! impl_par_chunks_iterator {
    ($iterator_type:ty) => {
        impl ParallelIterator for $iterator_type {
            type Item = ArrayIndicesTinyVec;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn opt_len(&self) -> Option<usize> {
                Some(self.len())
            }
        }

        impl IndexedParallelIterator for $iterator_type {
            fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
                callback.callback(self)
            }

            fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
                bridge(self, consumer)
            }

            fn len(&self) -> usize {
                self.range.end.saturating_sub(self.range.start)
            }
        }
    };
}

impl_par_chunks_iterator!(ParIndicesIterator<'_>);
impl_par_chunks_iterator!(ParIndicesIntoIterator);

impl<'a> Producer for ParIndicesIterator<'a> {
    type Item = ArrayIndicesTinyVec;
    type IntoIter = IndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIterator::new(self.subset, self.range)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left = ParIndicesIterator {
            subset: self.subset,
            range: self.range.start..self.range.start + index,
        };
        let right = ParIndicesIterator {
            subset: self.subset,
            range: (self.range.start + index)..self.range.end,
        };
        (left, right)
    }
}

impl Producer for ParIndicesIntoIterator {
    type Item = ArrayIndicesTinyVec;
    type IntoIter = IndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIntoIterator::new(self.subset, self.range)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left = ParIndicesIntoIterator {
            subset: self.subset.clone(),
            range: self.range.start..self.range.start + index,
        };
        let right = ParIndicesIntoIterator {
            subset: self.subset,
            range: (self.range.start + index)..self.range.end,
        };
        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArrayIndicesTinyVec;

    #[test]
    fn indices_iterator_partial() {
        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), 1..4);
        assert_eq!(indices.len(), 3);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(ArrayIndicesTinyVec::Heap(vec![1, 6])));
        assert_eq!(
            iter.next_back(),
            Some(ArrayIndicesTinyVec::Heap(vec![2, 6]))
        );
        assert_eq!(iter.next(), Some(ArrayIndicesTinyVec::Heap(vec![2, 5])));
        assert_eq!(iter.next(), None);

        assert_eq!(
            indices.into_par_iter().map(|v| v[0] + v[1]).sum::<u64>(),
            22
        );

        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), ..=0);
        assert_eq!(indices.len(), 1);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(ArrayIndicesTinyVec::Heap(vec![1, 5])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn indices_iterator_5d_overflowing_num_elements() {
        // The number of elements overflows a `u64`, but the indices of its leading elements do not
        let subset = ArraySubset::new_with_start_shape(vec![1; 5], vec![1 << 16; 5]).unwrap();
        let indices = Indices {
            subset,
            range: 0..3,
        };
        let mut iter = indices.iter();
        assert_eq!(
            iter.next(),
            Some(ArrayIndicesTinyVec::Heap(vec![1, 1, 1, 1, 1]))
        );
        assert_eq!(
            iter.next_back(),
            Some(ArrayIndicesTinyVec::Heap(vec![1, 1, 1, 1, 3]))
        );
        assert_eq!(
            iter.next(),
            Some(ArrayIndicesTinyVec::Heap(vec![1, 1, 1, 1, 2]))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn indices_iterator_max_index() {
        // Stepping past the last index along a dimension must not overflow
        let subset = ArraySubset::new_with_start_shape(vec![0, u64::MAX], vec![2, 1]).unwrap();
        let expected = vec![
            ArrayIndicesTinyVec::Heap(vec![0, u64::MAX]),
            ArrayIndicesTinyVec::Heap(vec![1, u64::MAX]),
        ];
        assert_eq!(subset.indices().iter().collect::<Vec<_>>(), expected);
        let subset = ArraySubset::new_with_start_shape(vec![u64::MAX - 1, 0], vec![2, 2]).unwrap();
        let expected = vec![
            ArrayIndicesTinyVec::Heap(vec![u64::MAX - 1, 0]),
            ArrayIndicesTinyVec::Heap(vec![u64::MAX - 1, 1]),
            ArrayIndicesTinyVec::Heap(vec![u64::MAX, 0]),
            ArrayIndicesTinyVec::Heap(vec![u64::MAX, 1]),
        ];
        assert_eq!(subset.indices().iter().collect::<Vec<_>>(), expected);
    }

    #[allow(clippy::reversed_empty_ranges)]
    #[test]
    fn indices_iterator_empty() {
        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), 5..5);
        assert_eq!(indices.len(), 0);
        assert!(indices.is_empty());

        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), 5..1);
        assert_eq!(indices.len(), 0);
        assert!(indices.is_empty());
    }
}

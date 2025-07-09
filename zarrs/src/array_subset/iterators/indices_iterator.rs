use std::iter::FusedIterator;

use crate::{
    array::{unravel_index, ArrayIndices},
    array_subset::ArraySubset,
};

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};

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
pub struct Indices {
    subset: ArraySubset,
    range: std::ops::Range<usize>,
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
    type Item = ArrayIndices;
    type IntoIter = IndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIterator {
            subset: &self.subset,
            range: self.range.clone(),
        }
    }
}

impl<'a> IntoParallelIterator for &'a Indices {
    type Item = ArrayIndices;
    type Iter = ParIndicesIterator<'a>;

    fn into_par_iter(self) -> Self::Iter {
        ParIndicesIterator {
            subset: &self.subset,
            range: self.range.clone(),
        }
    }
}

/// Serial indices iterator.
///
/// See [`Indices`].
pub struct IndicesIterator<'a> {
    subset: &'a ArraySubset,
    range: std::ops::Range<usize>,
}

impl<'a> IndicesIterator<'a> {
    /// Create a new indices iterator.
    #[must_use]
    pub(super) fn new(subset: &'a ArraySubset) -> Self {
        let length = subset.num_elements_usize();
        Self {
            subset,
            range: 0..length,
        }
    }

    /// Create a new indices iterator spanning an explicit index range.
    #[must_use]
    pub(super) fn new_with_start_end(
        subset: &'a ArraySubset,
        range: impl Into<std::ops::Range<usize>>,
    ) -> Self {
        Self {
            subset,
            range: range.into(),
        }
    }
}

impl Iterator for IndicesIterator<'_> {
    type Item = ArrayIndices;

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.start < self.range.end && self.range.start < self.subset.num_elements_usize() {
            let indices = self.subset.find_linearised_index(self.range.start);
            self.range.start += 1;
            Some(indices)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let length = self.range.end.saturating_sub(self.range.start);
        (length, Some(length))
    }
}

impl DoubleEndedIterator for IndicesIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.range.end > self.range.start {
            self.range.end -= 1;
            let indices = self.subset.find_linearised_index(self.range.end);
            Some(indices)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for IndicesIterator<'_> {}

impl FusedIterator for IndicesIterator<'_> {}

/// Parallel indices iterator.
///
/// See [`Indices`].
pub struct ParIndicesIterator<'a> {
    subset: &'a ArraySubset,
    range: std::ops::Range<usize>,
}

impl ParallelIterator for ParIndicesIterator<'_> {
    type Item = ArrayIndices;

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

impl IndexedParallelIterator for ParIndicesIterator<'_> {
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = ParIndicesIteratorProducer::from(&self);
        callback.callback(producer)
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.range.end.saturating_sub(self.range.start)
    }
}

#[derive(Debug)]
pub(super) struct ParIndicesIteratorProducer<'a> {
    pub(super) subset: &'a ArraySubset,
    pub(super) range: std::ops::Range<usize>,
}

impl<'a> Producer for ParIndicesIteratorProducer<'a> {
    type Item = ArrayIndices;
    type IntoIter = IndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIterator::new_with_start_end(self.subset, self.range)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left = ParIndicesIteratorProducer {
            subset: self.subset,
            range: self.range.start..self.range.start + index,
        };
        let right = ParIndicesIteratorProducer {
            subset: self.subset,
            range: (self.range.start + index)..self.range.end,
        };
        (left, right)
    }
}

impl<'a> From<&'a ParIndicesIterator<'_>> for ParIndicesIteratorProducer<'a> {
    fn from(iterator: &'a ParIndicesIterator<'_>) -> Self {
        Self {
            subset: iterator.subset,
            range: iterator.range.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_iterator_partial() {
        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), 1..4);
        assert_eq!(indices.len(), 3);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(vec![1, 6]));
        assert_eq!(iter.next_back(), Some(vec![2, 6]));
        assert_eq!(iter.next(), Some(vec![2, 5]));
        assert_eq!(iter.next(), None);

        assert_eq!(
            indices.into_par_iter().map(|v| v[0] + v[1]).sum::<u64>(),
            22
        );

        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), ..=0);
        assert_eq!(indices.len(), 1);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(vec![1, 5]));
        assert_eq!(iter.next(), None);
    }

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

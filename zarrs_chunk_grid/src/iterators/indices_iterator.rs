use std::iter::FusedIterator;

use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{ArrayIndices, ArraySubset, unravel_index};

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
    type Item = ArrayIndices;
    type IntoIter = IndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIterator::new(&self.subset, self.range.clone())
    }
}

impl<'a> IntoParallelRefIterator<'a> for &'a Indices {
    type Item = ArrayIndices;
    type Iter = ParIndicesIterator<'a>;

    fn par_iter(&self) -> Self::Iter {
        ParIndicesIterator {
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

impl IntoIterator for Indices {
    type Item = ArrayIndices;
    type IntoIter = IndicesIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        IndicesIntoIterator::new(self.subset, self.range)
    }
}

impl IntoParallelIterator for Indices {
    type Item = ArrayIndices;
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
    range: std::ops::Range<usize>,
    /// The indices of the next element from the front, if any.
    front: ArrayIndices,
}

/// Serial indices iterator.
///
/// See [`Indices`].
#[derive(Clone)]
pub struct IndicesIntoIterator {
    subset: ArraySubset,
    range: std::ops::Range<usize>,
    /// The indices of the next element from the front, if any.
    front: ArrayIndices,
}

impl<'a> IndicesIterator<'a> {
    pub(crate) fn new(subset: &'a ArraySubset, range: std::ops::Range<usize>) -> Self {
        let front = indices_at(subset, &range);
        Self {
            subset,
            range,
            front,
        }
    }
}

impl IndicesIntoIterator {
    pub(crate) fn new(subset: ArraySubset, range: std::ops::Range<usize>) -> Self {
        let front = indices_at(&subset, &range);
        Self {
            subset,
            range,
            front,
        }
    }
}

/// Compute indices from a linear index for dimensionality 1, adding subset start offset.
#[inline]
fn unravel_index_1d(index: u64, shape: &[u64], start: &[u64]) -> ArrayIndices {
    debug_assert_eq!(shape.len(), 1);
    debug_assert_eq!(start.len(), 1);
    tinyvec::tiny_vec!([u64; 4] => start[0] + (index % shape[0]))
}

/// Compute indices from a linear index for dimensionality 2, adding subset start offset.
#[inline]
fn unravel_index_2d(mut index: u64, shape: &[u64], start: &[u64]) -> ArrayIndices {
    debug_assert_eq!(shape.len(), 2);
    debug_assert_eq!(start.len(), 2);
    let i1 = start[1] + (index % shape[1]);
    index /= shape[1];
    let i0 = start[0] + (index % shape[0]);
    tinyvec::tiny_vec!([u64; 4] => i0, i1)
}

/// Compute indices from a linear index for dimensionality 3, adding subset start offset.
#[inline]
fn unravel_index_3d(mut index: u64, shape: &[u64], start: &[u64]) -> ArrayIndices {
    debug_assert_eq!(shape.len(), 3);
    debug_assert_eq!(start.len(), 3);
    let i2 = start[2] + (index % shape[2]);
    index /= shape[2];
    let i1 = start[1] + (index % shape[1]);
    index /= shape[1];
    let i0 = start[0] + (index % shape[0]);
    tinyvec::tiny_vec!([u64; 4] => i0, i1, i2)
}

/// Compute indices from a linear index for dimensionality 4, adding subset start offset.
#[inline]
fn unravel_index_4d(mut index: u64, shape: &[u64], start: &[u64]) -> ArrayIndices {
    debug_assert_eq!(shape.len(), 4);
    debug_assert_eq!(start.len(), 4);
    let i3 = start[3] + (index % shape[3]);
    index /= shape[3];
    let i2 = start[2] + (index % shape[2]);
    index /= shape[2];
    let i1 = start[1] + (index % shape[1]);
    index /= shape[1];
    let i0 = start[0] + (index % shape[0]);
    tinyvec::tiny_vec!([u64; 4] => i0, i1, i2, i3)
}

/// Compute indices from a linear index for dimensionality 5+, adding subset start offset.
#[inline]
fn unravel_index_nd(index: u64, shape: &[u64], start: &[u64]) -> Option<ArrayIndices> {
    let mut indices = unravel_index(index, shape)?;
    std::iter::zip(indices.iter_mut(), start).for_each(|(idx, st)| *idx += st);
    Some(indices)
}

/// The indices of the element at `index` in `subset`.
#[inline]
fn unravel_subset_index(subset: &ArraySubset, index: usize) -> Option<ArrayIndices> {
    let index = index as u64;
    let shape = subset.shape();
    let start = subset.start();
    match shape.len() {
        0 => Some(ArrayIndices::new()),
        1 => Some(unravel_index_1d(index, shape, start)),
        2 => Some(unravel_index_2d(index, shape, start)),
        3 => Some(unravel_index_3d(index, shape, start)),
        4 => Some(unravel_index_4d(index, shape, start)),
        _ => unravel_index_nd(index, shape, start),
    }
}

/// The indices of the first element of `range` in `subset`, or no indices if `range` is empty.
fn indices_at(subset: &ArraySubset, range: &std::ops::Range<usize>) -> ArrayIndices {
    if range.is_empty() {
        ArrayIndices::new()
    } else {
        unravel_subset_index(subset, range.start).unwrap_or_default()
    }
}

macro_rules! impl_indices_iterator {
    ($iterator_type:ty) => {
        impl Iterator for $iterator_type {
            type Item = ArrayIndices;

            /// Steps the indices of the front element like an odometer, by incrementing the innermost index and
            /// carrying into outer dimensions, so that no index is unravelled.
            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if self.range.is_empty() {
                    return None;
                }
                let indices = self.front.clone();
                self.range.start += 1;
                if !self.range.is_empty() {
                    let subset_start = self.subset.start();
                    let subset_shape = self.subset.shape();
                    for ((index, &start), &size) in std::iter::zip(&mut self.front, subset_start)
                        .zip(subset_shape)
                        .rev()
                    {
                        *index += 1;
                        if *index < start + size {
                            break;
                        }
                        *index = start;
                    }
                }
                Some(indices)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let length = self.range.end.saturating_sub(self.range.start);
                (length, Some(length))
            }
        }

        impl DoubleEndedIterator for $iterator_type {
            fn next_back(&mut self) -> Option<Self::Item> {
                let index = self.range.next_back()?;
                unravel_subset_index(&self.subset, index)
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
    type Item = ArrayIndices;
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
    type Item = ArrayIndices;
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
    use crate::ArrayIndices;

    #[test]
    fn indices_iterator_partial() {
        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), 1..4);
        assert_eq!(indices.len(), 3);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(ArrayIndices::Heap(vec![1, 6])));
        assert_eq!(iter.next_back(), Some(ArrayIndices::Heap(vec![2, 6])));
        assert_eq!(iter.next(), Some(ArrayIndices::Heap(vec![2, 5])));
        assert_eq!(iter.next(), None);

        assert_eq!(
            indices.into_par_iter().map(|v| v[0] + v[1]).sum::<u64>(),
            22
        );

        let indices =
            Indices::new_with_start_end(ArraySubset::new_with_ranges(&[1..3, 5..7]), ..=0);
        assert_eq!(indices.len(), 1);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(ArrayIndices::Heap(vec![1, 5])));
        assert_eq!(iter.next(), None);
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

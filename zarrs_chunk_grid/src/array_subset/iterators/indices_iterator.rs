use std::iter::FusedIterator;

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{array_subset::ArraySubset, unravel_index, ArrayIndicesTinyVec};

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
        IndicesIterator {
            subset: &self.subset,
            range: self.range.clone(),
        }
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
        IndicesIntoIterator {
            subset: self.subset,
            range: self.range,
        }
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
    pub(crate) subset: &'a ArraySubset,
    pub(crate) range: std::ops::Range<usize>,
}

/// Serial indices iterator.
///
/// See [`Indices`].
#[derive(Clone)]
pub struct IndicesIntoIterator {
    pub(crate) subset: ArraySubset,
    pub(crate) range: std::ops::Range<usize>,
}

macro_rules! impl_indices_iterator {
    ($iterator_type:ty) => {
        impl Iterator for $iterator_type {
            type Item = ArrayIndicesTinyVec;

            fn next(&mut self) -> Option<Self::Item> {
                if self.range.start >= self.range.end {
                    return None;
                }
                let mut indices = unravel_index(self.range.start as u64, self.subset.shape())?;
                std::iter::zip(indices.iter_mut(), self.subset.start())
                    .for_each(|(index, start)| *index += start);

                if self.range.start < self.range.end {
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

        impl DoubleEndedIterator for $iterator_type {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.range.end > self.range.start {
                    self.range.end -= 1;
                    let mut indices = unravel_index(self.range.end as u64, self.subset.shape())?;
                    std::iter::zip(indices.iter_mut(), self.subset.start())
                        .for_each(|(index, start)| *index += start);
                    Some(indices)
                } else {
                    None
                }
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
        IndicesIterator {
            subset: self.subset,
            range: self.range,
        }
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
        IndicesIntoIterator {
            subset: self.subset,
            range: self.range,
        }
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

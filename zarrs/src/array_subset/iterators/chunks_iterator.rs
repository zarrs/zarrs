use std::{iter::FusedIterator, num::NonZeroU64};

use itertools::izip;
use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use zarrs_metadata::ArrayShape;

use crate::{
    array::{chunk_shape_to_array_shape, ArrayIndices},
    array_subset::{
        iterators::indices_iterator::{IndicesIntoIterator, ParIndicesIntoIterator},
        ArraySubset, IncompatibleDimensionalityError,
    },
};

use super::{Indices, IndicesIterator, ParIndicesIterator};

/// Iterates over the regular sized chunks overlapping this array subset.
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// All chunks have the same size, and may extend over the bounds of the array subset since the start of the first chunk is aligned to the chunk size.
///
/// The iterator item is a ([`ArrayIndices`], [`ArraySubset`]) tuple corresponding to the chunk indices and array subset.
///
/// For example, consider a 4x3 array with element indices
/// ```text
/// (0, 0)  (0, 1)  (0, 2)
/// (1, 0)  (1, 1)  (1, 2)
/// (2, 0)  (2, 1)  (2, 2)
/// (3, 0)  (3, 1)  (3, 2)
/// ```
/// An 2x2 chunks iterator with an array subset covering the entire array will produce
/// ```rust,ignore
/// [
///     ((0, 0), ArraySubset{offset: (0,0), shape: (2, 2)}),
///     ((0, 1), ArraySubset{offset: (0,2), shape: (2, 2)}),
///     ((1, 0), ArraySubset{offset: (2,0), shape: (2, 2)}),
///     ((1, 1), ArraySubset{offset: (2,2), shape: (2, 2)}),
/// ]
/// ```
///
pub struct Chunks {
    pub(crate) indices: Indices,
    pub(crate) chunk_shape: Vec<u64>,
}

impl Chunks {
    /// Create a new chunks iterator.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_shape` does not match the dimensionality of `subset`.
    pub fn new(
        subset: &ArraySubset,
        chunk_shape: &[NonZeroU64],
    ) -> Result<Self, IncompatibleDimensionalityError> {
        if subset.dimensionality() == chunk_shape.len() {
            let chunk_shape = chunk_shape_to_array_shape(chunk_shape);
            Ok(match subset.end_inc() {
                Some(end) => {
                    let chunk_start: ArrayIndices = std::iter::zip(subset.start(), &chunk_shape)
                        .map(|(s, c)| s / c)
                        .collect();
                    let shape: ArrayIndices = izip!(end, &chunk_shape, &chunk_start)
                        .map(|(e, &c, &s)| (e / c).saturating_sub(s) + 1)
                        .collect();
                    let subset_chunks = ArraySubset::new_with_start_shape(chunk_start, shape)?;
                    Self {
                        indices: subset_chunks.indices(),
                        chunk_shape,
                    }
                }
                None => Self {
                    indices: ArraySubset::new_empty(subset.dimensionality()).indices(),
                    chunk_shape,
                },
            })
        } else {
            Err(IncompatibleDimensionalityError(
                chunk_shape.len(),
                subset.dimensionality(),
            ))
        }
    }

    /// Return the number of chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if the number of chunks is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a new serial iterator.
    #[must_use]
    pub fn iter(&self) -> ChunksIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a> IntoIterator for &'a Chunks {
    type Item = (ArrayIndices, ArraySubset);
    type IntoIter = ChunksIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ChunksIterator {
            inner: self.indices.iter(),
            chunk_shape: &self.chunk_shape,
        }
    }
}

impl<'a> IntoParallelRefIterator<'a> for &'a Chunks {
    type Item = (ArrayIndices, ArraySubset);
    type Iter = ParChunksIterator<'a>;

    fn par_iter(&self) -> Self::Iter {
        ParChunksIterator {
            inner: ParIndicesIterator {
                subset: &self.indices.subset,
                range: self.indices.range.clone(),
            },
            chunk_shape: &self.chunk_shape,
        }
    }
}

impl<'a> IntoParallelIterator for &'a Chunks {
    type Item = (ArrayIndices, ArraySubset);
    type Iter = ParChunksIterator<'a>;

    fn into_par_iter(self) -> Self::Iter {
        ParChunksIterator {
            inner: ParIndicesIterator {
                subset: &self.indices.subset,
                range: self.indices.range.clone(),
            },
            chunk_shape: &self.chunk_shape,
        }
    }
}

impl IntoIterator for Chunks {
    type Item = (ArrayIndices, ArraySubset);
    type IntoIter = ChunksIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        ChunksIntoIterator {
            inner: self.indices.into_iter(),
            chunk_shape: self.chunk_shape,
        }
    }
}

impl IntoParallelIterator for Chunks {
    type Item = (ArrayIndices, ArraySubset);
    type Iter = ParChunksIntoIterator;

    fn into_par_iter(self) -> Self::Iter {
        ParChunksIntoIterator {
            inner: self.indices.into_par_iter(),
            chunk_shape: self.chunk_shape,
        }
    }
}

/// Serial chunks iterator.
///
/// See [`Chunks`].
pub struct ChunksIterator<'a> {
    inner: IndicesIterator<'a>,
    chunk_shape: &'a [u64],
}

impl ChunksIterator<'_> {
    fn chunk_indices_with_subset(&self, chunk_indices: Vec<u64>) -> (Vec<u64>, ArraySubset) {
        let ranges =
            std::iter::zip(&chunk_indices, self.chunk_shape).map(|(i, c)| ((i * c)..(i * c) + c));
        let chunk_subset = ArraySubset::from(ranges);
        (chunk_indices, chunk_subset)
    }
}

impl Iterator for ChunksIterator<'_> {
    type Item = (ArrayIndices, ArraySubset);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|chunk_indices| self.chunk_indices_with_subset(chunk_indices))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for ChunksIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner
            .next_back()
            .map(|chunk_indices| self.chunk_indices_with_subset(chunk_indices))
    }
}

impl ExactSizeIterator for ChunksIterator<'_> {}

impl FusedIterator for ChunksIterator<'_> {}

/// Serial chunks iterator.
///
/// See [`Chunks`].
pub struct ChunksIntoIterator {
    inner: IndicesIntoIterator,
    chunk_shape: ArrayShape,
}

impl ChunksIntoIterator {
    fn chunk_indices_with_subset(&self, chunk_indices: Vec<u64>) -> (Vec<u64>, ArraySubset) {
        let ranges =
            std::iter::zip(&chunk_indices, &self.chunk_shape).map(|(i, c)| ((i * c)..(i * c) + c));
        let chunk_subset = ArraySubset::from(ranges);
        (chunk_indices, chunk_subset)
    }
}

impl Iterator for ChunksIntoIterator {
    type Item = (ArrayIndices, ArraySubset);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|chunk_indices| self.chunk_indices_with_subset(chunk_indices))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for ChunksIntoIterator {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner
            .next_back()
            .map(|chunk_indices| self.chunk_indices_with_subset(chunk_indices))
    }
}

impl ExactSizeIterator for ChunksIntoIterator {}

impl FusedIterator for ChunksIntoIterator {}

/// Parallel chunks iterator.
///
/// See [`Chunks`].
pub struct ParChunksIterator<'a> {
    inner: ParIndicesIterator<'a>,
    chunk_shape: &'a [u64],
}

impl ParallelIterator for ParChunksIterator<'_> {
    type Item = (Vec<u64>, ArraySubset);

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

impl IndexedParallelIterator for ParChunksIterator<'_> {
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl Producer for ParChunksIterator<'_> {
    type Item = (Vec<u64>, ArraySubset);
    type IntoIter = ChunksIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        ChunksIntoIterator {
            inner: IndicesIntoIterator {
                subset: self.inner.subset.clone(),
                range: self.inner.range,
            },
            chunk_shape: self.chunk_shape.to_vec(),
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.inner.split_at(index);
        (
            ParChunksIterator {
                inner: left,
                chunk_shape: self.chunk_shape,
            },
            ParChunksIterator {
                inner: right,
                chunk_shape: self.chunk_shape,
            },
        )
    }
}

/// Parallel chunks iterator.
///
/// See [`Chunks`].
pub struct ParChunksIntoIterator {
    pub(crate) inner: ParIndicesIntoIterator,
    pub(crate) chunk_shape: ArrayShape,
}

impl ParallelIterator for ParChunksIntoIterator {
    type Item = (Vec<u64>, ArraySubset);

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

impl IndexedParallelIterator for ParChunksIntoIterator {
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl Producer for ParChunksIntoIterator {
    type Item = (Vec<u64>, ArraySubset);
    type IntoIter = ChunksIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        ChunksIntoIterator {
            inner: IndicesIntoIterator {
                subset: self.inner.subset.clone(),
                range: self.inner.range,
            },
            chunk_shape: self.chunk_shape.clone(),
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.inner.split_at(index);
        (
            ParChunksIntoIterator {
                inner: left,
                chunk_shape: self.chunk_shape.clone(),
            },
            ParChunksIntoIterator {
                inner: right,
                chunk_shape: self.chunk_shape,
            },
        )
    }
}

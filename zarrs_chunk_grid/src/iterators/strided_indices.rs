use crate::ArrayIndicesTinyVec;

/// The linearised indices of a regular grid of positions in an array, in C order.
///
/// Only the dimensions with more than one position are stored, so iterating steps through no more dimensions than
/// necessary.
#[derive(Clone)]
pub(crate) struct StridedIndices {
    /// The number of positions.
    len: usize,
    /// The linearised index of the first position.
    first: u64,
    /// The number of positions along each dimension in which there is more than one, outermost first.
    counts: ArrayIndicesTinyVec,
    /// The distance between the linearised indices of adjacent positions along each dimension of `counts`.
    strides: ArrayIndicesTinyVec,
}

impl StridedIndices {
    /// The linearised indices of `counts` positions from `starts` along each dimension of an array with
    /// `array_shape`, where adjacent positions along a dimension are adjacent elements of the array.
    ///
    /// The positions must be within the array if there are any.
    ///
    /// # Panics
    /// Panics if the number of positions exceeds [`usize::MAX`].
    pub(crate) fn new(starts: &[u64], counts: &[u64], array_shape: &[u64]) -> Self {
        debug_assert_eq!(starts.len(), array_shape.len());
        debug_assert_eq!(counts.len(), array_shape.len());
        let len = usize::try_from(counts.iter().product::<u64>()).unwrap();
        let mut first = 0;
        let mut stride_counts = ArrayIndicesTinyVec::new();
        let mut strides = ArrayIndicesTinyVec::new();
        if len > 0 {
            let mut stride = 1;
            for ((&start, &count), &array_size) in
                std::iter::zip(starts, counts).zip(array_shape).rev()
            {
                first += start * stride;
                if count > 1 {
                    stride_counts.push(count);
                    strides.push(stride);
                }
                stride *= array_size;
            }
            stride_counts.reverse();
            strides.reverse();
        }
        Self {
            len,
            first,
            counts: stride_counts,
            strides,
        }
    }

    /// Return the number of positions.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The linearised index of the position at `index`.
    fn at(&self, mut index: u64) -> u64 {
        let mut linearised = self.first;
        for (&count, &stride) in std::iter::zip(&self.counts, &self.strides).rev() {
            linearised += (index % count) * stride;
            index /= count;
        }
        linearised
    }
}

/// An iterator position over [`StridedIndices`].
///
/// Positions are iterated forwards like an odometer, by adding the stride of the innermost dimension and carrying into
/// outer dimensions, so that no indices are unravelled or ravelled.
#[derive(Clone)]
pub(crate) struct StridedOdometer {
    /// The position of the next index from the front along each dimension of the counts.
    positions: ArrayIndicesTinyVec,
    /// The next linearised index from the front.
    front: u64,
    /// The positions of the remaining indices.
    range: std::ops::Range<usize>,
}

impl StridedOdometer {
    pub(crate) fn new(indices: &StridedIndices) -> Self {
        Self {
            positions: std::iter::repeat_n(0, indices.counts.len()).collect(),
            front: indices.first,
            range: 0..indices.len,
        }
    }

    #[inline]
    pub(crate) fn next(&mut self, indices: &StridedIndices) -> Option<u64> {
        if self.range.is_empty() {
            return None;
        }
        let linearised = self.front;
        self.range.start += 1;
        if !self.range.is_empty() {
            for ((position, &count), &stride) in
                std::iter::zip(&mut self.positions, &indices.counts)
                    .zip(&indices.strides)
                    .rev()
            {
                *position += 1;
                self.front += stride;
                if *position < count {
                    break;
                }
                *position = 0;
                self.front -= count * stride;
            }
        }
        Some(linearised)
    }

    pub(crate) fn next_back(&mut self, indices: &StridedIndices) -> Option<u64> {
        let index = self.range.next_back()?;
        Some(indices.at(index as u64))
    }

    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

use tinyvec::TinyVec;

use super::odometer::{GridDimension, Odometer, unravel};

/// The linearised indices of a regular grid of positions in an array, in C order.
///
/// Only the dimensions with more than one position are stored, and adjacent dimensions whose
/// strides are contiguous (`outer.stride == inner.count * inner.stride`) are coalesced, so
/// iterating steps through no more dimensions than necessary and the representation is canonical.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StridedIndices {
    /// The number of positions.
    num_elements: u64,
    /// The linearised index of the first position.
    first: u64,
    /// The dimensions in which there is more than one position, outermost first.
    dimensions: TinyVec<[StridedDimension; 4]>,
}

/// A dimension of [`StridedIndices`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StridedDimension {
    /// The number of positions.
    count: u64,
    /// The distance between the linearised indices of adjacent positions.
    stride: u64,
}

impl GridDimension for StridedDimension {
    fn count(&self) -> u64 {
        self.count
    }
}

impl StridedIndices {
    /// The linearised indices of `counts` positions from `starts` along each dimension of an array with
    /// `array_shape`, where adjacent positions along a dimension are adjacent elements of the array.
    ///
    /// The positions must be within the array if there are any.
    ///
    /// # Panics
    /// Panics if the linearised index of a position or the number of positions exceeds [`u64::MAX`].
    pub(crate) fn new(starts: &[u64], counts: &[u64], array_shape: &[u64]) -> Self {
        debug_assert_eq!(starts.len(), array_shape.len());
        debug_assert_eq!(counts.len(), array_shape.len());
        let mut num_elements = 0;
        let mut first = 0;
        let mut dimensions = TinyVec::<[StridedDimension; 4]>::new();
        if !counts.contains(&0) {
            // The linearised indices fit in a `u64` if the last does, since it is the greatest
            let mut last: u64 = 0;
            // The stride is only needed along dimensions with a nonzero index, so it may overflow beyond them
            let mut stride = Some(1u64);
            for ((&start, &count), &array_size) in
                std::iter::zip(starts, counts).zip(array_shape).rev()
            {
                let end = start + (count - 1);
                if end > 0 {
                    let stride = stride.expect("linearised indices exceed u64::MAX");
                    last = end
                        .checked_mul(stride)
                        .and_then(|offset| last.checked_add(offset))
                        .expect("linearised indices exceed u64::MAX");
                    first += start * stride;
                    if count > 1 {
                        if let Some(inner) = dimensions
                            .last_mut()
                            .filter(|inner| inner.count.checked_mul(inner.stride) == Some(stride))
                        {
                            inner.count = inner
                                .count
                                .checked_mul(count)
                                .expect("the number of positions exceeds u64::MAX");
                        } else {
                            dimensions.push(StridedDimension { count, stride });
                        }
                    }
                }
                stride = stride.and_then(|stride| stride.checked_mul(array_size));
            }
            dimensions.reverse();
            num_elements = dimensions
                .iter()
                .try_fold(1u64, |num_elements, dimension| {
                    num_elements.checked_mul(dimension.count)
                })
                .expect("the number of positions exceeds u64::MAX");
        }
        Self {
            num_elements,
            first,
            dimensions,
        }
    }

    /// No positions.
    pub(crate) fn empty() -> Self {
        Self {
            num_elements: 0,
            first: 0,
            dimensions: TinyVec::new(),
        }
    }

    /// The linearised indices of a grid of positions from `first`, with the `(count, stride)` of each dimension given
    /// innermost first.
    ///
    /// Dimensions with one position are dropped and adjacent contiguous dimensions are coalesced, as in
    /// [`new`](Self::new).
    ///
    /// # Panics
    /// Panics if the linearised index of a position or the number of positions exceeds [`u64::MAX`].
    pub(crate) fn from_dimensions_rev(
        first: u64,
        dimensions_rev: impl Iterator<Item = (u64, u64)>,
    ) -> Self {
        let mut num_elements = Some(1u64);
        let mut last = Some(first);
        let mut dimensions = TinyVec::<[StridedDimension; 4]>::new();
        for (count, stride) in dimensions_rev {
            if count == 0 {
                return Self::empty();
            }
            if count == 1 {
                continue;
            }
            num_elements = num_elements.and_then(|n| n.checked_mul(count));
            last = (count - 1)
                .checked_mul(stride)
                .and_then(|span| last?.checked_add(span));
            if let Some(inner) = dimensions
                .last_mut()
                .filter(|inner| inner.count.checked_mul(inner.stride) == Some(stride))
            {
                if let Some(coalesced) = inner.count.checked_mul(count) {
                    inner.count = coalesced;
                }
            } else {
                dimensions.push(StridedDimension { count, stride });
            }
        }
        let num_elements = num_elements.expect("the number of positions exceeds u64::MAX");
        last.expect("linearised indices exceed u64::MAX");
        dimensions.reverse();
        Self {
            num_elements,
            first,
            dimensions,
        }
    }

    /// Split off the innermost dimension if its stride is 1, returning the number of contiguous
    /// elements per remaining position (or 0 if there are no positions).
    pub(crate) fn take_contiguous_inner(&mut self) -> u64 {
        if self.num_elements == 0 {
            return 0;
        }
        if self
            .dimensions
            .last()
            .is_some_and(|inner| inner.stride == 1)
        {
            let inner = self.dimensions.pop().unwrap();
            self.num_elements /= inner.count;
            inner.count
        } else {
            1
        }
    }

    /// Return the number of positions.
    pub(crate) fn num_elements(&self) -> u64 {
        self.num_elements
    }

    /// Return the number of positions.
    ///
    /// # Panics
    /// Panics if the number of positions exceeds [`usize::MAX`].
    pub(crate) fn len(&self) -> usize {
        usize::try_from(self.num_elements).unwrap()
    }

    /// The linearised index of the position at `index`.
    fn at(&self, index: u64) -> u64 {
        let mut linearised = self.first;
        unravel(index, &self.dimensions, |dimension, position| {
            linearised += position * dimension.stride;
        });
        linearised
    }
}

/// An iterator position over [`StridedIndices`].
///
/// Positions are iterated forwards with an [`Odometer`], adding the stride of each dimension that steps rather than
/// ravelling indices.
#[derive(Clone)]
pub(crate) struct StridedOdometer {
    odometer: Odometer,
    /// The next linearised index from the front.
    front: u64,
}

impl StridedOdometer {
    /// # Panics
    /// Panics if the number of positions exceeds [`usize::MAX`].
    pub(crate) fn new(indices: &StridedIndices) -> Self {
        Self {
            odometer: Odometer::new(&indices.dimensions, 0..indices.len()),
            front: indices.first,
        }
    }

    #[inline]
    pub(crate) fn next(&mut self, indices: &StridedIndices) -> Option<u64> {
        let linearised = self.front;
        let front = &mut self.front;
        self.odometer
            .pop_front(&indices.dimensions, |dimension, position| {
                if position == 0 {
                    *front -= (dimension.count - 1) * dimension.stride;
                } else {
                    *front += dimension.stride;
                }
            })
            .then_some(linearised)
    }

    /// Remove the front `n` positions (or all remaining positions, if fewer).
    #[inline]
    pub(crate) fn skip(&mut self, indices: &StridedIndices, n: usize) {
        let front = &mut self.front;
        self.odometer
            .skip(&indices.dimensions, n, |dimension, from, to| {
                // Wrapping, since the position may step backwards along this dimension and forwards along an outer one
                *front = front
                    .wrapping_add(to * dimension.stride)
                    .wrapping_sub(from * dimension.stride);
            });
    }

    pub(crate) fn next_back(&mut self, indices: &StridedIndices) -> Option<u64> {
        let index = self.odometer.pop_back()?;
        Some(indices.at(index))
    }

    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        let length = self.odometer.len();
        (length, Some(length))
    }
}

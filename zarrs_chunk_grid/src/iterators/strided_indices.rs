use tinyvec::TinyVec;

use super::odometer::{GridDimension, Odometer, unravel};

/// The linearised indices of a regular grid of positions in an array, in C order.
///
/// Only the dimensions with more than one position are stored, so iterating steps through no more dimensions than
/// necessary.
#[derive(Clone)]
pub(crate) struct StridedIndices {
    /// The number of positions.
    num_elements: u64,
    /// The linearised index of the first position.
    first: u64,
    /// The dimensions in which there is more than one position, outermost first.
    dimensions: TinyVec<[StridedDimension; 4]>,
}

/// A dimension of [`StridedIndices`].
#[derive(Clone, Copy, Default)]
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
    pub(crate) fn new(starts: &[u64], counts: &[u64], array_shape: &[u64]) -> Self {
        debug_assert_eq!(starts.len(), array_shape.len());
        debug_assert_eq!(counts.len(), array_shape.len());
        let num_elements = counts.iter().product::<u64>();
        let mut first = 0;
        let mut dimensions = TinyVec::new();
        if num_elements > 0 {
            let mut stride = 1;
            for ((&start, &count), &array_size) in
                std::iter::zip(starts, counts).zip(array_shape).rev()
            {
                first += start * stride;
                if count > 1 {
                    dimensions.push(StridedDimension { count, stride });
                }
                stride *= array_size;
            }
            dimensions.reverse();
        }
        Self {
            num_elements,
            first,
            dimensions,
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

    pub(crate) fn next_back(&mut self, indices: &StridedIndices) -> Option<u64> {
        let index = self.odometer.pop_back()?;
        Some(indices.at(index))
    }

    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        let length = self.odometer.len();
        (length, Some(length))
    }
}

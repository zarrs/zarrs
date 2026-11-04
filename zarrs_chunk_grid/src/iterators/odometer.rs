use std::ops::Range;

use crate::ArrayIndices;

/// A dimension of a grid of positions.
pub(crate) trait GridDimension {
    /// The number of positions along the dimension.
    fn count(&self) -> u64;
}

/// Calls `f(dimension, position)` with the position along each of the `dimensions` of a grid of the element at
/// `index`, in C order, innermost dimension first.
///
/// `index` must be less than the number of elements in the grid, which need not fit in a [`u64`].
#[inline]
pub(crate) fn unravel<D: GridDimension>(
    mut index: u64,
    dimensions: &[D],
    mut f: impl FnMut(&D, u64),
) {
    for dimension in dimensions.iter().rev() {
        let count = dimension.count();
        f(dimension, index % count);
        index /= count;
    }
}

/// The positions of the remaining elements of a grid, in C order.
///
/// The front element steps to the next like an odometer, by incrementing its position along the innermost dimension
/// and carrying into outer dimensions, so that no index is unravelled.
#[derive(Clone)]
pub(crate) struct Odometer {
    /// The position of the front element along each dimension, if there are any elements.
    positions: ArrayIndices,
    /// The indices of the remaining elements.
    range: Range<usize>,
}

impl Odometer {
    /// An odometer over the elements in `range` of a grid with `dimensions`.
    pub(crate) fn new(dimensions: &[impl GridDimension], range: Range<usize>) -> Self {
        let mut positions = crate::zero_indices(dimensions.len());
        if !range.is_empty() {
            let mut positions_rev = positions.iter_mut().rev();
            unravel(range.start as u64, dimensions, |_, position| {
                *positions_rev.next().unwrap() = position;
            });
        }
        Self { positions, range }
    }

    /// The position of the front element along each dimension, if there are any elements.
    pub(crate) fn positions(&self) -> &[u64] {
        &self.positions
    }

    /// Remove the front element, returning `false` if there are no elements.
    ///
    /// `step(dimension, position)` is called for each of the `dimensions` along which the position of the front
    /// element changes.
    #[inline]
    pub(crate) fn pop_front<D: GridDimension>(
        &mut self,
        dimensions: &[D],
        mut step: impl FnMut(&D, u64),
    ) -> bool {
        if self.range.is_empty() {
            return false;
        }
        self.range.start += 1;
        if !self.range.is_empty() {
            for (position, dimension) in std::iter::zip(&mut self.positions, dimensions).rev() {
                // A position is less than its count, so this cannot overflow
                *position += 1;
                if *position < dimension.count() {
                    step(dimension, *position);
                    break;
                }
                *position = 0;
                step(dimension, 0);
            }
        }
        true
    }

    /// Remove the back element, returning its index in the grid.
    pub(crate) fn pop_back(&mut self) -> Option<u64> {
        self.range.next_back().map(|index| index as u64)
    }

    /// The number of remaining elements.
    pub(crate) fn len(&self) -> usize {
        self.range.len()
    }
}

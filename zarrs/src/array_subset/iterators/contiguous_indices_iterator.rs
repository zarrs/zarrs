use std::iter::FusedIterator;

use itertools::izip;

use crate::{
    array::ArrayIndices,
    array_subset::{indexers::{Indexer, IndexerEnum, RangeSubset, VIndex}, ArraySubset, IncompatibleArraySubsetAndShapeError},
};

use super::IndicesIterator;

/// Iterates over contiguous element indices in an array subset.
///
/// The iterator item is a tuple: (indices, # contiguous elements).
///
/// Iterates over the last dimension fastest (i.e. C-contiguous order).
/// For example, consider a 4x3 array with element indices
/// ```text
/// (0, 0)  (0, 1)  (0, 2)
/// (1, 0)  (1, 1)  (1, 2)
/// (2, 0)  (2, 1)  (2, 2)
/// (3, 0)  (3, 1)  (3, 2)
/// ```
/// An iterator with an array subset covering the entire array will produce
/// ```rust,ignore
/// [((0, 0), 9)]
/// ```
/// An iterator with an array subset corresponding to the lower right 2x2 region will produce
/// ```rust,ignore
/// [((2, 1), 2), ((3, 1), 2)]
/// ```
pub struct ContiguousIndices {
    subset_contiguous_start: ArraySubset,
    contiguous_elements: u64,
}

fn check_subset_array_shape(
    subset: &impl Indexer,
    array_shape: &[u64],
)-> Result<(), IncompatibleArraySubsetAndShapeError> {
    if !(subset.dimensionality() == array_shape.len()
            && std::iter::zip(subset.end_exc(), array_shape).all(|(end, shape)| end <= *shape))
        {
            let indexer_enum: IndexerEnum = subset.clone().to_enum();
            return Err(IncompatibleArraySubsetAndShapeError(
                indexer_enum.into(),
                array_shape.to_vec(),
            ));
        }
    Ok(())
}

impl ContiguousIndices {

    pub fn new_from_discontinuous(
        subset: &VIndex,
        array_shape: &[u64],
    ) -> Result<Self, IncompatibleArraySubsetAndShapeError> {
        if let Err(err) = check_subset_array_shape(subset, array_shape) {
            return Err(err);
        }
        Ok(Self { subset_contiguous_start: IndexerEnum::VIndex(subset.clone()).into(), contiguous_elements: 1 })
    }

    /// Create a new contiguous indices iterator.
    ///
    /// # Errors
    /// Returns [`IncompatibleArraySubsetAndShapeError`] if `array_shape` does not encapsulate `subset`.
    pub fn new_from_range_subset(
        subset: &RangeSubset,
        array_shape: &[u64],
    ) -> Result<Self, IncompatibleArraySubsetAndShapeError> {
        if let Err(err) = check_subset_array_shape(subset, array_shape) {
            return Err(err);
        } // TODO: certainly there is a cleaner way to do this?
        let mut contiguous = true;
        let mut contiguous_elements = 1;
        let mut shape_out: Vec<u64> = Vec::with_capacity(array_shape.len());
        for (&subset_start, &subset_size, &array_size, shape_out_i) in izip!(
            subset.start().iter().rev(),
            subset.shape().iter().rev(),
            array_shape.iter().rev(),
            shape_out.spare_capacity_mut().iter_mut().rev(),
        ) {
            if contiguous {
                contiguous_elements *= subset_size;
                shape_out_i.write(1);
                contiguous = subset_start == 0 && subset_size == array_size;
            } else {
                shape_out_i.write(subset_size);
            }
        }
        // SAFETY: each element is initialised
        unsafe { shape_out.set_len(array_shape.len()) };
        let ranges = subset
            .start()
            .iter()
            .zip(shape_out)
            .map(|(&st, sh)| st..(st + sh));
        let subset_contiguous_start = ArraySubset::from(ranges);
        // let inner = subset_contiguous_start.iter_indices();
        Ok(Self {
            subset_contiguous_start,
            contiguous_elements,
        })
    }

    /// Return the number of starting indices (i.e. the length of the iterator).
    #[must_use]
    pub fn len(&self) -> usize {
        self.subset_contiguous_start.num_elements_usize()
    }

    /// Returns true if the number of starting indices is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    #[must_use]
    pub fn contiguous_elements(&self) -> u64 {
        self.contiguous_elements
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    ///
    /// # Panics
    /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
    #[must_use]
    pub fn contiguous_elements_usize(&self) -> usize {
        usize::try_from(self.contiguous_elements).unwrap()
    }

    /// Create a new serial iterator.
    #[must_use]
    pub fn iter(&self) -> ContiguousIndicesIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'a> IntoIterator for &'a ContiguousIndices {
    type Item = ArrayIndices;
    type IntoIter = ContiguousIndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ContiguousIndicesIterator {
            inner: IndicesIterator::new(&self.subset_contiguous_start),
            contiguous_elements: self.contiguous_elements,
        }
    }
}

/// Serial contiguous indices iterator.
///
/// See [`ContiguousIndices`].
pub struct ContiguousIndicesIterator<'a> {
    inner: IndicesIterator<'a>,
    contiguous_elements: u64,
}

impl ContiguousIndicesIterator<'_> {
    /// Return the number of contiguous elements (fixed on each iteration).
    #[must_use]
    pub fn contiguous_elements(&self) -> u64 {
        self.contiguous_elements
    }

    /// Return the number of contiguous elements (fixed on each iteration).
    ///
    /// # Panics
    /// Panics if the number of contiguous elements exceeds [`usize::MAX`].
    #[must_use]
    pub fn contiguous_elements_usize(&self) -> usize {
        usize::try_from(self.contiguous_elements).unwrap()
    }
}

impl Iterator for ContiguousIndicesIterator<'_> {
    type Item = ArrayIndices;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for ContiguousIndicesIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl ExactSizeIterator for ContiguousIndicesIterator<'_> {}

impl FusedIterator for ContiguousIndicesIterator<'_> {}


#[cfg(test)]
mod tests {
    use crate::array_subset::indexers::{IndexerEnum, VIndex};

    use super::*;

    #[test]
    fn fully_contiguous() {
        let indices =
            ContiguousIndices::new_from_range_subset(&RangeSubset::new_with_ranges(&[0..4, 0..8]), &[4, 8]).unwrap();
        assert_eq!(indices.len(), 1);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(vec![0, 0]));
        assert_eq!(iter.contiguous_elements(), 32);
    }

    #[test]
    fn discontinuous_range_subset() {
        let indices =
            ContiguousIndices::new_from_range_subset(&RangeSubset::new_with_ranges(&[0..4, 0..7]), &[4, 8]).unwrap();
        assert_eq!(indices.len(), 4);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(vec![0, 0]));
        assert_eq!(iter.contiguous_elements(), 7);
        assert_eq!(iter.next(), Some(vec![1, 0]));
        assert_eq!(iter.next(), Some(vec![2, 0]));
        assert_eq!(iter.next_back(), Some(vec![3, 0]));
    }

    #[test]
    fn vindex() {
        let indices =
            ContiguousIndices::new_from_discontinuous(&VIndex::new_from_dimension_first_indices(vec![vec![0, 2, 3], vec![1, 2, 6]]).unwrap(), &[4, 8]).unwrap();
        assert_eq!(indices.len(), 3);
        let mut iter = indices.iter();
        assert_eq!(iter.next(), Some(vec![0, 1]));
        assert_eq!(iter.contiguous_elements(), 1);
        assert_eq!(iter.next(), Some(vec![2, 2]));
        assert_eq!(iter.next_back(), Some(vec![3, 6]));
    }

    #[test]
    fn vindex_bad_end() {
        assert!(ContiguousIndices::new_from_discontinuous(&VIndex::new_from_dimension_first_indices(vec![vec![0, 2, 3], vec![1, 2, 9]]).unwrap(), &[4, 8]).is_err());
    }

    #[test]
    fn range_susbet_bad_end() {
        assert!(ContiguousIndices::new_from_range_subset(&RangeSubset::new_with_ranges(&[0..4, 0..9]), &[4, 8]).is_err());
    }
}

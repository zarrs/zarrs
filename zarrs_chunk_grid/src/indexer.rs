//! Generic indexer support.

use std::ops::Range;
use thiserror::Error;

use crate::{
    ArrayIndices, ArrayIndicesTinyVec, ArrayShape, ArraySubsetTraits,
    IncompatibleDimensionalityError, MaybeSend, MaybeSync, ravel_indices,
};

/// An incompatible indexer and array shape error.
///
/// Raised if an indexer references an out-of-bounds element or the dimensionality differs.
#[derive(Clone, Debug, Error)]
pub enum IndexerError {
    /// The indexer dimensionality is incompatible.
    #[error("indexer is incompatible with array shape {_0:?}")]
    IncompatibleDimensionality(#[from] IncompatibleDimensionalityError),
    /// The indexer references out-of-bounds array indices.
    #[error(
        "indexer references array indices {_0:?} which are out-of-bounds of array shape {_1:?}"
    )]
    OutOfBounds(ArrayIndices, ArrayShape),
    /// The indexer has an incompatible length.
    #[error("indexer has an incompatible length {_0}, expected {_1}")]
    IncompatibleLength(u64, u64),
}

impl IndexerError {
    /// Create a new [`IndexerError`] where the dimensionality is incompatible.
    #[must_use]
    pub fn new_incompatible_dimensionality(got: usize, expected: usize) -> Self {
        IncompatibleDimensionalityError::new(got, expected).into()
    }

    /// Create a new [`IndexerError`] representing out-of-bounds indices.
    #[must_use]
    pub fn new_oob(indices: ArrayIndices, shape: ArrayShape) -> Self {
        Self::OutOfBounds(indices, shape)
    }

    /// Create a new [`IndexerError`] where the length is incompatible.
    #[must_use]
    pub fn new_incompatible_length(got: u64, expected: u64) -> Self {
        Self::IncompatibleLength(got, expected)
    }
}

/// This trait combines `MaybeSend` and `MaybeSync`
/// for an iterator over generic items.
pub trait IndexerIterator: Iterator + MaybeSend + MaybeSync {}
impl<T: Iterator + MaybeSend + MaybeSync> IndexerIterator for T {}

/// A trait for a generic indexer.
pub trait Indexer: MaybeSend + MaybeSync {
    /// Return the dimensionality of the indexer.
    #[must_use]
    fn dimensionality(&self) -> usize;

    /// The number of indices/elements.
    #[must_use]
    fn len(&self) -> u64;

    /// Returns if the indexer is empty (i.e. has a zero length).
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the output shape of the indexer.
    #[must_use]
    fn output_shape(&self) -> Vec<u64>;

    /// Returns an iterator over the indices of elements.
    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>>;

    /// Returns an iterator over the linearised indices of elements.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate the indices.
    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError>;

    /// Returns an iterator over contiguous sequences of linearised element indices.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate the indices.
    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError>;

    /// Return the byte ranges of the indexer in an array with `array_shape` and a fixed element size of `element_size`.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if the `array_shape` does not encapsulate indices of the indexer.
    fn iter_contiguous_byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Box<dyn IndexerIterator<Item = std::ops::Range<u64>>>, IndexerError> {
        let element_size_u64 = element_size as u64;
        let byte_ranges = self.iter_contiguous_linearised_indices(array_shape)?.map(
            move |(array_index, contiguous_elements)| {
                let byte_index = array_index * element_size_u64;
                byte_index..byte_index + contiguous_elements * element_size_u64
            },
        );
        Ok(Box::new(byte_ranges))
    }

    /// Return the indexer as an [`ArraySubsetTraits`].
    ///
    /// Returns [`None`] if the indexer does not represent a contiguous array subset.
    fn as_array_subset(&self) -> Option<&dyn ArraySubsetTraits> {
        None
    }
}

impl<T: Indexer> Indexer for &T {
    fn dimensionality(&self) -> usize {
        (**self).dimensionality()
    }

    fn len(&self) -> u64 {
        (**self).len()
    }

    fn output_shape(&self) -> Vec<u64> {
        (**self).output_shape()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        (**self).iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        (**self).iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        (**self).iter_contiguous_linearised_indices(array_shape)
    }

    fn as_array_subset(&self) -> Option<&dyn ArraySubsetTraits> {
        (**self).as_array_subset()
    }
}

impl<T: Indexer> Indexer for &[T] {
    fn dimensionality(&self) -> usize {
        self.first().map_or(0, T::dimensionality)
    }

    fn len(&self) -> u64 {
        self.iter().map(T::len).sum()
    }

    fn output_shape(&self) -> Vec<u64> {
        // flatten a slice of array subsets on output
        vec![self.len()]
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        let indices = self.iter().map(Indexer::iter_indices).collect::<Vec<_>>();

        Box::new(indices.into_iter().flat_map(IntoIterator::into_iter))
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        let linearised_indices = self
            .iter()
            .map(|indexer| indexer.iter_linearised_indices(array_shape))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Box::new(
            linearised_indices
                .into_iter()
                .flat_map(IntoIterator::into_iter),
        ))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        let contiguous_linearised_indices = self
            .iter()
            .map(|indexer| indexer.iter_contiguous_linearised_indices(array_shape))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Box::new(
            contiguous_linearised_indices
                .into_iter()
                .flat_map(IntoIterator::into_iter),
        ))
    }
}

impl<T: Indexer> Indexer for Vec<T> {
    fn dimensionality(&self) -> usize {
        self.as_slice().dimensionality()
    }

    fn len(&self) -> u64 {
        self.iter().map(T::len).sum()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn output_shape(&self) -> Vec<u64> {
        self.as_slice().output_shape()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        self.as_slice()
            .iter_contiguous_linearised_indices(array_shape)
    }
}

impl<T: Indexer, const N: usize> Indexer for [T; N] {
    fn dimensionality(&self) -> usize {
        self.as_slice().dimensionality()
    }

    fn len(&self) -> u64 {
        self.iter().map(T::len).sum()
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn output_shape(&self) -> Vec<u64> {
        self.as_slice().output_shape()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        self.as_slice()
            .iter_contiguous_linearised_indices(array_shape)
    }
}

impl Indexer for &[ArrayIndices] {
    fn dimensionality(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn len(&self) -> u64 {
        <[ArrayIndices]>::len(self) as u64
    }

    fn is_empty(&self) -> bool {
        <[ArrayIndices]>::is_empty(self)
    }

    fn output_shape(&self) -> Vec<u64> {
        vec![self.len()]
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        let indices: Vec<ArrayIndicesTinyVec> = self
            .iter()
            .map(|v| v.iter().copied().collect::<ArrayIndicesTinyVec>())
            .collect();
        Box::new(indices.into_iter())
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        let linearised_indices = self
            .iter()
            .map(|indices| {
                if indices.len() == array_shape.len() {
                    ravel_indices(indices, array_shape)
                        .ok_or_else(|| IndexerError::new_oob(indices.clone(), array_shape.to_vec()))
                } else {
                    Err(IncompatibleDimensionalityError::new(
                        self.dimensionality(),
                        array_shape.len(),
                    )
                    .into())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(linearised_indices.into_iter()))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        let contiguous_linearised_indices = self
            .iter()
            .map(|indices| {
                if indices.len() == array_shape.len() {
                    let index = ravel_indices(indices, array_shape).ok_or_else(|| {
                        IndexerError::new_oob(indices.clone(), array_shape.to_vec())
                    })?;
                    Ok((index, 1))
                } else {
                    Err(IncompatibleDimensionalityError::new(
                        self.dimensionality(),
                        array_shape.len(),
                    )
                    .into())
                }
            })
            .collect::<Result<Vec<_>, IndexerError>>()?;
        Ok(Box::new(contiguous_linearised_indices.into_iter()))
    }
}

impl Indexer for Vec<ArrayIndices> {
    fn dimensionality(&self) -> usize {
        self.as_slice().dimensionality()
    }

    fn len(&self) -> u64 {
        self.as_slice().len() as u64
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn output_shape(&self) -> Vec<u64> {
        self.as_slice().output_shape()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        self.as_slice()
            .iter_contiguous_linearised_indices(array_shape)
    }
}

impl<const N: usize> Indexer for [ArrayIndices; N] {
    fn dimensionality(&self) -> usize {
        self.as_slice().dimensionality()
    }

    fn len(&self) -> u64 {
        self.as_slice().len() as u64
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn output_shape(&self) -> Vec<u64> {
        self.as_slice().output_shape()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        self.as_slice()
            .iter_contiguous_linearised_indices(array_shape)
    }
}

// specialisation?
// impl<T> Indexer for T where T: Iterator<Item = ArrayIndices> {}

macro_rules! impl_indexer_for_ranges {
    ($ty:ty) => {
        impl Indexer for $ty {
            fn dimensionality(&self) -> usize {
                (*self).len()
            }

            fn len(&self) -> u64 {
                ArraySubsetTraits::num_elements(self)
            }

            fn output_shape(&self) -> Vec<u64> {
                self.iter().map(|r| r.end.saturating_sub(r.start)).collect()
            }

            fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
                Box::new(self.to_array_subset().indices().into_iter())
            }

            fn iter_linearised_indices(
                &self,
                array_shape: &[u64],
            ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
                Ok(Box::new(
                    self.to_array_subset()
                        .linearised_indices(array_shape)?
                        .into_iter(),
                ))
            }

            fn iter_contiguous_linearised_indices(
                &self,
                array_shape: &[u64],
            ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
                Ok(Box::new(
                    self.to_array_subset()
                        .contiguous_linearised_indices(array_shape)?
                        .into_iter(),
                ))
            }
        }
    };
}

impl_indexer_for_ranges!(&[Range<u64>]);
impl_indexer_for_ranges!(Vec<Range<u64>>);

impl<const N: usize> Indexer for [Range<u64>; N] {
    fn dimensionality(&self) -> usize {
        N
    }

    fn len(&self) -> u64 {
        ArraySubsetTraits::num_elements(self)
    }

    fn output_shape(&self) -> Vec<u64> {
        self.iter().map(|r| r.end.saturating_sub(r.start)).collect()
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndicesTinyVec>> {
        Box::new(self.to_array_subset().indices().into_iter())
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        Ok(Box::new(
            self.to_array_subset()
                .linearised_indices(array_shape)?
                .into_iter(),
        ))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        Ok(Box::new(
            self.to_array_subset()
                .contiguous_linearised_indices(array_shape)?
                .into_iter(),
        ))
    }

    fn as_array_subset(&self) -> Option<&dyn ArraySubsetTraits> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArraySubset;

    /// Test `as_array_subset` is Some for types implementing ArraySubsetTraits.
    #[test]
    fn test_as_array_subset() {
        let subset = ArraySubset::new_with_shape(vec![10, 10]);

        // Direct call on concrete type
        assert!(subset.as_array_subset().is_some());

        // Call through &ArraySubset (via impl<T: Indexer> Indexer for &T)
        let ref_subset: &ArraySubset = &subset;
        assert!(ref_subset.as_array_subset().is_some());

        // Call through &dyn ArraySubsetTraits
        let dyn_ref: &dyn ArraySubsetTraits = &subset;
        assert!(dyn_ref.as_array_subset().is_some());

        // Call through &dyn Indexer from ArraySubset
        let indexer_ref: &dyn Indexer = &subset;
        assert!(indexer_ref.as_array_subset().is_some());

        // Range array also supports as_array_subset
        let ranges: [Range<u64>; 2] = [0..10, 0..10];
        assert!(ranges.as_array_subset().is_some());

        // Reference to range array
        let ranges_ref: &[Range<u64>; 2] = &ranges;
        assert!(ranges_ref.as_array_subset().is_some());
    }
}

//! Generic indexer support.

use std::sync::Arc;

use enum_dispatch::enum_dispatch;
use zarrs_storage::byte_range::ByteRange;

use crate::{
    array::ArrayIndices,
    array_subset::{ArraySubset, IncompatibleIndexerAndShapeError},
};

/// A trait for a generic indexer.
#[enum_dispatch]
pub trait Indexer: Send + Sync + core::fmt::Debug {
    /// Return an [`Arc`] wrapped version of the indexer.
    fn to_arc(&self) -> Arc<dyn Indexer>;

    /// Return the dimensionality of the indexer.
    #[must_use]
    fn dimensionality(&self) -> usize;

    /// The number of indices/elements.
    #[must_use]
    fn len(&self) -> u64;

    /// Returns if the array subset is empty (i.e. has a zero length).
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the output shape of the indexer.
    #[must_use]
    fn output_shape(&self) -> &[u64];

    /// Returns an iterator over the indices of elements.
    fn indices(&self) -> impl IntoIterator<Item = ArrayIndices>
    where
        Self: Sized;

    /// Returns an iterator over the linearised indices of elements.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate the indices.
    fn linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<impl IntoIterator<Item = u64>, IncompatibleIndexerAndShapeError>
    where
        Self: Sized;

    /// Returns an iterator over contiguous sequences of element indices.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate the indices.
    fn contiguous_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<impl IntoIterator<Item = (ArrayIndices, u64)>, IncompatibleIndexerAndShapeError>
    where
        Self: Sized;

    /// Returns an iterator over contiguous sequences of linearised element indices.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate the indices.
    fn contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<impl IntoIterator<Item = (u64, u64)>, IncompatibleIndexerAndShapeError>
    where
        Self: Sized;

    /// Return the byte ranges of an array subset in an array with `array_shape` and `element_size`.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate this array subset.
    fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Vec<ByteRange>, IncompatibleIndexerAndShapeError>
    where
        Self: Sized,
    {
        let mut byte_ranges: Vec<ByteRange> = Vec::new();
        let contiguous_indices = self.contiguous_linearised_indices(array_shape)?;
        for (array_index, contiguous_elements) in contiguous_indices {
            let byte_index = array_index * element_size as u64;
            byte_ranges.push(ByteRange::FromStart(
                byte_index,
                Some(contiguous_elements * element_size as u64),
            ));
        }
        Ok(byte_ranges)
    }

    /// Return the indexer as an [`ArraySubset`].
    ///
    /// Returns [`None`] if the indexer is not an [`ArraySubset`].
    fn as_array_subset(&self) -> Option<&ArraySubset>;
}

/// Concrete implementations of generic indexers.
#[enum_dispatch(Indexer)]
#[derive(Debug)]
pub enum IndexerImpl {
    /// An [`ArraySubset`] indexer.
    ArraySubset(ArraySubset),
}

impl From<&ArraySubset> for IndexerImpl {
    fn from(value: &ArraySubset) -> Self {
        Self::ArraySubset(value.clone())
    }
}

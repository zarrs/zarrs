//! Generic indexer support.

use zarrs_storage::byte_range::ByteRange;

use crate::{
    array::ArrayIndices,
    array_subset::{ArraySubset, IncompatibleIndexerAndShapeError},
};

/// A trait for a generic indexer.
pub trait Indexer: Send + Sync + core::fmt::Debug {
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
    fn output_shape(&self) -> Vec<u64>;

    /// Returns an iterator over the indices of elements.
    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices>>;

    /// Returns an iterator over the linearised indices of elements.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate the indices.
    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64>>, IncompatibleIndexerAndShapeError>;

    // /// Returns an iterator over contiguous sequences of element indices.
    // ///
    // /// # Errors
    // /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate the indices.
    // fn iter_contiguous_indices(
    //     &self,
    //     array_shape: &[u64],
    // ) -> Result<Box<dyn Iterator<Item = (ArrayIndices, u64)>>, IncompatibleIndexerAndShapeError>;

    /// Returns an iterator over contiguous sequences of linearised element indices.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate the indices.
    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)>>, IncompatibleIndexerAndShapeError>;

    /// Return the byte ranges of an array subset in an array with `array_shape` and `element_size`.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate this array subset.
    fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Vec<ByteRange>, IncompatibleIndexerAndShapeError> {
        let mut byte_ranges: Vec<ByteRange> = Vec::new();
        let contiguous_indices = self.iter_contiguous_linearised_indices(array_shape)?;
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

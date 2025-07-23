//! Generic indexer support.

use zarrs_storage::byte_range::ByteRange;

use crate::{
    array::{ravel_indices, ArrayIndices},
    array_subset::{ArraySubset, IncompatibleIndexerAndShapeError},
};

/// A trait for a generic indexer.
pub trait Indexer: Send + Sync {
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
    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync>;

    /// Returns an iterator over the linearised indices of elements.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate the indices.
    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerAndShapeError>;

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
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerAndShapeError>;

    /// Return the byte ranges of an array subset in an array with `array_shape` and `element_size`.
    ///
    /// # Errors
    /// Returns [`IncompatibleIndexerAndShapeError`] if the `array_shape` does not encapsulate this array subset.
    // FIXME: Prefer to remove this? Or at least return an iterator
    fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Box<dyn Iterator<Item = ByteRange> + Send + Sync>, IncompatibleIndexerAndShapeError> {
        let element_size_u64 = element_size as u64;
        let byte_ranges = self.iter_contiguous_linearised_indices(array_shape)?.map(
            move |(array_index, contiguous_elements)| {
                let byte_index = array_index * element_size_u64;
                ByteRange::FromStart(
                    byte_index,
                    Some(contiguous_elements * element_size_u64),
                )
            }
        );
        Ok(Box::new(byte_ranges))
    }

    /// Return the indexer as an [`ArraySubset`].
    ///
    /// Returns [`None`] if the indexer is not an [`ArraySubset`].
    fn as_array_subset(&self) -> Option<&ArraySubset> {
        None
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

    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync> {
        let indices = self.iter().map(Indexer::iter_indices).collect::<Vec<_>>();

        Box::new(indices.into_iter().flat_map(IntoIterator::into_iter))
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerAndShapeError> {
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

    // fn iter_contiguous_indices(
    //     &self,
    //     array_shape: &[u64],
    // ) -> Result<Box<dyn Iterator<Item = (ArrayIndices, u64)>>, IncompatibleIndexerAndShapeError> {
    //     Ok(Box::new(self.contiguous_indices(array_shape)?.into_iter()))
    // }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerAndShapeError>
    {
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

    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerAndShapeError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerAndShapeError>
    {
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

    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerAndShapeError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerAndShapeError>
    {
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

    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync> {
        #[allow(clippy::unnecessary_to_owned)] // false positive
        Box::new(self.to_vec().into_iter())
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerAndShapeError> {
        let linearised_indices = self
            .iter()
            .map(|indices| {
                if indices.len() == array_shape.len() {
                    Ok(ravel_indices(indices, array_shape))
                } else {
                    Err(IncompatibleIndexerAndShapeError::new(array_shape.to_vec()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(linearised_indices.into_iter()))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerAndShapeError>
    {
        let contiguous_linearised_indices = self
            .iter()
            .map(|indices| {
                if indices.len() == array_shape.len() {
                    Ok((ravel_indices(indices, array_shape), 1))
                } else {
                    Err(IncompatibleIndexerAndShapeError::new(array_shape.to_vec()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
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

    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerAndShapeError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerAndShapeError>
    {
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

    fn iter_indices(&self) -> Box<dyn Iterator<Item = ArrayIndices> + Send + Sync> {
        self.as_slice().iter_indices()
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = u64> + Send + Sync>, IncompatibleIndexerAndShapeError> {
        self.as_slice().iter_linearised_indices(array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn Iterator<Item = (u64, u64)> + Send + Sync>, IncompatibleIndexerAndShapeError>
    {
        self.as_slice()
            .iter_contiguous_linearised_indices(array_shape)
    }
}

// specialisation?
// impl<T> Indexer for T where T: Iterator<Item = ArrayIndices> {}

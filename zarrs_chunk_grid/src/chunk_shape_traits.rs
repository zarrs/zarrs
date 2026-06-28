use std::num::{NonZeroU64, NonZeroUsize};

use zarrs_metadata::ChunkShape;

mod sealed {
    pub trait Sealed {}
}
impl<T> sealed::Sealed for T where T: AsRef<[u64]> {}

/// A trait for chunk shapes.
pub trait ChunkShapeTraits: AsRef<[u64]> + sealed::Sealed {
    /// Convert a chunk shape to an array shape.
    #[must_use]
    fn to_chunk_shape(&self) -> ChunkShape {
        self.as_ref().iter().copied().collect()
    }

    /// Return the number of elements.
    ///
    /// Equal to the product of the components of its shape.
    #[must_use]
    fn num_elements(&self) -> u64 {
        self.as_ref().iter().product::<u64>()
    }

    /// Return the number of elements as a usize.
    ///
    /// Equal to the product of the components of its shape.
    ///
    /// # Panics
    /// Panics if the number of elements exceeds [`usize::MAX`].
    #[must_use]
    fn num_elements_usize(&self) -> usize {
        usize::try_from(self.num_elements()).unwrap()
    }
}

impl<T> ChunkShapeTraits for T where T: AsRef<[u64]> {}

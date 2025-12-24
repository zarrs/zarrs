use std::num::{NonZeroU64, NonZeroUsize};

/// An array shape. Dimensions may be zero.
pub type ArrayShape = Vec<u64>;

/// A chunk shape. Dimensions must be non-zero.
pub type ChunkShape = Vec<NonZeroU64>;

/// A trait for chunk shapes.
pub trait ChunkShapeTraits: AsRef<[NonZeroU64]> {
    /// Convert a chunk shape to an array shape.
    #[must_use]
    fn to_array_shape(&self) -> ArrayShape {
        self.as_ref().iter().map(|i| i.get()).collect()
    }

    /// Return the number of elements.
    ///
    /// Equal to the product of the components of its shape.
    #[must_use]
    fn num_elements(&self) -> NonZeroU64 {
        unsafe {
            // Multiplying NonZeroU64 must result in NonZeroU64
            NonZeroU64::new_unchecked(self.num_elements_u64())
        }
    }

    /// Return the number of elements as a nonzero usize.
    ///
    /// Equal to the product of the components of its shape.
    ///
    /// # Panics
    /// Panics if the number of elements exceeds [`usize::MAX`].
    #[must_use]
    fn num_elements_nonzero_usize(&self) -> NonZeroUsize {
        unsafe {
            // Multiplying NonZeroU64 must result in NonZeroUsize
            NonZeroUsize::new_unchecked(usize::try_from(self.num_elements_u64()).unwrap())
        }
    }

    /// Return the number of elements as a u64.
    ///
    /// Equal to the product of the components of its shape.
    #[must_use]
    fn num_elements_u64(&self) -> u64 {
        self.as_ref()
            .iter()
            .copied()
            .map(NonZeroU64::get)
            .product::<u64>()
    }

    /// Return the number of elements as a usize.
    ///
    /// Equal to the product of the components of its shape.
    ///
    /// # Panics
    /// Panics if the number of elements exceeds [`usize::MAX`].
    #[must_use]
    fn num_elements_usize(&self) -> usize {
        usize::try_from(self.num_elements_u64()).unwrap()
    }
}

impl<T> ChunkShapeTraits for T where T: AsRef<[NonZeroU64]> {}

use std::num::NonZeroU64;

use zarrs_metadata::ChunkShape;

mod private {
    pub trait Sealed {}
}

impl private::Sealed for &[u64] {}
impl private::Sealed for Vec<u64> {}
impl private::Sealed for &[NonZeroU64] {}
impl private::Sealed for Vec<NonZeroU64> {}
impl<const N: usize> private::Sealed for [u64; N] {}
impl<const N: usize> private::Sealed for [NonZeroU64; N] {}

/// A trait for types that can be viewed as a chunk shape (`&[u64]`).
///
/// Implemented for common chunk shape types:
/// - `&[u64]`, `Vec<u64>`, `[u64; N]` — passed through directly
/// - `&[NonZeroU64]`, `Vec<NonZeroU64>`, `[NonZeroU64; N]` — cast using [`bytemuck`]
pub trait ChunkShapeView: private::Sealed {
    /// View this as a `[u64]` slice.
    fn as_shape(&self) -> &[u64];
}

impl ChunkShapeView for &[u64] {
    #[inline]
    fn as_shape(&self) -> &[u64] {
        self
    }
}

impl ChunkShapeView for Vec<u64> {
    #[inline]
    fn as_shape(&self) -> &[u64] {
        self.as_slice()
    }
}

impl ChunkShapeView for &[NonZeroU64] {
    #[inline]
    fn as_shape(&self) -> &[u64] {
        bytemuck::must_cast_slice(self)
    }
}

impl ChunkShapeView for Vec<NonZeroU64> {
    #[inline]
    fn as_shape(&self) -> &[u64] {
        bytemuck::must_cast_slice(self.as_slice())
    }
}

impl<const N: usize> ChunkShapeView for [u64; N] {
    #[inline]
    fn as_shape(&self) -> &[u64] {
        self.as_slice()
    }
}

impl<const N: usize> ChunkShapeView for [NonZeroU64; N] {
    #[inline]
    fn as_shape(&self) -> &[u64] {
        bytemuck::must_cast_slice(self.as_slice())
    }
}

/// A trait for chunk shapes.
pub trait ChunkShapeTraits: ChunkShapeView {
    /// Convert a chunk shape to an array shape.
    #[must_use]
    fn to_chunk_shape(&self) -> ChunkShape {
        self.as_shape().iter().copied().collect()
    }

    /// Return the number of elements.
    ///
    /// Equal to the product of the components of its shape.
    #[must_use]
    fn num_elements(&self) -> u64 {
        self.as_shape().iter().product::<u64>()
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

impl<T: ChunkShapeView> ChunkShapeTraits for T {}

//! The [`FromArrayBytes`] trait for converting [`ArrayBytes`] into other types.

use std::sync::Arc;

use super::element::ElementOwned;
use super::{ArrayBytes, ArrayError, DataType};

/// A trait for types that can be constructed from [`ArrayBytes`], an array shape and a [`DataType`].
pub trait FromArrayBytes: Sized {
    /// Convert [`ArrayBytes`] into `Self`.
    ///
    /// # Arguments
    /// * `bytes` - The array bytes to convert
    /// * `shape` - The shape of the array
    /// * `data_type` - The datatype of the array elements
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the conversion fails.
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError>;

    /// Convert an `Arc<ArrayBytes>` into `Self`.
    ///
    /// This method has a default implementation that unwraps the Arc (cloning if necessary).
    /// It is overridden for `Arc<ArrayBytes<'static>>` to avoid the clone.
    /// This is used by cached retrieval methods to avoid unnecessary copies.
    ///
    /// # Arguments
    /// * `bytes` - The array bytes to convert (wrapped in Arc)
    /// * `shape` - The shape of the array
    /// * `data_type` - The datatype of the array elements
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the conversion fails.
    fn from_array_bytes_arc(
        bytes: Arc<ArrayBytes<'static>>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        Self::from_array_bytes(Arc::unwrap_or_clone(bytes), shape, data_type)
    }
}

impl FromArrayBytes for ArrayBytes<'static> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        _shape: &[u64],
        _data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        Ok(bytes)
    }
}

impl FromArrayBytes for Arc<ArrayBytes<'static>> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        _shape: &[u64],
        _data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        Ok(Arc::new(bytes))
    }

    fn from_array_bytes_arc(
        bytes: Arc<ArrayBytes<'static>>,
        _shape: &[u64],
        _data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        Ok(bytes)
    }
}

impl<T: ElementOwned> FromArrayBytes for Vec<T> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        _shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        T::from_array_bytes(data_type, bytes)
    }
}

#[cfg(feature = "ndarray")]
impl<T: ElementOwned, D: ndarray::Dimension> FromArrayBytes for ndarray::Array<T, D> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let elements = T::from_array_bytes(data_type, bytes)?;
        let length = elements.len();
        let arrayd = ndarray::ArrayD::from_shape_vec(
            crate::array::iter_u64_to_usize(shape.iter()),
            elements,
        )
        .map_err(|_| {
            ArrayError::Other(format!(
                "`shape`: {shape:?} is not compatible with the number of elements: {length:?}"
            ))
        })?;
        arrayd.into_dimensionality::<D>().map_err(|_| {
            ArrayError::Other(format!(
                "`shape` {shape:?} is incompatible with requested dimensionality of size {}",
                D::NDIM.unwrap_or(0)
            ))
        })
    }
}

impl FromArrayBytes for super::Tensor {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let bytes = bytes.into_fixed()?;
        Ok(Self::new(bytes, data_type.clone(), shape.to_vec()))
    }
}

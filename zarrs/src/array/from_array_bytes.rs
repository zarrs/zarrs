//! The [`FromArrayBytes`] trait for converting [`ArrayBytes`] into other types.

use std::sync::Arc;

use super::array_data_order::reverse_axes;
use super::element::ElementOwned;
use super::{ArrayBytes, ArrayDataOrder, ArrayError, DataType};

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

    /// Convert [`ArrayBytes`] with elements in `data_order` into `Self`.
    ///
    /// The default implementation passes the reversed `shape` to [`from_array_bytes`](FromArrayBytes::from_array_bytes) if `data_order` is [`ArrayDataOrder::F`],
    /// since an F-order buffer has the same layout as a C-order buffer with reversed axes.
    /// It is overridden for [`ndarray::Array`] to return an F-order (column-major) array with the unreversed `shape`.
    ///
    /// # Arguments
    /// * `bytes` - The array bytes to convert
    /// * `shape` - The shape of the array
    /// * `data_type` - The datatype of the array elements
    /// * `data_order` - The order of the elements in `bytes`
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the conversion fails.
    fn from_array_bytes_with_order(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
        data_order: ArrayDataOrder,
    ) -> Result<Self, ArrayError> {
        match data_order {
            ArrayDataOrder::C => Self::from_array_bytes(bytes, shape, data_type),
            ArrayDataOrder::F => Self::from_array_bytes(bytes, &reverse_axes(shape), data_type),
        }
    }

    /// Convert an `Arc<ArrayBytes>` with elements in `data_order` into `Self`.
    ///
    /// Refer to [`from_array_bytes_with_order`](FromArrayBytes::from_array_bytes_with_order).
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the conversion fails.
    fn from_array_bytes_arc_with_order(
        bytes: Arc<ArrayBytes<'static>>,
        shape: &[u64],
        data_type: &DataType,
        data_order: ArrayDataOrder,
    ) -> Result<Self, ArrayError> {
        match data_order {
            ArrayDataOrder::C => Self::from_array_bytes_arc(bytes, shape, data_type),
            ArrayDataOrder::F => Self::from_array_bytes_with_order(
                Arc::unwrap_or_clone(bytes),
                shape,
                data_type,
                data_order,
            ),
        }
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

    fn from_array_bytes_arc_with_order(
        bytes: Arc<ArrayBytes<'static>>,
        _shape: &[u64],
        _data_type: &DataType,
        _data_order: ArrayDataOrder,
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
        Ok(T::from_array_bytes(data_type, bytes)?)
    }
}

#[cfg(feature = "ndarray")]
impl<T: ElementOwned, D: ndarray::Dimension> FromArrayBytes for ndarray::Array<T, D> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let elements: Vec<T> = T::from_array_bytes(data_type, bytes)?;
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

    fn from_array_bytes_with_order(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
        data_order: ArrayDataOrder,
    ) -> Result<Self, ArrayError> {
        match data_order {
            ArrayDataOrder::C => Self::from_array_bytes(bytes, shape, data_type),
            ArrayDataOrder::F => {
                // A C-order array with reversed axes is an F-order array once its axes are reversed
                let array_reversed =
                    ndarray::ArrayD::<T>::from_array_bytes(bytes, &reverse_axes(shape), data_type)?;
                array_reversed
                    .reversed_axes()
                    .into_dimensionality::<D>()
                    .map_err(|_| {
                        ArrayError::Other(format!(
                            "`shape` {shape:?} is incompatible with requested dimensionality of size {}",
                            D::NDIM.unwrap_or(0)
                        ))
                    })
            }
        }
    }
}

impl FromArrayBytes for super::Tensor<'static> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let bytes = bytes.into_fixed()?;
        Ok(Self::new(bytes, data_type.clone(), shape.to_vec()))
    }
}

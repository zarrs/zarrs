//! The [`FromArrayBytes`] trait for converting [`ArrayBytes`] into other types.

use std::sync::Arc;

use super::element::ElementOwned;
use super::{ArrayBytes, ArrayError, DataType, ElementOrder};

/// A trait for types that can be constructed from [`ArrayBytes`], an array shape and a [`DataType`].
pub trait FromArrayBytes: Sized {
    /// Convert [`ArrayBytes`] into `Self`.
    ///
    /// # Arguments
    /// * `bytes` - The array bytes to convert, with elements in `order`
    /// * `shape` - The shape of the array
    /// * `data_type` - The datatype of the array elements
    /// * `order` - The order of the elements in `bytes`
    ///
    /// Types without a layout of their own (e.g. [`Vec`] and [`ArrayBytes`]) hold the elements in `order`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the conversion fails.
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<Self, ArrayError>;

    /// Convert an `Arc<ArrayBytes>` into `Self`.
    ///
    /// This method has a default implementation that unwraps the Arc (cloning if necessary).
    /// It is overridden for `Arc<ArrayBytes<'static>>` to avoid the clone.
    /// This is used by cached retrieval methods to avoid unnecessary copies.
    ///
    /// # Arguments
    /// * `bytes` - The array bytes to convert (wrapped in Arc), with elements in `order`
    /// * `shape` - The shape of the array
    /// * `data_type` - The datatype of the array elements
    /// * `order` - The order of the elements in `bytes`
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the conversion fails.
    fn from_array_bytes_arc(
        bytes: Arc<ArrayBytes<'static>>,
        shape: &[u64],
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<Self, ArrayError> {
        Self::from_array_bytes(Arc::unwrap_or_clone(bytes), shape, data_type, order)
    }
}

impl FromArrayBytes for ArrayBytes<'static> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        _shape: &[u64],
        _data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<Self, ArrayError> {
        Ok(bytes)
    }
}

impl FromArrayBytes for Arc<ArrayBytes<'static>> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        _shape: &[u64],
        _data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<Self, ArrayError> {
        Ok(Arc::new(bytes))
    }

    fn from_array_bytes_arc(
        bytes: Arc<ArrayBytes<'static>>,
        _shape: &[u64],
        _data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<Self, ArrayError> {
        Ok(bytes)
    }
}

impl<T: ElementOwned> FromArrayBytes for Vec<T> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        _shape: &[u64],
        data_type: &DataType,
        _order: ElementOrder,
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
        order: ElementOrder,
    ) -> Result<Self, ArrayError> {
        use ndarray::ShapeBuilder;
        let elements: Vec<T> = T::from_array_bytes(data_type, bytes)?;
        let length = elements.len();
        let shape_usize = crate::array::iter_u64_to_usize(shape.iter());
        let arrayd = match order {
            ElementOrder::C => ndarray::ArrayD::from_shape_vec(shape_usize, elements),
            ElementOrder::F => ndarray::ArrayD::from_shape_vec(shape_usize.f(), elements),
        }
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

impl FromArrayBytes for super::Tensor<'static> {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<Self, ArrayError> {
        let bytes = bytes.into_fixed()?;
        Ok(Self::new(bytes, data_type.clone(), shape.to_vec()).with_order(order))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Tensor, data_type};

    #[test]
    fn from_array_bytes_order() {
        // A 2x3 array with elements 0..6 in C order, held in F order
        let shape = [2, 3];
        let f_order = vec![0u8, 3, 1, 4, 2, 5];
        let uint8 = data_type::uint8();
        let bytes = || ArrayBytes::new_flen(f_order.clone());

        // Types without a layout hold the elements in the given order
        let elements =
            <Vec<u8> as FromArrayBytes>::from_array_bytes(bytes(), &shape, &uint8, ElementOrder::F)
                .unwrap();
        assert_eq!(elements, f_order);

        let tensor = Tensor::from_array_bytes(bytes(), &shape, &uint8, ElementOrder::F).unwrap();
        assert_eq!(tensor.order(), ElementOrder::F);
        assert_eq!(tensor.bytes(), f_order.as_slice());

        #[cfg(feature = "ndarray")]
        {
            let array =
                ndarray::Array2::<u8>::from_array_bytes(bytes(), &shape, &uint8, ElementOrder::F)
                    .unwrap();
            assert_eq!(array, ndarray::array![[0u8, 1, 2], [3, 4, 5]]);
            assert!(array.t().is_standard_layout());
            let array = ndarray::ArrayD::<u8>::from_array_bytes(
                ArrayBytes::new_flen(vec![0u8, 1, 2, 3, 4, 5]),
                &shape,
                &uint8,
                ElementOrder::C,
            )
            .unwrap();
            assert!(array.is_standard_layout());
            assert_eq!(array, ndarray::array![[0u8, 1, 2], [3, 4, 5]].into_dyn());
        }
    }
}

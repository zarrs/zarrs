//! The [`IntoArrayBytes`] trait for converting input types into [`ArrayBytes`] for storage.

use super::{ArrayBytes, ArrayError, DataType, element::Element};

/// A trait for types that can be converted into [`ArrayBytes`] for storage.
pub trait IntoArrayBytes<'a> {
    /// Convert `self` into [`ArrayBytes`].
    ///
    /// # Arguments
    /// * `data_type` - The data type of the array.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the conversion fails.
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ArrayError>;
}

impl<'a> IntoArrayBytes<'a> for ArrayBytes<'a> {
    fn into_array_bytes(self, _data_type: &DataType) -> Result<ArrayBytes<'a>, ArrayError> {
        Ok(self)
    }
}

impl<T: Element> IntoArrayBytes<'static> for Vec<T> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'static>, ArrayError> {
        T::into_array_bytes(data_type, self)
    }
}

impl<'a, T: Element> IntoArrayBytes<'a> for &'a Vec<T> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ArrayError> {
        T::to_array_bytes(data_type, self)
    }
}

impl<'a, T: Element> IntoArrayBytes<'a> for &'a [T] {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ArrayError> {
        T::to_array_bytes(data_type, self)
    }
}

impl<'a, T: Element, const N: usize> IntoArrayBytes<'a> for &'a [T; N] {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ArrayError> {
        T::to_array_bytes(data_type, self)
    }
}

// #[cfg(feature = "ndarray")]
// impl<T: Element, D: ndarray::Dimension> IntoArrayBytes<'static>
//     for &ndarray::ArrayRef<T, D>
// {
//     fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'static>, ArrayError> {
//         let cow = super::ndarray_to_cow(self);
//         // Use Element::into_array_bytes which handles the conversion properly,
//         // then convert to owned to get 'static lifetime
//         T::into_array_bytes(data_type, &cow).map(ArrayBytes::into_owned)
//     }
// }

#[cfg(feature = "ndarray")]
impl<T: Element, D: ndarray::Dimension> IntoArrayBytes<'static> for ndarray::Array<T, D> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'static>, ArrayError> {
        let elements = if self.is_standard_layout() {
            self
        } else {
            self.as_standard_layout().into_owned()
        }
        .into_raw_vec_and_offset()
        .0;
        T::into_array_bytes(data_type, elements).map(ArrayBytes::into_owned)
    }
}

impl IntoArrayBytes<'static> for super::Tensor {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'static>, ArrayError> {
        let (bytes, tensor_data_type, _) = self.into_parts();
        if !tensor_data_type.eq(data_type.as_ref()) {
            return Err(ArrayError::IncompatibleElementType);
        }
        Ok(ArrayBytes::from(bytes))
    }
}

impl<'a> IntoArrayBytes<'a> for &'a super::Tensor {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ArrayError> {
        let (bytes, tensor_data_type, _) = self.as_parts();
        if !tensor_data_type.eq(data_type.as_ref()) {
            return Err(ArrayError::IncompatibleElementType);
        }
        Ok(ArrayBytes::from(bytes))
    }
}

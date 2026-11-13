//! The [`IntoArrayBytes`] trait for converting input types into [`ArrayBytes`] for storage.

use super::element::Element;
use super::{ArrayBytes, CowBytes, DataType, ElementError};

/// A trait for types that can be converted into [`ArrayBytes`] for storage.
pub trait IntoArrayBytes<'a> {
    /// Convert `self` into [`ArrayBytes`].
    ///
    /// # Arguments
    /// * `data_type` - The data type of the array.
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the conversion fails.
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError>;
}

impl<'a> IntoArrayBytes<'a> for ArrayBytes<'a> {
    fn into_array_bytes(self, _data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError> {
        Ok(self)
    }
}

/// Store raw bytes without a copy where possible.
///
/// The chunk is written directly from `self` if the codec chain passes its input through unchanged,
/// which requires a fixed length data type, native endianness, and no bytes-to-bytes codecs.
/// Any other configuration encodes into a new buffer, as usual.
impl IntoArrayBytes<'_> for &bytes::Bytes {
    fn into_array_bytes(self, _data_type: &DataType) -> Result<ArrayBytes<'static>, ElementError> {
        Ok(ArrayBytes::new_flen(CowBytes::Shared(self.clone())))
    }
}

impl<T: Element> IntoArrayBytes<'static> for Vec<T> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'static>, ElementError> {
        T::into_array_bytes(data_type, self)
    }
}

impl<'a, T: Element> IntoArrayBytes<'a> for &'a Vec<T> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError> {
        T::to_array_bytes(data_type, self)
    }
}

impl<'a, T: Element> IntoArrayBytes<'a> for &'a [T] {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError> {
        T::to_array_bytes(data_type, self)
    }
}

impl<'a, T: Element, const N: usize> IntoArrayBytes<'a> for &'a [T; N] {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError> {
        T::to_array_bytes(data_type, self)
    }
}

#[cfg(feature = "ndarray")]
impl<'a, T, S, D> IntoArrayBytes<'a> for &'a ndarray::ArrayBase<S, D>
where
    T: Element,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError> {
        if let Some(elements) = self.as_slice() {
            T::to_array_bytes(data_type, elements)
        } else {
            let elements = self
                .as_standard_layout()
                .into_owned()
                .into_raw_vec_and_offset()
                .0;
            Ok(T::into_array_bytes(data_type, elements)?.into_owned())
        }
    }
}

#[cfg(feature = "ndarray")]
impl<T: Element, D: ndarray::Dimension> IntoArrayBytes<'static> for ndarray::Array<T, D> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'static>, ElementError> {
        let elements = if self.is_standard_layout() {
            // The elements are contiguous from the offset, but an array sliced in place keeps its entire backing vector
            let len = self.len();
            let (mut elements, offset) = self.into_raw_vec_and_offset();
            elements.drain(..offset.unwrap_or(0));
            elements.truncate(len);
            elements
        } else {
            self.as_standard_layout()
                .into_owned()
                .into_raw_vec_and_offset()
                .0
        };
        Ok(T::into_array_bytes(data_type, elements)?.into_owned())
    }
}

impl<'a> IntoArrayBytes<'a> for super::Tensor<'a> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError> {
        let (bytes, tensor_data_type, _) = self.into_parts();
        if tensor_data_type != *data_type {
            return Err(ElementError::IncompatibleElementType);
        }
        Ok(ArrayBytes::from(bytes))
    }
}

impl<'a> IntoArrayBytes<'a> for &'a super::Tensor<'_> {
    fn into_array_bytes(self, data_type: &DataType) -> Result<ArrayBytes<'a>, ElementError> {
        let (bytes, tensor_data_type, _) = self.as_parts();
        if tensor_data_type != data_type {
            return Err(ElementError::IncompatibleElementType);
        }
        Ok(ArrayBytes::from(bytes))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "ndarray")]
    #[test]
    fn into_array_bytes_sliced_owned_ndarray() {
        use super::*;
        use crate::array::data_type;
        use ndarray::{Array2, s};

        // Rows 1..3 of a C-contiguous 4x3 array
        let mut array = Array2::from_shape_vec((4, 3), (0u8..12).collect()).unwrap();
        array.slice_collapse(s![1..3, ..]);
        assert!(array.is_standard_layout());
        assert_eq!(
            array.into_array_bytes(&data_type::uint8()).unwrap(),
            ArrayBytes::new_flen(vec![3u8, 4, 5, 6, 7, 8])
        );
    }
}

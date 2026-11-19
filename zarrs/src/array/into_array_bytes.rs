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

    /// Validate that `self` has a shape compatible with a target of `shape`.
    ///
    /// Only types with an inherent multidimensional layout (e.g. `ndarray` arrays and [`Tensor`](super::Tensor)) validate their shape here.
    /// Types without a shape of their own (e.g. slices, [`Vec`], [`ArrayBytes`], and `&bytes::Bytes`) accept any shape; their element or byte count is validated against the [`DataType`] via [`ArrayBytes::validate`] after conversion.
    ///
    /// # Errors
    /// Returns [`ElementError::IncompatibleShape`] if `self` has a layout with a shape other than `shape`.
    fn validate_shape(&self, _shape: &[u64]) -> Result<(), ElementError> {
        Ok(())
    }
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
fn validate_ndarray_shape<S, D>(
    array: &ndarray::ArrayBase<S, D>,
    shape: &[u64],
) -> Result<(), ElementError>
where
    S: ndarray::Data,
    D: ndarray::Dimension,
{
    let array_shape = array.shape().iter().map(|&size| size as u64);
    if !array_shape.clone().eq(shape.iter().copied()) {
        return Err(ElementError::IncompatibleShape {
            expected: shape.to_vec(),
            actual: array_shape.collect(),
        });
    }
    Ok(())
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

    fn validate_shape(&self, shape: &[u64]) -> Result<(), ElementError> {
        validate_ndarray_shape(self, shape)
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

    fn validate_shape(&self, shape: &[u64]) -> Result<(), ElementError> {
        validate_ndarray_shape(self, shape)
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

    fn validate_shape(&self, shape: &[u64]) -> Result<(), ElementError> {
        if self.shape() == shape {
            Ok(())
        } else {
            Err(ElementError::IncompatibleShape {
                expected: shape.to_vec(),
                actual: self.shape().to_vec(),
            })
        }
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

    fn validate_shape(&self, shape: &[u64]) -> Result<(), ElementError> {
        (*self).validate_shape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Tensor, data_type};

    #[test]
    fn into_array_bytes_validate_shape() {
        let elements = vec![0u8, 1, 2, 3, 4, 5];

        // Types without a layout accept any shape
        assert!(elements.validate_shape(&[2, 3]).is_ok());
        assert!(elements.validate_shape(&[6]).is_ok());
        assert!(elements.as_slice().validate_shape(&[3, 2]).is_ok());

        // Tensor validates its shape
        let tensor = Tensor::new(elements.clone(), data_type::uint8(), vec![2, 3]);
        let tensor_ref = &tensor;
        assert!(tensor.validate_shape(&[2, 3]).is_ok());
        assert!(<&Tensor<'_> as IntoArrayBytes>::validate_shape(&tensor_ref, &[2, 3]).is_ok());
        assert!(matches!(
            tensor.validate_shape(&[6]),
            Err(ElementError::IncompatibleShape { expected, actual })
                if expected == [6] && actual == [2, 3]
        ));
        assert!(matches!(
            <&Tensor<'_> as IntoArrayBytes>::validate_shape(&tensor_ref, &[3, 2]),
            Err(ElementError::IncompatibleShape { expected, actual })
                if expected == [3, 2] && actual == [2, 3]
        ));

        #[cfg(feature = "ndarray")]
        {
            let array = ndarray::Array2::from_shape_vec((2, 3), elements).unwrap();
            let array_ref = &array;
            assert!(array.validate_shape(&[2, 3]).is_ok());
            assert!(
                <&ndarray::Array2<u8> as IntoArrayBytes>::validate_shape(&array_ref, &[2, 3])
                    .is_ok()
            );
            assert!(matches!(
                array.validate_shape(&[6]),
                Err(ElementError::IncompatibleShape { expected, actual })
                    if expected == [6] && actual == [2, 3]
            ));
            assert!(matches!(
                <&ndarray::Array2<u8> as IntoArrayBytes>::validate_shape(&array_ref, &[3, 2]),
                Err(ElementError::IncompatibleShape { expected, actual })
                    if expected == [3, 2] && actual == [2, 3]
            ));
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn into_array_bytes_sliced_owned_ndarray() {
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

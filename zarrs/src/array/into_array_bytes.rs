//! The [`IntoArrayBytes`] trait for converting input types into [`ArrayBytes`] for storage.

use super::element::Element;
use super::{ArrayBytes, CowBytes, DataType, ElementError, ElementOrder};

/// A trait for types that can be converted into [`ArrayBytes`] for storage.
pub trait IntoArrayBytes<'a> {
    /// Convert `self` into [`ArrayBytes`] with elements in `order`.
    ///
    /// Types without a layout of their own (e.g. slices, [`Vec`] and [`ArrayBytes`]) are taken to hold their elements in `order`.
    /// Types with a layout (`ndarray` arrays and [`Tensor`](super::Tensor)) are reordered if necessary.
    ///
    /// # Arguments
    /// * `data_type` - The data type of the array.
    /// * `order` - The order of the elements in the returned bytes.
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the conversion fails.
    fn into_array_bytes(
        self,
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError>;

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

    /// Return the element order of `self`, or `order` if `self` has no layout of its own.
    ///
    /// Types without a layout of their own (e.g. slices, [`Vec`], [`ArrayBytes`], and `&bytes::Bytes`) use `order`.
    /// Types with a layout (`ndarray` arrays and [`Tensor`](super::Tensor)) use their own order.
    fn element_order(&self, order: ElementOrder) -> ElementOrder {
        order
    }
}

impl<'a> IntoArrayBytes<'a> for ArrayBytes<'a> {
    fn into_array_bytes(
        self,
        _data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Ok(self)
    }
}

/// Store raw bytes without a copy where possible.
///
/// The chunk is written directly from `self` if the codec chain passes its input through unchanged,
/// which requires a fixed length data type, native endianness, and no bytes-to-bytes codecs.
/// Any other configuration encodes into a new buffer, as usual.
impl IntoArrayBytes<'_> for &bytes::Bytes {
    fn into_array_bytes(
        self,
        _data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Ok(ArrayBytes::new_flen(CowBytes::Shared(self.clone())))
    }
}

impl<T: Element> IntoArrayBytes<'static> for Vec<T> {
    fn into_array_bytes(
        self,
        data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        T::into_array_bytes(data_type, self)
    }
}

impl<'a, T: Element> IntoArrayBytes<'a> for &'a Vec<T> {
    fn into_array_bytes(
        self,
        data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError> {
        T::to_array_bytes(data_type, self)
    }
}

impl<'a, T: Element> IntoArrayBytes<'a> for &'a [T] {
    fn into_array_bytes(
        self,
        data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError> {
        T::to_array_bytes(data_type, self)
    }
}

impl<'a, T: Element, const N: usize> IntoArrayBytes<'a> for &'a [T; N] {
    fn into_array_bytes(
        self,
        data_type: &DataType,
        _order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError> {
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
fn ndarray_element_order<S, D>(array: &ndarray::ArrayBase<S, D>) -> ElementOrder
where
    S: ndarray::Data,
    D: ndarray::Dimension,
{
    if !array.is_standard_layout() && array.t().is_standard_layout() {
        ElementOrder::F
    } else {
        ElementOrder::C
    }
}

#[cfg(feature = "ndarray")]
impl<'a, T, S, D> IntoArrayBytes<'a> for &'a ndarray::ArrayBase<S, D>
where
    T: Element,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
    fn into_array_bytes(
        self,
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError> {
        // The F order of an array is the C order of its reversed axes
        let view = match order {
            ElementOrder::C => self.view(),
            ElementOrder::F => self.t(),
        };
        if let Some(elements) = view.to_slice() {
            T::to_array_bytes(data_type, elements)
        } else {
            let elements = view
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

    fn element_order(&self, _order: ElementOrder) -> ElementOrder {
        ndarray_element_order(self)
    }
}

#[cfg(feature = "ndarray")]
impl<T: Element, D: ndarray::Dimension> IntoArrayBytes<'static> for ndarray::Array<T, D> {
    fn into_array_bytes(
        self,
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        // The F order of an array is the C order of its reversed axes
        let array = match order {
            ElementOrder::C => self,
            ElementOrder::F => self.reversed_axes(),
        };
        let elements = if array.is_standard_layout() {
            // The elements are contiguous from the offset, but an array sliced in place keeps its entire backing vector
            let len = array.len();
            let (mut elements, offset) = array.into_raw_vec_and_offset();
            elements.drain(..offset.unwrap_or(0));
            elements.truncate(len);
            elements
        } else {
            array
                .as_standard_layout()
                .into_owned()
                .into_raw_vec_and_offset()
                .0
        };
        Ok(T::into_array_bytes(data_type, elements)?.into_owned())
    }

    fn validate_shape(&self, shape: &[u64]) -> Result<(), ElementError> {
        validate_ndarray_shape(self, shape)
    }

    fn element_order(&self, _order: ElementOrder) -> ElementOrder {
        ndarray_element_order(self)
    }
}

impl<'a> IntoArrayBytes<'a> for super::Tensor<'a> {
    fn into_array_bytes(
        self,
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError> {
        let (bytes, tensor_data_type, shape, tensor_order) = self.into_parts();
        if tensor_data_type != *data_type {
            return Err(ElementError::IncompatibleElementType);
        }
        reorder_tensor_bytes(
            ArrayBytes::from(bytes),
            data_type,
            &shape,
            tensor_order,
            order,
        )
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

    fn element_order(&self, _order: ElementOrder) -> ElementOrder {
        self.order()
    }
}

impl<'a> IntoArrayBytes<'a> for &'a super::Tensor<'_> {
    fn into_array_bytes(
        self,
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<ArrayBytes<'a>, ElementError> {
        let (bytes, tensor_data_type, shape, tensor_order) = self.as_parts();
        if tensor_data_type != data_type {
            return Err(ElementError::IncompatibleElementType);
        }
        reorder_tensor_bytes(
            ArrayBytes::from(bytes),
            data_type,
            shape,
            tensor_order,
            order,
        )
    }

    fn validate_shape(&self, shape: &[u64]) -> Result<(), ElementError> {
        (*self).validate_shape(shape)
    }

    fn element_order(&self, order: ElementOrder) -> ElementOrder {
        (*self).element_order(order)
    }
}

/// Reorder the bytes of a tensor with `shape` from its order `from` to `to`.
fn reorder_tensor_bytes<'a>(
    bytes: ArrayBytes<'a>,
    data_type: &DataType,
    shape: &[u64],
    from: ElementOrder,
    to: ElementOrder,
) -> Result<ArrayBytes<'a>, ElementError> {
    super::array_bytes_internal::reorder(bytes, data_type, shape, from, to)
        .map_err(|err| ElementError::Other(err.to_string()))
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

    #[test]
    fn into_array_bytes_order() {
        // A 2x3 array with elements 0..6 in C order
        let c_order = vec![0u8, 1, 2, 3, 4, 5];
        let f_order = vec![0u8, 3, 1, 4, 2, 5];
        let uint8 = data_type::uint8();

        // Types without a layout are taken to be in the requested order
        assert_eq!(
            c_order.as_slice().element_order(ElementOrder::F),
            ElementOrder::F
        );
        assert_eq!(
            c_order
                .as_slice()
                .into_array_bytes(&uint8, ElementOrder::F)
                .unwrap(),
            ArrayBytes::new_flen(c_order.clone())
        );

        // Tensors are reordered if their order differs
        for (tensor_order, elements) in [(ElementOrder::C, &c_order), (ElementOrder::F, &f_order)] {
            for (order, expected) in [(ElementOrder::C, &c_order), (ElementOrder::F, &f_order)] {
                let tensor = Tensor::new(elements.clone(), uint8.clone(), vec![2, 3])
                    .with_order(tensor_order);
                let tensor_ref = &tensor;
                assert_eq!(tensor.element_order(order), tensor_order);
                assert_eq!(
                    <&Tensor<'_> as IntoArrayBytes>::element_order(&tensor_ref, order),
                    tensor_order
                );
                assert_eq!(
                    tensor_ref.into_array_bytes(&uint8, order).unwrap(),
                    ArrayBytes::new_flen(expected.clone())
                );
                assert_eq!(
                    tensor.into_array_bytes(&uint8, order).unwrap(),
                    ArrayBytes::new_flen(expected.clone())
                );
            }
        }

        #[cfg(feature = "ndarray")]
        {
            let c_array = ndarray::array![[0u8, 1, 2], [3, 4, 5]];
            let f_array = c_array
                .t()
                .as_standard_layout()
                .into_owned()
                .reversed_axes();
            assert_eq!(c_array, f_array);
            assert_eq!(c_array.element_order(ElementOrder::F), ElementOrder::C);
            assert_eq!(f_array.element_order(ElementOrder::C), ElementOrder::F);
            let f_array_ref = &f_array;
            assert_eq!(
                <&ndarray::Array2<u8> as IntoArrayBytes>::element_order(
                    &f_array_ref,
                    ElementOrder::C
                ),
                ElementOrder::F
            );
            for array in [&c_array, &f_array] {
                for (order, expected) in [(ElementOrder::C, &c_order), (ElementOrder::F, &f_order)]
                {
                    assert_eq!(
                        array.into_array_bytes(&uint8, order).unwrap(),
                        ArrayBytes::new_flen(expected.clone())
                    );
                    assert_eq!(
                        array.clone().into_array_bytes(&uint8, order).unwrap(),
                        ArrayBytes::new_flen(expected.clone())
                    );
                }
            }
            // An array already in the requested order is borrowed
            let ArrayBytes::Fixed(bytes) = f_array_ref
                .into_array_bytes(&uint8, ElementOrder::F)
                .unwrap()
            else {
                panic!()
            };
            assert_eq!(bytes.as_ptr(), f_array.as_ptr());
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn into_array_bytes_sliced_owned_ndarray() {
        use ndarray::{Array2, ShapeBuilder, s};
        let uint8 = data_type::uint8();
        let expected = ArrayBytes::new_flen(vec![3u8, 4, 5, 6, 7, 8]);

        // Rows 1..3 of a C-contiguous 4x3 array
        let mut c_array = Array2::from_shape_vec((4, 3), (0u8..12).collect()).unwrap();
        c_array.slice_collapse(s![1..3, ..]);
        assert!(c_array.is_standard_layout());
        assert_eq!(
            c_array.into_array_bytes(&uint8, ElementOrder::C).unwrap(),
            expected
        );

        // Columns 1..3 of an F-contiguous 3x4 array
        let mut f_array = Array2::from_shape_vec((3, 4).f(), (0u8..12).collect()).unwrap();
        f_array.slice_collapse(s![.., 1..3]);
        assert!(f_array.t().is_standard_layout());
        assert_eq!(
            f_array.into_array_bytes(&uint8, ElementOrder::F).unwrap(),
            expected
        );
    }
}

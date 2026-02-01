use std::mem::ManuallyDrop;

use ElementError::IncompatibleElementType as IET;
use itertools::Itertools;

use crate::array::data_type;

use super::{
    ArrayBytes, ArrayBytesOffsets, DataType, convert_from_bytes_slice, transmute_to_bytes,
    transmute_to_bytes_vec,
};

mod error;
mod numpy;

pub use error::ElementError;

/// A trait representing an array element type.
pub trait Element: Sized + Clone {
    /// Validate the data type.
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError>;

    /// Convert a slice of elements into [`ArrayBytes`].
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError>;

    /// Convert a vector of elements into [`ArrayBytes`].
    ///
    /// Avoids an extra copy compared to `to_array_bytes` when possible.
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError>;
}

/// A trait representing an owned array element type.
pub trait ElementOwned: Element {
    /// Convert bytes into a [`Vec<ElementOwned>`].
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError>;
}

impl Element for bool {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        data_type
            .is::<data_type::BoolDataType>()
            .then_some(())
            .ok_or(IET)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes(elements).into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes_vec(elements).into())
    }
}

impl ElementOwned for bool {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        let elements_u8 = convert_from_bytes_slice::<u8>(&bytes);
        if elements_u8.iter().all(|&u| u <= 1) {
            let length: usize = elements_u8.len();
            let capacity: usize = elements_u8.capacity();
            let mut manual_drop_vec = ManuallyDrop::new(elements_u8);
            let vec_ptr: *mut u8 = manual_drop_vec.as_mut_ptr();
            let ptr: *mut Self = vec_ptr.cast::<Self>();
            Ok(unsafe { Vec::from_raw_parts(ptr, length, capacity) })
        } else {
            Err(ElementError::InvalidElementValue)
        }
    }
}

/// Helper macro to implement `Element` for POD (plain old data) types.
/// Uses `TypeId` matching for data type validation.
macro_rules! impl_element_pod {
    ($raw_type:ty, $($data_type:ty),+ $(,)?) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
                let type_id = data_type.as_any().type_id();
                if $( type_id == std::any::TypeId::of::<$data_type>() )||+ {
                    Ok(())
                } else {
                    Err(IET)
                }
            }

            fn to_array_bytes<'a>(
                data_type: &DataType,
                elements: &'a [Self],
            ) -> Result<ArrayBytes<'a>, ElementError> {
                Self::validate_data_type(data_type)?;
                Ok(transmute_to_bytes(elements).into())
            }

            fn into_array_bytes(
                data_type: &DataType,
                elements: Vec<Self>,
            ) -> Result<ArrayBytes<'static>, ElementError> {
                Self::validate_data_type(data_type)?;
                Ok(transmute_to_bytes_vec(elements).into())
            }
        }

        impl ElementOwned for $raw_type {
            fn from_array_bytes(
                data_type: &DataType,
                bytes: ArrayBytes<'_>,
            ) -> Result<Vec<Self>, ElementError> {
                Self::validate_data_type(data_type)?;
                let bytes = bytes.into_fixed()?;
                Ok(convert_from_bytes_slice::<Self>(&bytes))
            }
        }
    };
}

impl_element_pod!(
    i8,
    data_type::Int8DataType,
    data_type::Int4DataType,
    data_type::Int2DataType
);
impl_element_pod!(i16, data_type::Int16DataType);
impl_element_pod!(i32, data_type::Int32DataType);
impl_element_pod!(
    i64,
    data_type::Int64DataType,
    data_type::NumpyDateTime64DataType,
    data_type::NumpyTimeDelta64DataType
);
impl_element_pod!(
    u8,
    data_type::UInt8DataType,
    data_type::UInt4DataType,
    data_type::UInt2DataType
);
impl_element_pod!(u16, data_type::UInt16DataType);
impl_element_pod!(u32, data_type::UInt32DataType);
impl_element_pod!(u64, data_type::UInt64DataType);
impl_element_pod!(half::f16, data_type::Float16DataType);
impl_element_pod!(f32, data_type::Float32DataType);
impl_element_pod!(f64, data_type::Float64DataType);
impl_element_pod!(half::bf16, data_type::BFloat16DataType);
impl_element_pod!(
    num::complex::Complex<half::bf16>,
    data_type::ComplexBFloat16DataType
);
impl_element_pod!(
    num::complex::Complex<half::f16>,
    data_type::ComplexFloat16DataType
);
impl_element_pod!(
    num::complex::Complex32,
    data_type::Complex64DataType,
    data_type::ComplexFloat32DataType
);
impl_element_pod!(
    num::complex::Complex64,
    data_type::Complex128DataType,
    data_type::ComplexFloat64DataType
);

impl<const N: usize> Element for &[u8; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        // RawBits and fixed size equal to N
        if data_type.is::<data_type::RawBitsDataType>() && data_type.fixed_size() == Some(N) {
            Ok(())
        } else {
            Err(IET)
        }
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes: Vec<u8> = elements.iter().flat_map(|i| i.iter()).copied().collect();
        Ok(bytes.into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes: Vec<u8> = elements.into_iter().flatten().copied().collect();
        Ok(bytes.into())
    }
}

impl<const N: usize> Element for [u8; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        // RawBits and fixed size equal to N
        if data_type.is::<data_type::RawBitsDataType>() && data_type.fixed_size() == Some(N) {
            Ok(())
        } else {
            Err(IET)
        }
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes(elements).into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes_vec(elements).into())
    }
}

impl<const N: usize> ElementOwned for [u8; N] {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        Ok(convert_from_bytes_slice::<Self>(&bytes))
    }
}

macro_rules! impl_element_string {
    ($raw_type:ty) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
                data_type
                    .is::<data_type::StringDataType>()
                    .then_some(())
                    .ok_or(IET)
            }

            fn to_array_bytes<'a>(
                data_type: &DataType,
                elements: &'a [Self],
            ) -> Result<ArrayBytes<'a>, ElementError> {
                Self::validate_data_type(data_type)?;

                // Calculate offsets
                let mut len: usize = 0;
                let mut offsets = Vec::with_capacity(elements.len());
                for element in elements {
                    offsets.push(len);
                    len = len.checked_add(element.len()).unwrap();
                }
                offsets.push(len);
                let offsets = unsafe {
                    // SAFETY: The offsets are monotonically increasing.
                    ArrayBytesOffsets::new_unchecked(offsets)
                };

                // Concatenate bytes
                let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap());
                for element in elements {
                    bytes.extend_from_slice(element.as_bytes());
                }
                let array_bytes = unsafe {
                    // SAFETY: The last offset is the length of the bytes.
                    ArrayBytes::new_vlen_unchecked(bytes, offsets)
                };
                Ok(array_bytes)
            }

            fn into_array_bytes(
                data_type: &DataType,
                elements: Vec<Self>,
            ) -> Result<ArrayBytes<'static>, ElementError> {
                Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
            }
        }
    };
}

impl_element_string!(&str);
impl_element_string!(String);

impl ElementOwned for String {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let (bytes, offsets) = bytes.into_variable()?.into_parts();
        let mut elements = Vec::with_capacity(offsets.len().saturating_sub(1));
        for (curr, next) in offsets.iter().tuple_windows() {
            elements.push(
                Self::from_utf8(bytes[*curr..*next].to_vec())
                    .map_err(|_| ElementError::InvalidElementValue)?,
            );
        }
        Ok(elements)
    }
}

macro_rules! impl_element_bytes {
    ($raw_type:ty) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
                data_type
                    .is::<crate::array::data_type::BytesDataType>()
                    .then_some(())
                    .ok_or(IET)
            }

            fn to_array_bytes<'a>(
                data_type: &DataType,
                elements: &'a [Self],
            ) -> Result<ArrayBytes<'a>, ElementError> {
                Self::validate_data_type(data_type)?;

                // Calculate offsets
                let mut len: usize = 0;
                let mut offsets = Vec::with_capacity(elements.len());
                for element in elements {
                    offsets.push(len);
                    len = len.checked_add(element.len()).unwrap();
                }
                offsets.push(len);
                let offsets = unsafe {
                    // SAFETY: The offsets are monotonically increasing.
                    ArrayBytesOffsets::new_unchecked(offsets)
                };

                // Concatenate bytes
                let bytes = elements.concat();

                let array_bytes = unsafe {
                    // SAFETY: The last offset is the length of the bytes.
                    ArrayBytes::new_vlen_unchecked(bytes, offsets)
                };
                Ok(array_bytes)
            }

            fn into_array_bytes(
                data_type: &DataType,
                elements: Vec<Self>,
            ) -> Result<ArrayBytes<'static>, ElementError> {
                Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
            }
        }
    };
}

impl_element_bytes!(&[u8]);
impl_element_bytes!(Vec<u8>);

impl ElementOwned for Vec<u8> {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let (bytes, offsets) = bytes.into_variable()?.into_parts();
        let mut elements = Vec::with_capacity(offsets.len().saturating_sub(1));
        for (curr, next) in offsets.iter().tuple_windows() {
            elements.push(bytes[*curr..*next].to_vec());
        }
        Ok(elements)
    }
}

#[cfg(feature = "float8")]
impl_element_pod!(float8::F8E4M3, data_type::Float8E4M3DataType);

#[cfg(feature = "float8")]
impl_element_pod!(float8::F8E5M2, data_type::Float8E5M2DataType);

#[cfg(feature = "float8")]
impl_element_pod!(
    num::Complex<float8::F8E4M3>,
    data_type::ComplexFloat8E4M3DataType
);

#[cfg(feature = "float8")]
impl_element_pod!(
    num::Complex<float8::F8E5M2>,
    data_type::ComplexFloat8E5M2DataType
);

impl<T> Element for Option<T>
where
    T: Element + Default,
{
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        let opt = data_type.as_optional().ok_or(IET)?;
        T::validate_data_type(opt.data_type())
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;

        let opt = data_type.as_optional().ok_or(IET)?;

        let num_elements = elements.len();

        // Create validity mask - one byte per element
        let mut mask = Vec::with_capacity(num_elements);

        // Create dense data - all elements, using default/zero for None values
        // We need to use a placeholder value for None elements
        let default_value = T::default();
        let mut dense_elements = Vec::with_capacity(num_elements);

        for element in elements {
            if let Some(value) = element {
                mask.push(1u8);
                dense_elements.push(value.clone());
            } else {
                mask.push(0u8);
                dense_elements.push(default_value.clone());
            }
        }

        // Convert all elements (dense) to ArrayBytes
        let data = T::into_array_bytes(opt.data_type(), dense_elements)?.into_owned();

        // Create optional ArrayBytes by adding mask to the data
        Ok(data.with_optional_mask(mask))
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

impl<T> ElementOwned for Option<T>
where
    T: ElementOwned + Clone + Default,
{
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;

        let opt = data_type.as_optional().ok_or(IET)?;

        // Extract mask and dense data from optional ArrayBytes
        let optional_bytes = bytes.into_optional()?;
        let (data, mask) = optional_bytes.into_parts();

        // Convert the dense inner data to a Vec<T>
        let dense_values = T::from_array_bytes(opt.data_type(), *data)?;

        // Build the result vector using mask to determine Some vs None
        let mut elements = Vec::with_capacity(mask.len());
        for (i, &mask_byte) in mask.iter().enumerate() {
            if mask_byte == 0 {
                // None value
                elements.push(None);
            } else {
                // Some value - take from dense data
                if i >= dense_values.len() {
                    return Err(ElementError::Other(format!(
                        "Not enough dense values for mask at index {i}"
                    )));
                }
                elements.push(Some(dense_values[i].clone()));
            }
        }

        Ok(elements)
    }
}

use std::mem::ManuallyDrop;

use itertools::Itertools;
use ArrayError::IncompatibleElementType as IET;

use super::{
    convert_from_bytes_slice, transmute_to_bytes, ArrayBytes, ArrayBytesOffsets, ArrayError,
    DataType,
};

mod numpy;

/// A trait representing an array element type.
pub trait Element: Sized + Clone {
    /// Validate the data type.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the data type is incompatible with [`Element`].
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError>;

    /// Convert a slice of elements into [`ArrayBytes`].
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the data type is incompatible with [`Element`].
    fn into_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ArrayError>;
}

/// A trait representing an owned array element type.
pub trait ElementOwned: Element {
    /// Convert bytes into a [`Vec<ElementOwned>`].
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the data type is incompatible with [`Element`].
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError>;
}

/// A marker trait for a fixed length element.
pub trait ElementFixedLength {}

impl ElementFixedLength for bool {}
impl ElementFixedLength for u8 {}
impl ElementFixedLength for u16 {}
impl ElementFixedLength for u32 {}
impl ElementFixedLength for u64 {}
impl ElementFixedLength for i8 {}
impl ElementFixedLength for i16 {}
impl ElementFixedLength for i32 {}
impl ElementFixedLength for i64 {}
impl ElementFixedLength for half::f16 {}
impl ElementFixedLength for half::bf16 {}
impl ElementFixedLength for f32 {}
impl ElementFixedLength for f64 {}
impl ElementFixedLength for num::complex::Complex<half::bf16> {}
impl ElementFixedLength for num::complex::Complex<half::f16> {}
impl ElementFixedLength for num::complex::Complex32 {}
impl ElementFixedLength for num::complex::Complex64 {}
impl<const N: usize> ElementFixedLength for [u8; N] {}
impl<T: ElementFixedLength> ElementFixedLength for Option<T> {}

impl Element for bool {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        (data_type == &DataType::Bool).then_some(()).ok_or(IET)
    }

    fn into_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes(elements).into())
    }
}

impl ElementOwned for bool {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
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
            Err(ArrayError::InvalidElementValue)
        }
    }
}

macro_rules! impl_element_pod {
    ($raw_type:ty, $pattern:pat $(if $guard:expr)? $(,)?) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
                if matches!(data_type, $pattern) {
                    Ok(())
                } else {
                    Err(IET)
                }
            }

            fn into_array_bytes<'a>(
                data_type: &DataType,
                elements: &'a [Self],
            ) -> Result<ArrayBytes<'a>, ArrayError> {
                Self::validate_data_type(data_type)?;
                Ok(transmute_to_bytes(elements).into())
            }
        }

        impl ElementOwned for $raw_type {
            fn from_array_bytes(
                data_type: &DataType,
                bytes: ArrayBytes<'_>,
            ) -> Result<Vec<Self>, ArrayError> {
                Self::validate_data_type(data_type)?;
                let bytes = bytes.into_fixed()?;
                Ok(convert_from_bytes_slice::<Self>(&bytes))
            }
        }
    };
}

impl_element_pod!(i8, DataType::Int8 | DataType::Int4 | DataType::Int2);
impl_element_pod!(i16, DataType::Int16);
impl_element_pod!(i32, DataType::Int32);
impl_element_pod!(
    i64,
    DataType::Int64
        | DataType::NumpyDateTime64 {
            unit: _,
            scale_factor: _
        }
        | DataType::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _
        }
);
impl_element_pod!(u8, DataType::UInt8 | DataType::UInt4 | DataType::UInt2);
impl_element_pod!(u16, DataType::UInt16);
impl_element_pod!(u32, DataType::UInt32);
impl_element_pod!(u64, DataType::UInt64);
impl_element_pod!(half::f16, DataType::Float16);
impl_element_pod!(f32, DataType::Float32);
impl_element_pod!(f64, DataType::Float64);
impl_element_pod!(half::bf16, DataType::BFloat16);
impl_element_pod!(num::complex::Complex<half::bf16>, DataType::ComplexBFloat16);
impl_element_pod!(num::complex::Complex<half::f16>, DataType::ComplexFloat16);
impl_element_pod!(
    num::complex::Complex32,
    DataType::Complex64 | DataType::ComplexFloat32
);
impl_element_pod!(
    num::complex::Complex64,
    DataType::Complex128 | DataType::ComplexFloat64
);

impl<const N: usize> Element for &[u8; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        if let DataType::RawBits(n) = data_type {
            (*n == N).then_some(()).ok_or(IET)
        } else {
            Err(IET)
        }
    }

    fn into_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let bytes: Vec<u8> = elements.iter().flat_map(|i| i.iter()).copied().collect();
        Ok(bytes.into())
    }
}

impl<const N: usize> Element for [u8; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        if let DataType::RawBits(n) = data_type {
            (*n == N).then_some(()).ok_or(IET)
        } else {
            Err(IET)
        }
    }

    fn into_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes(elements).into())
    }
}

impl<const N: usize> ElementOwned for [u8; N] {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        Ok(convert_from_bytes_slice::<Self>(&bytes))
    }
}

macro_rules! impl_element_string {
    ($raw_type:ty) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
                (data_type == &DataType::String).then_some(()).ok_or(IET)
            }

            fn into_array_bytes<'a>(
                data_type: &DataType,
                elements: &'a [Self],
            ) -> Result<ArrayBytes<'a>, ArrayError> {
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
        }
    };
}

impl_element_string!(&str);
impl_element_string!(String);

impl ElementOwned for String {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let (bytes, offsets) = bytes.into_variable()?;
        let mut elements = Vec::with_capacity(offsets.len().saturating_sub(1));
        for (curr, next) in offsets.iter().tuple_windows() {
            elements.push(
                Self::from_utf8(bytes[*curr..*next].to_vec())
                    .map_err(|_| ArrayError::InvalidElementValue)?,
            );
        }
        Ok(elements)
    }
}

macro_rules! impl_element_bytes {
    ($raw_type:ty) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
                (data_type == &DataType::Bytes).then_some(()).ok_or(IET)
            }

            fn into_array_bytes<'a>(
                data_type: &DataType,
                elements: &'a [Self],
            ) -> Result<ArrayBytes<'a>, ArrayError> {
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
        }
    };
}

impl_element_bytes!(&[u8]);
impl_element_bytes!(Vec<u8>);

impl ElementOwned for Vec<u8> {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let (bytes, offsets) = bytes.into_variable()?;
        let mut elements = Vec::with_capacity(offsets.len().saturating_sub(1));
        for (curr, next) in offsets.iter().tuple_windows() {
            elements.push(bytes[*curr..*next].to_vec());
        }
        Ok(elements)
    }
}

#[cfg(feature = "float8")]
impl_element_pod!(float8::F8E4M3, DataType::Float8E4M3);

#[cfg(feature = "float8")]
impl_element_pod!(float8::F8E5M2, DataType::Float8E5M2);

#[cfg(feature = "float8")]
impl_element_pod!(num::Complex<float8::F8E4M3>, DataType::ComplexFloat8E4M3);

#[cfg(feature = "float8")]
impl_element_pod!(num::Complex<float8::F8E5M2>, DataType::ComplexFloat8E5M2);

impl<T> Element for Option<T>
where
    T: Element + Default,
{
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        if let DataType::Optional(inner_data_type) = data_type {
            T::validate_data_type(inner_data_type)
        } else {
            Err(IET)
        }
    }

    fn into_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;

        let DataType::Optional(inner_data_type) = data_type else {
            return Err(IET);
        };

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
        let data = T::into_array_bytes(inner_data_type, &dense_elements)?.into_owned();

        // Create optional ArrayBytes by adding mask to the data
        Ok(data.with_optional_mask(mask))
    }
}

impl<T> ElementOwned for Option<T>
where
    T: ElementOwned + Clone + Default,
{
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;

        let DataType::Optional(inner_data_type) = data_type else {
            return Err(IET);
        };

        // Extract mask and dense data from optional ArrayBytes
        let optional_bytes = bytes.into_optional().map_err(|e| {
            ArrayError::Other(format!(
                "Expected optional ArrayBytes (with mask) for optional data type: {e}"
            ))
        })?;
        let (data, mask) = optional_bytes.into_parts();

        // Convert the dense inner data to a Vec<T>
        let dense_values = T::from_array_bytes(inner_data_type, *data)?;

        // Build the result vector using mask to determine Some vs None
        let mut elements = Vec::with_capacity(mask.len());
        for (i, &mask_byte) in mask.iter().enumerate() {
            if mask_byte == 0 {
                // None value
                elements.push(None);
            } else {
                // Some value - take from dense data
                if i >= dense_values.len() {
                    return Err(ArrayError::Other(format!(
                        "Not enough dense values for mask at index {i}"
                    )));
                }
                elements.push(Some(dense_values[i].clone()));
            }
        }

        Ok(elements)
    }
}

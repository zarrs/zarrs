use std::any::TypeId;

use crate::array::{
    ArrayBytes, DataType, convert_from_bytes_slice, transmute_to_bytes, transmute_to_bytes_vec,
};

use super::{Element, ElementError, ElementOwned};

use ElementError::IncompatibleElementType as IET;

/// Helper macro to implement `Element` for POD (plain old data) types.
/// Uses the data type's `compatible_element_types()` method for validation.
macro_rules! impl_element_pod {
    ($raw_type:ty) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
                let my_type_id = TypeId::of::<$raw_type>();
                if data_type.compatible_element_types().contains(&my_type_id) {
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

impl_element_pod!(i8);
impl_element_pod!(i16);
impl_element_pod!(i32);
impl_element_pod!(i64);
impl_element_pod!(u8);
impl_element_pod!(u16);
impl_element_pod!(u32);
impl_element_pod!(u64);
impl_element_pod!(half::f16);
impl_element_pod!(f32);
impl_element_pod!(f64);
impl_element_pod!(half::bf16);
impl_element_pod!(num::complex::Complex<half::bf16>);
impl_element_pod!(num::complex::Complex<half::f16>);
impl_element_pod!(num::complex::Complex32);
impl_element_pod!(num::complex::Complex64);

#[cfg(feature = "float8")]
impl_element_pod!(float8::F8E4M3);

#[cfg(feature = "float8")]
impl_element_pod!(float8::F8E5M2);

#[cfg(feature = "float8")]
impl_element_pod!(num::Complex<float8::F8E4M3>);

#[cfg(feature = "float8")]
impl_element_pod!(num::Complex<float8::F8E5M2>);

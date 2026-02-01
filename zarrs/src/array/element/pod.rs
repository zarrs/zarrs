use crate::array::{
    ArrayBytes, DataType, convert_from_bytes_slice, data_type, transmute_to_bytes,
    transmute_to_bytes_vec,
};

use super::{Element, ElementError, ElementOwned};

use ElementError::IncompatibleElementType as IET;

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

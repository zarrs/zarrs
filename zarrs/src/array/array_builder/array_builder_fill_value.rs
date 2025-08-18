use half::{bf16, f16};
use zarrs_data_type::FillValue;
use zarrs_metadata::v3::{
    FillValueMetadataV3, ZARR_NAN_BF16, ZARR_NAN_F16, ZARR_NAN_F32, ZARR_NAN_F64,
};

use crate::array::{ArrayCreateError, DataType};
use serde_json::Number;

/// An input that can be mapped to a fill value.
#[derive(Debug, PartialEq, Clone)]
pub struct ArrayBuilderFillValue(ArrayBuilderFillValueImpl);

#[derive(Debug, PartialEq, Clone)]
enum ArrayBuilderFillValueImpl {
    FillValue(FillValue),
    Metadata(FillValueMetadataV3),
}

impl ArrayBuilderFillValue {
    pub(crate) fn to_fill_value(
        &self,
        data_type: &DataType,
    ) -> Result<FillValue, ArrayCreateError> {
        match &self.0 {
            ArrayBuilderFillValueImpl::Metadata(metadata) => {
                Ok(data_type.fill_value_from_metadata(metadata)?)
            }
            ArrayBuilderFillValueImpl::FillValue(fill_value) => Ok(fill_value.clone()),
        }
    }
}

impl From<FillValue> for ArrayBuilderFillValue {
    fn from(value: FillValue) -> Self {
        Self(ArrayBuilderFillValueImpl::FillValue(value))
    }
}

impl From<FillValueMetadataV3> for ArrayBuilderFillValue {
    fn from(value: FillValueMetadataV3) -> Self {
        Self(ArrayBuilderFillValueImpl::Metadata(value))
    }
}

// TODO needs Rust specialisation, then everything below can be removed
// impl<T: Into<FillValueMetadataV3>> From<T> for ArrayBuilderFillValue {
//     fn from(value: T) -> Self {
//         Self(ArrayBuilderFillValueImpl::MetadataV3(value.into()))
//     }
// }

impl From<&[u8]> for ArrayBuilderFillValue {
    fn from(value: &[u8]) -> Self {
        Self(ArrayBuilderFillValueImpl::Metadata(
            FillValueMetadataV3::Array(
                value
                    .iter()
                    .map(|v| FillValueMetadataV3::from(*v))
                    .collect(),
            ),
        ))
    }
}

impl From<Vec<u8>> for ArrayBuilderFillValue {
    fn from(value: Vec<u8>) -> Self {
        Self(ArrayBuilderFillValueImpl::Metadata(
            FillValueMetadataV3::Array(value.into_iter().map(FillValueMetadataV3::from).collect()),
        ))
    }
}

impl From<&str> for ArrayBuilderFillValue {
    fn from(value: &str) -> Self {
        Self(ArrayBuilderFillValueImpl::Metadata(
            FillValueMetadataV3::String(value.to_string()),
        ))
    }
}

impl<const N: usize> From<[FillValueMetadataV3; N]> for ArrayBuilderFillValue {
    fn from(value: [FillValueMetadataV3; N]) -> Self {
        Self(ArrayBuilderFillValueImpl::Metadata(
            FillValueMetadataV3::Array(value.to_vec()),
        ))
    }
}

impl<const N: usize> From<&[FillValueMetadataV3; N]> for ArrayBuilderFillValue {
    fn from(value: &[FillValueMetadataV3; N]) -> Self {
        Self(ArrayBuilderFillValueImpl::Metadata(
            FillValueMetadataV3::Array(value.to_vec()),
        ))
    }
}

macro_rules! impl_from_for_int_fill_value_metadata_v3 {
    ($($t:ty),*) => {
        $(
            impl From<$t> for ArrayBuilderFillValue {
                fn from(value: $t) -> Self {
                    Self(ArrayBuilderFillValueImpl::Metadata(FillValueMetadataV3::Number(Number::from(value))))
                }
            }
        )*
    };
}

impl_from_for_int_fill_value_metadata_v3!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! impl_from_for_float_fill_value_metadata_v3 {
    ($type:ty, $nan_value:expr, $value_conversion:expr) => {
        impl From<$type> for ArrayBuilderFillValue {
            fn from(value: $type) -> Self {
                Self(ArrayBuilderFillValueImpl::Metadata(
                    if value.is_infinite() && value.is_sign_positive() {
                        FillValueMetadataV3::String("Infinity".to_string())
                    } else if value.is_infinite() && value.is_sign_negative() {
                        FillValueMetadataV3::String("-Infinity".to_string())
                    } else if value.to_bits() == $nan_value.to_bits() {
                        FillValueMetadataV3::String("NaN".to_string())
                    } else if value.is_nan() {
                        FillValueMetadataV3::String(bytes_to_hex_string(&value.to_be_bytes()))
                    } else {
                        FillValueMetadataV3::Number(
                            Number::from_f64($value_conversion(value))
                                .expect("already checked finite"),
                        )
                    },
                ))
            }
        }
    };
}

impl_from_for_float_fill_value_metadata_v3!(bf16, ZARR_NAN_BF16, f64::from);
impl_from_for_float_fill_value_metadata_v3!(f16, ZARR_NAN_F16, f64::from);
impl_from_for_float_fill_value_metadata_v3!(f32, ZARR_NAN_F32, f64::from);
impl_from_for_float_fill_value_metadata_v3!(f64, ZARR_NAN_F64, |v| v);

fn bytes_to_hex_string(v: &[u8]) -> String {
    let mut string = String::with_capacity(2 + v.len() * 2);
    string.push('0');
    string.push('x');
    for byte in v {
        string.push(char::from_digit((byte / 16).into(), 16).unwrap());
        string.push(char::from_digit((byte % 16).into(), 16).unwrap());
    }
    string
}

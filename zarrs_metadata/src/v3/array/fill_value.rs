//! Zarr V3 fill value metadata.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#fill-value>.
//!
//! Fill values metadata is serialised/deserialised into [`FillValueMetadataV3`].
//!
//! The interpretation of fill values is data type dependent.

use std::collections::HashMap;

use derive_more::From;
use half::{bf16, f16};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Number;

use super::nan_representations::{ZARR_NAN_BF16, ZARR_NAN_F16, ZARR_NAN_F32, ZARR_NAN_F64};

/// Zarr V3 fill value metadata.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, From)]
#[serde(untagged)]
pub enum FillValueMetadataV3 {
    /// Represents a JSON null value.
    Null,
    /// Represents a JSON boolean.
    Bool(bool),
    /// Represents a finite JSON number, whether integer or floating point.
    Number(Number),
    /// Represents a JSON string. This includes hex strings and non-finite float representations.
    String(String),
    /// Represents a JSON array.
    Array(Vec<FillValueMetadataV3>),
    /// Represents a JSON object.
    Object(HashMap<String, FillValueMetadataV3>),
}

impl std::fmt::Display for FillValueMetadataV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).expect("JSON serialisable")
        )
    }
}

impl FillValueMetadataV3 {
    /// Returns true if the value is a `null`. Returns `false` otherwise.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// If the value is a Null, returns (). Returns [`None`] otherwise.
    #[must_use]
    pub fn as_null(&self) -> Option<()> {
        match *self {
            Self::Null => Some(()),
            _ => None,
        }
    }

    /// If the value is a string, returns the associated `str`. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(string) => Some(string),
            _ => None,
        }
    }

    /// Returns a vector of bytes if the value is an array of integers in [0, 255].
    #[must_use]
    pub fn as_bytes(&self) -> Option<Vec<u8>> {
        match self {
            Self::Array(fvs) => fvs
                .iter()
                .map(|fv| u8::try_from(fv.as_u64()?).ok())
                .collect(),
            _ => None,
        }
    }

    /// If the value is an array, returns the associated elements. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_array(&self) -> Option<&[FillValueMetadataV3]> {
        match self {
            Self::Array(array) => Some(array),
            _ => None,
        }
    }

    /// If the value is an object, returns the associated [`HashMap`]. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_object(&self) -> Option<HashMap<String, FillValueMetadataV3>> {
        match self {
            Self::Object(object) => Some(object.clone()),
            _ => None,
        }
    }

    /// If the value is a Boolean, returns the associated [`bool`]. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(bool) => Some(*bool),
            _ => None,
        }
    }

    /// If the value is a number or non-finite float representation, represent it as [`bf16`] if possible. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_bf16(&self) -> Option<bf16> {
        match self {
            Self::String(string) => match string.as_str() {
                "Infinity" => Some(bf16::INFINITY),
                "-Infinity" => Some(bf16::NEG_INFINITY),
                "NaN" => Some(ZARR_NAN_BF16),
                _ => Some(bf16::from_be_bytes(
                    hex_string_to_be_bytes(string)?.try_into().ok()?,
                )),
            },
            Self::Number(number) => number.as_f64().map(bf16::from_f64),
            _ => None,
        }
    }

    /// If the value is a number or non-finite float representation, represent it as [`struct@f16`] if possible. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_f16(&self) -> Option<f16> {
        match self {
            Self::String(string) => match string.as_str() {
                "Infinity" => Some(f16::INFINITY),
                "-Infinity" => Some(f16::NEG_INFINITY),
                "NaN" => Some(ZARR_NAN_F16),
                _ => Some(f16::from_be_bytes(
                    hex_string_to_be_bytes(string)?.try_into().ok()?,
                )),
            },
            Self::Number(number) => number.as_f64().map(f16::from_f64),
            _ => None,
        }
    }

    /// If the value is a number or non-finite float representation, represent it as [`f32`] if possible. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::String(string) => match string.as_str() {
                "Infinity" => Some(f32::INFINITY),
                "-Infinity" => Some(f32::NEG_INFINITY),
                "NaN" => Some(ZARR_NAN_F32),
                _ => Some(f32::from_be_bytes(
                    hex_string_to_be_bytes(string)?.try_into().ok()?,
                )),
            },
            Self::Number(number) =>
            {
                #[allow(clippy::cast_possible_truncation)]
                number.as_f64().map(|f| f as f32)
            }
            _ => None,
        }
    }

    /// If the value is a number or non-finite float representation, represent it as [`f64`] if possible. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::String(string) => match string.as_str() {
                "Infinity" => Some(f64::INFINITY),
                "-Infinity" => Some(f64::NEG_INFINITY),
                "NaN" => Some(ZARR_NAN_F64),
                _ => Some(f64::from_be_bytes(
                    hex_string_to_be_bytes(string)?.try_into().ok()?,
                )),
            },
            Self::Number(number) => number.as_f64(),
            _ => None,
        }
    }

    /// If the value is an integer, represent it as [`u64`] if possible. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(number) => number.as_u64(),
            _ => None,
        }
    }

    /// If the value is an integer, represent it as [`i64`] if possible. Returns [`None`] otherwise.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(number) => number.as_i64(),
            _ => None,
        }
    }

    /// Convert fill value metadata to a custom structure.
    #[must_use]
    pub fn as_custom<T: DeserializeOwned>(&self) -> Option<T> {
        let custom = serde_json::from_value(serde_json::to_value(self).ok()?).ok()?;
        Some(custom)
    }
}

impl From<&[u8]> for FillValueMetadataV3 {
    fn from(value: &[u8]) -> Self {
        Self::Array(value.iter().map(|v| Self::from(*v)).collect())
    }
}

impl From<Vec<u8>> for FillValueMetadataV3 {
    fn from(value: Vec<u8>) -> Self {
        Self::Array(value.into_iter().map(Self::from).collect())
    }
}

impl From<&str> for FillValueMetadataV3 {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl<const N: usize> From<[FillValueMetadataV3; N]> for FillValueMetadataV3 {
    fn from(value: [FillValueMetadataV3; N]) -> Self {
        Self::Array(value.to_vec())
    }
}

macro_rules! impl_from_for_int_fill_value_metadata_v3 {
    ($($t:ty),*) => {
        $(
            impl From<$t> for FillValueMetadataV3 {
                fn from(value: $t) -> Self {
                    Self::Number(Number::from(value))
                }
            }
        )*
    };
}

impl_from_for_int_fill_value_metadata_v3!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! impl_from_for_float_fill_value_metadata_v3 {
    ($type:ty, $nan_value:expr, $value_conversion:expr) => {
        impl From<$type> for FillValueMetadataV3 {
            fn from(value: $type) -> Self {
                if value.is_infinite() && value.is_sign_positive() {
                    Self::String("Infinity".to_string())
                } else if value.is_infinite() && value.is_sign_negative() {
                    Self::String("-Infinity".to_string())
                } else if value.to_bits() == $nan_value.to_bits() {
                    Self::String("NaN".to_string())
                } else if value.is_nan() {
                    Self::String(bytes_to_hex_string(&value.to_be_bytes()))
                } else {
                    Self::Number(
                        Number::from_f64($value_conversion(value)).expect("already checked finite"),
                    )
                }
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

fn hex_string_to_be_bytes(s: &str) -> Option<Vec<u8>> {
    if s.starts_with("0x") && s.len() % 2 == 0 {
        (2..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect::<Result<Vec<_>, _>>()
            .ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_value_metadata_bool_false() {
        let json = r"false";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert!(!metadata.as_bool().unwrap());
    }

    #[test]
    fn fill_value_metadata_bool_true() {
        let json = r"true";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert!(metadata.as_bool().unwrap());
    }

    #[test]
    fn fill_value_metadata_uint() {
        let json = r"7";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(metadata.as_u64().unwrap(), 7);
    }

    #[test]
    fn fill_value_metadata_int() {
        let json = r"-7";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(metadata.as_i64().unwrap(), -7);
    }

    #[test]
    fn fill_value_metadata_float_number() {
        let json = r"7.5";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(metadata.as_f64().unwrap(), 7.5);
    }

    #[test]
    fn fill_value_metadata_float_infinity() {
        let json = r#""Infinity""#;
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(metadata.as_bf16().unwrap(), bf16::INFINITY);
        assert_eq!(metadata.as_f16().unwrap(), f16::INFINITY);
        assert_eq!(metadata.as_f32().unwrap(), f32::INFINITY);
        assert_eq!(metadata.as_f64().unwrap(), f64::INFINITY);
    }

    #[test]
    fn fill_value_metadata_float_neg_infinity() {
        let json = r#""-Infinity""#;
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(metadata.as_bf16().unwrap(), bf16::NEG_INFINITY);
        assert_eq!(metadata.as_f16().unwrap(), f16::NEG_INFINITY);
        assert_eq!(metadata.as_f32().unwrap(), f32::NEG_INFINITY);
        assert_eq!(metadata.as_f64().unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn fill_value_metadata_float_nan() {
        let json = r#""NaN""#;
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(
            metadata.as_bf16().unwrap().to_bits(),
            ZARR_NAN_BF16.to_bits()
        );
        assert_eq!(metadata.as_f16().unwrap().to_bits(), ZARR_NAN_F16.to_bits());
        assert_eq!(metadata.as_f32().unwrap().to_bits(), ZARR_NAN_F32.to_bits());
        assert_eq!(metadata.as_f64().unwrap().to_bits(), ZARR_NAN_F64.to_bits());
    }

    #[test]
    fn fill_value_metadata_float32_nan_standard() {
        let json = r#""0x7fc00000""#;
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert!(metadata.as_f16().is_none());
        assert!(metadata.as_bf16().is_none());
        assert_eq!(metadata.as_f32().unwrap().to_bits(), ZARR_NAN_F32.to_bits());
        assert!(metadata.as_f32().unwrap().is_nan());
        assert!(metadata.as_f64().is_none());
    }

    #[test]
    fn fill_value_metadata_float_nan_nonstandard() {
        let json = r#""0x7fc00001""#;
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert!(metadata.as_f16().is_none());
        assert!(metadata.as_bf16().is_none());
        assert_ne!(metadata.as_f32().unwrap().to_bits(), ZARR_NAN_F32.to_bits());
        assert!(metadata.as_f32().unwrap().is_nan());
        assert!(metadata.as_f64().is_none());
    }

    #[test]
    fn fill_value_metadata_float_hex_string() {
        let json = r#""0x3F800000""#;
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(metadata.as_f32().unwrap(), 1.0);
    }

    #[test]
    fn fill_value_metadata_float_complex() {
        let json = r#"["0x3F800000","NaN"]"#;
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        let metadata = metadata.as_array().unwrap();
        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata[0].as_f32().unwrap(), 1.0);
        assert_eq!(
            metadata[1].as_f32().unwrap().to_bits(),
            ZARR_NAN_F32.to_bits()
        );
    }

    #[test]
    fn fill_value_metadata_raw_bytes() {
        let json = r"[0,1,2,3]";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(metadata.as_bytes().unwrap(), [0, 1, 2, 3]);
    }

    #[test]
    fn fill_value_metadata_null() {
        let json = r"null";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert!(metadata.is_null());
    }

    #[test]
    fn fill_value_metadata_neg_array1() {
        let json = r"[-5]";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(
            metadata.as_array().unwrap(),
            [FillValueMetadataV3::from(-5)]
        );
    }

    #[test]
    fn fill_value_metadata_neg_array2() {
        let json = r"[-5, -5]";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_ne!(json, serde_json::to_string(&metadata).unwrap()); // [-5.0, -5.0]
        assert_eq!(
            metadata.as_array().unwrap(),
            [FillValueMetadataV3::from(-5), FillValueMetadataV3::from(-5)]
        );
    }

    #[test]
    fn fill_value_metadata_large_array() {
        let json = r"[256]";
        let metadata: FillValueMetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(json, serde_json::to_string(&metadata).unwrap());
        assert_eq!(
            metadata.as_array().unwrap(),
            [FillValueMetadataV3::from(256)]
        );
    }
}

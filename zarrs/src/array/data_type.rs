//! Zarr data types.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#data-types>.
//!
//! This submodule re-exports much of the [`zarrs_data_type`] crate.
//!
//! Use the factory functions (e.g. [`int8`], [`float32`], etc.) to create instances of built-in data types.
//!
//! Custom data types can be implemented by registering structs that implement the traits of [`zarrs_data_type`].
//! A custom data type guide can be found in [The `zarrs` book](https://book.zarrs.dev).
//!
#![doc = include_str!("../../doc/status/data_types.md")]

mod macros;

// Individual data type modules with substantial implementations
mod bool;
mod bytes;
mod complex_float;
mod complex_float8_e4m3;
mod complex_float8_e5m2;
mod complex_subfloat;
mod float;
mod float8_e4m3;
mod float8_e5m2;
mod int;
mod int2;
mod int4;
mod numpy_datetime64;
mod numpy_timedelta64;
mod raw_bits;
mod string;
mod subfloat;
mod uint;
mod uint2;
mod uint4;

use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::Arc;

use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::ExtensionName;

pub use zarrs_data_type::{
    DataType, DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePluginV2,
    DataTypePluginV3, DataTypeTraits, FillValue,
};
pub use zarrs_metadata_ext::data_type::NumpyTimeUnit;

pub use self::bool::BoolDataType;
pub use self::string::StringDataType;
pub use bytes::BytesDataType;
pub use complex_float::{
    Complex64DataType, Complex128DataType, ComplexBFloat16DataType, ComplexFloat16DataType,
    ComplexFloat32DataType, ComplexFloat64DataType,
};
pub use complex_float8_e4m3::ComplexFloat8E4M3DataType;
pub use complex_float8_e5m2::ComplexFloat8E5M2DataType;
pub use complex_subfloat::{
    ComplexFloat4E2M1FNDataType, ComplexFloat6E2M3FNDataType, ComplexFloat6E3M2FNDataType,
    ComplexFloat8E3M4DataType, ComplexFloat8E4M3B11FNUZDataType, ComplexFloat8E4M3FNUZDataType,
    ComplexFloat8E5M2FNUZDataType, ComplexFloat8E8M0FNUDataType,
};
pub use float::{BFloat16DataType, Float16DataType, Float32DataType, Float64DataType};
pub use float8_e4m3::Float8E4M3DataType;
pub use float8_e5m2::Float8E5M2DataType;
pub use int::{Int8DataType, Int16DataType, Int32DataType, Int64DataType};
pub use int2::Int2DataType;
pub use int4::Int4DataType;
pub use numpy_datetime64::NumpyDateTime64DataType;
pub use numpy_timedelta64::NumpyTimeDelta64DataType;
pub use raw_bits::RawBitsDataType;
pub use subfloat::{
    Float4E2M1FNDataType, Float6E2M3FNDataType, Float6E3M2FNDataType, Float8E3M4DataType,
    Float8E4M3B11FNUZDataType, Float8E4M3FNUZDataType, Float8E5M2FNUZDataType,
    Float8E8M0FNUDataType,
};
pub use uint::{UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType};
pub use uint2::UInt2DataType;
pub use uint4::UInt4DataType;
pub use zarrs_data_type::OptionalDataType;

// Integers
/// Create a `bool` data type.
#[must_use]
pub fn bool() -> DataType {
    Arc::new(BoolDataType).into()
}
/// Create an `int2` data type.
#[must_use]
pub fn int2() -> DataType {
    Arc::new(Int2DataType).into()
}
/// Create an `int4` data type.
#[must_use]
pub fn int4() -> DataType {
    Arc::new(Int4DataType).into()
}
/// Create an `int8` data type.
#[must_use]
pub fn int8() -> DataType {
    Arc::new(Int8DataType).into()
}
/// Create an `int16` data type.
#[must_use]
pub fn int16() -> DataType {
    Arc::new(Int16DataType).into()
}
/// Create an `int32` data type.
#[must_use]
pub fn int32() -> DataType {
    Arc::new(Int32DataType).into()
}
/// Create an `int64` data type.
#[must_use]
pub fn int64() -> DataType {
    Arc::new(Int64DataType).into()
}
/// Create a `uint2` data type.
#[must_use]
pub fn uint2() -> DataType {
    Arc::new(UInt2DataType).into()
}
/// Create a `uint4` data type.
#[must_use]
pub fn uint4() -> DataType {
    Arc::new(UInt4DataType).into()
}
/// Create a `uint8` data type.
#[must_use]
pub fn uint8() -> DataType {
    Arc::new(UInt8DataType).into()
}
/// Create a `uint16` data type.
#[must_use]
pub fn uint16() -> DataType {
    Arc::new(UInt16DataType).into()
}
/// Create a `uint32` data type.
#[must_use]
pub fn uint32() -> DataType {
    Arc::new(UInt32DataType).into()
}
/// Create a `uint64` data type.
#[must_use]
pub fn uint64() -> DataType {
    Arc::new(UInt64DataType).into()
}

// Standard floats
/// Create a `bfloat16` data type.
#[must_use]
pub fn bfloat16() -> DataType {
    Arc::new(BFloat16DataType).into()
}
/// Create a `float16` data type.
#[must_use]
pub fn float16() -> DataType {
    Arc::new(Float16DataType).into()
}
/// Create a `float32` data type.
#[must_use]
pub fn float32() -> DataType {
    Arc::new(Float32DataType).into()
}
/// Create a `float64` data type.
#[must_use]
pub fn float64() -> DataType {
    Arc::new(Float64DataType).into()
}

// Subfloats
/// Create a `float4_e2m1fn` data type.
#[must_use]
pub fn float4_e2m1fn() -> DataType {
    Arc::new(Float4E2M1FNDataType).into()
}
/// Create a `float6_e2m3fn` data type.
#[must_use]
pub fn float6_e2m3fn() -> DataType {
    Arc::new(Float6E2M3FNDataType).into()
}
/// Create a `float6_e3m2fn` data type.
#[must_use]
pub fn float6_e3m2fn() -> DataType {
    Arc::new(Float6E3M2FNDataType).into()
}
/// Create a `float8_e3m4` data type.
#[must_use]
pub fn float8_e3m4() -> DataType {
    Arc::new(Float8E3M4DataType).into()
}
/// Create a `float8_e4m3` data type.
#[must_use]
pub fn float8_e4m3() -> DataType {
    Arc::new(Float8E4M3DataType).into()
}
/// Create a `float8_e4m3b11fnuz` data type.
#[must_use]
pub fn float8_e4m3b11fnuz() -> DataType {
    Arc::new(Float8E4M3B11FNUZDataType).into()
}
/// Create a `float8_e4m3fnuz` data type.
#[must_use]
pub fn float8_e4m3fnuz() -> DataType {
    Arc::new(Float8E4M3FNUZDataType).into()
}
/// Create a `float8_e5m2` data type.
#[must_use]
pub fn float8_e5m2() -> DataType {
    Arc::new(Float8E5M2DataType).into()
}
/// Create a `float8_e5m2fnuz` data type.
#[must_use]
pub fn float8_e5m2fnuz() -> DataType {
    Arc::new(Float8E5M2FNUZDataType).into()
}
/// Create a `float8_e8m0fnu` data type.
#[must_use]
pub fn float8_e8m0fnu() -> DataType {
    Arc::new(Float8E8M0FNUDataType).into()
}

// Standard complex
/// Create a `complex64` data type.
#[must_use]
pub fn complex64() -> DataType {
    Arc::new(Complex64DataType).into()
}
/// Create a `complex128` data type.
#[must_use]
pub fn complex128() -> DataType {
    Arc::new(Complex128DataType).into()
}
/// Create a `complex_bfloat16` data type.
#[must_use]
pub fn complex_bfloat16() -> DataType {
    Arc::new(ComplexBFloat16DataType).into()
}
/// Create a `complex_float16` data type.
#[must_use]
pub fn complex_float16() -> DataType {
    Arc::new(ComplexFloat16DataType).into()
}
/// Create a `complex_float32` data type.
#[must_use]
pub fn complex_float32() -> DataType {
    Arc::new(ComplexFloat32DataType).into()
}
/// Create a `complex_float64` data type.
#[must_use]
pub fn complex_float64() -> DataType {
    Arc::new(ComplexFloat64DataType).into()
}

// Complex subfloats
/// Create a `complex_float4_e2m1fn` data type.
#[must_use]
pub fn complex_float4_e2m1fn() -> DataType {
    Arc::new(ComplexFloat4E2M1FNDataType).into()
}
/// Create a `complex_float6_e2m3fn` data type.
#[must_use]
pub fn complex_float6_e2m3fn() -> DataType {
    Arc::new(ComplexFloat6E2M3FNDataType).into()
}
/// Create a `complex_float6_e3m2fn` data type.
#[must_use]
pub fn complex_float6_e3m2fn() -> DataType {
    Arc::new(ComplexFloat6E3M2FNDataType).into()
}
/// Create a `complex_float8_e3m4` data type.
#[must_use]
pub fn complex_float8_e3m4() -> DataType {
    Arc::new(ComplexFloat8E3M4DataType).into()
}
/// Create a `complex_float8_e4m3` data type.
#[must_use]
pub fn complex_float8_e4m3() -> DataType {
    Arc::new(ComplexFloat8E4M3DataType).into()
}
/// Create a `complex_float8_e4m3b11fnuz` data type.
#[must_use]
pub fn complex_float8_e4m3b11fnuz() -> DataType {
    Arc::new(ComplexFloat8E4M3B11FNUZDataType).into()
}
/// Create a `complex_float8_e4m3fnuz` data type.
#[must_use]
pub fn complex_float8_e4m3fnuz() -> DataType {
    Arc::new(ComplexFloat8E4M3FNUZDataType).into()
}
/// Create a `complex_float8_e5m2` data type.
#[must_use]
pub fn complex_float8_e5m2() -> DataType {
    Arc::new(ComplexFloat8E5M2DataType).into()
}
/// Create a `complex_float8_e5m2fnuz` data type.
#[must_use]
pub fn complex_float8_e5m2fnuz() -> DataType {
    Arc::new(ComplexFloat8E5M2FNUZDataType).into()
}
/// Create a `complex_float8_e8m0fnu` data type.
#[must_use]
pub fn complex_float8_e8m0fnu() -> DataType {
    Arc::new(ComplexFloat8E8M0FNUDataType).into()
}

// Special types
/// Create a `string` data type.
#[must_use]
pub fn string() -> DataType {
    Arc::new(StringDataType).into()
}
/// Create a `bytes` data type.
#[must_use]
pub fn bytes() -> DataType {
    Arc::new(BytesDataType).into()
}
/// Create an `r*` (raw bits) data type with the given size in bytes.
#[must_use]
pub fn raw_bits(size_bytes: usize) -> DataType {
    Arc::new(RawBitsDataType::new(size_bytes)).into()
}

// NumPy time types
/// Create a `numpy.datetime64` data type.
#[must_use]
pub fn numpy_datetime64(unit: NumpyTimeUnit, scale_factor: NonZeroU32) -> DataType {
    Arc::new(NumpyDateTime64DataType::new(unit, scale_factor)).into()
}
/// Create a `numpy.timedelta64` data type.
#[must_use]
pub fn numpy_timedelta64(unit: NumpyTimeUnit, scale_factor: NonZeroU32) -> DataType {
    Arc::new(NumpyTimeDelta64DataType::new(unit, scale_factor)).into()
}

// Optional
/// Create an optional data type wrapping the given inner type.
#[must_use]
pub fn optional(inner: DataType) -> DataType {
    Arc::new(OptionalDataType::new(inner)).into()
}

/// Get the default V3 name for a V3 data type name.
///
/// This checks registered data type plugins (which include both built-in types and extensions).
/// Returns the default V3 name if a match is found, otherwise returns the input name unchanged.
#[must_use]
pub(crate) fn data_type_v3_default_name(metadata: &MetadataV3) -> Cow<'static, str> {
    let name = metadata.name().to_string();
    if let Ok(data_type) = DataType::from_metadata(metadata)
        && let Some(name) = data_type.name_v3()
    {
        return name;
    }
    Cow::Owned(name)
}

/// Get the default V2 name for a V2 data type name.
///
/// This checks registered data type plugins (which include both built-in types and extensions).
/// Returns the default V2 name if a match is found, otherwise returns the input name unchanged.
#[must_use]
pub(crate) fn data_type_v2_default_name(v2_name: &str) -> Cow<'static, str> {
    // For V2 we don't have full metadata, so create a minimal one from the name
    let metadata = MetadataV3::new(v2_name);
    if let Ok(data_type) = DataType::from_metadata(&metadata)
        && let Some(name) = data_type.name_v2()
    {
        return name;
    }
    Cow::Owned(v2_name.to_string())
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use zarrs_codec::ArrayBytes;
    use zarrs_metadata::DataTypeSize;

    use super::*;
    use crate::array::Element;
    use crate::metadata::FillValueMetadata;
    use crate::metadata::v3::{
        MetadataV3, ZARR_NAN_BF16, ZARR_NAN_F16, ZARR_NAN_F32, ZARR_NAN_F64,
    };

    /// Helper to create metadata from a data type
    fn data_type_metadata(data_type: &DataType) -> MetadataV3 {
        let name = data_type
            .name_v3()
            .map_or_else(String::new, |n| n.into_owned());
        let configuration = data_type.configuration_v3();
        if configuration.is_empty() {
            MetadataV3::new(name)
        } else {
            MetadataV3::new_with_configuration(name, configuration)
        }
    }

    #[test]
    fn data_type_unknown() {
        let json = r#""unknown""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(
            DataType::from_metadata(&metadata).unwrap_err().to_string(),
            "data type unknown is not supported"
        );
    }

    #[test]
    fn data_type_must_understand_false() {
        let json = r#"{"name":"unknown","must_understand": false}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(
            DataType::from_metadata(&metadata).unwrap_err().to_string(),
            r#"data type must not have `"must_understand": false`"#
        );
    }

    #[test]
    fn data_type_bool() {
        let json = r#""bool""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert!(data_type.is::<BoolDataType>());
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>("true").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), u8::from(true).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>("false").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), u8::from(false).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_int2() {
        let json = r#""int2""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Int2DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-1").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-1i8).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>("1").unwrap())
                .unwrap()
                .as_ne_bytes(),
            1i8.to_ne_bytes()
        );
        let metadata = serde_json::from_str::<FillValueMetadata>("-3").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadata>("2").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_int4() {
        let json = r#""int4""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Int4DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i8).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>("7").unwrap())
                .unwrap()
                .as_ne_bytes(),
            7i8.to_ne_bytes()
        );
        let metadata = serde_json::from_str::<FillValueMetadata>("8").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadata>("-9").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_int8() {
        let json = r#""int8""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Int8DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i8).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>("7").unwrap())
                .unwrap()
                .as_ne_bytes(),
            7i8.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_int16() {
        let json = r#""int16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Int16DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i16).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>("7").unwrap())
                .unwrap()
                .as_ne_bytes(),
            7i16.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_int32() {
        let json = r#""int32""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Int32DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i32).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>("7").unwrap())
                .unwrap()
                .as_ne_bytes(),
            7i32.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_int64() {
        let json = r#""int64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Int64DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i64).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>("7").unwrap())
                .unwrap()
                .as_ne_bytes(),
            7i64.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_uint2() {
        let json = r#""uint2""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<UInt2DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("3").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 3u8.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>("4").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadata>("-1").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_uint4() {
        let json = r#""uint4""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<UInt4DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("15").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 15u8.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
        let metadata = serde_json::from_str::<FillValueMetadata>("16").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadata>("-1").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_uint8() {
        let json = r#""uint8""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<UInt8DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 7u8.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_uint16() {
        let json = r#""uint16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<UInt16DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 7u16.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_uint32() {
        let json = r#""uint32""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<UInt32DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 7u32.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_uint64() {
        let json = r#""uint64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<UInt64DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("7").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 7u64.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_float32() {
        let json = r#""float32""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Float32DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-7.0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7.0f32).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>(r#""NaN""#).unwrap())
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F32.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""0x7fc00000""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            f32::NAN.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            f32::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""-Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            f32::NEG_INFINITY.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_float64() {
        let json = r#""float64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Float64DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("-7.0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7.0f64).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""0x7FF8000000000000""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F64.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>(r#""NaN""#).unwrap())
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F64.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            f64::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""-Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            f64::NEG_INFINITY.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_float4_e2m1fn() {
        let json = r#""float4_e2m1fn""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float4_e2m1fn");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0x0f""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [15]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_float6_e2m3fn() {
        let json = r#""float6_e2m3fn""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float6_e2m3fn");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0x3f""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [63]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_float6_e3m2fn() {
        let json = r#""float6_e3m2fn""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float6_e3m2fn");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0x3f""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [63]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_float8_e3m4() {
        let json = r#""float8_e3m4""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e3m4");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[cfg(not(feature = "float8"))]
    #[test]
    fn data_type_float8_e4m3() {
        let json = r#""float8_e4m3""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e4m3");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[cfg(feature = "float8")]
    #[test]
    fn data_type_float8_e4m3() {
        let json = r#""float8_e4m3""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e4m3");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        // Verify that the fill value represents -0.3125 in float8_e4m3 format
        assert_eq!(float8::F8E4M3::from_bits(170).to_f32(), -0.3125);
        // metadata_fill_value returns numeric value (with float8 feature enabled)
        let metadata_out = data_type.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(
            serde_json::from_str::<FillValueMetadata>(r"-0.3125").unwrap(),
            metadata_out
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""NaN""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert!(float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]).is_nan());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""Infinity""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""-Infinity""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::NEG_INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::NEG_ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-1").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::NEG_ONE
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"1").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::ONE
        );
    }

    #[test]
    fn data_type_float8_e4m3b11fnuz() {
        let json = r#""float8_e4m3b11fnuz""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e4m3b11fnuz");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_float8_e4m3fnuz() {
        let json = r#""float8_e4m3fnuz""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e4m3fnuz");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[cfg(not(feature = "float8"))]
    #[test]
    fn data_type_float8_e5m2() {
        let json = r#""float8_e5m2""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e5m2");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[cfg(feature = "float8")]
    #[test]
    fn data_type_float8_e5m2() {
        let json = r#""float8_e5m2""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e5m2");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        // Verify that the fill value represents -0.046875 in float8_e5m2 format
        assert_eq!(float8::F8E5M2::from_bits(170).to_f32(), -0.046875);
        // metadata_fill_value returns numeric value (with float8 feature enabled)
        let metadata_out = data_type.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(
            serde_json::from_str::<FillValueMetadata>(r"-0.046875").unwrap(),
            metadata_out
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""NaN""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert!(float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]).is_nan());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""Infinity""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""-Infinity""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::NEG_INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::NEG_ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-1").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::NEG_ONE
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"1").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::ONE
        );
    }

    #[test]
    fn data_type_float8_e5m2fnuz() {
        let json = r#""float8_e5m2fnuz""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e5m2fnuz");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_float8_e8m0fnu() {
        let json = r#""float8_e8m0fnu""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float8_e8m0fnu");

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_float16() {
        use half::f16;

        let json = r#""float16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "float16");

        let metadata = serde_json::from_str::<FillValueMetadata>("-7.0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            f16::from_f32_const(-7.0).to_ne_bytes()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>(r#""NaN""#).unwrap())
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F16.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            f16::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""-Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            f16::NEG_INFINITY.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_bfloat16() {
        use half::bf16;

        let json = r#""bfloat16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "bfloat16");

        let metadata = serde_json::from_str::<FillValueMetadata>("-7.0").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            bf16::from_f32_const(-7.0).to_ne_bytes()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    // &serde_json::from_str::<FillValueMetadata>(r#""0x7E00""#).unwrap()
                    &serde_json::from_str::<FillValueMetadata>(r#""0x7FC0""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_BF16.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(&serde_json::from_str::<FillValueMetadata>(r#""NaN""#).unwrap())
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_BF16.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            bf16::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_v3(
                    &serde_json::from_str::<FillValueMetadata>(r#""-Infinity""#).unwrap(),
                )
                .unwrap()
                .as_ne_bytes(),
            bf16::NEG_INFINITY.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_complex_bfloat16() {
        let json = r#""complex_bfloat16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<ComplexBFloat16DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            (half::bf16::from_f32(-7.0f32))
                .to_ne_bytes()
                .iter()
                .chain(half::bf16::INFINITY.to_ne_bytes().iter())
                .copied()
                .collect::<Vec<u8>>()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-7.0").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_complex_float16() {
        let json = r#""complex_float16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<ComplexFloat16DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            (half::f16::from_f32(-7.0f32))
                .to_ne_bytes()
                .iter()
                .chain(half::f16::INFINITY.to_ne_bytes().iter())
                .copied()
                .collect::<Vec<u8>>()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-7.0").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_complex_float32() {
        let json = r#""complex_float32""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<ComplexFloat32DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            (-7.0f32)
                .to_ne_bytes()
                .iter()
                .chain(f32::INFINITY.to_ne_bytes().iter())
                .copied()
                .collect::<Vec<u8>>()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-7.0").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_complexfloat64() {
        let json = r#""complex_float64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<ComplexFloat64DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            (-7.0f64)
                .to_ne_bytes()
                .iter()
                .chain(f64::INFINITY.to_ne_bytes().iter())
                .copied()
                .collect::<Vec<u8>>()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-7.0").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_complex64() {
        let json = r#""complex64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Complex64DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            (-7.0f32)
                .to_ne_bytes()
                .iter()
                .chain(f32::INFINITY.to_ne_bytes().iter())
                .copied()
                .collect::<Vec<u8>>()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-7.0").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_complex128() {
        let json = r#""complex128""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<Complex128DataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(),
            (-7.0f64)
                .to_ne_bytes()
                .iter()
                .chain(f64::INFINITY.to_ne_bytes().iter())
                .copied()
                .collect::<Vec<u8>>()
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r"-7.0").unwrap();
        assert!(data_type.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_r8() {
        let json = r#""r8""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "r8");
        assert_eq!(data_type.size(), DataTypeSize::Fixed(1));

        let metadata = serde_json::from_str::<FillValueMetadata>("[7]").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 7u8.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_r16() {
        let json = r#""r16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "r16");
        assert_eq!(data_type.size(), DataTypeSize::Fixed(2));

        let metadata = serde_json::from_str::<FillValueMetadata>("[0, 255]").unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(
            fill_value.as_ne_bytes(), // NOTE: Raw value bytes are always read as-is.
            &[0u8, 255u8]
        );
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_unknown1() {
        let json = r#"
    {
        "name": "datetime",
        "configuration": {
            "unit": "ns"
        }
    }"#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        println!("{json:?}");
        println!("{metadata:?}");
        assert_eq!(metadata.name(), "datetime");
        assert!(DataType::from_metadata(&metadata).is_err());
    }

    #[test]
    fn data_type_unknown2() {
        let json = r#""datetime""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        println!("{json:?}");
        println!("{metadata:?}");
        assert_eq!(metadata.name(), "datetime");
        assert!(DataType::from_metadata(&metadata).is_err());
    }

    #[test]
    fn data_type_unknown3() {
        let json = r#""ra""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        println!("{json:?}");
        println!("{metadata:?}");
        assert_eq!(metadata.name(), "ra");
        assert!(DataType::from_metadata(&metadata).is_err());
    }

    #[test]
    fn data_type_invalid() {
        let json = r#"
    {
        "name": "datetime",
        "notconfiguration": {
            "unit": "ns"
        }
    }"#;
        assert!(serde_json::from_str::<MetadataV3>(json).is_err());
    }

    #[test]
    fn data_type_raw_bits1() {
        let json = r#""r16""#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(data_type.size(), DataTypeSize::Fixed(2));
    }

    #[test]
    fn data_type_raw_bits2() {
        let json = r#"
    {
        "name": "r16"
    }"#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(data_type.size(), DataTypeSize::Fixed(2));
    }

    #[test]
    fn data_type_raw_bits_failure1() {
        let json = r#"
    {
        "name": "r5"
    }"#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        assert!(DataType::from_metadata(&metadata).is_err());
    }

    #[test]
    fn incompatible_fill_value_metadata() {
        let json = r#""bool""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert!(data_type.is::<BoolDataType>());

        let metadata = serde_json::from_str::<FillValueMetadata>("1").unwrap();
        assert_eq!(
            data_type.fill_value_v3(&metadata).unwrap_err().to_string(),
            "incompatible fill value metadata"
        );
    }

    #[test]
    fn incompatible_raw_bits_metadata() {
        let json = r#""r16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type, raw_bits(2));

        let metadata = serde_json::from_str::<FillValueMetadata>("[123]").unwrap();
        assert_eq!(serde_json::to_string(&metadata).unwrap(), "[123]");
        // assert_eq!(metadata.to_string(), "[123]");
        let fill_value_err = data_type.fill_value_v3(&metadata).unwrap_err();
        assert_eq!(
            fill_value_err.to_string(),
            "incompatible fill value metadata"
        );
    }

    #[test]
    fn float_fill_value() {
        assert_eq!(
            FillValueMetadata::from(half::f16::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(half::f16::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(ZARR_NAN_F16),
            serde_json::from_str(r#""NaN""#).unwrap()
        );
        let f16_nan_alt = unsafe { std::mem::transmute::<u16, half::f16>(0b01_11111_000000001) };
        assert!(f16_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadata::from(f16_nan_alt),
            serde_json::from_str(r#""0x7e01""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(bf16::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(bf16::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(ZARR_NAN_BF16),
            serde_json::from_str(r#""NaN""#).unwrap()
        );
        let bf16_nan_alt = unsafe { std::mem::transmute::<u16, bf16>(0b0_01111_11111000001) };
        assert!(bf16_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadata::from(bf16_nan_alt),
            serde_json::from_str(r#""0x7fc1""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(f32::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(f32::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(ZARR_NAN_F32),
            serde_json::from_str(r#""NaN""#).unwrap()
        );

        let f32_nan_alt = f32::from_bits(0b0_11111111_10000000000000000000001);
        assert!(f32_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadata::from(f32_nan_alt),
            serde_json::from_str(r#""0x7fc00001""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(f64::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(f64::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadata::from(ZARR_NAN_F64),
            serde_json::from_str(r#""NaN""#).unwrap()
        );
        let f64_nan_alt =
            f64::from_bits(0b0_11111111111_1000000000000000000000000000000000000000000000000001);
        assert!(f64_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadata::from(f64_nan_alt),
            serde_json::from_str(r#""0x7ff8000000000001""#).unwrap()
        );
    }

    #[test]
    fn incompatible_fill_value() {
        let err = DataTypeFillValueError;
        assert_eq!(err.to_string(), "fill value is incompatible with data type");
    }

    #[test]
    fn fill_value_from_metadata_failure() {
        let metadata = serde_json::from_str::<FillValueMetadata>("1").unwrap();
        assert!(bool().fill_value_v3(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadata>("false").unwrap();
        assert!(int8().fill_value_v3(&metadata).is_err());
        assert!(int16().fill_value_v3(&metadata).is_err());
        assert!(int32().fill_value_v3(&metadata).is_err());
        assert!(int64().fill_value_v3(&metadata).is_err());
        assert!(uint8().fill_value_v3(&metadata).is_err());
        assert!(uint16().fill_value_v3(&metadata).is_err());
        assert!(uint32().fill_value_v3(&metadata).is_err());
        assert!(uint64().fill_value_v3(&metadata).is_err());
        assert!(float16().fill_value_v3(&metadata).is_err());
        assert!(float32().fill_value_v3(&metadata).is_err());
        assert!(float64().fill_value_v3(&metadata).is_err());
        assert!(bfloat16().fill_value_v3(&metadata).is_err());
        assert!(complex64().fill_value_v3(&metadata).is_err());
        assert!(complex128().fill_value_v3(&metadata).is_err());
        assert!(raw_bits(1).fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn data_type_string() {
        let json = r#""string""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "string");
        assert_eq!(data_type.size(), DataTypeSize::Variable);

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""hello world""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), "hello world".as_bytes(),);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""Infinity""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), "Infinity".as_bytes(),);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadata>(r#""0x7fc00000""#).unwrap();
        let fill_value = data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), "0x7fc00000".as_bytes(),);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
    }

    #[test]
    fn data_type_bytes() {
        let json = r#""bytes""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "bytes");
        assert_eq!(data_type.size(), DataTypeSize::Variable);

        let expected_bytes = [0u8, 1, 2, 3];
        let metadata_from_arr: FillValueMetadata = serde_json::from_str(r"[0, 1, 2, 3]").unwrap();
        let fill_value_from_arr = data_type.fill_value_v3(&metadata_from_arr).unwrap();
        assert_eq!(fill_value_from_arr.as_ne_bytes(), expected_bytes,);

        let metadata_from_str: FillValueMetadata = serde_json::from_str(r#""AAECAw==""#).unwrap();
        let fill_value_from_str = data_type.fill_value_v3(&metadata_from_str).unwrap();
        assert_eq!(fill_value_from_str.as_ne_bytes(), expected_bytes,);

        // change to `metadata_from_str` when these land:
        // - https://github.com/zarr-developers/zarr-extensions/pull/38
        // - https://github.com/zarr-developers/zarr-python/pull/3559
        let expected_ser = metadata_from_arr;
        assert_eq!(
            expected_ser,
            data_type.metadata_fill_value(&fill_value_from_arr).unwrap()
        );
        assert_eq!(
            expected_ser,
            data_type.metadata_fill_value(&fill_value_from_str).unwrap()
        );
    }

    #[test]
    fn data_type_optional() {
        // Test optional int32
        let json =
            r#"{"name":"zarrs.optional","configuration":{"name":"int32","configuration":{}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "zarrs.optional");
        if let Some(opt) = data_type.as_optional() {
            assert!(opt.data_type().is::<Int32DataType>());
            assert_eq!(opt.data_type().size(), DataTypeSize::Fixed(4));
        } else {
            panic!("Expected Optional data type");
        }
        assert_eq!(data_type.size(), DataTypeSize::Fixed(4)); // inner type size (mask stored separately)

        // Test optional with complex configuration (numpy datetime64)
        let json = r#"{"name":"zarrs.optional","configuration":{"name":"numpy.datetime64","configuration":{"unit":"s","scale_factor":1}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert_eq!(data_type.name_v3().unwrap().as_ref(), "zarrs.optional");
        if let Some(opt) = data_type.as_optional() {
            assert!(opt.data_type().is::<NumpyDateTime64DataType>());
            if let Some(dt) = opt.data_type().downcast_ref::<NumpyDateTime64DataType>() {
                assert_eq!(dt.unit, NumpyTimeUnit::Second);
                assert_eq!(dt.scale_factor.get(), 1);
            } else {
                panic!("Expected NumpyDateTime64 inner type");
            }
        } else {
            panic!("Expected Optional data type");
        }

        // Test optional string (variable size)
        let json =
            r#"{"name":"zarrs.optional","configuration":{"name":"string","configuration":{}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        if let Some(opt) = data_type.as_optional() {
            assert!(opt.data_type().as_any().is::<StringDataType>());
            assert_eq!(opt.data_type().size(), DataTypeSize::Variable);
        } else {
            panic!("Expected Optional data type");
        }
        assert_eq!(data_type.size(), DataTypeSize::Variable);

        // Test metadata roundtrip
        let expected_metadata = data_type_metadata(&data_type);
        let roundtrip_data_type = DataType::from_metadata(&expected_metadata).unwrap();
        assert_eq!(data_type, roundtrip_data_type);
    }

    #[test]
    fn data_type_optional_invalid_configuration() {
        // Test missing configuration
        let json = r#"{"name":"zarrs.optional"}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let result = DataType::from_metadata(&metadata);
        assert!(result.is_err());

        // Test empty configuration
        let json = r#"{"name":"zarrs.optional","configuration":{}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let result = DataType::from_metadata(&metadata);
        assert!(result.is_err());

        // Test invalid inner data type
        let json = r#"{"name":"zarrs.optional","configuration":{"name":"unknown_type","configuration":{}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let result = DataType::from_metadata(&metadata);
        assert!(result.is_err());
    }

    #[test]
    fn data_type_optional_method() {
        // Single level optional
        let opt_u8 = uint8().to_optional();
        let expected = optional(uint8());
        assert_eq!(opt_u8, expected);
        assert!(opt_u8.is_optional());

        // Nested optional (2 levels)
        let nested_2 = uint8().to_optional().to_optional();
        let expected_2 = optional(optional(uint8()));
        assert_eq!(nested_2, expected_2);

        // Nested optional (3 levels)
        let nested_3 = uint16().to_optional().to_optional().to_optional();
        let expected_3 = optional(optional(optional(uint16())));
        assert_eq!(nested_3, expected_3);

        // Works with various inner types
        let opt_string = string().to_optional();
        let expected_string = optional(string());
        assert_eq!(opt_string, expected_string);

        let opt_f64 = float64().to_optional();
        let expected_f64 = optional(float64());
        assert_eq!(opt_f64, expected_f64);
    }

    #[test]
    fn data_type_optional_fill_value() {
        // Simple optional: None -> null
        let opt_data_type = uint8().to_optional();
        let fill_value = FillValue::new_optional_null();
        let metadata = opt_data_type.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(metadata, FillValueMetadata::Null);
        let roundtrip = opt_data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value, roundtrip);

        // Simple optional: Some(42) -> [42]
        let fill_value = FillValue::from(42u8).into_optional();
        let metadata = opt_data_type.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(
            metadata,
            FillValueMetadata::Array(vec![FillValueMetadata::from(42u8)])
        );
        let roundtrip = opt_data_type.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value, roundtrip);

        // Nested optional: None -> null
        let nested_opt = uint8().to_optional().to_optional();
        let fill_value = FillValue::new_optional_null();
        let metadata = nested_opt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(metadata, FillValueMetadata::Null);
        let roundtrip = nested_opt.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value, roundtrip);

        // Nested optional: Some(None) -> [null]
        let fill_value = FillValue::new_optional_null().into_optional();
        let metadata = nested_opt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(
            metadata,
            FillValueMetadata::Array(vec![FillValueMetadata::Null])
        );
        let roundtrip = nested_opt.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value, roundtrip);

        // Nested optional: Some(Some(42)) -> [[42]]
        let fill_value = FillValue::from(42u8).into_optional().into_optional();
        let metadata = nested_opt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(
            metadata,
            FillValueMetadata::Array(vec![FillValueMetadata::Array(vec![
                FillValueMetadata::from(42u8)
            ])])
        );
        let roundtrip = nested_opt.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value, roundtrip);

        // Triple nested: Some(Some(None)) -> [[null]]
        let triple_nested = uint8().to_optional().to_optional().to_optional();
        let fill_value = FillValue::new_optional_null()
            .into_optional()
            .into_optional();
        let metadata = triple_nested.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(
            metadata,
            FillValueMetadata::Array(vec![FillValueMetadata::Array(vec![
                FillValueMetadata::Null
            ])])
        );
        let roundtrip = triple_nested.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value, roundtrip);

        // Triple nested: Some(Some(Some(42))) -> [[[42]]]
        let fill_value = FillValue::from(42u8)
            .into_optional()
            .into_optional()
            .into_optional();
        let metadata = triple_nested.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(
            metadata,
            FillValueMetadata::Array(vec![FillValueMetadata::Array(vec![
                FillValueMetadata::Array(vec![FillValueMetadata::from(42u8)])
            ])])
        );
        let roundtrip = triple_nested.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value, roundtrip);
    }

    #[test]
    fn array_bytes_u32() {
        let data = vec![0u32, 1, 2, 3, 4];
        let n_elements = data.len();
        let bytes = Element::into_array_bytes(&uint32(), data).unwrap();
        let ArrayBytes::Fixed(bytes) = bytes else {
            panic!()
        };
        assert_eq!(bytes.len(), size_of::<u32>() * n_elements);
    }

    #[test]
    fn array_bytes_str() {
        let data = vec!["a", "bb", "ccc"];
        let bytes = Element::into_array_bytes(&string(), data).unwrap();
        let (bytes, offsets) = bytes.into_variable().unwrap().into_parts();
        assert_eq!(bytes, "abbccc".as_bytes());
        assert_eq!(*offsets, [0, 1, 3, 6]);
    }
}

//! Zarr data types.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#data-types>.
//!
//! This submodule re-exports much of the [`zarrs_data_type`] crate.
//!
//! Custom data types can be implemented by registering structs that implement the traits of [`zarrs_data_type`].
//! A custom data type guide can be found in [The `zarrs` book](https://book.zarrs.dev).
//!
#![doc = include_str!("../../doc/status/data_types.md")]

mod named_data_type;
use std::{fmt::Debug, mem::discriminant, num::NonZeroU32, sync::Arc};

pub use named_data_type::NamedDataType;
pub use zarrs_data_type::{
    DataTypeExtension, DataTypeExtensionBytesCodec, DataTypeExtensionBytesCodecError,
    DataTypeExtensionError, DataTypeExtensionPackBitsCodec, DataTypeFillValueError,
    DataTypeFillValueMetadataError, DataTypePlugin, FillValue,
};
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError, PluginUnsupportedError};

use crate::metadata::{
    v3::{FillValueMetadataV3, MetadataV3},
    ConfigurationSerialize, DataTypeSize,
};
use crate::metadata_ext::data_type::{
    numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1,
    numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1, NumpyTimeUnit,
};
use crate::registry::ExtensionAliasesDataTypeV3;

/// A data type.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[rustfmt::skip]
pub enum DataType {
    /// `bool` Boolean.
    Bool,
    /// `int2` Integer in `[-2, 1]`.
    Int2,
    /// `int4` Integer in `[-8, 7]`.
    Int4,
    /// `int8` Integer in `[-2^7, 2^7-1]`.
    Int8,
    /// `int16` Integer in `[-2^15, 2^15-1]`.
    Int16,
    /// `int32` Integer in `[-2^31, 2^31-1]`.
    Int32,
    /// `int64` Integer in `[-2^63, 2^63-1]`.
    Int64,
    /// `uint2` Integer in `[0, 3]`.
    UInt2,
    /// `uint4` Integer in `[0, 15]`.
    UInt4,
    /// `uint8` Integer in `[0, 2^8-1]`.
    UInt8,
    /// `uint16` Integer in `[0, 2^16-1]`.
    UInt16,
    /// `uint32` Integer in `[0, 2^32-1]`.
    UInt32,
    /// `uint64` Integer in `[0, 2^64-1]`.
    UInt64,
    /// `float4_e2m1fn` a 4-bit floating point representation: sign bit, 2 bit exponent (bias 1), 1 bit mantissa.
    /// - Extended range: no infinity, no NaN.
    /// - Subnormal numbers when biased exponent is 0.
    Float4E2M1FN,
    /// `float6_e2m3fn` a 6-bit floating point representation: sign bit, 2 bit exponent (bias 1), 3 bit mantissa.
    /// - Extended range: no infinity, no NaN.
    /// - Subnormal numbers when biased exponent is 0.
    Float6E2M3FN,
    /// `float6_e3m2fn` a 6-bit floating point representation: sign bit, 3 bit exponent (bias 3), 2 bit mantissa.
    /// - Extended range: no infinity, no NaN.
    /// - Subnormal numbers when biased exponent is 0.
    Float6E3M2FN,
    /// `float8_e3m4` an 8-bit floating point representation: sign bit, 3 bit exponent (bias 3), 4 bit mantissa.
    /// - IEEE 754-compliant, with NaN and +/-inf.
    //  - Subnormal numbers when biased exponent is 0.
    Float8E3M4,
    /// `float8_e4m3` an 8-bit floating point representation: sign bit, 4 bit exponent (bias 7), 3 bit mantissa.
    /// - IEEE 754-compliant, with NaN and +/-inf.
    //  - Subnormal numbers when biased exponent is 0.
    Float8E4M3,
    /// `float8_e4m3b11fnuz` an 8-bit floating point representation: sign bit, 4 bit exponent (bias 11), 3 bit mantissa.
    /// - Extended range: no infinity, NaN represented by `0b1000'0000`.
    /// - Subnormal numbers when biased exponent is 0.
    Float8E4M3B11FNUZ,
    /// `float8_e4m3fnuz` an 8-bit floating point representation: sign bit, 4 bit exponent (bias 8), 3 bit mantissa.
    /// - Extended range: no infinity, NaN represented by `0b1000'0000`.
    /// - Subnormal numbers when biased exponent is 0.
    Float8E4M3FNUZ,
    /// `float8_e5m2` an 8-bit floating point representation: sign bit, 5 bit exponent (bias 15), 2 bit mantissa.
    /// - IEEE 754-compliant, with NaN and +/-inf.
    /// - Subnormal numbers when biased exponent is 0.
    Float8E5M2,
    /// `float8_e5m2fnuz` an 8-bit floating point representation: sign bit, 5 bit exponent (bias 16), 2 bit mantissa.
    /// - Extended range: no infinity, NaN represented by `0b1000'0000`.
    /// - Subnormal numbers when biased exponent is 0.
    Float8E5M2FNUZ,
    /// `float8_e8m0fnu` an 8-bit floating point representation: no sign bit, 8 bit exponent (bias 127), 0 bit mantissa.
    /// - No zero, no infinity, NaN represented by `0b1111'1111`.
    /// - No subnormal numbers.
    Float8E8M0FNU,
    /// `bfloat16` brain floating point data type: sign bit, 5 bits exponent, 10 bits mantissa.
    BFloat16,
    /// `float16` IEEE 754 half-precision floating point: sign bit, 5 bits exponent, 10 bits mantissa.
    Float16,
    /// `float32` IEEE 754 single-precision floating point: sign bit, 8 bits exponent, 23 bits mantissa.
    Float32,
    /// `float64` IEEE 754 double-precision floating point: sign bit, 11 bits exponent, 52 bits mantissa.
    Float64,
    /// `complex_float32` real and complex components are each brain floating point data type.
    ComplexBFloat16,
    /// `complex_float32` real and complex components are each IEEE 754 half-precision floating point.
    ComplexFloat16,
    /// `complex_float32` real and complex components are each IEEE 754 single-precision floating point.
    ComplexFloat32,
    /// `complex_float64` real and complex components are each IEEE 754 double-precision floating point.
    ComplexFloat64,
    /// `complex_float4_e2m1fn` real and complex components are each the `float4_e2m1fn` type.
    ComplexFloat4E2M1FN,
    /// `complex_float6_e2m3fn` real and complex components are each the `float6_e2m3fn` type.
    ComplexFloat6E2M3FN,
    /// `complex_float6_e3m2fn` real and complex components are each the `float6_e3m2fn` type.
    ComplexFloat6E3M2FN,
    /// `complex_float8_e3m4` real and complex components are each the `float8_e3m4` type.
    ComplexFloat8E3M4,
    /// `complex_float8_e4m3` real and complex components are each the `float8_e4m3` type.
    ComplexFloat8E4M3,
    /// `complex_float8_e4m3b11fnuz` real and complex components are each the `float8_e4m3b11fnuz` type.
    ComplexFloat8E4M3B11FNUZ,
    /// `complex_float8_e4m3fnuz` real and complex components are each the `float8_e4m3fnuz` type.
    ComplexFloat8E4M3FNUZ,
    /// `complex_float8_e5m2` real and complex components are each the `float8_e5m2` type.
    ComplexFloat8E5M2,
    /// `complex_float8_e5m2fnuz` real and complex components are each the `float8_e5m2fnuz` type.
    ComplexFloat8E5M2FNUZ,
    /// `complex_float8_e8m0fnu` real and complex components are each the `float8_e8m0fnu` type.
    ComplexFloat8E8M0FNU,
    /// `complex64` real and complex components are each IEEE 754 single-precision floating point.
    Complex64,
    /// `complex128` real and complex components are each IEEE 754 double-precision floating point.
    Complex128,
    /// `r*` raw bits, variable size given by *, limited to be a multiple of 8.
    RawBits(usize), // the stored usize is the size in bytes
    /// A UTF-8 encoded string.
    String,
    /// Variable-sized binary data.
    Bytes,
    /// `numpy.datetime64` a 64-bit signed integer represents moments in time relative to the Unix epoch.
    ///
    /// This data type closely models the `datetime64` data type from `NumPy`.
    NumpyDateTime64{
        /// The `NumPy` temporal unit.
        unit: NumpyTimeUnit,
        /// The `NumPy` temporal scale factor.
        scale_factor: NonZeroU32,
    },
    /// `numpy.timedelta64` a 64-bit signed integer represents signed temporal durations.
    ///
    /// This data type closely models the `timedelta64` data type from `NumPy`.
    NumpyTimeDelta64{
        /// The `NumPy` temporal unit.
        unit: NumpyTimeUnit,
        /// The `NumPy` temporal scale factor.
        scale_factor: NonZeroU32,
    },
    /// An optional data type.
    Optional(Box<DataType>),
    /// An extension data type.
    Extension(Arc<dyn DataTypeExtension>),
}

impl PartialEq for DataType {
    fn eq(&self, other: &Self) -> bool {
        match (&self, other) {
            (DataType::RawBits(a), DataType::RawBits(b)) => a == b,
            (DataType::Extension(a), DataType::Extension(b)) => {
                a.name() == b.name() && a.configuration() == b.configuration()
            }
            _ => discriminant(self) == discriminant(other),
        }
    }
}

impl Eq for DataType {}

impl DataType {
    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Bool => zarrs_registry::data_type::BOOL.to_string(),
            Self::Int2 => zarrs_registry::data_type::INT2.to_string(),
            Self::Int4 => zarrs_registry::data_type::INT4.to_string(),
            Self::Int8 => zarrs_registry::data_type::INT8.to_string(),
            Self::Int16 => zarrs_registry::data_type::INT16.to_string(),
            Self::Int32 => zarrs_registry::data_type::INT32.to_string(),
            Self::Int64 => zarrs_registry::data_type::INT64.to_string(),
            Self::UInt2 => zarrs_registry::data_type::UINT2.to_string(),
            Self::UInt4 => zarrs_registry::data_type::UINT4.to_string(),
            Self::UInt8 => zarrs_registry::data_type::UINT8.to_string(),
            Self::UInt16 => zarrs_registry::data_type::UINT16.to_string(),
            Self::UInt32 => zarrs_registry::data_type::UINT32.to_string(),
            Self::UInt64 => zarrs_registry::data_type::UINT64.to_string(),
            Self::Float4E2M1FN => zarrs_registry::data_type::FLOAT4_E2M1FN.to_string(),
            Self::Float6E2M3FN => zarrs_registry::data_type::FLOAT6_E2M3FN.to_string(),
            Self::Float6E3M2FN => zarrs_registry::data_type::FLOAT6_E3M2FN.to_string(),
            Self::Float8E3M4 => zarrs_registry::data_type::FLOAT8_E3M4.to_string(),
            Self::Float8E4M3 => zarrs_registry::data_type::FLOAT8_E4M3.to_string(),
            Self::Float8E4M3B11FNUZ => zarrs_registry::data_type::FLOAT8_E4M3B11FNUZ.to_string(),
            Self::Float8E4M3FNUZ => zarrs_registry::data_type::FLOAT8_E4M3FNUZ.to_string(),
            Self::Float8E5M2 => zarrs_registry::data_type::FLOAT8_E5M2.to_string(),
            Self::Float8E5M2FNUZ => zarrs_registry::data_type::FLOAT8_E5M2FNUZ.to_string(),
            Self::Float8E8M0FNU => zarrs_registry::data_type::FLOAT8_E8M0FNU.to_string(),
            Self::BFloat16 => zarrs_registry::data_type::BFLOAT16.to_string(),
            Self::Float16 => zarrs_registry::data_type::FLOAT16.to_string(),
            Self::Float32 => zarrs_registry::data_type::FLOAT32.to_string(),
            Self::Float64 => zarrs_registry::data_type::FLOAT64.to_string(),
            Self::Complex64 => zarrs_registry::data_type::COMPLEX64.to_string(),
            Self::Complex128 => zarrs_registry::data_type::COMPLEX128.to_string(),
            Self::ComplexBFloat16 => zarrs_registry::data_type::COMPLEX_BFLOAT16.to_string(),
            Self::ComplexFloat16 => zarrs_registry::data_type::COMPLEX_FLOAT16.to_string(),
            Self::ComplexFloat32 => zarrs_registry::data_type::COMPLEX_FLOAT32.to_string(),
            Self::ComplexFloat64 => zarrs_registry::data_type::COMPLEX_FLOAT64.to_string(),
            Self::ComplexFloat4E2M1FN => {
                zarrs_registry::data_type::COMPLEX_FLOAT4_E2M1FN.to_string()
            }
            Self::ComplexFloat6E2M3FN => {
                zarrs_registry::data_type::COMPLEX_FLOAT6_E2M3FN.to_string()
            }
            Self::ComplexFloat6E3M2FN => {
                zarrs_registry::data_type::COMPLEX_FLOAT6_E3M2FN.to_string()
            }
            Self::ComplexFloat8E3M4 => zarrs_registry::data_type::COMPLEX_FLOAT8_E3M4.to_string(),
            Self::ComplexFloat8E4M3 => zarrs_registry::data_type::COMPLEX_FLOAT8_E4M3.to_string(),
            Self::ComplexFloat8E4M3B11FNUZ => {
                zarrs_registry::data_type::COMPLEX_FLOAT8_E4M3B11FNUZ.to_string()
            }
            Self::ComplexFloat8E4M3FNUZ => {
                zarrs_registry::data_type::COMPLEX_FLOAT8_E4M3FNUZ.to_string()
            }
            Self::ComplexFloat8E5M2 => zarrs_registry::data_type::COMPLEX_FLOAT8_E5M2.to_string(),
            Self::ComplexFloat8E5M2FNUZ => {
                zarrs_registry::data_type::COMPLEX_FLOAT8_E5M2FNUZ.to_string()
            }
            Self::ComplexFloat8E8M0FNU => {
                zarrs_registry::data_type::COMPLEX_FLOAT8_E8M0FNU.to_string()
            }
            Self::RawBits(size) => format!("r{}", size * 8),
            Self::String => zarrs_registry::data_type::STRING.to_string(),
            Self::Bytes => zarrs_registry::data_type::BYTES.to_string(),
            Self::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            } => zarrs_registry::data_type::NUMPY_DATETIME64.to_string(),
            Self::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => zarrs_registry::data_type::NUMPY_TIMEDELTA64.to_string(),
            Self::Optional(_data_type) => zarrs_registry::data_type::OPTIONAL.to_string(),
            Self::Extension(extension) => extension.name(),
        }
    }

    /// Returns the metadata.
    // TODO: Remove for configuration
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn metadata(&self) -> MetadataV3 {
        match self {
            Self::Bool => MetadataV3::new(zarrs_registry::data_type::BOOL),
            Self::Int2 => MetadataV3::new(zarrs_registry::data_type::INT2),
            Self::Int4 => MetadataV3::new(zarrs_registry::data_type::INT4),
            Self::Int8 => MetadataV3::new(zarrs_registry::data_type::INT8),
            Self::Int16 => MetadataV3::new(zarrs_registry::data_type::INT16),
            Self::Int32 => MetadataV3::new(zarrs_registry::data_type::INT32),
            Self::Int64 => MetadataV3::new(zarrs_registry::data_type::INT64),
            Self::UInt2 => MetadataV3::new(zarrs_registry::data_type::UINT2),
            Self::UInt4 => MetadataV3::new(zarrs_registry::data_type::UINT4),
            Self::UInt8 => MetadataV3::new(zarrs_registry::data_type::UINT8),
            Self::UInt16 => MetadataV3::new(zarrs_registry::data_type::UINT16),
            Self::UInt32 => MetadataV3::new(zarrs_registry::data_type::UINT32),
            Self::UInt64 => MetadataV3::new(zarrs_registry::data_type::UINT64),
            Self::Float4E2M1FN => MetadataV3::new(zarrs_registry::data_type::FLOAT4_E2M1FN),
            Self::Float6E2M3FN => MetadataV3::new(zarrs_registry::data_type::FLOAT6_E2M3FN),
            Self::Float6E3M2FN => MetadataV3::new(zarrs_registry::data_type::FLOAT6_E3M2FN),
            Self::Float8E3M4 => MetadataV3::new(zarrs_registry::data_type::FLOAT8_E3M4),
            Self::Float8E4M3 => MetadataV3::new(zarrs_registry::data_type::FLOAT8_E4M3),
            Self::Float8E4M3B11FNUZ => {
                MetadataV3::new(zarrs_registry::data_type::FLOAT8_E4M3B11FNUZ)
            }
            Self::Float8E4M3FNUZ => MetadataV3::new(zarrs_registry::data_type::FLOAT8_E4M3FNUZ),
            Self::Float8E5M2 => MetadataV3::new(zarrs_registry::data_type::FLOAT8_E5M2),
            Self::Float8E5M2FNUZ => MetadataV3::new(zarrs_registry::data_type::FLOAT8_E5M2FNUZ),
            Self::Float8E8M0FNU => MetadataV3::new(zarrs_registry::data_type::FLOAT8_E8M0FNU),
            Self::BFloat16 => MetadataV3::new(zarrs_registry::data_type::BFLOAT16),
            Self::Float16 => MetadataV3::new(zarrs_registry::data_type::FLOAT16),
            Self::Float32 => MetadataV3::new(zarrs_registry::data_type::FLOAT32),
            Self::Float64 => MetadataV3::new(zarrs_registry::data_type::FLOAT64),
            Self::Complex64 => MetadataV3::new(zarrs_registry::data_type::COMPLEX64),
            Self::Complex128 => MetadataV3::new(zarrs_registry::data_type::COMPLEX128),
            Self::ComplexBFloat16 => MetadataV3::new(zarrs_registry::data_type::COMPLEX_BFLOAT16),
            Self::ComplexFloat16 => MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT16),
            Self::ComplexFloat32 => MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT32),
            Self::ComplexFloat64 => MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT64),
            Self::ComplexFloat4E2M1FN => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT4_E2M1FN)
            }
            Self::ComplexFloat6E2M3FN => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT6_E2M3FN)
            }
            Self::ComplexFloat6E3M2FN => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT6_E3M2FN)
            }
            Self::ComplexFloat8E3M4 => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT8_E3M4)
            }
            Self::ComplexFloat8E4M3 => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT8_E4M3)
            }
            Self::ComplexFloat8E4M3B11FNUZ => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT8_E4M3B11FNUZ)
            }
            Self::ComplexFloat8E4M3FNUZ => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT8_E4M3FNUZ)
            }
            Self::ComplexFloat8E5M2 => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT8_E5M2)
            }
            Self::ComplexFloat8E5M2FNUZ => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT8_E5M2FNUZ)
            }
            Self::ComplexFloat8E8M0FNU => {
                MetadataV3::new(zarrs_registry::data_type::COMPLEX_FLOAT8_E8M0FNU)
            }
            Self::RawBits(size) => MetadataV3::new(format!("r{}", size * 8)),
            Self::String => MetadataV3::new(zarrs_registry::data_type::STRING),
            Self::Bytes => MetadataV3::new(zarrs_registry::data_type::BYTES),
            Self::NumpyDateTime64 { unit, scale_factor } => MetadataV3::new_with_configuration(
                zarrs_registry::data_type::NUMPY_DATETIME64,
                NumpyDateTime64DataTypeConfigurationV1 {
                    unit: *unit,
                    scale_factor: *scale_factor,
                },
            ),
            Self::NumpyTimeDelta64 { unit, scale_factor } => MetadataV3::new_with_configuration(
                zarrs_registry::data_type::NUMPY_TIMEDELTA64,
                NumpyTimeDelta64DataTypeConfigurationV1 {
                    unit: *unit,
                    scale_factor: *scale_factor,
                },
            ),
            Self::Optional(data_type) => {
                let configuration = data_type
                    .metadata()
                    .configuration()
                    .cloned()
                    .unwrap_or_default();
                MetadataV3::new_with_configuration(
                    "optional",
                    zarrs_metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1 {
                        name: data_type.name(),
                        configuration,
                    },
                )
            }
            Self::Extension(ext) => {
                MetadataV3::new_with_configuration(ext.name(), ext.configuration())
            }
        }
    }

    /// Returns the [`DataTypeSize`].
    #[must_use]
    pub fn size(&self) -> DataTypeSize {
        match self {
            Self::Bool
            | Self::Int2
            | Self::Int4
            | Self::Int8
            | Self::UInt2
            | Self::UInt4
            | Self::UInt8
            | Self::Float4E2M1FN
            | Self::Float6E2M3FN
            | Self::Float6E3M2FN
            | Self::Float8E3M4
            | Self::Float8E4M3
            | Self::Float8E4M3B11FNUZ
            | Self::Float8E4M3FNUZ
            | Self::Float8E5M2
            | Self::Float8E5M2FNUZ
            | Self::Float8E8M0FNU => DataTypeSize::Fixed(1),
            Self::Int16
            | Self::UInt16
            | Self::Float16
            | Self::BFloat16
            | Self::ComplexFloat4E2M1FN
            | Self::ComplexFloat6E2M3FN
            | Self::ComplexFloat6E3M2FN
            | Self::ComplexFloat8E3M4
            | Self::ComplexFloat8E4M3
            | Self::ComplexFloat8E4M3B11FNUZ
            | Self::ComplexFloat8E4M3FNUZ
            | Self::ComplexFloat8E5M2
            | Self::ComplexFloat8E5M2FNUZ
            | Self::ComplexFloat8E8M0FNU => DataTypeSize::Fixed(2),
            Self::Int32
            | Self::UInt32
            | Self::Float32
            | Self::ComplexFloat16
            | Self::ComplexBFloat16 => DataTypeSize::Fixed(4),
            Self::Int64
            | Self::UInt64
            | Self::Float64
            | Self::Complex64
            | Self::ComplexFloat32
            | Self::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | Self::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => DataTypeSize::Fixed(8),
            Self::Complex128 | Self::ComplexFloat64 => DataTypeSize::Fixed(16),
            Self::RawBits(size) => DataTypeSize::Fixed(*size),
            Self::String | Self::Bytes => DataTypeSize::Variable,
            Self::Optional(inner) => inner.size(),
            Self::Extension(extension) => extension.size(),
        }
    }

    /// Returns true if this is an optional data type.
    #[must_use]
    pub fn is_optional(&self) -> bool {
        matches!(self, Self::Optional(_))
    }

    /// Returns the size in bytes of a fixed-size data type, otherwise returns [`None`].
    #[must_use]
    pub fn fixed_size(&self) -> Option<usize> {
        match self.size() {
            DataTypeSize::Fixed(size) => Some(size),
            DataTypeSize::Variable => None,
        }
    }

    /// Returns `true` if the data type has a fixed size.
    #[must_use]
    pub fn is_fixed(&self) -> bool {
        matches!(self.size(), DataTypeSize::Fixed(_))
    }

    /// Returns `true` if the data type has a variable size.
    #[must_use]
    pub fn is_variable(&self) -> bool {
        matches!(self.size(), DataTypeSize::Variable)
    }

    /// Create a data type from metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered data type plugin.
    #[allow(clippy::too_many_lines)]
    pub fn from_metadata(
        metadata: &MetadataV3,
        data_type_aliases: &ExtensionAliasesDataTypeV3,
    ) -> Result<Self, PluginCreateError> {
        if !metadata.must_understand() {
            return Err(PluginCreateError::Other(
                r#"data type must not have `"must_understand": false`"#.to_string(),
            ));
        }

        let identifier = data_type_aliases.identifier(metadata.name());
        if metadata.name() != identifier {
            log::info!(
                "Using data type alias `{}` for `{}`",
                metadata.name(),
                identifier
            );
        }

        if let Some(configuration) = metadata.configuration() {
            match identifier {
                zarrs_registry::data_type::NUMPY_DATETIME64 => {
                    use crate::metadata_ext::data_type::numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;
                    let NumpyDateTime64DataTypeConfigurationV1 { unit, scale_factor } =
                        NumpyDateTime64DataTypeConfigurationV1::try_from_configuration(
                            configuration.clone(),
                        )
                        .map_err(|_| {
                            PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                                zarrs_registry::data_type::NUMPY_DATETIME64,
                                "data_type",
                                metadata.to_string(),
                            ))
                        })?;
                    return Ok(Self::NumpyDateTime64 { unit, scale_factor });
                }
                zarrs_registry::data_type::NUMPY_TIMEDELTA64 => {
                    use crate::metadata_ext::data_type::numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1;
                    let NumpyTimeDelta64DataTypeConfigurationV1 { unit, scale_factor } =
                        NumpyTimeDelta64DataTypeConfigurationV1::try_from_configuration(
                            configuration.clone(),
                        )
                        .map_err(|_| {
                            PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                                zarrs_registry::data_type::NUMPY_TIMEDELTA64,
                                "data_type",
                                metadata.to_string(),
                            ))
                        })?;
                    return Ok(Self::NumpyTimeDelta64 { unit, scale_factor });
                }
                zarrs_registry::data_type::OPTIONAL => {
                    use crate::metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1;
                    let OptionalDataTypeConfigurationV1 {
                        name,
                        configuration,
                    } = OptionalDataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            zarrs_registry::data_type::OPTIONAL,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;

                    // Create metadata for the inner data type
                    let inner_metadata = if configuration.is_empty() {
                        MetadataV3::new(name)
                    } else {
                        MetadataV3::new_with_configuration(name, configuration)
                    };

                    // Recursively parse the inner data type
                    let inner_data_type = Self::from_metadata(&inner_metadata, data_type_aliases)?;
                    return Ok(Self::Optional(Box::new(inner_data_type)));
                }
                _ => {}
            }
        }

        if metadata.configuration_is_none_or_empty() {
            // Data types with no configuration
            match identifier {
                zarrs_registry::data_type::BOOL => return Ok(Self::Bool),
                zarrs_registry::data_type::INT2 => return Ok(Self::Int2),
                zarrs_registry::data_type::INT4 => return Ok(Self::Int4),
                zarrs_registry::data_type::INT8 => return Ok(Self::Int8),
                zarrs_registry::data_type::INT16 => return Ok(Self::Int16),
                zarrs_registry::data_type::INT32 => return Ok(Self::Int32),
                zarrs_registry::data_type::INT64 => return Ok(Self::Int64),
                zarrs_registry::data_type::UINT2 => return Ok(Self::UInt2),
                zarrs_registry::data_type::UINT4 => return Ok(Self::UInt4),
                zarrs_registry::data_type::UINT8 => return Ok(Self::UInt8),
                zarrs_registry::data_type::UINT16 => return Ok(Self::UInt16),
                zarrs_registry::data_type::UINT32 => return Ok(Self::UInt32),
                zarrs_registry::data_type::UINT64 => return Ok(Self::UInt64),
                zarrs_registry::data_type::FLOAT4_E2M1FN => return Ok(Self::Float4E2M1FN),
                zarrs_registry::data_type::FLOAT6_E2M3FN => return Ok(Self::Float6E2M3FN),
                zarrs_registry::data_type::FLOAT6_E3M2FN => return Ok(Self::Float6E3M2FN),
                zarrs_registry::data_type::FLOAT8_E3M4 => return Ok(Self::Float8E3M4),
                zarrs_registry::data_type::FLOAT8_E4M3 => return Ok(Self::Float8E4M3),
                zarrs_registry::data_type::FLOAT8_E4M3B11FNUZ => {
                    return Ok(Self::Float8E4M3B11FNUZ);
                }
                zarrs_registry::data_type::FLOAT8_E4M3FNUZ => return Ok(Self::Float8E4M3FNUZ),
                zarrs_registry::data_type::FLOAT8_E5M2 => return Ok(Self::Float8E5M2),
                zarrs_registry::data_type::FLOAT8_E5M2FNUZ => return Ok(Self::Float8E5M2FNUZ),
                zarrs_registry::data_type::FLOAT8_E8M0FNU => return Ok(Self::Float8E8M0FNU),
                zarrs_registry::data_type::BFLOAT16 => return Ok(Self::BFloat16),
                zarrs_registry::data_type::FLOAT16 => return Ok(Self::Float16),
                zarrs_registry::data_type::FLOAT32 => return Ok(Self::Float32),
                zarrs_registry::data_type::FLOAT64 => return Ok(Self::Float64),
                zarrs_registry::data_type::COMPLEX_BFLOAT16 => return Ok(Self::ComplexBFloat16),
                zarrs_registry::data_type::COMPLEX_FLOAT16 => return Ok(Self::ComplexFloat16),
                zarrs_registry::data_type::COMPLEX_FLOAT32 => return Ok(Self::ComplexFloat32),
                zarrs_registry::data_type::COMPLEX_FLOAT64 => return Ok(Self::ComplexFloat64),
                zarrs_registry::data_type::COMPLEX64 => return Ok(Self::Complex64),
                zarrs_registry::data_type::COMPLEX128 => return Ok(Self::Complex128),
                zarrs_registry::data_type::STRING => return Ok(Self::String),
                zarrs_registry::data_type::BYTES => return Ok(Self::Bytes),
                name => {
                    if name.starts_with('r') && name.len() > 1 {
                        if let Ok(size_bits) = metadata.name()[1..].parse::<usize>() {
                            if size_bits % 8 == 0 {
                                let size_bytes = size_bits / 8;
                                return Ok(Self::RawBits(size_bytes));
                            }
                            return Err(PluginUnsupportedError::new(
                                name.to_string(),
                                "data type".to_string(),
                            )
                            .into());
                        }
                    }
                }
            }
        }

        // Try an extension
        for plugin in inventory::iter::<DataTypePlugin> {
            if plugin.match_name(identifier) {
                return plugin.create(&metadata.clone()).map(DataType::Extension);
            }
        }

        // The data type is not supported
        Err(
            PluginUnsupportedError::new(metadata.name().to_string(), "data type".to_string())
                .into(),
        )
    }

    /// Create a fill value from metadata.
    ///
    /// # Errors
    ///
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    #[allow(clippy::too_many_lines)]
    pub fn fill_value_from_metadata(
        &self,
        fill_value: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        NamedDataType::new(self.name(), self.clone()).fill_value_from_metadata(fill_value)
    }

    /// Create fill value metadata.
    ///
    /// # Errors
    ///
    /// Returns an [`DataTypeFillValueError`] if the metadata cannot be created from the fill value.
    #[allow(clippy::too_many_lines)]
    pub fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
        let error = || DataTypeFillValueError::new(self.name(), fill_value.clone());
        match self {
            Self::Bool => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                match bytes[0] {
                    0 => Ok(FillValueMetadataV3::from(false)),
                    1 => Ok(FillValueMetadataV3::from(true)),
                    _ => Err(error()),
                }
            }
            Self::Int2 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = i8::from_ne_bytes(bytes);
                if (-2..2).contains(&number) {
                    Ok(FillValueMetadataV3::from(number))
                } else {
                    Err(error())
                }
            }
            Self::Int4 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = i8::from_ne_bytes(bytes);
                if (-8..8).contains(&number) {
                    Ok(FillValueMetadataV3::from(number))
                } else {
                    Err(error())
                }
            }
            Self::Int8 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = i8::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::Int16 => {
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = i16::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::Int32 => {
                let bytes: [u8; 4] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = i32::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::Int64 => {
                let bytes: [u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = i64::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::UInt2 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = u8::from_ne_bytes(bytes);
                if (0..4).contains(&number) {
                    Ok(FillValueMetadataV3::from(number))
                } else {
                    Err(error())
                }
            }
            Self::UInt4 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = u8::from_ne_bytes(bytes);
                if (0..16).contains(&number) {
                    Ok(FillValueMetadataV3::from(number))
                } else {
                    Err(error())
                }
            }
            Self::UInt8 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = u8::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::UInt16 => {
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = u16::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::UInt32 => {
                let bytes: [u8; 4] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = u32::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::UInt64 => {
                let bytes: [u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = u64::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::Float8E4M3 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                #[cfg(feature = "float8")]
                {
                    let number = float8::F8E4M3::from_bits(bytes[0]);
                    Ok(FillValueMetadataV3::from(number.to_f64()))
                }
                #[cfg(not(feature = "float8"))]
                Ok(FillValueMetadataV3::from(byte_to_hex_string(bytes[0])))
            }
            Self::Float8E5M2 => {
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                #[cfg(feature = "float8")]
                {
                    let number = float8::F8E5M2::from_bits(bytes[0]);
                    Ok(FillValueMetadataV3::from(number.to_f64()))
                }
                #[cfg(not(feature = "float8"))]
                Ok(FillValueMetadataV3::from(byte_to_hex_string(bytes[0])))
            }
            Self::Float4E2M1FN
            | Self::Float6E2M3FN
            | Self::Float6E3M2FN
            | Self::Float8E3M4
            | Self::Float8E4M3B11FNUZ
            | Self::Float8E4M3FNUZ
            | Self::Float8E5M2FNUZ
            | Self::Float8E8M0FNU => {
                // FIXME: Support normal floating point fill value metadata for these data types.
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                Ok(FillValueMetadataV3::from(byte_to_hex_string(bytes[0])))
            }
            Self::ComplexFloat8E4M3 => {
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                #[cfg(feature = "float8")]
                {
                    let re = float8::F8E4M3::from_bits(bytes[0]);
                    let im = float8::F8E4M3::from_bits(bytes[1]);
                    let re = FillValueMetadataV3::from(re.to_f64());
                    let im = FillValueMetadataV3::from(im.to_f64());
                    Ok(FillValueMetadataV3::from([re, im]))
                }
                #[cfg(not(feature = "float8"))]
                {
                    let hex_string_re = FillValueMetadataV3::from(byte_to_hex_string(bytes[0]));
                    let hex_string_im = FillValueMetadataV3::from(byte_to_hex_string(bytes[1]));
                    Ok(FillValueMetadataV3::from([hex_string_re, hex_string_im]))
                }
            }
            Self::ComplexFloat8E5M2 => {
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                #[cfg(feature = "float8")]
                {
                    let re = float8::F8E5M2::from_bits(bytes[0]);
                    let im = float8::F8E5M2::from_bits(bytes[1]);
                    let re = FillValueMetadataV3::from(re.to_f64());
                    let im = FillValueMetadataV3::from(im.to_f64());
                    Ok(FillValueMetadataV3::from([re, im]))
                }
                #[cfg(not(feature = "float8"))]
                {
                    let hex_string_re = FillValueMetadataV3::from(byte_to_hex_string(bytes[0]));
                    let hex_string_im = FillValueMetadataV3::from(byte_to_hex_string(bytes[1]));
                    Ok(FillValueMetadataV3::from([hex_string_re, hex_string_im]))
                }
            }
            Self::ComplexFloat4E2M1FN
            | Self::ComplexFloat6E2M3FN
            | Self::ComplexFloat6E3M2FN
            | Self::ComplexFloat8E3M4
            | Self::ComplexFloat8E4M3B11FNUZ
            | Self::ComplexFloat8E4M3FNUZ
            | Self::ComplexFloat8E5M2FNUZ
            | Self::ComplexFloat8E8M0FNU => {
                // FIXME: Support normal floating point fill value metadata for these data types.
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let hex_string_re = FillValueMetadataV3::from(byte_to_hex_string(bytes[0]));
                let hex_string_im = FillValueMetadataV3::from(byte_to_hex_string(bytes[1]));
                Ok(FillValueMetadataV3::from([hex_string_re, hex_string_im]))
            }
            Self::BFloat16 => {
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = half::bf16::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::Float16 => {
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = half::f16::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::Float32 => {
                let bytes: [u8; 4] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = f32::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::Float64 => {
                let bytes: [u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = f64::from_ne_bytes(bytes);
                Ok(FillValueMetadataV3::from(number))
            }
            Self::ComplexBFloat16 => {
                let bytes: &[u8; 4] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let re =
                    half::bf16::from_ne_bytes(unsafe { bytes[0..2].try_into().unwrap_unchecked() });
                let im =
                    half::bf16::from_ne_bytes(unsafe { bytes[2..4].try_into().unwrap_unchecked() });
                let re = FillValueMetadataV3::from(re);
                let im = FillValueMetadataV3::from(im);
                Ok(FillValueMetadataV3::from([re, im]))
            }
            Self::ComplexFloat16 => {
                let bytes: &[u8; 4] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let re =
                    half::f16::from_ne_bytes(unsafe { bytes[0..2].try_into().unwrap_unchecked() });
                let im =
                    half::f16::from_ne_bytes(unsafe { bytes[2..4].try_into().unwrap_unchecked() });
                let re = FillValueMetadataV3::from(re);
                let im = FillValueMetadataV3::from(im);
                Ok(FillValueMetadataV3::from([re, im]))
            }
            Self::Complex64 | Self::ComplexFloat32 => {
                let bytes: &[u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let re = f32::from_ne_bytes(unsafe { bytes[0..4].try_into().unwrap_unchecked() });
                let im = f32::from_ne_bytes(unsafe { bytes[4..8].try_into().unwrap_unchecked() });
                let re = FillValueMetadataV3::from(re);
                let im = FillValueMetadataV3::from(im);
                Ok(FillValueMetadataV3::from([re, im]))
            }
            Self::Complex128 | Self::ComplexFloat64 => {
                let bytes: &[u8; 16] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let re = f64::from_ne_bytes(unsafe { bytes[0..8].try_into().unwrap_unchecked() });
                let im = f64::from_ne_bytes(unsafe { bytes[8..16].try_into().unwrap_unchecked() });
                let re = FillValueMetadataV3::from(re);
                let im = FillValueMetadataV3::from(im);
                Ok(FillValueMetadataV3::from([re, im]))
            }
            Self::RawBits(size) => {
                let bytes = fill_value.as_ne_bytes();
                if bytes.len() == *size {
                    Ok(FillValueMetadataV3::from(bytes))
                } else {
                    Err(error())
                }
            }
            Self::String => Ok(FillValueMetadataV3::from(
                String::from_utf8(fill_value.as_ne_bytes().to_vec()).map_err(|_| error())?,
            )),
            // Array representation [0, 255, 13, 74].
            // Replace with base64 implementation below when these land:
            // - https://github.com/zarr-developers/zarr-extensions/pull/38
            // - https://github.com/zarr-developers/zarr-python/pull/3559
            Self::Bytes => Ok(FillValueMetadataV3::from(fill_value.as_ne_bytes().to_vec())),
            // Self::Bytes => {
            //     let s = BASE64_STANDARD.encode(fill_value.as_ne_bytes());
            //     Ok(FillValueMetadataV3::from(s))
            // }
            Self::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | Self::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => {
                let bytes: [u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                let number = i64::from_ne_bytes(bytes);
                if number == i64::MIN {
                    Ok(FillValueMetadataV3::from("NaT"))
                } else {
                    Ok(FillValueMetadataV3::from(number))
                }
            }
            Self::Optional(data_type) => {
                if fill_value.size() == 0 {
                    Ok(FillValueMetadataV3::Null)
                } else {
                    data_type.metadata_fill_value(fill_value)
                }
            }
            Self::Extension(extension) => extension.metadata_fill_value(fill_value),
        }
    }
}

impl core::fmt::Display for DataType {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

fn byte_to_hex_string(byte: u8) -> String {
    let mut string = String::with_capacity(4);
    string.push('0');
    string.push('x');
    string.push(char::from_digit((byte / 16).into(), 16).unwrap());
    string.push(char::from_digit((byte % 16).into(), 16).unwrap());
    string
}

fn subfloat_hex_string_to_fill_value(fill_value: &FillValueMetadataV3) -> Option<FillValue> {
    if let Some(s) = fill_value.as_str() {
        if s.starts_with("0x") && s.len() == 4 {
            return u8::from_str_radix(&s[2..4], 16).ok().map(FillValue::from);
        }
    }
    None
}

fn complex_subfloat_hex_string_to_fill_value(
    fill_value: &FillValueMetadataV3,
) -> Option<FillValue> {
    if let Some([re, im]) = fill_value.as_array() {
        if let (Some(re), Some(im)) = (
            subfloat_hex_string_to_fill_value(re),
            subfloat_hex_string_to_fill_value(im),
        ) {
            return Some(FillValue::from([re.as_ne_bytes()[0], im.as_ne_bytes()[0]]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metadata::v3::{ZARR_NAN_BF16, ZARR_NAN_F16, ZARR_NAN_F32, ZARR_NAN_F64};

    #[test]
    fn data_type_unknown() {
        let json = r#""unknown""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default())
                .unwrap_err()
                .to_string(),
            "data type unknown is not supported"
        );
    }

    #[test]
    fn data_type_must_understand_false() {
        let json = r#"{"name":"unknown","must_understand": false}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        assert_eq!(
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default())
                .unwrap_err()
                .to_string(),
            r#"data type must not have `"must_understand": false`"#
        );
    }

    #[test]
    fn data_type_bool() {
        let json = r#""bool""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(format!("{}", data_type), zarrs_registry::data_type::BOOL);
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Bool);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("true").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), u8::from(true).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>("false").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Int2);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-1").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-1i8).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>("1").unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            1i8.to_ne_bytes()
        );
        let metadata = serde_json::from_str::<FillValueMetadataV3>("-3").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadataV3>("2").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
    }

    #[test]
    fn data_type_int4() {
        let json = r#""int4""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Int4);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i8).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>("7").unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            7i8.to_ne_bytes()
        );
        let metadata = serde_json::from_str::<FillValueMetadataV3>("8").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadataV3>("-9").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
    }

    #[test]
    fn data_type_int8() {
        let json = r#""int8""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Int8);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i8).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>("7").unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            7i8.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_int16() {
        let json = r#""int16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Int16);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i16).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>("7").unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            7i16.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_int32() {
        let json = r#""int32""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Int32);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i32).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>("7").unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            7i32.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_int64() {
        let json = r#""int64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Int64);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7i64).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>("7").unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            7i64.to_ne_bytes()
        );
    }

    #[test]
    fn data_type_uint2() {
        let json = r#""uint2""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::UInt2);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("3").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 3u8.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>("4").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadataV3>("-1").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
    }

    #[test]
    fn data_type_uint4() {
        let json = r#""uint4""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::UInt4);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("15").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), 15u8.to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );
        let metadata = serde_json::from_str::<FillValueMetadataV3>("16").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadataV3>("-1").unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err());
    }

    #[test]
    fn data_type_uint8() {
        let json = r#""uint8""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::UInt8);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::UInt16);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::UInt32);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::UInt64);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("7").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Float32);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7.0").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7.0f32).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""NaN""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F32.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""0x7fc00000""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            f32::NAN.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""Infinity""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            f32::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""-Infinity""#).unwrap()
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Float64);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7.0").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), (-7.0f64).to_ne_bytes());
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""0x7FF8000000000000""#)
                        .unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F64.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""NaN""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F64.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""Infinity""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            f64::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""-Infinity""#).unwrap()
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float4_e2m1fn");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0x0f""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float6_e2m3fn");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0x3f""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float6_e3m2fn");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0x3f""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e3m4");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e4m3");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e4m3");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        let metadata2 = serde_json::from_str::<FillValueMetadataV3>(r#"-0.3125"#).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata2,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""NaN""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert!(float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]).is_nan());

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""Infinity""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""-Infinity""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::NEG_INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"0"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-0"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::NEG_ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-1"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::NEG_ONE
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"1"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E4M3::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E4M3::ONE
        );
    }

    #[test]
    fn data_type_float8_e4m3b11fnuz() {
        let json = r#""float8_e4m3b11fnuz""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e4m3b11fnuz");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e4m3fnuz");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e5m2");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e5m2");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        let metadata2 = serde_json::from_str::<FillValueMetadataV3>(r#"-0.046875"#).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [170]);
        assert_eq!(
            metadata2,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""NaN""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert!(float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]).is_nan());

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""Infinity""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""-Infinity""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::NEG_INFINITY
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"0"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-0"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::NEG_ZERO
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-1"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::NEG_ONE
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"1"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(
            float8::F8E5M2::from_bits(fill_value.as_ne_bytes()[0]),
            float8::F8E5M2::ONE
        );
    }

    #[test]
    fn data_type_float8_e5m2fnuz() {
        let json = r#""float8_e5m2fnuz""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e5m2fnuz");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float8_e8m0fnu");

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0xaa""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "float16");

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7.0").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""NaN""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_F16.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""Infinity""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            f16::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""-Infinity""#).unwrap()
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "bfloat16");

        let metadata = serde_json::from_str::<FillValueMetadataV3>("-7.0").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
                .fill_value_from_metadata(
                    // &serde_json::from_str::<FillValueMetadataV3>(r#""0x7E00""#).unwrap()
                    &serde_json::from_str::<FillValueMetadataV3>(r#""0x7FC0""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_BF16.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""NaN""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            ZARR_NAN_BF16.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""Infinity""#).unwrap()
                )
                .unwrap()
                .as_ne_bytes(),
            bf16::INFINITY.to_ne_bytes()
        );

        assert_eq!(
            data_type
                .fill_value_from_metadata(
                    &serde_json::from_str::<FillValueMetadataV3>(r#""-Infinity""#).unwrap()
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::ComplexBFloat16);

        let metadata =
            serde_json::from_str::<FillValueMetadataV3>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-7.0"#).unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err())
    }

    #[test]
    fn data_type_complex_float16() {
        let json = r#""complex_float16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::ComplexFloat16);

        let metadata =
            serde_json::from_str::<FillValueMetadataV3>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-7.0"#).unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err())
    }

    #[test]
    fn data_type_complex_float32() {
        let json = r#""complex_float32""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::ComplexFloat32);

        let metadata =
            serde_json::from_str::<FillValueMetadataV3>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-7.0"#).unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err())
    }

    #[test]
    fn data_type_complexfloat64() {
        let json = r#""complex_float64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::ComplexFloat64);

        let metadata =
            serde_json::from_str::<FillValueMetadataV3>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-7.0"#).unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err())
    }

    #[test]
    fn data_type_complex64() {
        let json = r#""complex64""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Complex64);

        let metadata =
            serde_json::from_str::<FillValueMetadataV3>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-7.0"#).unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err())
    }

    #[test]
    fn data_type_complex128() {
        let json = r#""complex128""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Complex128);

        let metadata =
            serde_json::from_str::<FillValueMetadataV3>(r#"[-7.0, "Infinity"]"#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#"-7.0"#).unwrap();
        assert!(data_type.fill_value_from_metadata(&metadata).is_err())
    }

    #[test]
    fn data_type_r8() {
        let json = r#""r8""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "r8");
        assert_eq!(data_type.size(), DataTypeSize::Fixed(1));

        let metadata = serde_json::from_str::<FillValueMetadataV3>("[7]").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "r16");
        assert_eq!(data_type.size(), DataTypeSize::Fixed(2));

        let metadata = serde_json::from_str::<FillValueMetadataV3>("[0, 255]").unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        assert!(
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).is_err()
        );
    }

    #[test]
    fn data_type_unknown2() {
        let json = r#""datetime""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        println!("{json:?}");
        println!("{metadata:?}");
        assert_eq!(metadata.name(), "datetime");
        assert!(
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).is_err()
        );
    }

    #[test]
    fn data_type_unknown3() {
        let json = r#""ra""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        println!("{json:?}");
        println!("{metadata:?}");
        assert_eq!(metadata.name(), "ra");
        assert!(
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).is_err()
        );
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
        let data_type: DataType =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(data_type.size(), DataTypeSize::Fixed(2));
    }

    #[test]
    fn data_type_raw_bits2() {
        let json = r#"
    {
        "name": "r16"
    }"#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        let data_type: DataType =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(data_type.size(), DataTypeSize::Fixed(2));
    }

    #[test]
    fn data_type_raw_bits_failure1() {
        let json = r#"
    {
        "name": "r5"
    }"#;
        let metadata = serde_json::from_str::<MetadataV3>(json).unwrap();
        assert!(
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).is_err()
        );
    }

    #[test]
    fn incompatible_fill_value_metadata() {
        let json = r#""bool""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::Bool);

        let metadata = serde_json::from_str::<FillValueMetadataV3>("1").unwrap();
        assert_eq!(
            data_type
                .fill_value_from_metadata(&metadata)
                .unwrap_err()
                .to_string(),
            "incompatible fill value 1 for data type bool"
        );
    }

    #[test]
    fn incompatible_raw_bits_metadata() {
        let json = r#""r16""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type, DataType::RawBits(2));

        let metadata = serde_json::from_str::<FillValueMetadataV3>("[123]").unwrap();
        assert_eq!(serde_json::to_string(&metadata).unwrap(), "[123]");
        // assert_eq!(metadata.to_string(), "[123]");
        let fill_value_err = data_type.fill_value_from_metadata(&metadata).unwrap_err();
        assert_eq!(
            fill_value_err.to_string(),
            "incompatible fill value [123] for data type r16"
        );
    }

    #[test]
    fn float_fill_value() {
        assert_eq!(
            FillValueMetadataV3::from(half::f16::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(half::f16::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(ZARR_NAN_F16),
            serde_json::from_str(r#""NaN""#).unwrap()
        );
        let f16_nan_alt = unsafe { std::mem::transmute::<u16, half::f16>(0b01_11111_000000001) };
        assert!(f16_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadataV3::from(f16_nan_alt),
            serde_json::from_str(r#""0x7e01""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(bf16::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(bf16::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(ZARR_NAN_BF16),
            serde_json::from_str(r#""NaN""#).unwrap()
        );
        let bf16_nan_alt = unsafe { std::mem::transmute::<u16, bf16>(0b0_01111_11111000001) };
        assert!(bf16_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadataV3::from(bf16_nan_alt),
            serde_json::from_str(r#""0x7fc1""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(f32::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(f32::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(ZARR_NAN_F32),
            serde_json::from_str(r#""NaN""#).unwrap()
        );

        let f32_nan_alt = f32::from_bits(0b0_11111111_10000000000000000000001);
        assert!(f32_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadataV3::from(f32_nan_alt),
            serde_json::from_str(r#""0x7fc00001""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(f64::INFINITY),
            serde_json::from_str(r#""Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(f64::NEG_INFINITY),
            serde_json::from_str(r#""-Infinity""#).unwrap()
        );
        assert_eq!(
            FillValueMetadataV3::from(ZARR_NAN_F64),
            serde_json::from_str(r#""NaN""#).unwrap()
        );
        let f64_nan_alt =
            f64::from_bits(0b0_11111111111_1000000000000000000000000000000000000000000000000001);
        assert!(f64_nan_alt.is_nan());
        assert_eq!(
            FillValueMetadataV3::from(f64_nan_alt),
            serde_json::from_str(r#""0x7ff8000000000001""#).unwrap()
        );
    }

    #[test]
    fn incompatible_fill_value() {
        let err =
            DataTypeFillValueError::new(zarrs_registry::data_type::BOOL.to_string(), 1.0f32.into());
        assert_eq!(
            err.to_string(),
            "incompatible fill value [0, 0, 128, 63] for data type bool"
        );
    }

    #[test]
    fn fill_value_from_metadata_failure() {
        let metadata = serde_json::from_str::<FillValueMetadataV3>("1").unwrap();
        assert!(DataType::Bool.fill_value_from_metadata(&metadata).is_err());
        let metadata = serde_json::from_str::<FillValueMetadataV3>("false").unwrap();
        assert!(DataType::Int8.fill_value_from_metadata(&metadata).is_err());
        assert!(DataType::Int16.fill_value_from_metadata(&metadata).is_err());
        assert!(DataType::Int32.fill_value_from_metadata(&metadata).is_err());
        assert!(DataType::Int64.fill_value_from_metadata(&metadata).is_err());
        assert!(DataType::UInt8.fill_value_from_metadata(&metadata).is_err());
        assert!(DataType::UInt16
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::UInt32
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::UInt64
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::Float16
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::Float32
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::Float64
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::BFloat16
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::Complex64
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::Complex128
            .fill_value_from_metadata(&metadata)
            .is_err());
        assert!(DataType::RawBits(1)
            .fill_value_from_metadata(&metadata)
            .is_err());
    }

    #[test]
    fn data_type_string() {
        let json = r#""string""#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "string");
        assert_eq!(data_type.size(), DataTypeSize::Variable);

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""hello world""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), "hello world".as_bytes(),);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""Infinity""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), "Infinity".as_bytes(),);
        assert_eq!(
            metadata,
            data_type.metadata_fill_value(&fill_value).unwrap()
        );

        let metadata = serde_json::from_str::<FillValueMetadataV3>(r#""0x7fc00000""#).unwrap();
        let fill_value = data_type.fill_value_from_metadata(&metadata).unwrap();
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
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(json, serde_json::to_string(&data_type.metadata()).unwrap());
        assert_eq!(data_type.name(), "bytes");
        assert_eq!(data_type.size(), DataTypeSize::Variable);

        let expected_bytes = [0u8, 1, 2, 3];
        let metadata_from_arr: FillValueMetadataV3 =
            serde_json::from_str(r#"[0, 1, 2, 3]"#).unwrap();
        let fill_value_from_arr = data_type
            .fill_value_from_metadata(&metadata_from_arr)
            .unwrap();
        assert_eq!(fill_value_from_arr.as_ne_bytes(), expected_bytes,);

        let metadata_from_str: FillValueMetadataV3 = serde_json::from_str(r#""AAECAw==""#).unwrap();
        let fill_value_from_str = data_type
            .fill_value_from_metadata(&metadata_from_str)
            .unwrap();
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
        let json = r#"{"name":"optional","configuration":{"name":"int32","configuration":{}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(data_type.name(), "optional");
        if let DataType::Optional(inner) = &data_type {
            assert_eq!(inner.name(), "int32");
            assert_eq!(inner.size(), DataTypeSize::Fixed(4));
        } else {
            panic!("Expected Optional data type");
        }
        assert_eq!(data_type.size(), DataTypeSize::Fixed(4)); // inner type size (mask stored separately)

        // Test optional with complex configuration (numpy datetime64)
        let json = r#"{"name":"optional","configuration":{"name":"numpy.datetime64","configuration":{"unit":"s","scale_factor":1}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        assert_eq!(data_type.name(), "optional");
        if let DataType::Optional(inner) = &data_type {
            assert_eq!(inner.name(), "numpy.datetime64");
            if let DataType::NumpyDateTime64 { unit, scale_factor } = inner.as_ref() {
                assert_eq!(*unit, NumpyTimeUnit::Second);
                assert_eq!(scale_factor.get(), 1);
            } else {
                panic!("Expected NumpyDateTime64 inner type");
            }
        } else {
            panic!("Expected Optional data type");
        }

        // Test optional string (variable size)
        let json = r#"{"name":"optional","configuration":{"name":"string","configuration":{}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type =
            DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
        if let DataType::Optional(inner) = &data_type {
            assert_eq!(inner.name(), "string");
            assert_eq!(inner.size(), DataTypeSize::Variable);
        } else {
            panic!("Expected Optional data type");
        }
        assert_eq!(data_type.size(), DataTypeSize::Variable);

        // Test metadata roundtrip
        let expected_metadata = data_type.metadata();
        let roundtrip_data_type =
            DataType::from_metadata(&expected_metadata, &ExtensionAliasesDataTypeV3::default())
                .unwrap();
        assert_eq!(data_type, roundtrip_data_type);
    }

    #[test]
    fn data_type_optional_invalid_configuration() {
        // Test missing configuration
        let json = r#"{"name":"optional"}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let result = DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default());
        assert!(result.is_err());

        // Test empty configuration
        let json = r#"{"name":"optional","configuration":{}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let result = DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default());
        assert!(result.is_err());

        // Test invalid inner data type
        let json =
            r#"{"name":"optional","configuration":{"name":"unknown_type","configuration":{}}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let result = DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default());
        assert!(result.is_err());
    }
}

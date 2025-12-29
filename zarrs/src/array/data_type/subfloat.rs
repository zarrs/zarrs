//! Subfloat data type markers and implementations.

use super::macros::{impl_subfloat_data_type, register_data_type_plugin};

// Subfloats - No V2 equivalents

/// The `float4_e2m1fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float4E2M1FNDataType;
zarrs_plugin::impl_extension_aliases!(Float4E2M1FNDataType, "float4_e2m1fn");

/// The `float6_e2m3fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E2M3FNDataType;
zarrs_plugin::impl_extension_aliases!(Float6E2M3FNDataType, "float6_e2m3fn");

/// The `float6_e3m2fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E3M2FNDataType;
zarrs_plugin::impl_extension_aliases!(Float6E3M2FNDataType, "float6_e3m2fn");

/// The `float8_e3m4` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E3M4DataType;
zarrs_plugin::impl_extension_aliases!(Float8E3M4DataType, "float8_e3m4");

/// The `float8_e4m3` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3DataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3DataType, "float8_e4m3");

/// The `float8_e4m3b11fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3B11FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3B11FNUZDataType, "float8_e4m3b11fnuz");

/// The `float8_e4m3fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3FNUZDataType, "float8_e4m3fnuz");

/// The `float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2DataType;
zarrs_plugin::impl_extension_aliases!(Float8E5M2DataType, "float8_e5m2");

/// The `float8_e5m2fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E5M2FNUZDataType, "float8_e5m2fnuz");

/// The `float8_e8m0fnu` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E8M0FNUDataType;
zarrs_plugin::impl_extension_aliases!(Float8E8M0FNUDataType, "float8_e8m0fnu");

// DataTypeExtension implementations
// The second parameter is the bit size for packbits codec
impl_subfloat_data_type!(Float4E2M1FNDataType, 4);
impl_subfloat_data_type!(Float6E2M3FNDataType, 6);
impl_subfloat_data_type!(Float6E3M2FNDataType, 6);
impl_subfloat_data_type!(Float8E3M4DataType, 8);
impl_subfloat_data_type!(Float8E4M3B11FNUZDataType, 8);
impl_subfloat_data_type!(Float8E4M3FNUZDataType, 8);
impl_subfloat_data_type!(Float8E5M2FNUZDataType, 8);
impl_subfloat_data_type!(Float8E8M0FNUDataType, 8);

// Float8E4M3 and Float8E5M2 have special implementations when float8 feature is enabled
// to support NaN, Infinity, and numeric fill values
#[cfg(not(feature = "float8"))]
impl_subfloat_data_type!(Float8E4M3DataType, 8);
#[cfg(not(feature = "float8"))]
impl_subfloat_data_type!(Float8E5M2DataType, 8);

// Special Float8E4M3 implementation with float8 feature support
#[cfg(feature = "float8")]
mod float8_e4m3_impl {
    use super::Float8E4M3DataType;
    use std::borrow::Cow;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeExtensionBytesCodec, DataTypeExtensionBytesCodecError,
        DataTypeExtensionPackBitsCodec, DataTypeFillValueError, DataTypeFillValueMetadataError,
        FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, v3::FillValueMetadataV3};
    use zarrs_plugin::ExtensionIdentifier;

    impl DataTypeExtension for Float8E4M3DataType {
        fn identifier(&self) -> &'static str {
            <Self as ExtensionIdentifier>::IDENTIFIER
        }

        fn configuration(&self) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(1)
        }

        fn fill_value(
            &self,
            fill_value_metadata: &FillValueMetadataV3,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            let err = || {
                DataTypeFillValueMetadataError::new(
                    self.identifier().to_string(),
                    fill_value_metadata.clone(),
                )
            };

            // Handle hex string like "0xaa"
            if let Some(s) = fill_value_metadata.as_str() {
                if let Some(hex) = s.strip_prefix("0x")
                    && let Ok(byte) = u8::from_str_radix(hex, 16)
                {
                    return Ok(FillValue::from(byte));
                }
                // Handle special float values
                match s {
                    "NaN" => return Ok(FillValue::from(float8::F8E4M3::NAN.to_bits())),
                    "Infinity" => return Ok(FillValue::from(float8::F8E4M3::INFINITY.to_bits())),
                    "-Infinity" => {
                        return Ok(FillValue::from(float8::F8E4M3::NEG_INFINITY.to_bits()));
                    }
                    _ => {}
                }
            }

            // Handle numeric values (float or integer) - always convert via float8
            // This ensures that numeric values like 1, -1, 0.5 etc are properly
            // converted to their float8 representation
            if let Some(f) = fill_value_metadata.as_f64() {
                let f8 = float8::F8E4M3::from_f64(f);
                return Ok(FillValue::from(f8.to_bits()));
            }

            Err(err())
        }

        fn metadata_fill_value(
            &self,
            fill_value: &FillValue,
        ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
            let error =
                || DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone());
            let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
            let f8 = float8::F8E4M3::from_bits(bytes[0]);

            // Return special values as strings, numeric values as floats
            if f8.is_nan() {
                Ok(FillValueMetadataV3::from("NaN".to_string()))
            } else if f8 == float8::F8E4M3::INFINITY {
                Ok(FillValueMetadataV3::from("Infinity".to_string()))
            } else if f8 == float8::F8E4M3::NEG_INFINITY {
                Ok(FillValueMetadataV3::from("-Infinity".to_string()))
            } else {
                Ok(FillValueMetadataV3::from(f8.to_f64()))
            }
        }

        fn codec_bytes(&self) -> Option<&dyn DataTypeExtensionBytesCodec> {
            Some(self)
        }

        fn codec_packbits(&self) -> Option<&dyn DataTypeExtensionPackBitsCodec> {
            Some(self)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl DataTypeExtensionBytesCodec for Float8E4M3DataType {
        fn encode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }

        fn decode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }
    }

    impl DataTypeExtensionPackBitsCodec for Float8E4M3DataType {
        fn component_size_bits(&self) -> u64 {
            8
        }
        fn num_components(&self) -> u64 {
            1
        }
        fn sign_extension(&self) -> bool {
            false
        }
    }
}

// Special Float8E5M2 implementation with float8 feature support
#[cfg(feature = "float8")]
mod float8_e5m2_impl {
    use super::Float8E5M2DataType;
    use std::borrow::Cow;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeExtensionBytesCodec, DataTypeExtensionBytesCodecError,
        DataTypeExtensionPackBitsCodec, DataTypeFillValueError, DataTypeFillValueMetadataError,
        FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, v3::FillValueMetadataV3};
    use zarrs_plugin::ExtensionIdentifier;

    impl DataTypeExtension for Float8E5M2DataType {
        fn identifier(&self) -> &'static str {
            <Self as ExtensionIdentifier>::IDENTIFIER
        }

        fn configuration(&self) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(1)
        }

        fn fill_value(
            &self,
            fill_value_metadata: &FillValueMetadataV3,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            let err = || {
                DataTypeFillValueMetadataError::new(
                    self.identifier().to_string(),
                    fill_value_metadata.clone(),
                )
            };

            // Handle hex string like "0xaa"
            if let Some(s) = fill_value_metadata.as_str() {
                if let Some(hex) = s.strip_prefix("0x")
                    && let Ok(byte) = u8::from_str_radix(hex, 16)
                {
                    return Ok(FillValue::from(byte));
                }
                // Handle special float values
                match s {
                    "NaN" => return Ok(FillValue::from(float8::F8E5M2::NAN.to_bits())),
                    "Infinity" => return Ok(FillValue::from(float8::F8E5M2::INFINITY.to_bits())),
                    "-Infinity" => {
                        return Ok(FillValue::from(float8::F8E5M2::NEG_INFINITY.to_bits()));
                    }
                    _ => {}
                }
            }

            // Handle numeric values (float or integer) - always convert via float8
            // This ensures that numeric values like 1, -1, 0.5 etc are properly
            // converted to their float8 representation
            if let Some(f) = fill_value_metadata.as_f64() {
                let f8 = float8::F8E5M2::from_f64(f);
                return Ok(FillValue::from(f8.to_bits()));
            }

            Err(err())
        }

        fn metadata_fill_value(
            &self,
            fill_value: &FillValue,
        ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
            let error =
                || DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone());
            let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
            let f8 = float8::F8E5M2::from_bits(bytes[0]);

            // Return special values as strings, numeric values as floats
            if f8.is_nan() {
                Ok(FillValueMetadataV3::from("NaN".to_string()))
            } else if f8 == float8::F8E5M2::INFINITY {
                Ok(FillValueMetadataV3::from("Infinity".to_string()))
            } else if f8 == float8::F8E5M2::NEG_INFINITY {
                Ok(FillValueMetadataV3::from("-Infinity".to_string()))
            } else {
                Ok(FillValueMetadataV3::from(f8.to_f64()))
            }
        }

        fn codec_bytes(&self) -> Option<&dyn DataTypeExtensionBytesCodec> {
            Some(self)
        }

        fn codec_packbits(&self) -> Option<&dyn DataTypeExtensionPackBitsCodec> {
            Some(self)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl DataTypeExtensionBytesCodec for Float8E5M2DataType {
        fn encode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }

        fn decode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }
    }

    impl DataTypeExtensionPackBitsCodec for Float8E5M2DataType {
        fn component_size_bits(&self) -> u64 {
            8
        }
        fn num_components(&self) -> u64 {
            1
        }
        fn sign_extension(&self) -> bool {
            false
        }
    }
}

// Plugin registrations
register_data_type_plugin!(Float4E2M1FNDataType);
register_data_type_plugin!(Float6E2M3FNDataType);
register_data_type_plugin!(Float6E3M2FNDataType);
register_data_type_plugin!(Float8E3M4DataType);
register_data_type_plugin!(Float8E4M3DataType);
register_data_type_plugin!(Float8E4M3B11FNUZDataType);
register_data_type_plugin!(Float8E4M3FNUZDataType);
register_data_type_plugin!(Float8E5M2DataType);
register_data_type_plugin!(Float8E5M2FNUZDataType);
register_data_type_plugin!(Float8E8M0FNUDataType);

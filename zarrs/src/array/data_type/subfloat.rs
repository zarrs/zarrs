//! Subfloat data types (sub-byte floating point formats).

use super::macros::{impl_bytes_codec_passthrough, impl_packbits_codec, register_data_type_plugin};

/// Macro to implement `DataTypeExtension` for subfloat types (single-byte floating point formats).
macro_rules! impl_subfloat_data_type {
    ($marker:ty) => {
        impl zarrs_data_type::DataTypeExtension for $marker {
            fn identifier(&self) -> &'static str {
                <Self as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER
            }

            fn configuration(&self) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed(1)
            }

            fn fill_value(
                &self,
                fill_value_metadata: &zarrs_metadata::v3::FillValueMetadataV3,
            ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError>
            {
                let err = || {
                    zarrs_data_type::DataTypeFillValueMetadataError::new(
                        self.identifier().to_string(),
                        fill_value_metadata.clone(),
                    )
                };
                // Subfloats use hex string representation like "0x00"
                if let Some(s) = fill_value_metadata.as_str() {
                    if let Some(hex) = s.strip_prefix("0x") {
                        if let Ok(byte) = u8::from_str_radix(hex, 16) {
                            return Ok(zarrs_data_type::FillValue::from(byte));
                        }
                    }
                }
                // Also accept integer values in range
                if let Some(int) = fill_value_metadata.as_u64() {
                    if let Ok(byte) = u8::try_from(int) {
                        return Ok(zarrs_data_type::FillValue::from(byte));
                    }
                }
                Err(err())
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<
                zarrs_metadata::v3::FillValueMetadataV3,
                zarrs_data_type::DataTypeFillValueError,
            > {
                let error = || {
                    zarrs_data_type::DataTypeFillValueError::new(
                        self.identifier().to_string(),
                        fill_value.clone(),
                    )
                };
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                // Return as hex string
                Ok(zarrs_metadata::v3::FillValueMetadataV3::from(format!(
                    "0x{:02x}",
                    bytes[0]
                )))
            }

            fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
                Some(self)
            }

            fn codec_packbits(
                &self,
            ) -> Option<&dyn zarrs_data_type::DataTypeExtensionPackBitsCodec> {
                Some(self)
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

/// The `float4_e2m1fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float4E2M1FNDataType;
zarrs_plugin::impl_extension_aliases!(Float4E2M1FNDataType, "float4_e2m1fn");
register_data_type_plugin!(Float4E2M1FNDataType);

/// The `float6_e2m3fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E2M3FNDataType;
zarrs_plugin::impl_extension_aliases!(Float6E2M3FNDataType, "float6_e2m3fn");
register_data_type_plugin!(Float6E2M3FNDataType);

/// The `float6_e3m2fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E3M2FNDataType;
zarrs_plugin::impl_extension_aliases!(Float6E3M2FNDataType, "float6_e3m2fn");
register_data_type_plugin!(Float6E3M2FNDataType);

/// The `float8_e3m4` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E3M4DataType;
zarrs_plugin::impl_extension_aliases!(Float8E3M4DataType, "float8_e3m4");
register_data_type_plugin!(Float8E3M4DataType);

/// The `float8_e4m3b11fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3B11FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3B11FNUZDataType, "float8_e4m3b11fnuz");
register_data_type_plugin!(Float8E4M3B11FNUZDataType);

/// The `float8_e4m3fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3FNUZDataType, "float8_e4m3fnuz");
register_data_type_plugin!(Float8E4M3FNUZDataType);

/// The `float8_e5m2fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E5M2FNUZDataType, "float8_e5m2fnuz");
register_data_type_plugin!(Float8E5M2FNUZDataType);

/// The `float8_e8m0fnu` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E8M0FNUDataType;
zarrs_plugin::impl_extension_aliases!(Float8E8M0FNUDataType, "float8_e8m0fnu");
register_data_type_plugin!(Float8E8M0FNUDataType);

// DataTypeExtension implementations for subfloats
impl_subfloat_data_type!(Float4E2M1FNDataType);
impl_subfloat_data_type!(Float6E2M3FNDataType);
impl_subfloat_data_type!(Float6E3M2FNDataType);
impl_subfloat_data_type!(Float8E3M4DataType);
impl_subfloat_data_type!(Float8E4M3B11FNUZDataType);
impl_subfloat_data_type!(Float8E4M3FNUZDataType);
impl_subfloat_data_type!(Float8E5M2FNUZDataType);
impl_subfloat_data_type!(Float8E8M0FNUDataType);

// PackBits codec implementations for subfloats
impl_packbits_codec!(Float4E2M1FNDataType, 4, float, 1);
impl_packbits_codec!(Float6E2M3FNDataType, 6, float, 1);
impl_packbits_codec!(Float6E3M2FNDataType, 6, float, 1);
impl_packbits_codec!(Float8E3M4DataType, 8, float, 1);
impl_packbits_codec!(Float8E4M3B11FNUZDataType, 8, float, 1);
impl_packbits_codec!(Float8E4M3FNUZDataType, 8, float, 1);
impl_packbits_codec!(Float8E5M2FNUZDataType, 8, float, 1);
impl_packbits_codec!(Float8E8M0FNUDataType, 8, float, 1);

// Bytes codec implementations for subfloats (passthrough - single byte, no endianness conversion)
impl_bytes_codec_passthrough!(Float4E2M1FNDataType);
impl_bytes_codec_passthrough!(Float6E2M3FNDataType);
impl_bytes_codec_passthrough!(Float6E3M2FNDataType);
impl_bytes_codec_passthrough!(Float8E3M4DataType);
impl_bytes_codec_passthrough!(Float8E4M3B11FNUZDataType);
impl_bytes_codec_passthrough!(Float8E4M3FNUZDataType);
impl_bytes_codec_passthrough!(Float8E5M2FNUZDataType);
impl_bytes_codec_passthrough!(Float8E8M0FNUDataType);

//! Subfloat data types (sub-byte floating point formats).

use super::macros::register_data_type_plugin;

/// Macro to implement `DataTypeTraits` for subfloat types (single-byte floating point formats).
macro_rules! impl_subfloat_data_type {
    ($marker:ty) => {
        impl zarrs_data_type::DataTypeTraits for $marker {
            fn configuration(
                &self,
                _version: zarrs_plugin::ZarrVersions,
            ) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed(1)
            }

            fn fill_value(
                &self,
                fill_value_metadata: &zarrs_metadata::FillValueMetadata,
                _version: zarrs_plugin::ZarrVersions,
            ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError>
            {
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
                Err(zarrs_data_type::DataTypeFillValueMetadataError)
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError>
            {
                let bytes: [u8; 1] = fill_value
                    .as_ne_bytes()
                    .try_into()
                    .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
                // Return as hex string
                Ok(zarrs_metadata::FillValueMetadata::from(format!(
                    "0x{:02x}",
                    bytes[0]
                )))
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
register_data_type_plugin!(Float4E2M1FNDataType);
zarrs_plugin::impl_extension_aliases!(Float4E2M1FNDataType, v3: "float4_e2m1fn");

/// The `float6_e2m3fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E2M3FNDataType;
register_data_type_plugin!(Float6E2M3FNDataType);
zarrs_plugin::impl_extension_aliases!(Float6E2M3FNDataType, v3: "float6_e2m3fn");

/// The `float6_e3m2fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E3M2FNDataType;
register_data_type_plugin!(Float6E3M2FNDataType);
zarrs_plugin::impl_extension_aliases!(Float6E3M2FNDataType, v3: "float6_e3m2fn");

/// The `float8_e3m4` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E3M4DataType;
register_data_type_plugin!(Float8E3M4DataType);
zarrs_plugin::impl_extension_aliases!(Float8E3M4DataType, v3: "float8_e3m4");

/// The `float8_e4m3b11fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3B11FNUZDataType;
register_data_type_plugin!(Float8E4M3B11FNUZDataType);
zarrs_plugin::impl_extension_aliases!(Float8E4M3B11FNUZDataType, v3: "float8_e4m3b11fnuz");

/// The `float8_e4m3fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3FNUZDataType;
register_data_type_plugin!(Float8E4M3FNUZDataType);
zarrs_plugin::impl_extension_aliases!(Float8E4M3FNUZDataType, v3: "float8_e4m3fnuz");

/// The `float8_e5m2fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2FNUZDataType;
register_data_type_plugin!(Float8E5M2FNUZDataType);
zarrs_plugin::impl_extension_aliases!(Float8E5M2FNUZDataType, v3: "float8_e5m2fnuz");

/// The `float8_e8m0fnu` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E8M0FNUDataType;
register_data_type_plugin!(Float8E8M0FNUDataType);
zarrs_plugin::impl_extension_aliases!(Float8E8M0FNUDataType, v3: "float8_e8m0fnu");

// DataTypeTraits implementations for subfloats
impl_subfloat_data_type!(Float4E2M1FNDataType);
impl_subfloat_data_type!(Float6E2M3FNDataType);
impl_subfloat_data_type!(Float6E3M2FNDataType);
impl_subfloat_data_type!(Float8E3M4DataType);
impl_subfloat_data_type!(Float8E4M3B11FNUZDataType);
impl_subfloat_data_type!(Float8E4M3FNUZDataType);
impl_subfloat_data_type!(Float8E5M2FNUZDataType);
impl_subfloat_data_type!(Float8E8M0FNUDataType);

// PackBits codec implementations for subfloats
use crate::array::codec::impl_packbits_codec;
impl_packbits_codec!(Float4E2M1FNDataType, 4, float, 1);
impl_packbits_codec!(Float6E2M3FNDataType, 6, float, 1);
impl_packbits_codec!(Float6E3M2FNDataType, 6, float, 1);
impl_packbits_codec!(Float8E3M4DataType, 8, float, 1);
impl_packbits_codec!(Float8E4M3B11FNUZDataType, 8, float, 1);
impl_packbits_codec!(Float8E4M3FNUZDataType, 8, float, 1);
impl_packbits_codec!(Float8E5M2FNUZDataType, 8, float, 1);
impl_packbits_codec!(Float8E8M0FNUDataType, 8, float, 1);

// Bytes codec implementations for subfloats (passthrough - single byte, no endianness conversion)
use crate::array::codec::impl_bytes_codec_passthrough;
impl_bytes_codec_passthrough!(Float4E2M1FNDataType);
impl_bytes_codec_passthrough!(Float6E2M3FNDataType);
impl_bytes_codec_passthrough!(Float6E3M2FNDataType);
impl_bytes_codec_passthrough!(Float8E3M4DataType);
impl_bytes_codec_passthrough!(Float8E4M3B11FNUZDataType);
impl_bytes_codec_passthrough!(Float8E4M3FNUZDataType);
impl_bytes_codec_passthrough!(Float8E5M2FNUZDataType);
impl_bytes_codec_passthrough!(Float8E8M0FNUDataType);

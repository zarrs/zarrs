//! Complex subfloat data types (two subfloats packed together).

use crate::{impl_bytes_codec_passthrough, impl_packbits_codec};

use super::macros::register_data_type_plugin;

/// Macro to implement `DataTypeExtension` for complex subfloat types (two subfloats packed together).
macro_rules! impl_complex_subfloat_data_type {
    ($marker:ty) => {
        impl zarrs_data_type::DataTypeExtension for $marker {
            fn identifier(&self) -> &'static str {
                <Self as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER
            }

            fn configuration(&self) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed(2)
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
                // Complex subfloats use array of two hex strings like ["0x00", "0x00"]
                if let Some([re, im]) = fill_value_metadata.as_array() {
                    let parse_hex = |v: &zarrs_metadata::v3::FillValueMetadataV3| -> Option<u8> {
                        if let Some(s) = v.as_str() {
                            if let Some(hex) = s.strip_prefix("0x") {
                                return u8::from_str_radix(hex, 16).ok();
                            }
                        }
                        if let Some(int) = v.as_u64() {
                            return u8::try_from(int).ok();
                        }
                        None
                    };
                    if let (Some(re_byte), Some(im_byte)) = (parse_hex(re), parse_hex(im)) {
                        return Ok(zarrs_data_type::FillValue::from([re_byte, im_byte]));
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
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                // Return as array of hex strings
                Ok(zarrs_metadata::v3::FillValueMetadataV3::from(vec![
                    zarrs_metadata::v3::FillValueMetadataV3::from(format!("0x{:02x}", bytes[0])),
                    zarrs_metadata::v3::FillValueMetadataV3::from(format!("0x{:02x}", bytes[1])),
                ]))
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

/// The `complex_float4_e2m1fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat4E2M1FNDataType;
register_data_type_plugin!(ComplexFloat4E2M1FNDataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat4E2M1FNDataType, "complex_float4_e2m1fn");

/// The `complex_float6_e2m3fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat6E2M3FNDataType;
register_data_type_plugin!(ComplexFloat6E2M3FNDataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat6E2M3FNDataType, "complex_float6_e2m3fn");

/// The `complex_float6_e3m2fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat6E3M2FNDataType;
register_data_type_plugin!(ComplexFloat6E3M2FNDataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat6E3M2FNDataType, "complex_float6_e3m2fn");

/// The `complex_float8_e3m4` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E3M4DataType;
register_data_type_plugin!(ComplexFloat8E3M4DataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E3M4DataType, "complex_float8_e3m4");

/// The `complex_float8_e4m3b11fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3B11FNUZDataType;
register_data_type_plugin!(ComplexFloat8E4M3B11FNUZDataType);
zarrs_plugin::impl_extension_aliases!(
    ComplexFloat8E4M3B11FNUZDataType,
    "complex_float8_e4m3b11fnuz"
);

/// The `complex_float8_e4m3fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3FNUZDataType;
register_data_type_plugin!(ComplexFloat8E4M3FNUZDataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E4M3FNUZDataType, "complex_float8_e4m3fnuz");

/// The `complex_float8_e5m2fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E5M2FNUZDataType;
register_data_type_plugin!(ComplexFloat8E5M2FNUZDataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E5M2FNUZDataType, "complex_float8_e5m2fnuz");

/// The `complex_float8_e8m0fnu` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E8M0FNUDataType;
register_data_type_plugin!(ComplexFloat8E8M0FNUDataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E8M0FNUDataType, "complex_float8_e8m0fnu");

// DataTypeExtension implementations for complex subfloats
impl_complex_subfloat_data_type!(ComplexFloat4E2M1FNDataType);
impl_complex_subfloat_data_type!(ComplexFloat6E2M3FNDataType);
impl_complex_subfloat_data_type!(ComplexFloat6E3M2FNDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E3M4DataType);
impl_complex_subfloat_data_type!(ComplexFloat8E4M3B11FNUZDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E4M3FNUZDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E5M2FNUZDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E8M0FNUDataType);

// PackBits codec implementations for complex subfloats
impl_packbits_codec!(ComplexFloat4E2M1FNDataType, 4, float, 2);
impl_packbits_codec!(ComplexFloat6E2M3FNDataType, 6, float, 2);
impl_packbits_codec!(ComplexFloat6E3M2FNDataType, 6, float, 2);
impl_packbits_codec!(ComplexFloat8E3M4DataType, 8, float, 2);
impl_packbits_codec!(ComplexFloat8E4M3B11FNUZDataType, 8, float, 2);
impl_packbits_codec!(ComplexFloat8E4M3FNUZDataType, 8, float, 2);
impl_packbits_codec!(ComplexFloat8E5M2FNUZDataType, 8, float, 2);
impl_packbits_codec!(ComplexFloat8E8M0FNUDataType, 8, float, 2);

// Bytes codec implementations for complex subfloats (passthrough - two single-byte components)
impl_bytes_codec_passthrough!(ComplexFloat4E2M1FNDataType);
impl_bytes_codec_passthrough!(ComplexFloat6E2M3FNDataType);
impl_bytes_codec_passthrough!(ComplexFloat6E3M2FNDataType);
impl_bytes_codec_passthrough!(ComplexFloat8E3M4DataType);
impl_bytes_codec_passthrough!(ComplexFloat8E4M3B11FNUZDataType);
impl_bytes_codec_passthrough!(ComplexFloat8E4M3FNUZDataType);
impl_bytes_codec_passthrough!(ComplexFloat8E5M2FNUZDataType);
impl_bytes_codec_passthrough!(ComplexFloat8E8M0FNUDataType);

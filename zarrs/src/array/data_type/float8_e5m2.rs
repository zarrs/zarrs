//! The `float8_e5m2` data type.

use super::macros::{impl_bytes_codec_passthrough, impl_packbits_codec, register_data_type_plugin};

/// The `float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2DataType;
register_data_type_plugin!(Float8E5M2DataType);
zarrs_plugin::impl_extension_aliases!(Float8E5M2DataType, "float8_e5m2");

// Default implementation when float8 feature is not enabled
#[cfg(not(feature = "float8"))]
mod impl_default {
    use super::Float8E5M2DataType;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue,
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
            // Subfloats use hex string representation like "0x00"
            if let Some(s) = fill_value_metadata.as_str() {
                if let Some(hex) = s.strip_prefix("0x") {
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        return Ok(FillValue::from(byte));
                    }
                }
            }
            // Also accept integer values in range
            if let Some(int) = fill_value_metadata.as_u64() {
                if let Ok(byte) = u8::try_from(int) {
                    return Ok(FillValue::from(byte));
                }
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
            // Return as hex string
            Ok(FillValueMetadataV3::from(format!("0x{:02x}", bytes[0])))
        }

        fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
            Some(self)
        }

        fn codec_packbits(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPackBitsCodec> {
            Some(self)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}

// Special Float8E5M2 implementation with float8 feature support
#[cfg(feature = "float8")]
mod impl_float8 {
    use super::Float8E5M2DataType;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue,
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

        fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
            Some(self)
        }

        fn codec_packbits(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPackBitsCodec> {
            Some(self)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}

// Codec implementations (same for both feature configurations)
impl_packbits_codec!(Float8E5M2DataType, 8, float, 1);
impl_bytes_codec_passthrough!(Float8E5M2DataType);

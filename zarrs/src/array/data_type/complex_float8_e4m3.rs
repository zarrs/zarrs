//! The `complex_float8_e4m3` data type.

use super::macros::register_data_type_plugin;

/// The `complex_float8_e4m3` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3DataType;
register_data_type_plugin!(ComplexFloat8E4M3DataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E4M3DataType, "complex_float8_e4m3");

// Default implementation when float8 feature is not enabled
#[cfg(not(feature = "float8"))]
mod impl_default {
    use super::ComplexFloat8E4M3DataType;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, v3::FillValueMetadataV3};
    use zarrs_plugin::ExtensionIdentifier;

    impl DataTypeExtension for ComplexFloat8E4M3DataType {
        fn identifier(&self) -> &'static str {
            <Self as ExtensionIdentifier>::IDENTIFIER
        }

        fn configuration(&self) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(2)
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
            // Complex subfloats use array of two hex strings like ["0x00", "0x00"]
            if let Some([re, im]) = fill_value_metadata.as_array() {
                let parse_hex = |v: &FillValueMetadataV3| -> Option<u8> {
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
                    return Ok(FillValue::from([re_byte, im_byte]));
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
            let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
            // Return as array of hex strings
            Ok(FillValueMetadataV3::from(vec![
                FillValueMetadataV3::from(format!("0x{:02x}", bytes[0])),
                FillValueMetadataV3::from(format!("0x{:02x}", bytes[1])),
            ]))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}

// Special ComplexFloat8E4M3 implementation with float8 feature support
#[cfg(feature = "float8")]
mod impl_float8 {
    use super::ComplexFloat8E4M3DataType;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, v3::FillValueMetadataV3};
    use zarrs_plugin::ExtensionIdentifier;

    impl DataTypeExtension for ComplexFloat8E4M3DataType {
        fn identifier(&self) -> &'static str {
            <Self as ExtensionIdentifier>::IDENTIFIER
        }

        fn configuration(&self) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(2)
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

            // Complex fill values are arrays of two elements [re, im]
            if let Some([re, im]) = fill_value_metadata.as_array() {
                let parse_component = |v: &FillValueMetadataV3| -> Option<u8> {
                    // Handle hex string like "0xaa"
                    if let Some(s) = v.as_str() {
                        if let Some(hex) = s.strip_prefix("0x") {
                            return u8::from_str_radix(hex, 16).ok();
                        }
                        // Handle special float values
                        match s {
                            "NaN" => return Some(float8::F8E4M3::NAN.to_bits()),
                            "Infinity" => return Some(float8::F8E4M3::INFINITY.to_bits()),
                            "-Infinity" => return Some(float8::F8E4M3::NEG_INFINITY.to_bits()),
                            _ => {}
                        }
                    }
                    // Handle numeric values (float or integer) - convert via float8
                    if let Some(f) = v.as_f64() {
                        return Some(float8::F8E4M3::from_f64(f).to_bits());
                    }
                    None
                };
                if let (Some(re_byte), Some(im_byte)) = (parse_component(re), parse_component(im)) {
                    return Ok(FillValue::from([re_byte, im_byte]));
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
            let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;

            let component_to_metadata = |byte: u8| -> FillValueMetadataV3 {
                let f8 = float8::F8E4M3::from_bits(byte);
                if f8.is_nan() {
                    FillValueMetadataV3::from("NaN".to_string())
                } else if f8 == float8::F8E4M3::INFINITY {
                    FillValueMetadataV3::from("Infinity".to_string())
                } else if f8 == float8::F8E4M3::NEG_INFINITY {
                    FillValueMetadataV3::from("-Infinity".to_string())
                } else {
                    FillValueMetadataV3::from(f8.to_f64())
                }
            };

            Ok(FillValueMetadataV3::from(vec![
                component_to_metadata(bytes[0]),
                component_to_metadata(bytes[1]),
            ]))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}

crate::array::codec::impl_packbits_codec!(ComplexFloat8E4M3DataType, 8, float, 2);
crate::array::codec::impl_bytes_codec_passthrough!(ComplexFloat8E4M3DataType);

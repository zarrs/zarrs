//! The `float8_e4m3` data type.

use super::macros::register_data_type_plugin;

/// The `float8_e4m3` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3DataType;
register_data_type_plugin!(Float8E4M3DataType);
zarrs_plugin::impl_extension_aliases!(Float8E4M3DataType, v3: "float8_e4m3");

// Default implementation when float8 feature is not enabled
#[cfg(not(feature = "float8"))]
mod impl_default {
    use super::Float8E4M3DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for Float8E4M3DataType {
        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(1)
        }

        fn fill_value(
            &self,
            fill_value_metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
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
            Err(DataTypeFillValueMetadataError)
        }

        fn metadata_fill_value(
            &self,
            fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            let bytes: [u8; 1] = fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError)?;
            // Return as hex string
            Ok(FillValueMetadata::from(format!("0x{:02x}", bytes[0])))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        // No compatible element types when float8 feature is disabled
    }
}

// Special Float8E4M3 implementation with float8 feature support
#[cfg(feature = "float8")]
mod impl_float8 {
    use super::Float8E4M3DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for Float8E4M3DataType {
        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(1)
        }

        fn fill_value(
            &self,
            fill_value_metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
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

            Err(DataTypeFillValueMetadataError)
        }

        fn metadata_fill_value(
            &self,
            fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            let bytes: [u8; 1] = fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError)?;
            let f8 = float8::F8E4M3::from_bits(bytes[0]);

            // Return special values as strings, numeric values as floats
            if f8.is_nan() {
                Ok(FillValueMetadata::from("NaN".to_string()))
            } else if f8 == float8::F8E4M3::INFINITY {
                Ok(FillValueMetadata::from("Infinity".to_string()))
            } else if f8 == float8::F8E4M3::NEG_INFINITY {
                Ok(FillValueMetadata::from("-Infinity".to_string()))
            } else {
                Ok(FillValueMetadata::from(f8.to_f64()))
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
            const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<float8::F8E4M3>()];
            &TYPES
        }
    }
}

zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(Float8E4M3DataType, 8, float, 1);
zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(Float8E4M3DataType, 1);

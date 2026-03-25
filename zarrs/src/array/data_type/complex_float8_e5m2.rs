//! The `complex_float8_e5m2` data type.

use super::macros::register_data_type_plugin;

/// The `complex_float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E5M2DataType;
register_data_type_plugin!(ComplexFloat8E5M2DataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E5M2DataType, v3: "complex_float8_e5m2");

// Default implementation when float8 feature is not enabled
#[cfg(not(feature = "float8"))]
mod impl_default {
    use super::ComplexFloat8E5M2DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for ComplexFloat8E5M2DataType {
        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(2)
        }

        fn fill_value(
            &self,
            fill_value_metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            // Complex subfloats use array of two hex strings like ["0x00", "0x00"]
            if let Some([re, im]) = fill_value_metadata.as_array() {
                let parse_hex = |v: &FillValueMetadata| -> Option<u8> {
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
            Err(DataTypeFillValueMetadataError)
        }

        fn metadata_fill_value(
            &self,
            fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            let bytes: [u8; 2] = fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError)?;
            // Return as array of hex strings
            Ok(FillValueMetadata::from(vec![
                FillValueMetadata::from(format!("0x{:02x}", bytes[0])),
                FillValueMetadata::from(format!("0x{:02x}", bytes[1])),
            ]))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        // No compatible element types when float8 feature is disabled
    }
}

// Special ComplexFloat8E5M2 implementation with float8 feature support
#[cfg(feature = "float8")]
mod impl_float8 {
    use super::ComplexFloat8E5M2DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for ComplexFloat8E5M2DataType {
        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(2)
        }

        fn fill_value(
            &self,
            fill_value_metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            // Complex fill values are arrays of two elements [re, im]
            if let Some([re, im]) = fill_value_metadata.as_array() {
                let parse_component = |v: &FillValueMetadata| -> Option<u8> {
                    // Handle hex string like "0xaa"
                    if let Some(s) = v.as_str() {
                        if let Some(hex) = s.strip_prefix("0x") {
                            return u8::from_str_radix(hex, 16).ok();
                        }
                        // Handle special float values
                        match s {
                            "NaN" => return Some(float8::F8E5M2::NAN.to_bits()),
                            "Infinity" => return Some(float8::F8E5M2::INFINITY.to_bits()),
                            "-Infinity" => return Some(float8::F8E5M2::NEG_INFINITY.to_bits()),
                            _ => {}
                        }
                    }
                    // Handle numeric values (float or integer) - convert via float8
                    if let Some(f) = v.as_f64() {
                        return Some(float8::F8E5M2::from_f64(f).to_bits());
                    }
                    None
                };
                if let (Some(re_byte), Some(im_byte)) = (parse_component(re), parse_component(im)) {
                    return Ok(FillValue::from([re_byte, im_byte]));
                }
            }
            Err(DataTypeFillValueMetadataError)
        }

        fn metadata_fill_value(
            &self,
            fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            let bytes: [u8; 2] = fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError)?;

            let component_to_metadata = |byte: u8| -> FillValueMetadata {
                let f8 = float8::F8E5M2::from_bits(byte);
                if f8.is_nan() {
                    FillValueMetadata::from("NaN".to_string())
                } else if f8 == float8::F8E5M2::INFINITY {
                    FillValueMetadata::from("Infinity".to_string())
                } else if f8 == float8::F8E5M2::NEG_INFINITY {
                    FillValueMetadata::from("-Infinity".to_string())
                } else {
                    FillValueMetadata::from(f8.to_f64())
                }
            };

            Ok(FillValueMetadata::from(vec![
                component_to_metadata(bytes[0]),
                component_to_metadata(bytes[1]),
            ]))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
            const TYPES: [std::any::TypeId; 1] =
                [std::any::TypeId::of::<num::Complex<float8::F8E5M2>>()];
            &TYPES
        }
    }
}

zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(
    ComplexFloat8E5M2DataType,
    8,
    float,
    2
);
zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(ComplexFloat8E5M2DataType, 1);

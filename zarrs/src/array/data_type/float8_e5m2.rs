//! The `float8_e5m2` data type.

use super::macros::register_data_type_plugin;

/// The `float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2DataType;
register_data_type_plugin!(Float8E5M2DataType);
zarrs_plugin::impl_extension_aliases!(Float8E5M2DataType, v3: "float8_e5m2");

// Default implementation when no concrete float8 element implementation is enabled
#[cfg(not(any(feature = "float8", feature = "microfloat")))]
mod impl_default {
    use super::Float8E5M2DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for Float8E5M2DataType {
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

// Special Float8E5M2 implementation with float8 feature support
#[cfg(all(feature = "float8", not(feature = "microfloat")))]
mod impl_float8 {
    use super::Float8E5M2DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for Float8E5M2DataType {
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
            let f8 = float8::F8E5M2::from_bits(bytes[0]);

            // Return special values as strings, numeric values as floats
            if f8.is_nan() {
                Ok(FillValueMetadata::from("NaN".to_string()))
            } else if f8 == float8::F8E5M2::INFINITY {
                Ok(FillValueMetadata::from("Infinity".to_string()))
            } else if f8 == float8::F8E5M2::NEG_INFINITY {
                Ok(FillValueMetadata::from("-Infinity".to_string()))
            } else {
                Ok(FillValueMetadata::from(f8.to_f64()))
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
            const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<float8::F8E5M2>()];
            &TYPES
        }
    }
}

// Special Float8E5M2 implementation with microfloat feature support
#[cfg(all(not(feature = "float8"), feature = "microfloat"))]
mod impl_microfloat {
    use super::Float8E5M2DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for Float8E5M2DataType {
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
            if let Some(s) = fill_value_metadata.as_str() {
                if let Some(hex) = s.strip_prefix("0x")
                    && let Ok(byte) = u8::from_str_radix(hex, 16)
                {
                    return Ok(FillValue::from(byte));
                }
                match s {
                    "NaN" => return Ok(FillValue::from(microfloat::f8e5m2::NAN.to_bits())),
                    "Infinity" => {
                        return Ok(FillValue::from(microfloat::f8e5m2::INFINITY.to_bits()));
                    }
                    "-Infinity" => {
                        return Ok(FillValue::from(microfloat::f8e5m2::NEG_INFINITY.to_bits()));
                    }
                    _ => {}
                }
            }

            if let Some(f) = fill_value_metadata.as_f64() {
                return Ok(FillValue::from(microfloat::f8e5m2::from_f64(f).to_bits()));
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
            let f8 = microfloat::f8e5m2::from_bits(bytes[0]);

            if f8.is_nan() {
                Ok(FillValueMetadata::from("NaN".to_string()))
            } else if f8 == microfloat::f8e5m2::INFINITY {
                Ok(FillValueMetadata::from("Infinity".to_string()))
            } else if f8 == microfloat::f8e5m2::NEG_INFINITY {
                Ok(FillValueMetadata::from("-Infinity".to_string()))
            } else {
                Ok(FillValueMetadata::from(f8.to_f64()))
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
            const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<microfloat::f8e5m2>()];
            &TYPES
        }
    }
}

// Special Float8E5M2 implementation with float8 and microfloat feature support
#[cfg(all(feature = "float8", feature = "microfloat"))]
mod impl_float8_microfloat {
    use super::Float8E5M2DataType;
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    impl DataTypeTraits for Float8E5M2DataType {
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
            let f8 = float8::F8E5M2::from_bits(bytes[0]);

            // Return special values as strings, numeric values as floats
            if f8.is_nan() {
                Ok(FillValueMetadata::from("NaN".to_string()))
            } else if f8 == float8::F8E5M2::INFINITY {
                Ok(FillValueMetadata::from("Infinity".to_string()))
            } else if f8 == float8::F8E5M2::NEG_INFINITY {
                Ok(FillValueMetadata::from("-Infinity".to_string()))
            } else {
                Ok(FillValueMetadata::from(f8.to_f64()))
            }
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
            const TYPES: [std::any::TypeId; 2] = [
                std::any::TypeId::of::<float8::F8E5M2>(),
                std::any::TypeId::of::<microfloat::f8e5m2>(),
            ];
            &TYPES
        }
    }
}

zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(Float8E5M2DataType, 8, float, 1);
zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(Float8E5M2DataType, 1);

//! Subfloat data types (sub-byte floating point formats).

use super::macros::register_data_type_plugin;

// Prefer microfloat, fall back to float8 for types available in both crates
#[cfg(feature = "microfloat")]
type SelectedFloat8E4M3 = microfloat::f8e4m3;
#[cfg(all(not(feature = "microfloat"), feature = "float8"))]
type SelectedFloat8E4M3 = float8::F8E4M3;

#[cfg(feature = "microfloat")]
type SelectedFloat8E5M2 = microfloat::f8e5m2;
#[cfg(all(not(feature = "microfloat"), feature = "float8"))]
type SelectedFloat8E5M2 = float8::F8E5M2;

/// Macro to implement `DataTypeTraits` for subfloat types (microfloat only).
macro_rules! impl_subfloat_data_type {
    ($marker:ty, $float_type:ty) => {
        impl zarrs_data_type::DataTypeTraits for $marker {
            fn configuration(
                &self,
                _version: zarrs_plugin::ZarrVersion,
            ) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed(1)
            }

            fn fill_value(
                &self,
                fill_value_metadata: &zarrs_metadata::FillValueMetadata,
                _version: zarrs_plugin::ZarrVersion,
            ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError>
            {
                if let Some(s) = fill_value_metadata.as_str() {
                    if let Some(hex) = s.strip_prefix("0x")
                        && let Ok(byte) = u8::from_str_radix(hex, 16)
                    {
                        return Ok(zarrs_data_type::FillValue::from(byte));
                    }
                    #[cfg(feature = "microfloat")]
                    {
                        match s {
                            "NaN" => {
                                return Ok(zarrs_data_type::FillValue::from(
                                    <$float_type>::NAN.to_bits(),
                                ));
                            }
                            "Infinity" => {
                                return Ok(zarrs_data_type::FillValue::from(
                                    <$float_type>::INFINITY.to_bits(),
                                ));
                            }
                            "-Infinity" => {
                                return Ok(zarrs_data_type::FillValue::from(
                                    <$float_type>::NEG_INFINITY.to_bits(),
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                #[cfg(feature = "microfloat")]
                {
                    if let Some(f) = fill_value_metadata.as_f64() {
                        return Ok(zarrs_data_type::FillValue::from(
                            <$float_type>::from_f64(f).to_bits(),
                        ));
                    }
                }
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
                #[cfg(feature = "microfloat")]
                {
                    let value = <$float_type>::from_bits(bytes[0]);
                    if value.is_nan() {
                        Ok(zarrs_metadata::FillValueMetadata::from("NaN".to_string()))
                    } else if value.is_infinite() {
                        if value.is_sign_negative() {
                            Ok(zarrs_metadata::FillValueMetadata::from(
                                "-Infinity".to_string(),
                            ))
                        } else {
                            Ok(zarrs_metadata::FillValueMetadata::from(
                                "Infinity".to_string(),
                            ))
                        }
                    } else {
                        Ok(zarrs_metadata::FillValueMetadata::from(value.to_f64()))
                    }
                }
                #[cfg(not(feature = "microfloat"))]
                {
                    Ok(zarrs_metadata::FillValueMetadata::from(format!(
                        "0x{:02x}",
                        bytes[0]
                    )))
                }
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
                #[cfg(feature = "microfloat")]
                {
                    const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<$float_type>()];
                    &TYPES
                }
                #[cfg(not(feature = "microfloat"))]
                {
                    &[]
                }
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

/// The `float8_e4m3` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3DataType;
register_data_type_plugin!(Float8E4M3DataType);
zarrs_plugin::impl_extension_aliases!(Float8E4M3DataType, v3: "float8_e4m3");

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

/// The `float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2DataType;
register_data_type_plugin!(Float8E5M2DataType);
zarrs_plugin::impl_extension_aliases!(Float8E5M2DataType, v3: "float8_e5m2");

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

// DataTypeTraits implementations for subfloats (microfloat only)
impl_subfloat_data_type!(Float4E2M1FNDataType, microfloat::f4e2m1fn);
impl_subfloat_data_type!(Float6E2M3FNDataType, microfloat::f6e2m3fn);
impl_subfloat_data_type!(Float6E3M2FNDataType, microfloat::f6e3m2fn);
impl_subfloat_data_type!(Float8E3M4DataType, microfloat::f8e3m4);
impl_subfloat_data_type!(Float8E4M3B11FNUZDataType, microfloat::f8e4m3b11fnuz);
impl_subfloat_data_type!(Float8E4M3FNUZDataType, microfloat::f8e4m3fnuz);
impl_subfloat_data_type!(Float8E5M2FNUZDataType, microfloat::f8e5m2fnuz);
impl_subfloat_data_type!(Float8E8M0FNUDataType, microfloat::f8e8m0fnu);

// DataTypeTraits implementations for float8 types (microfloat or float8 crate)
impl zarrs_data_type::DataTypeTraits for Float8E4M3DataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(1)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        _version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        if let Some(s) = fill_value_metadata.as_str() {
            if let Some(hex) = s.strip_prefix("0x")
                && let Ok(byte) = u8::from_str_radix(hex, 16)
            {
                return Ok(zarrs_data_type::FillValue::from(byte));
            }
            #[cfg(any(feature = "float8", feature = "microfloat"))]
            {
                match s {
                    "NaN" => {
                        return Ok(zarrs_data_type::FillValue::from(
                            SelectedFloat8E4M3::NAN.to_bits(),
                        ));
                    }
                    "Infinity" => {
                        return Ok(zarrs_data_type::FillValue::from(
                            SelectedFloat8E4M3::INFINITY.to_bits(),
                        ));
                    }
                    "-Infinity" => {
                        return Ok(zarrs_data_type::FillValue::from(
                            SelectedFloat8E4M3::NEG_INFINITY.to_bits(),
                        ));
                    }
                    _ => {}
                }
            }
        }
        #[cfg(any(feature = "float8", feature = "microfloat"))]
        {
            if let Some(f) = fill_value_metadata.as_f64() {
                return Ok(zarrs_data_type::FillValue::from(
                    SelectedFloat8E4M3::from_f64(f).to_bits(),
                ));
            }
        }
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
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes: [u8; 1] = fill_value
            .as_ne_bytes()
            .try_into()
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        #[cfg(any(feature = "float8", feature = "microfloat"))]
        {
            let f8 = SelectedFloat8E4M3::from_bits(bytes[0]);
            if f8.is_nan() {
                Ok(zarrs_metadata::FillValueMetadata::from("NaN".to_string()))
            } else if f8 == SelectedFloat8E4M3::INFINITY {
                Ok(zarrs_metadata::FillValueMetadata::from(
                    "Infinity".to_string(),
                ))
            } else if f8 == SelectedFloat8E4M3::NEG_INFINITY {
                Ok(zarrs_metadata::FillValueMetadata::from(
                    "-Infinity".to_string(),
                ))
            } else {
                Ok(zarrs_metadata::FillValueMetadata::from(f8.to_f64()))
            }
        }
        #[cfg(not(any(feature = "float8", feature = "microfloat")))]
        {
            Ok(zarrs_metadata::FillValueMetadata::from(format!(
                "0x{:02x}",
                bytes[0]
            )))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
        #[cfg(all(feature = "float8", feature = "microfloat"))]
        {
            const TYPES: [std::any::TypeId; 2] = [
                std::any::TypeId::of::<float8::F8E4M3>(),
                std::any::TypeId::of::<microfloat::f8e4m3>(),
            ];
            &TYPES
        }
        #[cfg(all(feature = "float8", not(feature = "microfloat")))]
        {
            const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<float8::F8E4M3>()];
            &TYPES
        }
        #[cfg(all(not(feature = "float8"), feature = "microfloat"))]
        {
            const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<microfloat::f8e4m3>()];
            &TYPES
        }
        #[cfg(not(any(feature = "float8", feature = "microfloat")))]
        {
            &[]
        }
    }
}

impl zarrs_data_type::DataTypeTraits for Float8E5M2DataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(1)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        _version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        if let Some(s) = fill_value_metadata.as_str() {
            if let Some(hex) = s.strip_prefix("0x")
                && let Ok(byte) = u8::from_str_radix(hex, 16)
            {
                return Ok(zarrs_data_type::FillValue::from(byte));
            }
            #[cfg(any(feature = "float8", feature = "microfloat"))]
            {
                match s {
                    "NaN" => {
                        return Ok(zarrs_data_type::FillValue::from(
                            SelectedFloat8E5M2::NAN.to_bits(),
                        ));
                    }
                    "Infinity" => {
                        return Ok(zarrs_data_type::FillValue::from(
                            SelectedFloat8E5M2::INFINITY.to_bits(),
                        ));
                    }
                    "-Infinity" => {
                        return Ok(zarrs_data_type::FillValue::from(
                            SelectedFloat8E5M2::NEG_INFINITY.to_bits(),
                        ));
                    }
                    _ => {}
                }
            }
        }
        #[cfg(any(feature = "float8", feature = "microfloat"))]
        {
            if let Some(f) = fill_value_metadata.as_f64() {
                return Ok(zarrs_data_type::FillValue::from(
                    SelectedFloat8E5M2::from_f64(f).to_bits(),
                ));
            }
        }
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
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes: [u8; 1] = fill_value
            .as_ne_bytes()
            .try_into()
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        #[cfg(any(feature = "float8", feature = "microfloat"))]
        {
            let f8 = SelectedFloat8E5M2::from_bits(bytes[0]);
            if f8.is_nan() {
                Ok(zarrs_metadata::FillValueMetadata::from("NaN".to_string()))
            } else if f8 == SelectedFloat8E5M2::INFINITY {
                Ok(zarrs_metadata::FillValueMetadata::from(
                    "Infinity".to_string(),
                ))
            } else if f8 == SelectedFloat8E5M2::NEG_INFINITY {
                Ok(zarrs_metadata::FillValueMetadata::from(
                    "-Infinity".to_string(),
                ))
            } else {
                Ok(zarrs_metadata::FillValueMetadata::from(f8.to_f64()))
            }
        }
        #[cfg(not(any(feature = "float8", feature = "microfloat")))]
        {
            Ok(zarrs_metadata::FillValueMetadata::from(format!(
                "0x{:02x}",
                bytes[0]
            )))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
        #[cfg(all(feature = "float8", feature = "microfloat"))]
        {
            const TYPES: [std::any::TypeId; 2] = [
                std::any::TypeId::of::<float8::F8E5M2>(),
                std::any::TypeId::of::<microfloat::f8e5m2>(),
            ];
            &TYPES
        }
        #[cfg(all(feature = "float8", not(feature = "microfloat")))]
        {
            const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<float8::F8E5M2>()];
            &TYPES
        }
        #[cfg(all(not(feature = "float8"), feature = "microfloat"))]
        {
            const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<microfloat::f8e5m2>()];
            &TYPES
        }
        #[cfg(not(any(feature = "float8", feature = "microfloat")))]
        {
            &[]
        }
    }
}

// PackBits codec implementations for subfloats
use zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits;
impl_pack_bits_data_type_traits!(Float4E2M1FNDataType, 4, float, 1);
impl_pack_bits_data_type_traits!(Float6E2M3FNDataType, 6, float, 1);
impl_pack_bits_data_type_traits!(Float6E3M2FNDataType, 6, float, 1);
impl_pack_bits_data_type_traits!(Float8E3M4DataType, 8, float, 1);
impl_pack_bits_data_type_traits!(Float8E4M3DataType, 8, float, 1);
impl_pack_bits_data_type_traits!(Float8E4M3B11FNUZDataType, 8, float, 1);
impl_pack_bits_data_type_traits!(Float8E4M3FNUZDataType, 8, float, 1);
impl_pack_bits_data_type_traits!(Float8E5M2DataType, 8, float, 1);
impl_pack_bits_data_type_traits!(Float8E5M2FNUZDataType, 8, float, 1);
impl_pack_bits_data_type_traits!(Float8E8M0FNUDataType, 8, float, 1);

// Bytes codec implementations for subfloats (passthrough - single byte, no endianness conversion)
use zarrs_data_type::codec_traits::impl_bytes_data_type_traits;
impl_bytes_data_type_traits!(Float4E2M1FNDataType, 1);
impl_bytes_data_type_traits!(Float6E2M3FNDataType, 1);
impl_bytes_data_type_traits!(Float6E3M2FNDataType, 1);
impl_bytes_data_type_traits!(Float8E3M4DataType, 1);
impl_bytes_data_type_traits!(Float8E4M3DataType, 1);
impl_bytes_data_type_traits!(Float8E4M3B11FNUZDataType, 1);
impl_bytes_data_type_traits!(Float8E4M3FNUZDataType, 1);
impl_bytes_data_type_traits!(Float8E5M2DataType, 1);
impl_bytes_data_type_traits!(Float8E5M2FNUZDataType, 1);
impl_bytes_data_type_traits!(Float8E8M0FNUDataType, 1);

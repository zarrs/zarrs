//! Subfloat data types (sub-byte floating point formats).

use super::macros::register_data_type_plugin;

/// Macro to implement `DataTypeTraits` for subfloat types (microfloat only).
macro_rules! impl_subfloat_data_type {
    ($marker:ty, $float_type:ty $(, $float8_type:ty)?) => {
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
                #[cfg(all(feature = "float8", feature = "microfloat"))]
                {
                    const TYPES: [std::any::TypeId; impl_subfloat_data_type!(@num_types $($float8_type)?)] = [
                        std::any::TypeId::of::<$float_type>(),
                        $(std::any::TypeId::of::<$float8_type>(),)?
                    ];
                    &TYPES
                }
                #[cfg(all(not(feature = "float8"), feature = "microfloat"))]
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
    (@num_types) => {
        1
    };
    (@num_types $float8_type:ty) => {
        2
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
impl_subfloat_data_type!(Float8E4M3DataType, microfloat::f8e4m3, float8::F8E4M3);
impl_subfloat_data_type!(Float8E4M3B11FNUZDataType, microfloat::f8e4m3b11fnuz);
impl_subfloat_data_type!(Float8E4M3FNUZDataType, microfloat::f8e4m3fnuz);
impl_subfloat_data_type!(Float8E5M2DataType, microfloat::f8e5m2, float8::F8E5M2);
impl_subfloat_data_type!(Float8E5M2FNUZDataType, microfloat::f8e5m2fnuz);
impl_subfloat_data_type!(Float8E8M0FNUDataType, microfloat::f8e8m0fnu);

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

// CastValue codec implementations for subfloats
#[cfg(feature = "microfloat")]
use zarrs_data_type::codec_traits::impl_cast_value_data_type_traits_float;
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(Float4E2M1FNDataType, microfloat, microfloat::f4e2m1fn, 4);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(Float6E2M3FNDataType, microfloat, microfloat::f6e2m3fn, 6);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(Float6E3M2FNDataType, microfloat, microfloat::f6e3m2fn, 6);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(Float8E3M4DataType, microfloat, microfloat::f8e3m4, 8);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(Float8E4M3DataType, microfloat, microfloat::f8e4m3, 8);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(
    Float8E4M3B11FNUZDataType,
    microfloat,
    microfloat::f8e4m3b11fnuz,
    8
);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(
    Float8E4M3FNUZDataType,
    microfloat,
    microfloat::f8e4m3fnuz,
    8
);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(Float8E5M2DataType, microfloat, microfloat::f8e5m2, 8);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(
    Float8E5M2FNUZDataType,
    microfloat,
    microfloat::f8e5m2fnuz,
    8
);
#[cfg(feature = "microfloat")]
impl_cast_value_data_type_traits_float!(
    Float8E8M0FNUDataType,
    microfloat,
    microfloat::f8e8m0fnu,
    8
);

// ScaleOffset codec implementations for subfloats
// Floats allow infinity/NaN as valid results (no overflow error).
#[cfg(feature = "microfloat")]
use zarrs_data_type::codec_traits::impl_scale_offset_data_type_traits;
#[cfg(feature = "microfloat")]
use zarrs_data_type::codec_traits::scale_offset::{ScaleOffsetDataTypeTraits, ScaleOffsetError};

#[cfg(feature = "microfloat")]
macro_rules! impl_subfloat_scale_offset {
    ($marker:ty, $float_type:ty) => {
        impl ScaleOffsetDataTypeTraits for $marker {
            fn scale_offset_encode(
                &self,
                bytes: &mut [u8],
                offset: Option<&[u8]>,
                scale: Option<&[u8]>,
            ) -> Result<(), ScaleOffsetError> {
                let offset: f64 = match offset {
                    Some([byte]) => <$float_type>::from_bits(*byte).to_f32().into(),
                    Some(_) => return Err(ScaleOffsetError::InvalidElementBytes),
                    None => 0.0,
                };
                let scale: f64 = match scale {
                    Some([byte]) => <$float_type>::from_bits(*byte).to_f32().into(),
                    Some(_) => return Err(ScaleOffsetError::InvalidElementBytes),
                    None => 1.0,
                };
                for chunk in bytes.as_chunks_mut::<1>().0 {
                    let value: f64 = <$float_type>::from_bits(chunk[0]).to_f32().into();
                    let result = (value - offset) * scale;
                    chunk[0] = <$float_type>::from_f64(result).to_bits();
                }
                Ok(())
            }

            fn scale_offset_decode(
                &self,
                bytes: &mut [u8],
                offset: Option<&[u8]>,
                scale: Option<&[u8]>,
            ) -> Result<(), ScaleOffsetError> {
                let offset: f64 = match offset {
                    Some([byte]) => <$float_type>::from_bits(*byte).to_f32().into(),
                    Some(_) => return Err(ScaleOffsetError::InvalidElementBytes),
                    None => 0.0,
                };
                let scale: f64 = match scale {
                    Some([byte]) => <$float_type>::from_bits(*byte).to_f32().into(),
                    Some(_) => return Err(ScaleOffsetError::InvalidElementBytes),
                    None => 1.0,
                };
                for chunk in bytes.as_chunks_mut::<1>().0 {
                    let value: f64 = <$float_type>::from_bits(chunk[0]).to_f32().into();
                    let result = (value / scale) + offset;
                    chunk[0] = <$float_type>::from_f64(result).to_bits();
                }
                Ok(())
            }
        }
    };
}

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float4E2M1FNDataType, microfloat::f4e2m1fn);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float4E2M1FNDataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float6E2M3FNDataType, microfloat::f6e2m3fn);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float6E2M3FNDataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float6E3M2FNDataType, microfloat::f6e3m2fn);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float6E3M2FNDataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float8E3M4DataType, microfloat::f8e3m4);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float8E3M4DataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float8E4M3DataType, microfloat::f8e4m3);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float8E4M3DataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float8E4M3B11FNUZDataType, microfloat::f8e4m3b11fnuz);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float8E4M3B11FNUZDataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float8E4M3FNUZDataType, microfloat::f8e4m3fnuz);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float8E4M3FNUZDataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float8E5M2DataType, microfloat::f8e5m2);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float8E5M2DataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float8E5M2FNUZDataType, microfloat::f8e5m2fnuz);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float8E5M2FNUZDataType);

#[cfg(feature = "microfloat")]
impl_subfloat_scale_offset!(Float8E8M0FNUDataType, microfloat::f8e8m0fnu);
#[cfg(feature = "microfloat")]
impl_scale_offset_data_type_traits!(Float8E8M0FNUDataType);

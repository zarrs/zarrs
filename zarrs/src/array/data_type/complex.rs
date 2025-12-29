//! Complex data type markers and implementations.

use super::macros::{
    impl_complex_data_type, impl_complex_subfloat_data_type, register_data_type_plugin,
};
use zarrs_data_type::{
    DataTypeExtensionBitroundCodec, DataTypeExtensionPackBitsCodec, DataTypeExtensionPcodecCodec,
    PcodecElementType, round_bytes_float16, round_bytes_float32, round_bytes_float64,
};

// Complex floats - V2: <c8, <c16 (and > variants)

/// The `complex_bfloat16` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexBFloat16DataType;
zarrs_plugin::impl_extension_aliases!(ComplexBFloat16DataType, "complex_bfloat16");

/// The `complex_float16` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat16DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat16DataType, "complex_float16");

/// The `complex_float32` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat32DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat32DataType, "complex_float32");

/// The `complex_float64` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat64DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat64DataType, "complex_float64");

/// The `complex64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Complex64DataType;
zarrs_plugin::impl_extension_aliases!(Complex64DataType, "complex64",
    v3: "complex64", [],
    v2: "<c8", ["<c8", ">c8"]
);

/// The `complex128` data type.
#[derive(Debug, Clone, Copy)]
pub struct Complex128DataType;
zarrs_plugin::impl_extension_aliases!(Complex128DataType, "complex128",
    v3: "complex128", [],
    v2: "<c16", ["<c16", ">c16"]
);

// Complex subfloats - No V2 equivalents

/// The `complex_float4_e2m1fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat4E2M1FNDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat4E2M1FNDataType, "complex_float4_e2m1fn");

/// The `complex_float6_e2m3fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat6E2M3FNDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat6E2M3FNDataType, "complex_float6_e2m3fn");

/// The `complex_float6_e3m2fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat6E3M2FNDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat6E3M2FNDataType, "complex_float6_e3m2fn");

/// The `complex_float8_e3m4` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E3M4DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E3M4DataType, "complex_float8_e3m4");

/// The `complex_float8_e4m3` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E4M3DataType, "complex_float8_e4m3");

/// The `complex_float8_e4m3b11fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3B11FNUZDataType;
zarrs_plugin::impl_extension_aliases!(
    ComplexFloat8E4M3B11FNUZDataType,
    "complex_float8_e4m3b11fnuz"
);

/// The `complex_float8_e4m3fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3FNUZDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E4M3FNUZDataType, "complex_float8_e4m3fnuz");

/// The `complex_float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E5M2DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E5M2DataType, "complex_float8_e5m2");

/// The `complex_float8_e5m2fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E5M2FNUZDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E5M2FNUZDataType, "complex_float8_e5m2fnuz");

/// The `complex_float8_e8m0fnu` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E8M0FNUDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E8M0FNUDataType, "complex_float8_e8m0fnu");

// DataTypeExtension implementations for standard complex floats
// All complex types support: pcodec, bitround, packbits
impl_complex_data_type!(ComplexBFloat16DataType, 4, bf16; pcodec, bitround, packbits);
impl_complex_data_type!(ComplexFloat16DataType, 4, f16; pcodec, bitround, packbits);
impl_complex_data_type!(ComplexFloat32DataType, 8, f32; pcodec, bitround, packbits);
impl_complex_data_type!(ComplexFloat64DataType, 16, f64; pcodec, bitround, packbits);
impl_complex_data_type!(Complex64DataType, 8, f32; pcodec, bitround, packbits);
impl_complex_data_type!(Complex128DataType, 16, f64; pcodec, bitround, packbits);

// DataTypeExtension implementations for complex subfloats
// The second parameter is the bit size of each component for packbits codec
impl_complex_subfloat_data_type!(ComplexFloat4E2M1FNDataType, 4);
impl_complex_subfloat_data_type!(ComplexFloat6E2M3FNDataType, 6);
impl_complex_subfloat_data_type!(ComplexFloat6E3M2FNDataType, 6);
impl_complex_subfloat_data_type!(ComplexFloat8E3M4DataType, 8);
impl_complex_subfloat_data_type!(ComplexFloat8E4M3B11FNUZDataType, 8);
impl_complex_subfloat_data_type!(ComplexFloat8E4M3FNUZDataType, 8);
impl_complex_subfloat_data_type!(ComplexFloat8E5M2FNUZDataType, 8);
impl_complex_subfloat_data_type!(ComplexFloat8E8M0FNUDataType, 8);

// ComplexFloat8E4M3 and ComplexFloat8E5M2 have special implementations when float8 feature is enabled
#[cfg(not(feature = "float8"))]
impl_complex_subfloat_data_type!(ComplexFloat8E4M3DataType, 8);
#[cfg(not(feature = "float8"))]
impl_complex_subfloat_data_type!(ComplexFloat8E5M2DataType, 8);

// Special ComplexFloat8E4M3 implementation with float8 feature support
#[cfg(feature = "float8")]
mod complex_float8_e4m3_impl {
    use super::ComplexFloat8E4M3DataType;
    use std::borrow::Cow;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeExtensionBytesCodec, DataTypeExtensionBytesCodecError,
        DataTypeExtensionPackBitsCodec, DataTypeFillValueError, DataTypeFillValueMetadataError,
        FillValue,
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

        fn codec_bytes(&self) -> Option<&dyn DataTypeExtensionBytesCodec> {
            Some(self)
        }

        fn codec_packbits(&self) -> Option<&dyn DataTypeExtensionPackBitsCodec> {
            Some(self)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl DataTypeExtensionBytesCodec for ComplexFloat8E4M3DataType {
        fn encode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }

        fn decode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }
    }

    impl DataTypeExtensionPackBitsCodec for ComplexFloat8E4M3DataType {
        fn component_size_bits(&self) -> u64 {
            8
        }
        fn num_components(&self) -> u64 {
            2
        }
        fn sign_extension(&self) -> bool {
            false
        }
    }
}

// Special ComplexFloat8E5M2 implementation with float8 feature support
#[cfg(feature = "float8")]
mod complex_float8_e5m2_impl {
    use super::ComplexFloat8E5M2DataType;
    use std::borrow::Cow;
    use zarrs_data_type::{
        DataTypeExtension, DataTypeExtensionBytesCodec, DataTypeExtensionBytesCodecError,
        DataTypeExtensionPackBitsCodec, DataTypeFillValueError, DataTypeFillValueMetadataError,
        FillValue,
    };
    use zarrs_metadata::{Configuration, DataTypeSize, v3::FillValueMetadataV3};
    use zarrs_plugin::ExtensionIdentifier;

    impl DataTypeExtension for ComplexFloat8E5M2DataType {
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
                let f8 = float8::F8E5M2::from_bits(byte);
                if f8.is_nan() {
                    FillValueMetadataV3::from("NaN".to_string())
                } else if f8 == float8::F8E5M2::INFINITY {
                    FillValueMetadataV3::from("Infinity".to_string())
                } else if f8 == float8::F8E5M2::NEG_INFINITY {
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

        fn codec_bytes(&self) -> Option<&dyn DataTypeExtensionBytesCodec> {
            Some(self)
        }

        fn codec_packbits(&self) -> Option<&dyn DataTypeExtensionPackBitsCodec> {
            Some(self)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl DataTypeExtensionBytesCodec for ComplexFloat8E5M2DataType {
        fn encode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }

        fn decode<'a>(
            &self,
            bytes: Cow<'a, [u8]>,
            _endianness: Option<zarrs_metadata::Endianness>,
        ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
            Ok(bytes)
        }
    }

    impl DataTypeExtensionPackBitsCodec for ComplexFloat8E5M2DataType {
        fn component_size_bits(&self) -> u64 {
            8
        }
        fn num_components(&self) -> u64 {
            2
        }
        fn sign_extension(&self) -> bool {
            false
        }
    }
}

// Plugin registrations
register_data_type_plugin!(ComplexBFloat16DataType);
register_data_type_plugin!(ComplexFloat16DataType);
register_data_type_plugin!(ComplexFloat32DataType);
register_data_type_plugin!(ComplexFloat64DataType);
register_data_type_plugin!(Complex64DataType);
register_data_type_plugin!(Complex128DataType);
register_data_type_plugin!(ComplexFloat4E2M1FNDataType);
register_data_type_plugin!(ComplexFloat6E2M3FNDataType);
register_data_type_plugin!(ComplexFloat6E3M2FNDataType);
register_data_type_plugin!(ComplexFloat8E3M4DataType);
register_data_type_plugin!(ComplexFloat8E4M3DataType);
register_data_type_plugin!(ComplexFloat8E4M3B11FNUZDataType);
register_data_type_plugin!(ComplexFloat8E4M3FNUZDataType);
register_data_type_plugin!(ComplexFloat8E5M2DataType);
register_data_type_plugin!(ComplexFloat8E5M2FNUZDataType);
register_data_type_plugin!(ComplexFloat8E8M0FNUDataType);

// ============================================================================
// Codec extension trait implementations
// ============================================================================

// --- Bitround codec ---
// Complex types apply rounding component-wise. component_size is the size of each component.
// For complex64 (f32+f32), component_size=4, mantissa_bits=23
// For complex128 (f64+f64), component_size=8, mantissa_bits=52
// For complex_bfloat16, component_size=2, mantissa_bits=7
// For complex_float16, component_size=2, mantissa_bits=10

impl DataTypeExtensionBitroundCodec for ComplexBFloat16DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        Some(7)
    }

    fn component_size(&self) -> usize {
        2
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        round_bytes_float16(bytes, keepbits, 7);
    }
}

impl DataTypeExtensionBitroundCodec for ComplexFloat16DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        Some(10)
    }

    fn component_size(&self) -> usize {
        2
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        round_bytes_float16(bytes, keepbits, 10);
    }
}

impl DataTypeExtensionBitroundCodec for ComplexFloat32DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        Some(23)
    }

    fn component_size(&self) -> usize {
        4
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        round_bytes_float32(bytes, keepbits, 23);
    }
}

impl DataTypeExtensionBitroundCodec for ComplexFloat64DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        Some(52)
    }

    fn component_size(&self) -> usize {
        8
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        round_bytes_float64(bytes, keepbits, 52);
    }
}

impl DataTypeExtensionBitroundCodec for Complex64DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        Some(23)
    }

    fn component_size(&self) -> usize {
        4
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        round_bytes_float32(bytes, keepbits, 23);
    }
}

impl DataTypeExtensionBitroundCodec for Complex128DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        Some(52)
    }

    fn component_size(&self) -> usize {
        8
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        round_bytes_float64(bytes, keepbits, 52);
    }
}

// --- Pcodec codec ---
// Pcodec supports complex types by treating them as pairs of float components

impl DataTypeExtensionPcodecCodec for ComplexBFloat16DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        None // bfloat16 not supported by pcodec
    }
}

impl DataTypeExtensionPcodecCodec for ComplexFloat16DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F16)
    }

    fn pcodec_elements_per_element(&self) -> usize {
        2 // complex = 2 components
    }
}

impl DataTypeExtensionPcodecCodec for ComplexFloat32DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F32)
    }

    fn pcodec_elements_per_element(&self) -> usize {
        2
    }
}

impl DataTypeExtensionPcodecCodec for ComplexFloat64DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F64)
    }

    fn pcodec_elements_per_element(&self) -> usize {
        2
    }
}

impl DataTypeExtensionPcodecCodec for Complex64DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F32)
    }

    fn pcodec_elements_per_element(&self) -> usize {
        2
    }
}

impl DataTypeExtensionPcodecCodec for Complex128DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F64)
    }

    fn pcodec_elements_per_element(&self) -> usize {
        2
    }
}

// --- PackBits codec ---
// Complex types have two components, so num_components = 2

impl DataTypeExtensionPackBitsCodec for ComplexBFloat16DataType {
    fn component_size_bits(&self) -> u64 {
        16
    }

    fn num_components(&self) -> u64 {
        2
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

impl DataTypeExtensionPackBitsCodec for ComplexFloat16DataType {
    fn component_size_bits(&self) -> u64 {
        16
    }

    fn num_components(&self) -> u64 {
        2
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

impl DataTypeExtensionPackBitsCodec for ComplexFloat32DataType {
    fn component_size_bits(&self) -> u64 {
        32
    }

    fn num_components(&self) -> u64 {
        2
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

impl DataTypeExtensionPackBitsCodec for ComplexFloat64DataType {
    fn component_size_bits(&self) -> u64 {
        64
    }

    fn num_components(&self) -> u64 {
        2
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

impl DataTypeExtensionPackBitsCodec for Complex64DataType {
    fn component_size_bits(&self) -> u64 {
        32
    }

    fn num_components(&self) -> u64 {
        2
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

impl DataTypeExtensionPackBitsCodec for Complex128DataType {
    fn component_size_bits(&self) -> u64 {
        64
    }

    fn num_components(&self) -> u64 {
        2
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

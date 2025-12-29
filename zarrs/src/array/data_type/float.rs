//! Float data type markers and implementations.

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};
use zarrs_data_type::{
    DataTypeExtensionBitroundCodec, DataTypeExtensionFixedScaleOffsetCodec,
    DataTypeExtensionPackBitsCodec, DataTypeExtensionPcodecCodec, DataTypeExtensionZfpCodec,
    FixedScaleOffsetElementType, PcodecElementType, ZfpPromotion, ZfpType, round_bytes_float16,
    round_bytes_float32, round_bytes_float64,
};

// Standard floats - V2: <f2, <f4, <f8 (and > variants), no bfloat16

/// The `bfloat16` data type.
#[derive(Debug, Clone, Copy)]
pub struct BFloat16DataType;
zarrs_plugin::impl_extension_aliases!(BFloat16DataType, "bfloat16");

/// The `float16` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float16DataType;
zarrs_plugin::impl_extension_aliases!(Float16DataType, "float16",
    v3: "float16", [],
    v2: "<f2", ["<f2", ">f2"]
);

/// The `float32` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float32DataType;
zarrs_plugin::impl_extension_aliases!(Float32DataType, "float32",
    v3: "float32", [],
    v2: "<f4", ["<f4", ">f4"]
);

/// The `float64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float64DataType;
zarrs_plugin::impl_extension_aliases!(Float64DataType, "float64",
    v3: "float64", [],
    v2: "<f8", ["<f8", ">f8"]
);

// DataTypeExtension implementations
// All float types support: pcodec, bitround, fixedscaleoffset, zfp, packbits
impl_data_type_extension_numeric!(BFloat16DataType, 2, bf16; pcodec, bitround, fixedscaleoffset, zfp, packbits);
impl_data_type_extension_numeric!(Float16DataType, 2, f16; pcodec, bitround, fixedscaleoffset, zfp, packbits);
impl_data_type_extension_numeric!(Float32DataType, 4, f32; pcodec, bitround, fixedscaleoffset, zfp, packbits);
impl_data_type_extension_numeric!(Float64DataType, 8, f64; pcodec, bitround, fixedscaleoffset, zfp, packbits);

// Plugin registrations
register_data_type_plugin!(BFloat16DataType);
register_data_type_plugin!(Float16DataType);
register_data_type_plugin!(Float32DataType);
register_data_type_plugin!(Float64DataType);

// ============================================================================
// Codec extension trait implementations
// ============================================================================

// --- Bitround codec ---
// Float types have fixed mantissa bits: float16=10, bfloat16=7, float32=23, float64=52

impl DataTypeExtensionBitroundCodec for BFloat16DataType {
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

impl DataTypeExtensionBitroundCodec for Float16DataType {
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

impl DataTypeExtensionBitroundCodec for Float32DataType {
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

impl DataTypeExtensionBitroundCodec for Float64DataType {
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
// Pcodec supports float16, float32, float64 (not bfloat16)

impl DataTypeExtensionPcodecCodec for BFloat16DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        None // bfloat16 not supported by pcodec
    }
}

impl DataTypeExtensionPcodecCodec for Float16DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F16)
    }
}

impl DataTypeExtensionPcodecCodec for Float32DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F32)
    }
}

impl DataTypeExtensionPcodecCodec for Float64DataType {
    fn pcodec_element_type(&self) -> Option<PcodecElementType> {
        Some(PcodecElementType::F64)
    }
}

// --- FixedScaleOffset codec ---
// FixedScaleOffset supports float32, float64 (not float16/bfloat16)

impl DataTypeExtensionFixedScaleOffsetCodec for BFloat16DataType {
    fn fixedscaleoffset_element_type(&self) -> Option<FixedScaleOffsetElementType> {
        None // bfloat16 not supported
    }
}

impl DataTypeExtensionFixedScaleOffsetCodec for Float16DataType {
    fn fixedscaleoffset_element_type(&self) -> Option<FixedScaleOffsetElementType> {
        None // float16 not supported
    }
}

impl DataTypeExtensionFixedScaleOffsetCodec for Float32DataType {
    fn fixedscaleoffset_element_type(&self) -> Option<FixedScaleOffsetElementType> {
        Some(FixedScaleOffsetElementType::F32)
    }
}

impl DataTypeExtensionFixedScaleOffsetCodec for Float64DataType {
    fn fixedscaleoffset_element_type(&self) -> Option<FixedScaleOffsetElementType> {
        Some(FixedScaleOffsetElementType::F64)
    }
}

// --- ZFP codec ---
// ZFP natively supports float32 (zfp_type_float) and float64 (zfp_type_double)

impl DataTypeExtensionZfpCodec for BFloat16DataType {
    fn zfp_type(&self) -> Option<ZfpType> {
        None // bfloat16 not supported by zfp
    }
}

impl DataTypeExtensionZfpCodec for Float16DataType {
    fn zfp_type(&self) -> Option<ZfpType> {
        None // float16 not supported by zfp
    }
}

impl DataTypeExtensionZfpCodec for Float32DataType {
    fn zfp_type(&self) -> Option<ZfpType> {
        Some(ZfpType::Float)
    }

    fn zfp_promotion(&self) -> ZfpPromotion {
        ZfpPromotion::None
    }
}

impl DataTypeExtensionZfpCodec for Float64DataType {
    fn zfp_type(&self) -> Option<ZfpType> {
        Some(ZfpType::Double)
    }

    fn zfp_promotion(&self) -> ZfpPromotion {
        ZfpPromotion::None
    }
}

// --- PackBits codec ---
// All float types support packbits (byte-aligned encoding)

impl DataTypeExtensionPackBitsCodec for BFloat16DataType {
    fn component_size_bits(&self) -> u64 {
        16
    }
    fn num_components(&self) -> u64 {
        1
    }
    fn sign_extension(&self) -> bool {
        false // floats don't use sign extension
    }
}

impl DataTypeExtensionPackBitsCodec for Float16DataType {
    fn component_size_bits(&self) -> u64 {
        16
    }
    fn num_components(&self) -> u64 {
        1
    }
    fn sign_extension(&self) -> bool {
        false
    }
}

impl DataTypeExtensionPackBitsCodec for Float32DataType {
    fn component_size_bits(&self) -> u64 {
        32
    }
    fn num_components(&self) -> u64 {
        1
    }
    fn sign_extension(&self) -> bool {
        false
    }
}

impl DataTypeExtensionPackBitsCodec for Float64DataType {
    fn component_size_bits(&self) -> u64 {
        64
    }
    fn num_components(&self) -> u64 {
        1
    }
    fn sign_extension(&self) -> bool {
        false
    }
}

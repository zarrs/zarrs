//! Standard float data types (`bfloat16`, `float16`, `float32`, `float64`).

use crate::{
    impl_bitround_codec, impl_fixedscaleoffset_codec, impl_packbits_codec, impl_pcodec_codec,
    impl_zfp_codec,
};

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};

/// The `bfloat16` data type.
#[derive(Debug, Clone, Copy)]
pub struct BFloat16DataType;
register_data_type_plugin!(BFloat16DataType);
zarrs_plugin::impl_extension_aliases!(BFloat16DataType, "bfloat16");

/// The `float16` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float16DataType;
register_data_type_plugin!(Float16DataType);
zarrs_plugin::impl_extension_aliases!(Float16DataType, "float16",
    v3: "float16", [],
    v2: "<f2", ["<f2", ">f2"]
);

/// The `float32` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float32DataType;
register_data_type_plugin!(Float32DataType);
zarrs_plugin::impl_extension_aliases!(Float32DataType, "float32",
    v3: "float32", [],
    v2: "<f4", ["<f4", ">f4"]
);

/// The `float64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float64DataType;
register_data_type_plugin!(Float64DataType);
zarrs_plugin::impl_extension_aliases!(Float64DataType, "float64",
    v3: "float64", [],
    v2: "<f8", ["<f8", ">f8"]
);

// DataTypeExtension implementations for standard floats
impl_data_type_extension_numeric!(BFloat16DataType, 2, bf16);
impl_data_type_extension_numeric!(Float16DataType, 2, f16);
impl_data_type_extension_numeric!(Float32DataType, 4, f32);
impl_data_type_extension_numeric!(Float64DataType, 8, f64);

// Bitround codec implementations for standard floats
impl_bitround_codec!(BFloat16DataType, 2, float16, 7);
impl_bitround_codec!(Float16DataType, 2, float16, 10);
impl_bitround_codec!(Float32DataType, 4, float32, 23);
impl_bitround_codec!(Float64DataType, 8, float64, 52);

// Pcodec implementations for standard floats
// impl_pcodec_codec!(BFloat16DataType, BF16, 1);
impl_pcodec_codec!(Float16DataType, F16, 1);
impl_pcodec_codec!(Float32DataType, F32, 1);
impl_pcodec_codec!(Float64DataType, F64, 1);

// FixedScaleOffset implementations for standard floats
// impl_fixedscaleoffset_codec!(BFloat16DataType, BF16);
// impl_fixedscaleoffset_codec!(Float16DataType, F16);
impl_fixedscaleoffset_codec!(Float32DataType, F32);
impl_fixedscaleoffset_codec!(Float64DataType, F64);

// ZFP implementations for standard floats
impl_zfp_codec!(BFloat16DataType, None);
impl_zfp_codec!(Float16DataType, None);
impl_zfp_codec!(Float32DataType, Float);
impl_zfp_codec!(Float64DataType, Double);

// PackBits implementations for standard floats
impl_packbits_codec!(BFloat16DataType, 16, float, 1);
impl_packbits_codec!(Float16DataType, 16, float, 1);
impl_packbits_codec!(Float32DataType, 32, float, 1);
impl_packbits_codec!(Float64DataType, 64, float, 1);

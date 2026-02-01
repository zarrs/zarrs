//! Standard float data types (`bfloat16`, `float16`, `float32`, `float64`).

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};

/// The `bfloat16` data type.
#[derive(Debug, Clone, Copy)]
pub struct BFloat16DataType;
register_data_type_plugin!(BFloat16DataType);
zarrs_plugin::impl_extension_aliases!(BFloat16DataType, v3: "bfloat16");

/// The `float16` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float16DataType;
register_data_type_plugin!(Float16DataType);
zarrs_plugin::impl_extension_aliases!(Float16DataType,
    v3: "float16", [],
    v2: "<f2", ["<f2", ">f2"]
);

/// The `float32` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float32DataType;
register_data_type_plugin!(Float32DataType);
zarrs_plugin::impl_extension_aliases!(Float32DataType,
    v3: "float32", [],
    v2: "<f4", ["<f4", ">f4"]
);

/// The `float64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float64DataType;
register_data_type_plugin!(Float64DataType);
zarrs_plugin::impl_extension_aliases!(Float64DataType,
    v3: "float64", [],
    v2: "<f8", ["<f8", ">f8"]
);

// DataTypeTraits implementations for standard floats
impl_data_type_extension_numeric!(BFloat16DataType, 2, bf16);
impl_data_type_extension_numeric!(Float16DataType, 2, f16);
impl_data_type_extension_numeric!(Float32DataType, 4, f32);
impl_data_type_extension_numeric!(Float64DataType, 8, f64);

// Bitround codec implementations for standard floats
#[cfg(feature = "bitround")]
mod impl_bitround {
    use zarrs_data_type::codec_traits::impl_bitround_codec;
    impl_bitround_codec!(super::BFloat16DataType, 2, float16, 7);
    impl_bitround_codec!(super::Float16DataType, 2, float16, 10);
    impl_bitround_codec!(super::Float32DataType, 4, float32, 23);
    impl_bitround_codec!(super::Float64DataType, 8, float64, 52);
}

// Pcodec implementations for standard floats
#[cfg(feature = "pcodec")]
mod impl_pcodec {
    use crate::array::codec::impl_pcodec_data_type_traits;
    // impl_pcodec_data_type_traits!(super::BFloat16DataType, BF16, 1);
    impl_pcodec_data_type_traits!(super::Float16DataType, F16, 1);
    impl_pcodec_data_type_traits!(super::Float32DataType, F32, 1);
    impl_pcodec_data_type_traits!(super::Float64DataType, F64, 1);
}

// FixedScaleOffset implementations for standard floats
use crate::array::codec::impl_fixed_scale_offset_data_type_traits;
// impl_fixed_scale_offset_data_type_traits!(BFloat16DataType, BF16);
// impl_fixed_scale_offset_data_type_traits!(Float16DataType, F16);
impl_fixed_scale_offset_data_type_traits!(Float32DataType, F32);
impl_fixed_scale_offset_data_type_traits!(Float64DataType, F64);

// ZFP implementations for standard floats
#[cfg(feature = "zfp")]
mod impl_zfp {
    use crate::array::codec::impl_zfp_data_type_traits;
    impl_zfp_data_type_traits!(super::Float32DataType, Float32);
    impl_zfp_data_type_traits!(super::Float64DataType, Float64);
}

// PackBits implementations for standard floats
use zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits;
impl_pack_bits_data_type_traits!(BFloat16DataType, 16, float, 1);
impl_pack_bits_data_type_traits!(Float16DataType, 16, float, 1);
impl_pack_bits_data_type_traits!(Float32DataType, 32, float, 1);
impl_pack_bits_data_type_traits!(Float64DataType, 64, float, 1);

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
use zarrs_data_type::codec_traits::impl_bitround_codec;
impl_bitround_codec!(BFloat16DataType, 2, float16, 7);
impl_bitround_codec!(Float16DataType, 2, float16, 10);
impl_bitround_codec!(Float32DataType, 4, float32, 23);
impl_bitround_codec!(Float64DataType, 8, float64, 52);

// CastValue codec implementations for standard floats
use zarrs_data_type::codec_traits::impl_cast_value_data_type_traits_float;
impl_cast_value_data_type_traits_float!(BFloat16DataType, bf16);
impl_cast_value_data_type_traits_float!(Float16DataType, f16);
impl_cast_value_data_type_traits_float!(Float32DataType, f32);
impl_cast_value_data_type_traits_float!(Float64DataType, f64);

// Pcodec implementations for standard floats
use zarrs_data_type::codec_traits::impl_pcodec_data_type_traits;
// impl_pcodec_data_type_traits!(BFloat16DataType, BF16, 1);
impl_pcodec_data_type_traits!(Float16DataType, F16, 1);
impl_pcodec_data_type_traits!(Float32DataType, F32, 1);
impl_pcodec_data_type_traits!(Float64DataType, F64, 1);

// FixedScaleOffset implementations for standard floats
use zarrs_data_type::codec_traits::impl_fixed_scale_offset_data_type_traits;
// impl_fixed_scale_offset_data_type_traits!(BFloat16DataType, BF16);
// impl_fixed_scale_offset_data_type_traits!(Float16DataType, F16);
impl_fixed_scale_offset_data_type_traits!(Float32DataType, F32);
impl_fixed_scale_offset_data_type_traits!(Float64DataType, F64);

// ScaleOffset implementations for standard floats
// Floats allow infinity/NaN as valid results (no overflow error).
use half::{bf16, f16};
use zarrs_data_type::codec_traits::impl_scale_offset_data_type_traits;
use zarrs_data_type::codec_traits::scale_offset::{
    ScaleOffsetDataTypeTraits, ScaleOffsetError, scale_offset_float,
};

impl ScaleOffsetDataTypeTraits for BFloat16DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<bf16, _, _>(bytes, offset, scale, |v, o, s| (v - o) * s)
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<bf16, _, _>(bytes, offset, scale, |v, o, s| (v / s) + o)
    }
}
impl_scale_offset_data_type_traits!(BFloat16DataType);

impl ScaleOffsetDataTypeTraits for Float16DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<f16, _, _>(bytes, offset, scale, |v, o, s| (v - o) * s)
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<f16, _, _>(bytes, offset, scale, |v, o, s| (v / s) + o)
    }
}
impl_scale_offset_data_type_traits!(Float16DataType);

impl ScaleOffsetDataTypeTraits for Float32DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<f32, _, _>(bytes, offset, scale, |v, o, s| (v - o) * s)
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<f32, _, _>(bytes, offset, scale, |v, o, s| (v / s) + o)
    }
}
impl_scale_offset_data_type_traits!(Float32DataType);

impl ScaleOffsetDataTypeTraits for Float64DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<f64, _, _>(bytes, offset, scale, |v, o, s| (v - o) * s)
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        scale_offset_float::<f64, _, _>(bytes, offset, scale, |v, o, s| (v / s) + o)
    }
}
impl_scale_offset_data_type_traits!(Float64DataType);

// ZFP implementations for standard floats
use zarrs_data_type::codec_traits::impl_zfp_data_type_traits;
impl_zfp_data_type_traits!(Float32DataType, Float32);
impl_zfp_data_type_traits!(Float64DataType, Float64);

// PackBits implementations for standard floats
use zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits;
impl_pack_bits_data_type_traits!(BFloat16DataType, 16, float, 1);
impl_pack_bits_data_type_traits!(Float16DataType, 16, float, 1);
impl_pack_bits_data_type_traits!(Float32DataType, 32, float, 1);
impl_pack_bits_data_type_traits!(Float64DataType, 64, float, 1);

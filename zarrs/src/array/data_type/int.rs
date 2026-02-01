//! Standard signed integer data types (`int8`, `int16`, `int32`, `int64`).

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};

/// The `int8` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int8DataType;
register_data_type_plugin!(Int8DataType);
zarrs_plugin::impl_extension_aliases!(Int8DataType,
    v3: "int8", [],
    v2: "|i1", ["|i1"]
);
impl_data_type_extension_numeric!(Int8DataType, 1, i8);

/// The `int16` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int16DataType;
register_data_type_plugin!(Int16DataType);
zarrs_plugin::impl_extension_aliases!(Int16DataType,
    v3: "int16", [],
    v2: "<i2", ["<i2", ">i2"]
);
impl_data_type_extension_numeric!(Int16DataType, 2, i16);

/// The `int32` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int32DataType;
register_data_type_plugin!(Int32DataType);
zarrs_plugin::impl_extension_aliases!(Int32DataType,
    v3: "int32", [],
    v2: "<i4", ["<i4", ">i4"]
);
impl_data_type_extension_numeric!(Int32DataType, 4, i32);

/// The `int64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int64DataType;
register_data_type_plugin!(Int64DataType);
zarrs_plugin::impl_extension_aliases!(Int64DataType,
    v3: "int64", [],
    v2: "<i8", ["<i8", ">i8"]
);
impl_data_type_extension_numeric!(Int64DataType, 8, i64);

// Bitround codec implementations for standard integers
#[cfg(feature = "bitround")]
mod impl_bitround {
    use zarrs_data_type::codec_traits::impl_bitround_codec;
    impl_bitround_codec!(super::Int8DataType, 1, int8);
    impl_bitround_codec!(super::Int16DataType, 2, int16);
    impl_bitround_codec!(super::Int32DataType, 4, int32);
    impl_bitround_codec!(super::Int64DataType, 8, int64);
}

// Pcodec implementations for standard integers (int8 not supported)
#[cfg(feature = "pcodec")]
mod impl_pcodec {
    use crate::array::codec::impl_pcodec_data_type_traits;
    impl_pcodec_data_type_traits!(super::Int16DataType, I16, 1);
    impl_pcodec_data_type_traits!(super::Int32DataType, I32, 1);
    impl_pcodec_data_type_traits!(super::Int64DataType, I64, 1);
}

// FixedScaleOffset implementations for standard integers
use crate::array::codec::impl_fixed_scale_offset_data_type_traits;
impl_fixed_scale_offset_data_type_traits!(Int8DataType, I8);
impl_fixed_scale_offset_data_type_traits!(Int16DataType, I16);
impl_fixed_scale_offset_data_type_traits!(Int32DataType, I32);
impl_fixed_scale_offset_data_type_traits!(Int64DataType, I64);

// ZFP implementations for standard integers
#[cfg(feature = "zfp")]
mod impl_zfp {
    use crate::array::codec::impl_zfp_data_type_traits;
    impl_zfp_data_type_traits!(super::Int8DataType, Int8);
    impl_zfp_data_type_traits!(super::Int16DataType, Int16);
    impl_zfp_data_type_traits!(super::Int32DataType, Int32);
    impl_zfp_data_type_traits!(super::Int64DataType, Int64);
}

// PackBits implementations for standard integers
use zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits;
impl_pack_bits_data_type_traits!(Int8DataType, 8, signed, 1);
impl_pack_bits_data_type_traits!(Int16DataType, 16, signed, 1);
impl_pack_bits_data_type_traits!(Int32DataType, 32, signed, 1);
impl_pack_bits_data_type_traits!(Int64DataType, 64, signed, 1);

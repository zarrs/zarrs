//! Standard signed integer data types (`int8`, `int16`, `int32`, `int64`).

use crate::{
    impl_bitround_codec, impl_fixedscaleoffset_codec, impl_packbits_codec, impl_pcodec_codec,
    impl_zfp_codec,
};

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};

/// The `int8` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int8DataType;
register_data_type_plugin!(Int8DataType);
zarrs_plugin::impl_extension_aliases!(Int8DataType, "int8",
    v3: "int8", [],
    v2: "|i1", ["|i1"]
);
impl_data_type_extension_numeric!(Int8DataType, 1, i8);

/// The `int16` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int16DataType;
register_data_type_plugin!(Int16DataType);
zarrs_plugin::impl_extension_aliases!(Int16DataType, "int16",
    v3: "int16", [],
    v2: "<i2", ["<i2", ">i2"]
);
impl_data_type_extension_numeric!(Int16DataType, 2, i16);

/// The `int32` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int32DataType;
register_data_type_plugin!(Int32DataType);
zarrs_plugin::impl_extension_aliases!(Int32DataType, "int32",
    v3: "int32", [],
    v2: "<i4", ["<i4", ">i4"]
);
impl_data_type_extension_numeric!(Int32DataType, 4, i32);

/// The `int64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int64DataType;
register_data_type_plugin!(Int64DataType);
zarrs_plugin::impl_extension_aliases!(Int64DataType, "int64",
    v3: "int64", [],
    v2: "<i8", ["<i8", ">i8"]
);
impl_data_type_extension_numeric!(Int64DataType, 8, i64);

// Bitround codec implementations for standard integers
impl_bitround_codec!(Int8DataType, 1, int8);
impl_bitround_codec!(Int16DataType, 2, int16);
impl_bitround_codec!(Int32DataType, 4, int32);
impl_bitround_codec!(Int64DataType, 8, int64);

// Pcodec implementations for standard integers (int8 not supported)
impl_pcodec_codec!(Int16DataType, I16, 1);
impl_pcodec_codec!(Int32DataType, I32, 1);
impl_pcodec_codec!(Int64DataType, I64, 1);

// FixedScaleOffset implementations for standard integers
impl_fixedscaleoffset_codec!(Int8DataType, I8);
impl_fixedscaleoffset_codec!(Int16DataType, I16);
impl_fixedscaleoffset_codec!(Int32DataType, I32);
impl_fixedscaleoffset_codec!(Int64DataType, I64);

// ZFP implementations for standard integers
impl_zfp_codec!(Int8DataType, Int32, I8ToI32);
impl_zfp_codec!(Int16DataType, Int32, I16ToI32);
impl_zfp_codec!(Int32DataType, Int32);
impl_zfp_codec!(Int64DataType, Int64);

// PackBits implementations for standard integers
impl_packbits_codec!(Int8DataType, 8, signed, 1);
impl_packbits_codec!(Int16DataType, 16, signed, 1);
impl_packbits_codec!(Int32DataType, 32, signed, 1);
impl_packbits_codec!(Int64DataType, 64, signed, 1);

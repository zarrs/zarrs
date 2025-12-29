//! Standard unsigned integer data types (`uint8`, `uint16`, `uint32`, `uint64`).

use super::macros::{
    impl_bitround_codec, impl_data_type_extension_numeric, impl_fixedscaleoffset_codec,
    impl_packbits_codec, impl_pcodec_codec, impl_zfp_codec, register_data_type_plugin,
};

/// The `uint8` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt8DataType;
register_data_type_plugin!(UInt8DataType);
zarrs_plugin::impl_extension_aliases!(UInt8DataType, "uint8",
    v3: "uint8", [],
    v2: "|u1", ["|u1"]
);
impl_data_type_extension_numeric!(UInt8DataType, 1, u8; bitround, fixedscaleoffset, zfp, packbits);

/// The `uint16` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt16DataType;
register_data_type_plugin!(UInt16DataType);
zarrs_plugin::impl_extension_aliases!(UInt16DataType, "uint16",
    v3: "uint16", [],
    v2: "<u2", ["<u2", ">u2"]
);
impl_data_type_extension_numeric!(UInt16DataType, 2, u16; pcodec, bitround, fixedscaleoffset, zfp, packbits);

/// The `uint32` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt32DataType;
register_data_type_plugin!(UInt32DataType);
zarrs_plugin::impl_extension_aliases!(UInt32DataType, "uint32",
    v3: "uint32", [],
    v2: "<u4", ["<u4", ">u4"]
);
impl_data_type_extension_numeric!(UInt32DataType, 4, u32; pcodec, bitround, fixedscaleoffset, zfp, packbits);

/// The `uint64` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt64DataType;
register_data_type_plugin!(UInt64DataType);
zarrs_plugin::impl_extension_aliases!(UInt64DataType, "uint64",
    v3: "uint64", [],
    v2: "<u8", ["<u8", ">u8"]
);
impl_data_type_extension_numeric!(UInt64DataType, 8, u64; pcodec, bitround, fixedscaleoffset, zfp, packbits);

// Bitround codec implementations for standard unsigned integers
impl_bitround_codec!(UInt8DataType, 1, uint8);
impl_bitround_codec!(UInt16DataType, 2, uint16);
impl_bitround_codec!(UInt32DataType, 4, uint32);
impl_bitround_codec!(UInt64DataType, 8, uint64);

// Pcodec implementations for standard unsigned integers (uint8 not supported)
impl_pcodec_codec!(UInt16DataType, U16);
impl_pcodec_codec!(UInt32DataType, U32);
impl_pcodec_codec!(UInt64DataType, U64);

// FixedScaleOffset implementations for standard unsigned integers
impl_fixedscaleoffset_codec!(UInt8DataType, U8);
impl_fixedscaleoffset_codec!(UInt16DataType, U16);
impl_fixedscaleoffset_codec!(UInt32DataType, U32);
impl_fixedscaleoffset_codec!(UInt64DataType, U64);

// ZFP implementations for standard unsigned integers
impl_zfp_codec!(UInt8DataType, Int32, U8ToI32);
impl_zfp_codec!(UInt16DataType, Int32, U16ToI32);
impl_zfp_codec!(UInt32DataType, Int32, U32ToI32);
impl_zfp_codec!(UInt64DataType, Int64, U64ToI64);

// PackBits implementations for standard unsigned integers
impl_packbits_codec!(UInt8DataType, 8, unsigned, 1);
impl_packbits_codec!(UInt16DataType, 16, unsigned, 1);
impl_packbits_codec!(UInt32DataType, 32, unsigned, 1);
impl_packbits_codec!(UInt64DataType, 64, unsigned, 1);

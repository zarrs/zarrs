//! Standard unsigned integer data types (`uint8`, `uint16`, `uint32`, `uint64`).

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};

/// The `uint8` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt8DataType;
register_data_type_plugin!(UInt8DataType);
zarrs_plugin::impl_extension_aliases!(UInt8DataType,
    v3: "uint8", [],
    v2: "|u1", ["|u1"]
);
impl_data_type_extension_numeric!(UInt8DataType, 1, u8);

/// The `uint16` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt16DataType;
register_data_type_plugin!(UInt16DataType);
zarrs_plugin::impl_extension_aliases!(UInt16DataType,
    v3: "uint16", [],
    v2: "<u2", ["<u2", ">u2"]
);
impl_data_type_extension_numeric!(UInt16DataType, 2, u16);

/// The `uint32` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt32DataType;
register_data_type_plugin!(UInt32DataType);
zarrs_plugin::impl_extension_aliases!(UInt32DataType,
    v3: "uint32", [],
    v2: "<u4", ["<u4", ">u4"]
);
impl_data_type_extension_numeric!(UInt32DataType, 4, u32);

/// The `uint64` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt64DataType;
register_data_type_plugin!(UInt64DataType);
zarrs_plugin::impl_extension_aliases!(UInt64DataType,
    v3: "uint64", [],
    v2: "<u8", ["<u8", ">u8"]
);
impl_data_type_extension_numeric!(UInt64DataType, 8, u64);

// Bitround codec implementations for standard unsigned integers
use zarrs_data_type::codec_traits::impl_bitround_codec;
impl_bitround_codec!(UInt8DataType, 1, uint8);
impl_bitround_codec!(UInt16DataType, 2, uint16);
impl_bitround_codec!(UInt32DataType, 4, uint32);
impl_bitround_codec!(UInt64DataType, 8, uint64);

// Pcodec implementations for standard unsigned integers (uint8 not supported)
use zarrs_data_type::codec_traits::impl_pcodec_data_type_traits;
impl_pcodec_data_type_traits!(UInt16DataType, U16, 1);
impl_pcodec_data_type_traits!(UInt32DataType, U32, 1);
impl_pcodec_data_type_traits!(UInt64DataType, U64, 1);

// FixedScaleOffset implementations for standard unsigned integers
use zarrs_data_type::codec_traits::impl_fixed_scale_offset_data_type_traits;
impl_fixed_scale_offset_data_type_traits!(UInt8DataType, U8);
impl_fixed_scale_offset_data_type_traits!(UInt16DataType, U16);
impl_fixed_scale_offset_data_type_traits!(UInt32DataType, U32);
impl_fixed_scale_offset_data_type_traits!(UInt64DataType, U64);

// ZFP implementations for standard unsigned integers
use zarrs_data_type::codec_traits::impl_zfp_data_type_traits;
impl_zfp_data_type_traits!(UInt8DataType, UInt8);
impl_zfp_data_type_traits!(UInt16DataType, UInt16);
impl_zfp_data_type_traits!(UInt32DataType, UInt32);
impl_zfp_data_type_traits!(UInt64DataType, UInt64);

// PackBits implementations for standard unsigned integers
use zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits;
impl_pack_bits_data_type_traits!(UInt8DataType, 8, unsigned, 1);
impl_pack_bits_data_type_traits!(UInt16DataType, 16, unsigned, 1);
impl_pack_bits_data_type_traits!(UInt32DataType, 32, unsigned, 1);
impl_pack_bits_data_type_traits!(UInt64DataType, 64, unsigned, 1);

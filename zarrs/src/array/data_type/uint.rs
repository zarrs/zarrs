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

// CastValue codec implementations for standard unsigned integers
use zarrs_data_type::codec_traits::impl_cast_value_data_type_traits_unsigned_integer;
impl_cast_value_data_type_traits_unsigned_integer!(UInt8DataType, u8, 8);
impl_cast_value_data_type_traits_unsigned_integer!(UInt16DataType, u16, 16);
impl_cast_value_data_type_traits_unsigned_integer!(UInt32DataType, u32, 32);
impl_cast_value_data_type_traits_unsigned_integer!(UInt64DataType, u64, 64);

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

// ScaleOffset implementations for standard unsigned integers
use zarrs_data_type::codec_traits::impl_scale_offset_data_type_traits;
use zarrs_data_type::codec_traits::scale_offset::{
    ScaleOffsetDataTypeTraits, ScaleOffsetError, scale_offset_decode_int, scale_offset_encode_int,
};

impl ScaleOffsetDataTypeTraits for UInt8DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u8 = match offset {
            Some(bytes) => u8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u8 = match scale {
            Some(bytes) => u8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<1>().0 {
            let value = u8::from_ne_bytes(*chunk);
            let result = scale_offset_encode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u8 = match offset {
            Some(bytes) => u8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u8 = match scale {
            Some(bytes) => u8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<1>().0 {
            let value = u8::from_ne_bytes(*chunk);
            let result = scale_offset_decode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }
}
impl_scale_offset_data_type_traits!(UInt8DataType);

impl ScaleOffsetDataTypeTraits for UInt16DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u16 = match offset {
            Some(bytes) => u16::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u16 = match scale {
            Some(bytes) => u16::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<2>().0 {
            let value = u16::from_ne_bytes(*chunk);
            let result = scale_offset_encode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u16 = match offset {
            Some(bytes) => u16::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u16 = match scale {
            Some(bytes) => u16::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<2>().0 {
            let value = u16::from_ne_bytes(*chunk);
            let result = scale_offset_decode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }
}
impl_scale_offset_data_type_traits!(UInt16DataType);

impl ScaleOffsetDataTypeTraits for UInt32DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u32 = match offset {
            Some(bytes) => u32::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u32 = match scale {
            Some(bytes) => u32::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<4>().0 {
            let value = u32::from_ne_bytes(*chunk);
            let result = scale_offset_encode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u32 = match offset {
            Some(bytes) => u32::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u32 = match scale {
            Some(bytes) => u32::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<4>().0 {
            let value = u32::from_ne_bytes(*chunk);
            let result = scale_offset_decode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }
}
impl_scale_offset_data_type_traits!(UInt32DataType);

impl ScaleOffsetDataTypeTraits for UInt64DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u64 = match offset {
            Some(bytes) => u64::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u64 = match scale {
            Some(bytes) => u64::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<8>().0 {
            let value = u64::from_ne_bytes(*chunk);
            let result = scale_offset_encode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: u64 = match offset {
            Some(bytes) => u64::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: u64 = match scale {
            Some(bytes) => u64::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<8>().0 {
            let value = u64::from_ne_bytes(*chunk);
            let result = scale_offset_decode_int(&value, &offset, &scale)?;
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }
}
impl_scale_offset_data_type_traits!(UInt64DataType);

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

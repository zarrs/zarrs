//! The `zfp` array to bytes codec.
//!
//! <div class="warning">
//! This is a registered codec that originated in `zarrs`. It may not be supported by other Zarr V3 implementations.
//! </div>
//!
//! [zfp](https://zfp.io/) is a compressed number format for 1D to 4D arrays of 32/64-bit floating point or integer data.
//! 8/16-bit integer types are supported through promotion to 32-bit in accordance with the [zfp utility functions](https://zfp.readthedocs.io/en/release1.0.1/low-level-api.html#utility-functions).
//!
//! This codec requires the `zfp` feature, which is disabled by default.
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification:
//! - <https://codec.zarrs.dev/array_to_bytes/zfp>
//!
//! This codec is similar to `numcodecs.zfpy` with the following except:
//! - `"mode"`s are specified as strings
//! - reversible mode and expert mode are supported
//! - a zfp header is **not** written, as it is redundant with the codec metadata
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `zfp`
//! - `zarrs.zfp`
//! - `https://codec.zarrs.dev/array_to_bytes/zfp`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `zfp`
//!
//! ### Codec `configuration` Examples - [`ZfpCodecConfiguration`]:
//! #### Encode in fixed rate mode with 10.5 compressed bits per value
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": "fixed_rate",
//!     "rate": 10.5
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
//! # let configuration: ZfpCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! #### Encode in fixed precision mode with 19 uncompressed bits per value
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": "fixed_precision",
//!     "precision": 19
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
//! # let configuration: ZfpCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! #### Encode in fixed accuracy mode with a tolerance of 0.05
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": "fixed_accuracy",
//!     "tolerance": 0.05
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
//! # let configuration: ZfpCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! #### Encode in reversible mode
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": "reversible"
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
//! # let configuration: ZfpCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! #### Encode in expert mode
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": "expert",
//!     "minbits": 1,
//!     "maxbits": 13,
//!     "maxprec": 19,
//!     "minexp": -2
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
//! # let configuration: ZfpCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();

mod zfp_array;
mod zfp_bitstream;
mod zfp_codec;
mod zfp_field;
mod zfp_stream;

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_plugin::ExtensionIdentifier;
pub use zfp_codec::ZfpCodec;
use zfp_sys::{
    zfp_decompress, zfp_exec_policy_zfp_exec_omp, zfp_field_alloc, zfp_field_free,
    zfp_field_set_pointer, zfp_read_header, zfp_stream_close, zfp_stream_open, zfp_stream_rewind,
    zfp_stream_set_bit_stream, zfp_stream_set_execution,
};

use self::zfp_array::ZfpArray;
use self::zfp_bitstream::ZfpBitstream;
use self::zfp_field::ZfpField;
use self::zfp_stream::ZfpStream;
use crate::array::codec::{Codec, CodecError, CodecPlugin};
use crate::array::{ChunkShapeTraits, DataType, convert_from_bytes_slice, transmute_to_bytes_vec};
use crate::metadata::v3::MetadataV3;
pub use crate::metadata_ext::codec::zfp::{
    ZfpCodecConfiguration, ZfpCodecConfigurationV1, ZfpMode,
};
use crate::plugin::{PluginCreateError, PluginMetadataInvalidError};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(ZfpCodec::IDENTIFIER, ZfpCodec::matches_name, ZfpCodec::default_name, create_codec_zfp)
}
zarrs_plugin::impl_extension_aliases!(ZfpCodec, "zfp",
    v3: "zfp", ["zarrs.zfp", "https://codec.zarrs.dev/array_to_bytes/zfp"]
);

pub(crate) fn create_codec_zfp(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ZfpCodecConfiguration = metadata.to_configuration().map_err(|_| {
        PluginMetadataInvalidError::new(ZfpCodec::IDENTIFIER, "codec", metadata.to_string())
    })?;
    let codec = Arc::new(ZfpCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

/// The zfp data type for dispatching to the zfp library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZfpType {
    /// 32-bit integer (`zfp_type_int32`)
    Int32,
    /// 64-bit integer (`zfp_type_int64`)
    Int64,
    /// 32-bit float (`zfp_type_float`)
    Float,
    /// 64-bit float (`zfp_type_double`)
    Double,
}

/// Promotion strategy for types that need to be promoted before zfp encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZfpPromotion {
    /// No promotion needed - use the type directly
    None,
    /// Promote i8 to i32 with left shift of 23 bits
    I8ToI32,
    /// Promote u8 to i32: (i32(u) - 0x80) << 23
    U8ToI32,
    /// Promote i16 to i32 with left shift of 15 bits
    I16ToI32,
    /// Promote u16 to i32: (i32(u) - 0x8000) << 15
    U16ToI32,
    /// Promote u32 to i32: min(u, `i32::MAX`) as i32
    U32ToI32,
    /// Promote u64 to i64: min(u, `i64::MAX`) as i64
    U64ToI64,
}

/// Traits for a data type supporting the `zfp` codec.
///
/// The zfp codec provides lossy and lossless compression for floating point and integer data.
pub trait ZfpCodecDataTypeTraits {
    /// Returns the zfp type for this data type.
    ///
    /// Returns `None` if the data type is not supported by zfp.
    fn zfp_type(&self) -> Option<ZfpType>;

    /// Returns the promotion strategy for encoding.
    ///
    /// For types that don't need promotion, returns `ZfpPromotion::None`.
    fn zfp_promotion(&self) -> ZfpPromotion {
        ZfpPromotion::None
    }
}

// Generate the codec support infrastructure using the generic macro
crate::array::codec::define_data_type_support!(Zfp, ZfpCodecDataTypeTraits);

/// Macro to implement `ZfpCodecDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// // Unsupported type:
/// impl_zfp_codec!(BFloat16DataType, None);
///
/// // Native ZFP type (no promotion):
/// impl_zfp_codec!(Int32DataType, Int32);
/// impl_zfp_codec!(Float32DataType, Float);
///
/// // Promoted type:
/// impl_zfp_codec!(Int8DataType, Int32, I8ToI32);
/// impl_zfp_codec!(Int16DataType, Int32, I16ToI32);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_zfp_codec {
    // Unsupported type - still implements the trait but returns None, and registers support
    ($marker:ty, None) => {
        impl $crate::array::codec::ZfpCodecDataTypeTraits for $marker {
            fn zfp_type(&self) -> Option<$crate::array::codec::ZfpType> {
                None
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::ZfpPlugin,
            $crate::array::codec::ZfpCodecDataTypeTraits
        );
    };
    // Native type (no promotion needed)
    ($marker:ty, $zfp_type:ident) => {
        impl $crate::array::codec::ZfpCodecDataTypeTraits for $marker {
            fn zfp_type(&self) -> Option<$crate::array::codec::ZfpType> {
                Some($crate::array::codec::ZfpType::$zfp_type)
            }
            fn zfp_promotion(&self) -> $crate::array::codec::ZfpPromotion {
                $crate::array::codec::ZfpPromotion::None
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::ZfpPlugin,
            $crate::array::codec::ZfpCodecDataTypeTraits
        );
    };
    // Promoted type
    ($marker:ty, $zfp_type:ident, $promotion:ident) => {
        impl $crate::array::codec::ZfpCodecDataTypeTraits for $marker {
            fn zfp_type(&self) -> Option<$crate::array::codec::ZfpType> {
                Some($crate::array::codec::ZfpType::$zfp_type)
            }
            fn zfp_promotion(&self) -> $crate::array::codec::ZfpPromotion {
                $crate::array::codec::ZfpPromotion::$promotion
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::ZfpPlugin,
            $crate::array::codec::ZfpCodecDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_zfp_codec as impl_zfp_codec;

fn zfp_type_to_sys(zfp_type: ZfpType) -> zfp_sys::zfp_type {
    match zfp_type {
        ZfpType::Int32 => zfp_sys::zfp_type_zfp_type_int32,
        ZfpType::Int64 => zfp_sys::zfp_type_zfp_type_int64,
        ZfpType::Float => zfp_sys::zfp_type_zfp_type_float,
        ZfpType::Double => zfp_sys::zfp_type_zfp_type_double,
    }
}

fn zarr_to_zfp_data_type(data_type: &DataType) -> Option<zfp_sys::zfp_type> {
    let zfp = get_zfp_support(data_type)?;
    zfp.zfp_type().map(zfp_type_to_sys)
}

#[allow(clippy::cast_possible_wrap)]
fn promote_before_zfp_encoding(
    decoded_value: &[u8],
    data_type: &DataType,
) -> Result<ZfpArray, CodecError> {
    let zfp = get_zfp_support(data_type).ok_or_else(|| {
        CodecError::UnsupportedDataType(data_type.clone(), ZfpCodec::IDENTIFIER.to_string())
    })?;
    let zfp_type = zfp.zfp_type().ok_or_else(|| {
        CodecError::UnsupportedDataType(data_type.clone(), ZfpCodec::IDENTIFIER.to_string())
    })?;
    let promotion = zfp.zfp_promotion();

    match (zfp_type, promotion) {
        // Direct copy - no promotion needed
        (ZfpType::Int32, ZfpPromotion::None) => Ok(ZfpArray::Int32(
            convert_from_bytes_slice::<i32>(decoded_value),
        )),
        (ZfpType::Int64, ZfpPromotion::None) => Ok(ZfpArray::Int64(
            convert_from_bytes_slice::<i64>(decoded_value),
        )),
        (ZfpType::Float, ZfpPromotion::None) => Ok(ZfpArray::Float(
            convert_from_bytes_slice::<f32>(decoded_value),
        )),
        (ZfpType::Double, ZfpPromotion::None) => Ok(ZfpArray::Double(convert_from_bytes_slice::<
            f64,
        >(decoded_value))),
        // i8 -> i32 with left shift 23
        (ZfpType::Int32, ZfpPromotion::I8ToI32) => {
            let values = convert_from_bytes_slice::<i8>(decoded_value);
            let promoted = values.into_iter().map(|i| i32::from(i) << 23).collect();
            Ok(ZfpArray::Int32(promoted))
        }
        // u8 -> i32 with offset and shift
        (ZfpType::Int32, ZfpPromotion::U8ToI32) => {
            let values = convert_from_bytes_slice::<u8>(decoded_value);
            let promoted = values
                .into_iter()
                .map(|i| (i32::from(i) - 0x80) << 23)
                .collect();
            Ok(ZfpArray::Int32(promoted))
        }
        // i16 -> i32 with left shift 15
        (ZfpType::Int32, ZfpPromotion::I16ToI32) => {
            let values = convert_from_bytes_slice::<i16>(decoded_value);
            let promoted = values.into_iter().map(|i| i32::from(i) << 15).collect();
            Ok(ZfpArray::Int32(promoted))
        }
        // u16 -> i32 with offset and shift
        (ZfpType::Int32, ZfpPromotion::U16ToI32) => {
            let values = convert_from_bytes_slice::<u16>(decoded_value);
            let promoted = values
                .into_iter()
                .map(|i| (i32::from(i) - 0x8000) << 15)
                .collect();
            Ok(ZfpArray::Int32(promoted))
        }
        // u32 -> i32 clamping to i32::MAX
        (ZfpType::Int32, ZfpPromotion::U32ToI32) => {
            let values = convert_from_bytes_slice::<u32>(decoded_value);
            let promoted = values
                .into_iter()
                .map(|u| core::cmp::min(u, i32::MAX as u32) as i32)
                .collect();
            Ok(ZfpArray::Int32(promoted))
        }
        // u64 -> i64 clamping to i64::MAX
        (ZfpType::Int64, ZfpPromotion::U64ToI64) => {
            let values = convert_from_bytes_slice::<u64>(decoded_value);
            let promoted = values
                .into_iter()
                .map(|u| core::cmp::min(u, i64::MAX as u64) as i64)
                .collect();
            Ok(ZfpArray::Int64(promoted))
        }
        // Invalid combinations
        _ => Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            ZfpCodec::IDENTIFIER.to_string(),
        )),
    }
}

fn init_zfp_decoding_output(
    shape: &[NonZeroU64],
    data_type: &DataType,
) -> Result<ZfpArray, CodecError> {
    let zfp = get_zfp_support(data_type).ok_or_else(|| {
        CodecError::UnsupportedDataType(data_type.clone(), ZfpCodec::IDENTIFIER.to_string())
    })?;
    let zfp_type = zfp.zfp_type().ok_or_else(|| {
        CodecError::UnsupportedDataType(data_type.clone(), ZfpCodec::IDENTIFIER.to_string())
    })?;

    let num_elements = shape.num_elements_usize();
    match zfp_type {
        ZfpType::Int32 => Ok(ZfpArray::Int32(vec![0; num_elements])),
        ZfpType::Int64 => Ok(ZfpArray::Int64(vec![0; num_elements])),
        ZfpType::Float => Ok(ZfpArray::Float(vec![0.0; num_elements])),
        ZfpType::Double => Ok(ZfpArray::Double(vec![0.0; num_elements])),
    }
}

#[allow(clippy::cast_sign_loss)]
fn demote_after_zfp_decoding(array: ZfpArray, data_type: &DataType) -> Result<Vec<u8>, CodecError> {
    let zfp = get_zfp_support(data_type).ok_or_else(|| {
        CodecError::UnsupportedDataType(data_type.clone(), ZfpCodec::IDENTIFIER.to_string())
    })?;
    let promotion = zfp.zfp_promotion();

    match (array, promotion) {
        // Direct copy - no demotion needed
        (ZfpArray::Int32(vec), ZfpPromotion::None) => Ok(transmute_to_bytes_vec(vec)),
        (ZfpArray::Int64(vec), ZfpPromotion::None) => Ok(transmute_to_bytes_vec(vec)),
        (ZfpArray::Float(vec), ZfpPromotion::None) => Ok(transmute_to_bytes_vec(vec)),
        (ZfpArray::Double(vec), ZfpPromotion::None) => Ok(transmute_to_bytes_vec(vec)),
        // i32 -> i8 with right shift and clamp
        (ZfpArray::Int32(vec), ZfpPromotion::I8ToI32) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| i8::try_from((i >> 23).clamp(-0x80, 0x7f)).unwrap())
                .collect::<Vec<_>>(),
        )),
        // i32 -> u8 with right shift, offset, and clamp
        (ZfpArray::Int32(vec), ZfpPromotion::U8ToI32) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| u8::try_from(((i >> 23) + 0x80).clamp(0x00, 0xff)).unwrap())
                .collect::<Vec<_>>(),
        )),
        // i32 -> i16 with right shift and clamp
        (ZfpArray::Int32(vec), ZfpPromotion::I16ToI32) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| i16::try_from((i >> 15).clamp(-0x8000, 0x7fff)).unwrap())
                .collect::<Vec<_>>(),
        )),
        // i32 -> u16 with right shift, offset, and clamp
        (ZfpArray::Int32(vec), ZfpPromotion::U16ToI32) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| u16::try_from(((i >> 15) + 0x8000).clamp(0x0000, 0xffff)).unwrap())
                .collect::<Vec<_>>(),
        )),
        // i32 -> u32 clamping to 0
        (ZfpArray::Int32(vec), ZfpPromotion::U32ToI32) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| core::cmp::max(i, 0) as u32)
                .collect::<Vec<_>>(),
        )),
        // i64 -> u64 clamping to 0
        (ZfpArray::Int64(vec), ZfpPromotion::U64ToI64) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| core::cmp::max(i, 0) as u64)
                .collect::<Vec<_>>(),
        )),
        // Invalid combinations
        _ => Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            ZfpCodec::IDENTIFIER.to_string(),
        )),
    }
}

fn zfp_decode(
    zfp_mode: &ZfpMode,
    write_header: bool,
    encoded_value: &mut [u8],
    shape: &[NonZeroU64],
    data_type: &DataType,
    parallel: bool,
) -> Result<Vec<u8>, CodecError> {
    let mut array = init_zfp_decoding_output(shape, data_type)?;
    let zfp_type = array.zfp_type();
    let stream = ZfpStream::new(zfp_mode, zfp_type)
        .ok_or_else(|| CodecError::from("failed to create zfp stream"))?;

    let bitstream = ZfpBitstream::new(encoded_value)
        .ok_or_else(|| CodecError::from("failed to create zfp bitstream"))?;
    if write_header {
        let ret = unsafe {
            let field = zfp_field_alloc();
            let stream = zfp_stream_open(bitstream.as_bitstream());
            zfp_stream_open(bitstream.as_bitstream());
            zfp_read_header(stream, field, zfp_sys::ZFP_HEADER_FULL);
            zfp_field_set_pointer(field, array.as_mut_ptr());
            let ret = zfp_decompress(stream, field);
            zfp_stream_close(stream);
            zfp_field_free(field);
            ret
        };
        if ret == 0 {
            return Err(CodecError::from("zfp decompression failed"));
        }
    } else {
        let field = ZfpField::new(
            &mut array,
            &shape
                .iter()
                .map(|u| usize::try_from(u.get()).unwrap())
                .collect::<Vec<usize>>(),
        )
        .ok_or_else(|| CodecError::from("failed to create zfp field"))?;
        let ret = unsafe {
            zfp_stream_set_bit_stream(stream.as_zfp_stream(), bitstream.as_bitstream());
            zfp_stream_rewind(stream.as_zfp_stream());
            if parallel {
                zfp_stream_set_execution(stream.as_zfp_stream(), zfp_exec_policy_zfp_exec_omp);
            }
            zfp_decompress(stream.as_zfp_stream(), field.as_zfp_field())
        };
        if ret == 0 {
            return Err(CodecError::from("zfp decompression failed"));
        }
    }

    demote_after_zfp_decoding(array, data_type)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use num::traits::AsPrimitive;

    use super::*;
    use crate::array::codec::array_to_array::squeeze::SqueezeCodec;
    use crate::array::codec::{ArrayToBytesCodecTraits, BytesPartialDecoderTraits, CodecOptions};
    use crate::array::element::ElementOwned;
    use crate::array::{
        ArrayBytes, ChunkShape, ChunkShapeTraits, CodecChain, FillValue, data_type,
    };
    use crate::array_subset::ArraySubset;

    const JSON_REVERSIBLE: &str = r#"{
        "mode": "reversible"
    }"#;

    fn json_fixedrate(rate: f32) -> String {
        format!(r#"{{ "mode": "fixed_rate", "rate": {rate} }}"#)
    }

    fn json_fixedprecision(precision: u32) -> String {
        format!(r#"{{ "mode": "fixed_precision", "precision": {precision} }}"#)
    }

    fn json_fixedaccuracy(tolerance: f32) -> String {
        format!(r#"{{ "mode": "fixed_accuracy", "tolerance": {tolerance} }}"#)
    }

    fn chunk_shape() -> ChunkShape {
        ChunkShape::from(vec![
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ])
    }

    fn codec_zfp_round_trip<
        T: core::fmt::Debug + std::cmp::PartialEq + ElementOwned + Copy + 'static,
    >(
        chunk_shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        configuration: &str,
    ) where
        u64: num::traits::AsPrimitive<T>,
    {
        let elements: Vec<T> = (0..chunk_shape.num_elements_u64())
            .map(|i: u64| i.as_())
            .collect();
        let bytes = T::to_array_bytes(data_type, &elements).unwrap();

        let configuration: ZfpCodecConfiguration = serde_json::from_str(configuration).unwrap();
        let codec = CodecChain::new(
            vec![Arc::new(SqueezeCodec::new())],
            Arc::new(ZfpCodec::new_with_configuration(&configuration).unwrap()),
            vec![],
        );

        let encoded = codec
            .encode(
                bytes.clone(),
                chunk_shape,
                data_type,
                fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded.clone(),
                chunk_shape,
                data_type,
                fill_value,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_owned();
        let decoded_elements = T::from_array_bytes(data_type, decoded).unwrap();
        assert_eq!(elements, decoded_elements);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i8() {
        codec_zfp_round_trip::<i8>(
            &chunk_shape(),
            &data_type::int8(),
            &0i8.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u8() {
        codec_zfp_round_trip::<u8>(
            &chunk_shape(),
            &data_type::uint8(),
            &0u8.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i16() {
        codec_zfp_round_trip::<i16>(
            &chunk_shape(),
            &data_type::int16(),
            &0i16.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u16() {
        codec_zfp_round_trip::<u16>(
            &chunk_shape(),
            &data_type::uint16(),
            &0u16.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i32() {
        codec_zfp_round_trip::<i32>(
            &chunk_shape(),
            &data_type::int32(),
            &0i32.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u32() {
        codec_zfp_round_trip::<u32>(
            &chunk_shape(),
            &data_type::uint32(),
            &0u32.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i64() {
        codec_zfp_round_trip::<i64>(
            &chunk_shape(),
            &data_type::int64(),
            &0i64.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u64() {
        codec_zfp_round_trip::<u64>(
            &chunk_shape(),
            &data_type::uint64(),
            &0u64.into(),
            JSON_REVERSIBLE,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_f32() {
        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            JSON_REVERSIBLE,
        );
        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            &json_fixedrate(2.5),
        );
        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            &json_fixedaccuracy(1.0),
        );
        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            &json_fixedprecision(13),
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_f64() {
        codec_zfp_round_trip::<f64>(
            &chunk_shape(),
            &data_type::float64(),
            &0.0f64.into(),
            JSON_REVERSIBLE,
        );
        codec_zfp_round_trip::<f64>(
            &chunk_shape(),
            &data_type::float64(),
            &0.0f64.into(),
            &json_fixedrate(2.5),
        );
        codec_zfp_round_trip::<f64>(
            &chunk_shape(),
            &data_type::float64(),
            &0.0f64.into(),
            &json_fixedaccuracy(1.0),
        );
        codec_zfp_round_trip::<f64>(
            &chunk_shape(),
            &data_type::float64(),
            &0.0f64.into(),
            &json_fixedprecision(16),
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_partial_decode() {
        let chunk_shape = ChunkShape::from(vec![
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ]);
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let elements: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let configuration: ZfpCodecConfiguration = serde_json::from_str(JSON_REVERSIBLE).unwrap();
        let codec = Arc::new(ZfpCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_regions = [
            ArraySubset::new_with_shape(vec![1, 2, 3]),
            ArraySubset::new_with_ranges(&[0..3, 1..3, 2..3]),
        ];

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // zfp partial decoder does not hold bytes

        for (decoded_region, expected) in decoded_regions.into_iter().zip([
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 8.0, 14.0, 17.0, 23.0, 26.0],
        ]) {
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .unwrap();

            let decoded_partial_chunk: Vec<f32> = decoded_partial_chunk
                .into_fixed()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect();
            assert_eq!(decoded_partial_chunk, expected);
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn codec_zfp_async_partial_decode() {
        let chunk_shape = ChunkShape::from(vec![
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ]);
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let elements: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let configuration: ZfpCodecConfiguration = serde_json::from_str(JSON_REVERSIBLE).unwrap();
        let codec = Arc::new(ZfpCodec::new_with_configuration(&configuration).unwrap());

        let max_encoded_size = codec
            .encoded_representation(&chunk_shape, &data_type, &fill_value)
            .unwrap();
        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        assert!((encoded.len() as u64) <= max_encoded_size.size().unwrap());

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .await
            .unwrap();

        let decoded_regions = [
            ArraySubset::new_with_shape(vec![1, 2, 3]),
            ArraySubset::new_with_ranges(&[0..3, 1..3, 2..3]),
        ];

        for (decoded_region, expected) in decoded_regions.into_iter().zip([
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 8.0, 14.0, 17.0, 23.0, 26.0],
        ]) {
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .await
                .unwrap();

            let decoded_partial_chunk: Vec<f32> = decoded_partial_chunk
                .into_fixed()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect();
            assert_eq!(decoded_partial_chunk, expected);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_f32_6d() {
        let chunk_shape = || {
            ChunkShape::from(vec![
                NonZeroU64::new(4).unwrap(),
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(3).unwrap(),
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(1).unwrap(),
            ])
        };

        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            JSON_REVERSIBLE,
        );
        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            &json_fixedrate(2.5),
        );
        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            &json_fixedaccuracy(1.0),
        );
        codec_zfp_round_trip::<f32>(
            &chunk_shape(),
            &data_type::float32(),
            &0.0f32.into(),
            &json_fixedprecision(13),
        );
    }
}

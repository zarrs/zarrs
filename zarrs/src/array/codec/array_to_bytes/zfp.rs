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
//! None
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
//! # use zarrs_metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
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
//! # use zarrs_metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
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
//! # use zarrs_metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
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
//! # use zarrs_metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
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
//! # use zarrs_metadata_ext::codec::zfp::ZfpCodecConfigurationV1;
//! # let configuration: ZfpCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();

mod zfp_array;
mod zfp_bitstream;
mod zfp_codec;
mod zfp_field;
mod zfp_partial_decoder;
mod zfp_stream;

use std::sync::Arc;

pub use zarrs_metadata_ext::codec::zfp::{ZfpCodecConfiguration, ZfpCodecConfigurationV1, ZfpMode};
pub use zfp_codec::ZfpCodec;

use zfp_sys::{
    zfp_decompress, zfp_exec_policy_zfp_exec_omp, zfp_field_alloc, zfp_field_free,
    zfp_field_set_pointer, zfp_read_header, zfp_stream_close, zfp_stream_open, zfp_stream_rewind,
    zfp_stream_set_bit_stream, zfp_stream_set_execution,
};

use crate::{
    array::{
        codec::{Codec, CodecError, CodecPlugin},
        convert_from_bytes_slice, transmute_to_bytes_vec, ChunkRepresentation, DataType,
    },
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};
use zarrs_registry::codec::ZFP;

use self::{
    zfp_array::ZfpArray, zfp_bitstream::ZfpBitstream, zfp_field::ZfpField, zfp_stream::ZfpStream,
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(ZFP, is_identifier_zfp, create_codec_zfp)
}

fn is_identifier_zfp(identifier: &str) -> bool {
    identifier == ZFP
}

pub(crate) fn create_codec_zfp(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ZfpCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(ZFP, "codec", metadata.to_string()))?;
    let codec = Arc::new(ZfpCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

macro_rules! unsupported_dtypes {
    // TODO: Add support for extensions?
    // TODO: Add support for complex types (as a dimension of size 2)?
    () => {
        DataType::Bool
            | DataType::Int2
            | DataType::Int4
            | DataType::UInt2
            | DataType::UInt4
            | DataType::Float4E2M1FN
            | DataType::Float6E2M3FN
            | DataType::Float6E3M2FN
            | DataType::Float8E3M4
            | DataType::Float8E4M3
            | DataType::Float8E4M3B11FNUZ
            | DataType::Float8E4M3FNUZ
            | DataType::Float8E5M2
            | DataType::Float8E5M2FNUZ
            | DataType::Float8E8M0FNU
            | DataType::BFloat16
            | DataType::Float16
            | DataType::ComplexBFloat16
            | DataType::ComplexFloat16
            | DataType::ComplexFloat32
            | DataType::ComplexFloat64
            | DataType::ComplexFloat4E2M1FN
            | DataType::ComplexFloat6E2M3FN
            | DataType::ComplexFloat6E3M2FN
            | DataType::ComplexFloat8E3M4
            | DataType::ComplexFloat8E4M3
            | DataType::ComplexFloat8E4M3B11FNUZ
            | DataType::ComplexFloat8E4M3FNUZ
            | DataType::ComplexFloat8E5M2
            | DataType::ComplexFloat8E5M2FNUZ
            | DataType::ComplexFloat8E8M0FNU
            | DataType::Complex64
            | DataType::Complex128
            | DataType::RawBits(_)
            | DataType::String
            | DataType::Bytes
            | DataType::Extension(_)
    };
}
use unsupported_dtypes;

const fn zarr_to_zfp_data_type(data_type: &DataType) -> Option<zfp_sys::zfp_type> {
    match data_type {
        DataType::Int8
        | DataType::UInt8
        | DataType::Int16
        | DataType::UInt16
        | DataType::Int32
        | DataType::UInt32 => Some(zfp_sys::zfp_type_zfp_type_int32),
        DataType::Int64
        | DataType::UInt64
        | DataType::NumpyDateTime64 {
            unit: _,
            scale_factor: _,
        }
        | DataType::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _,
        } => Some(zfp_sys::zfp_type_zfp_type_int64),
        DataType::Float32 => Some(zfp_sys::zfp_type_zfp_type_float),
        DataType::Float64 => Some(zfp_sys::zfp_type_zfp_type_double),
        unsupported_dtypes!() => None,
    }
}

fn promote_before_zfp_encoding(
    decoded_value: &[u8],
    decoded_representation: &ChunkRepresentation,
) -> Result<ZfpArray, CodecError> {
    #[allow(clippy::cast_possible_wrap)]
    match decoded_representation.data_type() {
        DataType::Int8 => {
            let decoded_value = convert_from_bytes_slice::<i8>(decoded_value);
            let decoded_value_promoted = decoded_value
                .into_iter()
                .map(|i| i32::from(i) << 23)
                .collect();
            Ok(ZfpArray::Int32(decoded_value_promoted))
        }
        DataType::UInt8 => {
            let decoded_value = convert_from_bytes_slice::<u8>(decoded_value);
            let decoded_value_promoted = decoded_value
                .into_iter()
                .map(|i| (i32::from(i) - 0x80) << 23)
                .collect();
            Ok(ZfpArray::Int32(decoded_value_promoted))
        }
        DataType::Int16 => {
            let decoded_value = convert_from_bytes_slice::<i16>(decoded_value);
            let decoded_value_promoted = decoded_value
                .into_iter()
                .map(|i| i32::from(i) << 15)
                .collect();
            Ok(ZfpArray::Int32(decoded_value_promoted))
        }
        DataType::UInt16 => {
            let decoded_value = convert_from_bytes_slice::<u16>(decoded_value);
            let decoded_value_promoted = decoded_value
                .into_iter()
                .map(|i| (i32::from(i) - 0x8000) << 15)
                .collect();
            Ok(ZfpArray::Int32(decoded_value_promoted))
        }
        DataType::Int32 => Ok(ZfpArray::Int32(convert_from_bytes_slice::<i32>(
            decoded_value,
        ))),
        DataType::UInt32 => {
            let u = convert_from_bytes_slice::<u32>(decoded_value);
            let i = u
                .into_iter()
                .map(|u| core::cmp::min(u, i32::MAX as u32) as i32)
                .collect();
            Ok(ZfpArray::Int32(i))
        }
        DataType::Int64
        | DataType::NumpyDateTime64 {
            unit: _,
            scale_factor: _,
        }
        | DataType::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _,
        } => Ok(ZfpArray::Int64(convert_from_bytes_slice::<i64>(
            decoded_value,
        ))),
        DataType::UInt64 => {
            let u = convert_from_bytes_slice::<u64>(decoded_value);
            let i = u
                .into_iter()
                .map(|u| core::cmp::min(u, i64::MAX as u64) as i64)
                .collect();
            Ok(ZfpArray::Int64(i))
        }
        DataType::Float32 => Ok(ZfpArray::Float(convert_from_bytes_slice::<f32>(
            decoded_value,
        ))),
        DataType::Float64 => Ok(ZfpArray::Double(convert_from_bytes_slice::<f64>(
            decoded_value,
        ))),
        unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
            decoded_representation.data_type().clone(),
            ZFP.to_string(),
        )),
    }
}

fn init_zfp_decoding_output(
    decoded_representation: &ChunkRepresentation,
) -> Result<ZfpArray, CodecError> {
    let num_elements = decoded_representation.num_elements_usize();
    match decoded_representation.data_type() {
        DataType::Int8
        | DataType::UInt8
        | DataType::Int16
        | DataType::UInt16
        | DataType::Int32
        | DataType::UInt32 => Ok(ZfpArray::Int32(vec![0; num_elements])),
        DataType::Int64
        | DataType::UInt64
        | DataType::NumpyDateTime64 {
            unit: _,
            scale_factor: _,
        }
        | DataType::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _,
        } => Ok(ZfpArray::Int64(vec![0; num_elements])),
        DataType::Float32 => Ok(ZfpArray::Float(vec![0.0; num_elements])),
        DataType::Float64 => Ok(ZfpArray::Double(vec![0.0; num_elements])),
        unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
            decoded_representation.data_type().clone(),
            ZFP.to_string(),
        )),
    }
}

fn demote_after_zfp_decoding(
    array: ZfpArray,
    decoded_representation: &ChunkRepresentation,
) -> Result<Vec<u8>, CodecError> {
    #[allow(clippy::cast_sign_loss)]
    match (decoded_representation.data_type(), array) {
        (DataType::Int32, ZfpArray::Int32(vec)) => Ok(transmute_to_bytes_vec(vec)),
        (DataType::UInt32, ZfpArray::Int32(vec)) => {
            let vec = vec
                .into_iter()
                .map(|i| core::cmp::max(i, 0) as u32)
                .collect();
            Ok(transmute_to_bytes_vec(vec))
        }
        (DataType::Int64, ZfpArray::Int64(vec)) => Ok(transmute_to_bytes_vec(vec)),
        (DataType::UInt64, ZfpArray::Int64(vec)) => {
            let vec = vec
                .into_iter()
                .map(|i| core::cmp::max(i, 0) as u64)
                .collect();
            Ok(transmute_to_bytes_vec(vec))
        }
        (DataType::Float32, ZfpArray::Float(vec)) => Ok(transmute_to_bytes_vec(vec)),
        (DataType::Float64, ZfpArray::Double(vec)) => Ok(transmute_to_bytes_vec(vec)),
        (DataType::Int8, ZfpArray::Int32(vec)) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| i8::try_from((i >> 23).clamp(-0x80, 0x7f)).unwrap())
                .collect(),
        )),
        (DataType::UInt8, ZfpArray::Int32(vec)) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| u8::try_from(((i >> 23) + 0x80).clamp(0x00, 0xff)).unwrap())
                .collect(),
        )),
        (DataType::Int16, ZfpArray::Int32(vec)) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| i16::try_from((i >> 15).clamp(-0x8000, 0x7fff)).unwrap())
                .collect(),
        )),
        (DataType::UInt16, ZfpArray::Int32(vec)) => Ok(transmute_to_bytes_vec(
            vec.into_iter()
                .map(|i| u16::try_from(((i >> 15) + 0x8000).clamp(0x0000, 0xffff)).unwrap())
                .collect(),
        )),
        _ => Err(CodecError::UnsupportedDataType(
            decoded_representation.data_type().clone(),
            ZFP.to_string(),
        )),
    }
}

fn zfp_decode(
    zfp_mode: &ZfpMode,
    write_header: bool,
    encoded_value: &mut [u8],
    decoded_representation: &ChunkRepresentation,
    parallel: bool,
) -> Result<Vec<u8>, CodecError> {
    let mut array = init_zfp_decoding_output(decoded_representation)?;
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
            &decoded_representation
                .shape()
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

    demote_after_zfp_decoding(array, decoded_representation)
}

#[cfg(test)]
mod tests {
    use num::traits::AsPrimitive;
    use std::{num::NonZeroU64, sync::Arc};

    use crate::{
        array::{
            codec::{
                array_to_array::squeeze::SqueezeCodec, ArrayToBytesCodecTraits,
                BytesPartialDecoderTraits, CodecOptions,
            },
            element::ElementOwned,
            ArrayBytes, CodecChain,
        },
        array_subset::ArraySubset,
    };

    use super::*;

    const JSON_REVERSIBLE: &'static str = r#"{
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

    fn chunk_shape() -> Vec<NonZeroU64> {
        vec![
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ]
    }

    fn codec_zfp_round_trip<
        T: core::fmt::Debug + std::cmp::PartialEq + ElementOwned + Copy + 'static,
    >(
        chunk_representation: &ChunkRepresentation,
        configuration: &str,
    ) where
        u64: num::traits::AsPrimitive<T>,
    {
        let elements: Vec<T> = (0..chunk_representation.num_elements())
            .map(|i: u64| i.as_())
            .collect();
        let bytes = T::into_array_bytes(chunk_representation.data_type(), &elements).unwrap();

        let configuration: ZfpCodecConfiguration = serde_json::from_str(configuration).unwrap();
        let codec = CodecChain::new(
            vec![Arc::new(SqueezeCodec::new())],
            Arc::new(ZfpCodec::new_with_configuration(&configuration).unwrap()),
            vec![],
        );

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_owned();
        let decoded_elements =
            T::from_array_bytes(chunk_representation.data_type(), decoded).unwrap();
        assert_eq!(elements, decoded_elements);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i8() {
        codec_zfp_round_trip::<i8>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Int8, 0i8).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<i8>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::Int8, 0i8.into()).unwrap(),
        //     &json_fixedprecision(8),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u8() {
        codec_zfp_round_trip::<u8>(
            &ChunkRepresentation::new(chunk_shape(), DataType::UInt8, 0u8).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<u8>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::UInt8, 0u8.into()).unwrap(),
        //     &json_fixedprecision(8),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i16() {
        codec_zfp_round_trip::<i16>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Int16, 0i16).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<i16>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::Int16, 0i16.into()).unwrap(),
        //     &json_fixedprecision(16),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u16() {
        codec_zfp_round_trip::<u16>(
            &ChunkRepresentation::new(chunk_shape(), DataType::UInt16, 0u16).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<u16>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::UInt16, 0u16.into()).unwrap(),
        //     &json_fixedprecision(16),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i32() {
        codec_zfp_round_trip::<i32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Int32, 0i32).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<i32>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::Int32, 0i32.into()).unwrap(),
        //     &json_fixedprecision(32),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u32() {
        codec_zfp_round_trip::<u32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::UInt32, 0u32).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<u32>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::UInt32, 0u32.into()).unwrap(),
        //     &json_fixedprecision(32),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_i64() {
        codec_zfp_round_trip::<i64>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Int64, 0i64).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<i64>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::Int64, 0i64.into()).unwrap(),
        //     &json_fixedprecision(64),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_u64() {
        codec_zfp_round_trip::<u64>(
            &ChunkRepresentation::new(chunk_shape(), DataType::UInt64, 0u64).unwrap(),
            JSON_REVERSIBLE,
        );
        // codec_zfp_round_trip::<u64>(
        //     &ChunkRepresentation::new(chunk_shape(), DataType::UInt64, 0u64).unwrap(),
        //     &json_fixedprecision(64),
        // );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_f32() {
        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            JSON_REVERSIBLE,
        );
        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            &json_fixedrate(2.5),
        );
        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            &json_fixedaccuracy(1.0),
        );
        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            &json_fixedprecision(13),
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_f64() {
        codec_zfp_round_trip::<f64>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float64, 0.0f64).unwrap(),
            JSON_REVERSIBLE,
        );
        codec_zfp_round_trip::<f64>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float64, 0.0f64).unwrap(),
            &json_fixedrate(2.5),
        );
        codec_zfp_round_trip::<f64>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float64, 0.0f64).unwrap(),
            &json_fixedaccuracy(1.0),
        );
        codec_zfp_round_trip::<f64>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float64, 0.0f64).unwrap(),
            &json_fixedprecision(16),
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_partial_decode() {
        let chunk_shape = vec![
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, DataType::Float32, 0.0f32).unwrap();
        let elements: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let configuration: ZfpCodecConfiguration = serde_json::from_str(JSON_REVERSIBLE).unwrap();
        let codec = Arc::new(ZfpCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
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
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size(), input_handle.size()); // zfp partial decoder does not hold bytes

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
                .chunks(size_of::<f32>())
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(decoded_partial_chunk, expected);
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn codec_zfp_async_partial_decode() {
        let chunk_shape = vec![
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, DataType::Float32, 0.0f32).unwrap();
        let elements: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let configuration: ZfpCodecConfiguration = serde_json::from_str(JSON_REVERSIBLE).unwrap();
        let codec = Arc::new(ZfpCodec::new_with_configuration(&configuration).unwrap());

        let max_encoded_size = codec.encoded_representation(&chunk_representation).unwrap();
        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert!((encoded.len() as u64) <= max_encoded_size.size().unwrap());

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &chunk_representation,
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
                .chunks(size_of::<f32>())
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(decoded_partial_chunk, expected);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zfp_round_trip_f32_6d() {
        let chunk_shape = || {
            vec![
                NonZeroU64::new(4).unwrap(),
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(3).unwrap(),
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(1).unwrap(),
            ]
        };

        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            JSON_REVERSIBLE,
        );
        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            &json_fixedrate(2.5),
        );
        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            &json_fixedaccuracy(1.0),
        );
        codec_zfp_round_trip::<f32>(
            &ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32).unwrap(),
            &json_fixedprecision(13),
        );
    }
}

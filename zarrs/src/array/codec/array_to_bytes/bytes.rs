//! The `bytes` array to bytes codec (Core).
//!
//! Encodes arrays of fixed-size numeric data types as little endian or big endian in lexicographical order.
//!
//! ### Compatible Implementations:
//! This is a core codec and should be compatible with all Zarr V3 implementations that support it.
//!
//! ### Specification
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html>
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/bytes>
//!
//! ### Specification Deviations
//! The `bytes` specification defines a fixed set of supported data types, whereas the `bytes` codec in `zarrs` supports any fixed size data type that implements the [`DataTypeExtensionBytesCodec`](zarrs_data_type::DataTypeExtensionBytesCodec) trait.
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `bytes`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`BytesCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "endian": "little"
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::bytes::BytesCodecConfiguration;
//! # serde_json::from_str::<BytesCodecConfiguration>(JSON).unwrap();
//! ```

mod bytes_codec;
mod bytes_codec_partial;

use std::sync::Arc;

use crate::metadata::Endianness;

pub use zarrs_metadata_ext::codec::bytes::{BytesCodecConfiguration, BytesCodecConfigurationV1};
use zarrs_registry::codec::BYTES;

pub use bytes_codec::BytesCodec;
pub(crate) use bytes_codec_partial::BytesCodecPartial;

use crate::{
    array::{
        codec::{Codec, CodecPlugin},
        DataType,
    },
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(BYTES, is_identifier_bytes, create_codec_bytes)
}

fn is_identifier_bytes(identifier: &str) -> bool {
    identifier == BYTES
}

pub(crate) fn create_codec_bytes(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: BytesCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(BYTES, "codec", metadata.to_string()))?;
    let codec = Arc::new(BytesCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

/// Reverse the endianness of bytes for a given data type.
pub(crate) fn reverse_endianness(v: &mut [u8], data_type: &DataType) {
    match data_type {
        DataType::Bool
        | DataType::Int2
        | DataType::Int4
        | DataType::Int8
        | DataType::UInt2
        | DataType::UInt4
        | DataType::UInt8
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
        | DataType::RawBits(_) => {}
        DataType::Int16
        | DataType::UInt16
        | DataType::Float16
        | DataType::BFloat16
        | DataType::ComplexFloat16
        | DataType::ComplexBFloat16 => {
            let swap = |chunk: &mut [u8]| {
                let bytes = u16::from_ne_bytes(unsafe { chunk.try_into().unwrap_unchecked() });
                chunk.copy_from_slice(bytes.swap_bytes().to_ne_bytes().as_slice());
            };
            v.chunks_exact_mut(2).for_each(swap);
        }
        DataType::Int32
        | DataType::UInt32
        | DataType::Float32
        | DataType::Complex64
        | DataType::ComplexFloat32 => {
            let swap = |chunk: &mut [u8]| {
                let bytes = u32::from_ne_bytes(unsafe { chunk.try_into().unwrap_unchecked() });
                chunk.copy_from_slice(bytes.swap_bytes().to_ne_bytes().as_slice());
            };
            v.chunks_exact_mut(4).for_each(swap);
        }
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Complex128
        | DataType::ComplexFloat64
        | DataType::NumpyDateTime64 {
            unit: _,
            scale_factor: _,
        }
        | DataType::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _,
        } => {
            let swap = |chunk: &mut [u8]| {
                let bytes = u64::from_ne_bytes(unsafe { chunk.try_into().unwrap_unchecked() });
                chunk.copy_from_slice(bytes.swap_bytes().to_ne_bytes().as_slice());
            };
            v.chunks_exact_mut(8).for_each(swap);
        }
        // Variable-sized data types and extensions are not supported and are rejected outside of this function
        DataType::Extension(_) | DataType::String | DataType::Bytes => {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroU64, sync::Arc};

    use crate::{
        array::{
            codec::{
                ArrayToBytesCodecTraits, BytesPartialDecoderTraits, CodecOptions, CodecTraits,
            },
            ArrayBytes, ChunkRepresentation, ChunkShape, Endianness, FillValue,
        },
        array_subset::ArraySubset,
    };

    use super::*;

    #[test]
    fn codec_bytes_configuration_big() {
        let codec_configuration: BytesCodecConfiguration =
            serde_json::from_str(r#"{"endian":"big"}"#).unwrap();
        let codec = BytesCodec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec.configuration(BYTES).unwrap();
        assert_eq!(
            serde_json::to_string(&configuration).unwrap(),
            r#"{"endian":"big"}"#
        );
    }

    #[test]
    fn codec_bytes_configuration_little() {
        let codec_configuration: BytesCodecConfiguration =
            serde_json::from_str(r#"{"endian":"little"}"#).unwrap();
        let codec = BytesCodec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec.configuration(BYTES).unwrap();
        assert_eq!(
            serde_json::to_string(&configuration).unwrap(),
            r#"{"endian":"little"}"#
        );
    }

    #[test]
    fn codec_bytes_configuration_none() {
        let codec_configuration: BytesCodecConfiguration = serde_json::from_str(r#"{}"#).unwrap();
        let codec = BytesCodec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec.configuration(BYTES).unwrap();
        assert_eq!(serde_json::to_string(&configuration).unwrap(), r#"{}"#);
    }

    fn codec_bytes_round_trip_impl(
        endianness: Option<Endianness>,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let chunk_shape = vec![NonZeroU64::new(10).unwrap(), NonZeroU64::new(10).unwrap()];
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, data_type, fill_value).unwrap();
        let size = chunk_representation.num_elements_usize()
            * chunk_representation.data_type().fixed_size().unwrap();
        let bytes: ArrayBytes = (0..size).map(|s| s as u8).collect::<Vec<_>>().into();

        let codec = BytesCodec::new(endianness);

        let encoded = codec.encode(
            bytes.clone(),
            &chunk_representation,
            &CodecOptions::default(),
        )?;
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded);
        Ok(())
    }

    #[test]
    fn codec_bytes_round_trip_f32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), DataType::Float32, 0.0f32).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), DataType::Float32, 0.0f32).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), DataType::UInt32, 0u32).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), DataType::UInt32, 0u32).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u16() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), DataType::UInt16, 0u16).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), DataType::UInt16, 0u16).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u8() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), DataType::UInt8, 0u8).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), DataType::UInt8, 0u8).unwrap();
        codec_bytes_round_trip_impl(None, DataType::UInt8, 0u8).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_i32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), DataType::Int32, 0).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), DataType::Int32, 0).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_i32_endianness_none() {
        assert!(codec_bytes_round_trip_impl(None, DataType::Int32, 0).is_err());
    }

    #[test]
    fn codec_bytes_round_trip_complex64() {
        codec_bytes_round_trip_impl(
            Some(Endianness::Big),
            DataType::Complex64,
            num::complex::Complex32::new(0.0, 0.0),
        )
        .unwrap();
        codec_bytes_round_trip_impl(
            Some(Endianness::Little),
            DataType::Complex64,
            num::complex::Complex32::new(0.0, 0.0),
        )
        .unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_complex128() {
        codec_bytes_round_trip_impl(
            Some(Endianness::Big),
            DataType::Complex128,
            num::complex::Complex64::new(0.0, 0.0),
        )
        .unwrap();
        codec_bytes_round_trip_impl(
            Some(Endianness::Little),
            DataType::Complex128,
            num::complex::Complex64::new(0.0, 0.0),
        )
        .unwrap();
    }

    #[test]
    fn codec_bytes_partial_decode() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, 0u8).unwrap();
        let elements: Vec<u8> = (0..chunk_representation.num_elements() as u8).collect();
        let bytes: ArrayBytes = elements.into();

        let codec = Arc::new(BytesCodec::new(None));

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..3, 0..1]);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // bytes partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_region, &CodecOptions::default())
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .chunks(size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_bytes_async_partial_decode() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, 0u8).unwrap();
        let elements: Vec<u8> = (0..chunk_representation.num_elements() as u8).collect();
        let bytes: ArrayBytes = elements.into();

        let codec = Arc::new(BytesCodec::new(None));

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..3, 0..1]);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_region, &CodecOptions::default())
            .await
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .chunks(size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }
}

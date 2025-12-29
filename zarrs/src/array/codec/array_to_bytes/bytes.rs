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
//! # use zarrs::metadata_ext::codec::bytes::BytesCodecConfiguration;
//! # serde_json::from_str::<BytesCodecConfiguration>(JSON).unwrap();
//! ```

mod bytes_codec;
mod bytes_codec_partial;

use std::sync::Arc;

pub use bytes_codec::BytesCodec;
pub(crate) use bytes_codec_partial::BytesCodecPartial;
use zarrs_plugin::ExtensionIdentifier;

use crate::metadata::Endianness;
pub use crate::metadata_ext::codec::bytes::{BytesCodecConfiguration, BytesCodecConfigurationV1};
use crate::{
    array::{
        DataType,
        codec::{Codec, CodecPlugin},
        data_type::DataTypeExt,
    },
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(BytesCodec::IDENTIFIER, BytesCodec::matches_name, BytesCodec::default_name, create_codec_bytes)
}

pub(crate) fn create_codec_bytes(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    if metadata.name() == "binary" {
        crate::warn_deprecated_extension("binary", "codec", Some("bytes"));
    }
    let configuration: BytesCodecConfiguration = metadata.to_configuration().map_err(|_| {
        PluginMetadataInvalidError::new(BytesCodec::IDENTIFIER, "codec", metadata.to_string())
    })?;
    let codec = Arc::new(BytesCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

/// Reverse the endianness of bytes for a given data type.
pub(crate) fn reverse_endianness(v: &mut [u8], data_type: &DataType) {
    // Get the fixed size of the data type. Variable-sized types are not supported.
    let Some(size) = data_type.fixed_size() else {
        // Variable-sized data types are rejected outside of this function
        unreachable!()
    };

    match size {
        // Single-byte types don't need endianness reversal
        1 => {}
        2 => {
            for chunk in v.as_chunks_mut::<2>().0 {
                let bytes = u16::from_ne_bytes(*chunk);
                *chunk = bytes.swap_bytes().to_ne_bytes();
            }
        }
        4 => {
            for chunk in v.as_chunks_mut::<4>().0 {
                let bytes = u32::from_ne_bytes(*chunk);
                *chunk = bytes.swap_bytes().to_ne_bytes();
            }
        }
        8 => {
            for chunk in v.as_chunks_mut::<8>().0 {
                let bytes = u64::from_ne_bytes(*chunk);
                *chunk = bytes.swap_bytes().to_ne_bytes();
            }
        }
        // Other sizes: swap bytes pairwise for each element
        _ => {
            for chunk in v.chunks_exact_mut(size) {
                chunk.reverse();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroU64, sync::Arc};

    use super::*;
    use crate::{
        array::{
            ArrayBytes, ChunkShape, ChunkShapeTraits, Endianness, FillValue,
            codec::{
                ArrayToBytesCodecTraits, BytesPartialDecoderTraits, CodecMetadataOptions,
                CodecOptions, CodecTraits,
            },
            data_type::DataTypeExt,
            data_types,
        },
        array_subset::ArraySubset,
    };

    #[test]
    fn codec_bytes_configuration_big() {
        let codec_configuration: BytesCodecConfiguration =
            serde_json::from_str(r#"{"endian":"big"}"#).unwrap();
        let codec = BytesCodec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec
            .configuration(BytesCodec::IDENTIFIER, &CodecMetadataOptions::default())
            .unwrap();
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
        let configuration = codec
            .configuration(BytesCodec::IDENTIFIER, &CodecMetadataOptions::default())
            .unwrap();
        assert_eq!(
            serde_json::to_string(&configuration).unwrap(),
            r#"{"endian":"little"}"#
        );
    }

    #[test]
    fn codec_bytes_configuration_none() {
        let codec_configuration: BytesCodecConfiguration = serde_json::from_str(r"{}").unwrap();
        let codec = BytesCodec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec
            .configuration(BytesCodec::IDENTIFIER, &CodecMetadataOptions::default())
            .unwrap();
        assert_eq!(serde_json::to_string(&configuration).unwrap(), r"{}");
    }

    fn codec_bytes_round_trip_impl(
        endianness: Option<Endianness>,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let chunk_shape = ChunkShape::from(vec![
            NonZeroU64::new(10).unwrap(),
            NonZeroU64::new(10).unwrap(),
        ]);
        let fill_value = fill_value.into();
        let size = chunk_shape.num_elements_u64() as usize * data_type.fixed_size().unwrap();
        let bytes: ArrayBytes = (0..size).map(|s| s as u8).collect::<Vec<_>>().into();

        let codec = BytesCodec::new(endianness);

        let encoded = codec.encode(
            bytes.clone(),
            &chunk_shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        )?;
        let decoded = codec
            .decode(
                encoded,
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(bytes, decoded);
        Ok(())
    }

    #[test]
    fn codec_bytes_round_trip_f32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_types::float32(), 0.0f32).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_types::float32(), 0.0f32)
            .unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_types::uint32(), 0u32).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_types::uint32(), 0u32).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u16() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_types::uint16(), 0u16).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_types::uint16(), 0u16).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u8() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_types::uint8(), 0u8).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_types::uint8(), 0u8).unwrap();
        codec_bytes_round_trip_impl(None, data_types::uint8(), 0u8).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_i32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_types::int32(), 0).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_types::int32(), 0).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_i32_endianness_none() {
        assert!(codec_bytes_round_trip_impl(None, data_types::int32(), 0).is_err());
    }

    #[test]
    fn codec_bytes_round_trip_complex64() {
        codec_bytes_round_trip_impl(
            Some(Endianness::Big),
            data_types::complex64(),
            num::complex::Complex32::new(0.0, 0.0),
        )
        .unwrap();
        codec_bytes_round_trip_impl(
            Some(Endianness::Little),
            data_types::complex64(),
            num::complex::Complex32::new(0.0, 0.0),
        )
        .unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_complex128() {
        codec_bytes_round_trip_impl(
            Some(Endianness::Big),
            data_types::complex128(),
            num::complex::Complex64::new(0.0, 0.0),
        )
        .unwrap();
        codec_bytes_round_trip_impl(
            Some(Endianness::Little),
            data_types::complex128(),
            num::complex::Complex64::new(0.0, 0.0),
        )
        .unwrap();
    }

    #[test]
    fn codec_bytes_partial_decode() {
        let chunk_shape: ChunkShape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_types::uint8();
        let fill_value = FillValue::from(0u8);

        let elements: Vec<u8> = (0..chunk_shape.num_elements_u64() as u8).collect();
        let bytes: ArrayBytes = elements.into();

        let codec = Arc::new(BytesCodec::new(None));

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..3, 0..1]);
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
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // bytes partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_region, &CodecOptions::default())
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .as_chunks::<1>()
            .0
            .iter()
            .map(|b| u8::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_bytes_async_partial_decode() {
        let chunk_shape: ChunkShape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_types::uint8();
        let fill_value = FillValue::from(0u8);
        let elements: Vec<u8> = (0..chunk_shape.num_elements_u64() as u8).collect();
        let bytes: ArrayBytes = elements.into();

        let codec = Arc::new(BytesCodec::new(None));

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..3, 0..1]);
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
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_region, &CodecOptions::default())
            .await
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .as_chunks::<1>()
            .0
            .iter()
            .map(|b| u8::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }
}

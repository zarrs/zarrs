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
//! The `bytes` specification defines a fixed set of supported data types, whereas the `bytes` codec in `zarrs` supports any fixed size data type that implements the [`BytesCodecDataTypeTraits`] trait.
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
use zarrs_metadata::v3::MetadataV3;

use crate::array::DataType;
use crate::array::codec::{Codec, CodecError, CodecPluginV3};
use crate::metadata::Endianness;
pub use crate::metadata_ext::codec::bytes::{BytesCodecConfiguration, BytesCodecConfigurationV1};
use crate::plugin::{PluginConfigurationInvalidError, PluginCreateError};

zarrs_plugin::impl_extension_aliases!(BytesCodec,
    v3: "bytes", ["endian"]
);

// Register the V3 codec (bytes is V3-only).
inventory::submit! {
    CodecPluginV3::new::<BytesCodec>(create_codec_bytes_v3)
}

pub(crate) fn create_codec_bytes_v3(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    if metadata.name() == "binary" {
        crate::warn_deprecated_extension("binary", "codec", Some("bytes"));
    }
    let configuration: BytesCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let codec = Arc::new(BytesCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

use std::borrow::Cow;

/// Traits for a data type supporting the `bytes` codec.
pub trait BytesCodecDataTypeTraits {
    /// Encode the bytes of a fixed-size data type to a specified endianness for the `bytes` codec.
    ///
    /// Returns the input bytes unmodified for fixed-size data where endianness is not applicable (i.e. the bytes are serialised directly from the in-memory representation).
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `endianness` is [`None`] but must be specified or the `bytes` do not have the correct length.
    #[allow(unused_variables)]
    fn encode<'a>(
        &self,
        bytes: Cow<'a, [u8]>,
        endianness: Option<Endianness>,
    ) -> Result<Cow<'a, [u8]>, CodecError>;

    /// Decode the bytes of a fixed-size data type from a specified endianness for the `bytes` codec.
    ///
    /// This performs the inverse operation of [`encode`](BytesCodecDataTypeTraits::encode).
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `endianness` is [`None`] but must be specified or the `bytes` do not have the correct length.
    #[allow(unused_variables)]
    fn decode<'a>(
        &self,
        bytes: Cow<'a, [u8]>,
        endianness: Option<Endianness>,
    ) -> Result<Cow<'a, [u8]>, CodecError>;
}

// Generate the codec support infrastructure using the generic macro
zarrs_codec::define_data_type_support!(Bytes, BytesCodecDataTypeTraits);

/// Macro to implement a passthrough `BytesCodecDataTypeTraits` for data types and register support.
///
/// This is useful for single-byte types and other types where no byte-swapping
/// or transformation is needed during encoding/decoding.
///
/// # Usage
/// ```ignore
/// crate::array::codec::array_to_bytes::bytes::impl_bytes_codec_passthrough!(BoolDataType);
/// crate::array::codec::array_to_bytes::bytes::impl_bytes_codec_passthrough!(UInt4DataType);
/// crate::array::codec::array_to_bytes::bytes::impl_bytes_codec_passthrough!(Float8E4M3DataType);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_bytes_codec_passthrough {
    ($marker:ty) => {
        impl $crate::array::codec::BytesCodecDataTypeTraits for $marker {
            fn encode<'a>(
                &self,
                bytes: ::std::borrow::Cow<'a, [u8]>,
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<::std::borrow::Cow<'a, [u8]>, $crate::array::codec::CodecError> {
                Ok(bytes)
            }

            fn decode<'a>(
                &self,
                bytes: ::std::borrow::Cow<'a, [u8]>,
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<::std::borrow::Cow<'a, [u8]>, $crate::array::codec::CodecError> {
                Ok(bytes)
            }
        }
        zarrs_codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BytesPlugin,
            $crate::array::codec::BytesCodecDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_bytes_codec_passthrough as impl_bytes_codec_passthrough;

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
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::codec::{
        ArrayToBytesCodecTraits, BytesPartialDecoderTraits, CodecMetadataOptions, CodecOptions,
        CodecTraits,
    };
    use crate::array::{
        ArrayBytes, ArraySubset, ChunkShape, ChunkShapeTraits, Endianness, FillValue, data_type,
    };

    #[test]
    fn codec_bytes_configuration_big() {
        let codec_configuration: BytesCodecConfiguration =
            serde_json::from_str(r#"{"endian":"big"}"#).unwrap();
        let codec = BytesCodec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec
            .configuration_v3(&CodecMetadataOptions::default())
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
            .configuration_v3(&CodecMetadataOptions::default())
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
            .configuration_v3(&CodecMetadataOptions::default())
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
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_type::float32(), 0.0f32).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_type::float32(), 0.0f32)
            .unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_type::uint32(), 0u32).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_type::uint32(), 0u32).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u16() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_type::uint16(), 0u16).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_type::uint16(), 0u16).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_u8() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_type::uint8(), 0u8).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_type::uint8(), 0u8).unwrap();
        codec_bytes_round_trip_impl(None, data_type::uint8(), 0u8).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_i32() {
        codec_bytes_round_trip_impl(Some(Endianness::Big), data_type::int32(), 0).unwrap();
        codec_bytes_round_trip_impl(Some(Endianness::Little), data_type::int32(), 0).unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_i32_endianness_none() {
        assert!(codec_bytes_round_trip_impl(None, data_type::int32(), 0).is_err());
    }

    #[test]
    fn codec_bytes_round_trip_complex64() {
        codec_bytes_round_trip_impl(
            Some(Endianness::Big),
            data_type::complex64(),
            num::complex::Complex32::new(0.0, 0.0),
        )
        .unwrap();
        codec_bytes_round_trip_impl(
            Some(Endianness::Little),
            data_type::complex64(),
            num::complex::Complex32::new(0.0, 0.0),
        )
        .unwrap();
    }

    #[test]
    fn codec_bytes_round_trip_complex128() {
        codec_bytes_round_trip_impl(
            Some(Endianness::Big),
            data_type::complex128(),
            num::complex::Complex64::new(0.0, 0.0),
        )
        .unwrap();
        codec_bytes_round_trip_impl(
            Some(Endianness::Little),
            data_type::complex128(),
            num::complex::Complex64::new(0.0, 0.0),
        )
        .unwrap();
    }

    #[test]
    fn codec_bytes_partial_decode() {
        let chunk_shape: ChunkShape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_type::uint8();
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
        let data_type = data_type::uint8();
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

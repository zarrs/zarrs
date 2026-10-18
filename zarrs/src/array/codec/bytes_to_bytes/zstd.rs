//! The `zstd` bytes to bytes codec.
//!
//! Applies [Zstd](https://tools.ietf.org/html/rfc8878) compression.
//!
//! ### Compatible Implementations
//! This is expected to become a standardised extension.
//!
//! Some implementations have compatibility issues due to how the content size is encoded:
//! - <https://github.com/zarr-developers/numcodecs/issues/424>
//! - <https://github.com/google/neuroglancer/issues/625>
//!
//! ### Specification:
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/zstd>
//! - <https://github.com/zarr-developers/zarr-specs/pull/256>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `zstd`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `zstd`
//!
//! ### Codec `configuration` Example - [`ZstdCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "level": 1,
//!     "checksum": true
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zstd::ZstdCodecConfiguration;
//! # serde_json::from_str::<ZstdCodecConfiguration>(JSON).unwrap();

mod zstd_codec;

use std::sync::Arc;

use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;
pub use zstd_codec::ZstdCodec;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::zstd::{
    ZstdCodecConfiguration, ZstdCodecConfigurationNumcodecs, ZstdCodecConfigurationV1,
    ZstdCompressionLevel,
};

zarrs_plugin::impl_extension_aliases!(ZstdCodec, v3: "zstd", v2: "zstd");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<ZstdCodec>()
}

// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<ZstdCodec>()
}

impl CodecTraitsV3 for ZstdCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: ZstdCodecConfigurationV1 = metadata.to_typed_configuration()?;
        let codec = ZstdCodec::new(configuration.level.into(), configuration.checksum);
        Ok(Codec::BytesToBytes(Arc::new(codec)))
    }
}

impl CodecTraitsV2 for ZstdCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, zarrs_codec::CodecCreateError> {
        // Support both the plain `numcodecs` configuration (`{"level": ..}`) and the
        // newer configuration with an additional `checksum` field (e.g. as written by
        // recent `zarr-python`/`numcodecs` releases), see
        // https://github.com/zarr-developers/numcodecs/issues/424
        let configuration: ZstdCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = ZstdCodec::new_with_configuration(&configuration)?;
        Ok(Codec::BytesToBytes(Arc::new(codec)))
    }
}

#[cfg(test)]
mod tests {
    use crate::array::ComputeResources;
    use crate::array::codec::test_util::partial_decode_bytes;
    use std::sync::Arc;
    use zarrs_codec::CowBytes;

    use super::*;
    use crate::array::BytesRepresentation;
    use zarrs_codec::{BytesToBytesCodecTraits, CodecError, CodecOptions};
    use zarrs_storage::byte_range::ByteRange;

    const JSON_VALID: &str = r#"{
    "level": 22,
    "checksum": false
}"#;

    #[test]
    fn codec_zstd_v2_numcodecs_level_only() {
        // Legacy `numcodecs` configuration: no `checksum` field.
        let metadata: MetadataV2 = serde_json::from_str(r#"{"id": "zstd", "level": 6}"#).unwrap();
        let codec = <ZstdCodec as CodecTraitsV2>::create(&metadata).unwrap();
        assert!(matches!(codec, Codec::BytesToBytes(_)));
    }

    #[test]
    fn codec_zstd_v2_with_checksum() {
        // Configuration as written by recent `numcodecs`/`zarr-python` releases, which
        // includes an additional `checksum` field (e.g. the janelia-cosem-datasets bucket
        // on S3). This previously failed with a "unknown field `checksum`" error because
        // `CodecTraitsV2::create` only accepted `ZstdCodecConfigurationNumcodecs`.
        let metadata: MetadataV2 =
            serde_json::from_str(r#"{"id": "zstd", "checksum": false, "level": 6}"#).unwrap();
        let codec = <ZstdCodec as CodecTraitsV2>::create(&metadata).unwrap();
        assert!(matches!(codec, Codec::BytesToBytes(_)));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zstd_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ZstdCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = ZstdCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(
                CowBytes::Borrowed(&bytes),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &bytes_representation,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        assert_eq!(bytes, decoded.to_vec());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zstd_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ZstdCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(ZstdCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(
                CowBytes::from(bytes),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let decoded_partial_chunk = partial_decode_bytes(
            codec.clone(),
            &encoded,
            bytes_representation.size().unwrap(),
            &decoded_regions,
        )
        .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn decodes_reuse_the_decompression_context() {
        let codec = ZstdCodec::new(1, false);
        let options = CodecOptions::default();
        let encode = |decoded: &[u8]| -> Vec<u8> {
            codec
                .encode(
                    CowBytes::from(decoded),
                    &options,
                    &ComputeResources::default(),
                )
                .unwrap()
                .to_vec()
        };
        let decode = |encoded: Vec<u8>, len: usize| -> Result<Vec<u8>, CodecError> {
            codec
                .decode(
                    CowBytes::from(encoded),
                    &BytesRepresentation::FixedSize(len as u64),
                    &options,
                    &ComputeResources::default(),
                )
                .map(|decoded| decoded.to_vec())
        };
        let first = b"first".repeat(100);
        let second = b"the second frame".repeat(50);
        let (first_encoded, second_encoded) = (encode(&first), encode(&second));

        // A failed decode does not affect later decodes on the thread
        let mut corrupt = first_encoded.clone();
        let middle = corrupt.len() / 2;
        corrupt[middle..].fill(0xff);
        assert!(decode(corrupt, first.len()).is_err());
        for _ in 0..2 {
            assert_eq!(decode(first_encoded.clone(), first.len()).unwrap(), first);
            assert_eq!(
                decode(second_encoded.clone(), second.len()).unwrap(),
                second
            );
        }

        // A decode while the context of the thread is in use uses another context
        let nested = zstd_codec::with_decompressor(|_| {
            Ok(decode(second_encoded.clone(), second.len()).unwrap())
        })
        .unwrap();
        assert_eq!(nested, second);
    }

    #[test]
    fn decode_into() {
        let codec = ZstdCodec::new(1, false);
        crate::array::codec::bytes_to_bytes::check_bytes_decode_into(
            &codec,
            &b"decode directly into this buffer".repeat(32),
            1,
            |encoded| {
                // A truncated stream, and bytes that are not a zstd stream
                vec![encoded[..encoded.len() / 2].to_vec(), vec![0u8; 64]]
            },
        );
    }

    #[test]
    fn decode_into_unknown_content_size() {
        let codec = ZstdCodec::new(1, false);
        let options = CodecOptions::default();
        let decoded = b"streamed frame without content size".repeat(32);
        let len = decoded.len();
        // The streaming encoder does not write the content size into the frame header
        let encoded = zstd::stream::encode_all(decoded.as_slice(), 1).unwrap();
        assert!(matches!(
            zstd::zstd_safe::get_frame_content_size(&encoded),
            Ok(None)
        ));
        let decode_into = |output: &mut [u8]| {
            codec.decode_into(
                CowBytes::from(encoded.as_slice()),
                &BytesRepresentation::UnboundedSize,
                output,
                &options,
                &ComputeResources::default(),
            )
        };

        let mut output = vec![0; len];
        decode_into(&mut output).unwrap();
        assert_eq!(output, decoded);

        // A too-small output returns the bounded decoder's error without a second decode.
        assert!(matches!(
            decode_into(&mut output[..len - 1]),
            Err(CodecError::IOError(_))
        ));
        // Successful decompression reports its actual length for a too-large output.
        assert!(matches!(
            decode_into(&mut vec![0; len + 1]),
            Err(CodecError::UnexpectedChunkDecodedSize(_))
        ));
    }

    #[test]
    fn decode_into_oversized_frame_does_not_materialise_output() {
        let codec = ZstdCodec::new(1, false);
        let options = CodecOptions::default();
        let decoded = vec![0u8; 8 * 1024 * 1024];
        let known_size = codec
            .encode(
                CowBytes::from(decoded.as_slice()),
                &options,
                &ComputeResources::default(),
            )
            .unwrap();
        let unknown_size = zstd::stream::encode_all(decoded.as_slice(), 1).unwrap();
        for (encoded, size_known) in [(known_size, true), (CowBytes::from(unknown_size), false)] {
            let mut output = [0u8; 8];
            // The old allocating retry returned UnexpectedChunkDecodedSize only after expanding the entire 8 MiB
            // frame. A size in the frame header is compared without decompressing, otherwise decompressing fails
            // once the output is full.
            let result = codec.decode_into(
                encoded,
                &BytesRepresentation::FixedSize(8),
                &mut output,
                &options,
                &ComputeResources::default(),
            );
            if size_known {
                assert!(matches!(
                    result,
                    Err(CodecError::UnexpectedChunkDecodedSize(_))
                ));
            } else {
                assert!(matches!(result, Err(CodecError::IOError(_))));
            }
        }
    }
}

//! The `gzip` bytes to bytes codec (Core).
//!
//! Applies [gzip](https://datatracker.ietf.org/doc/html/rfc1952) compression.
//!
//! ### Compatible Implementations
//! This is a core codec and should be compatible with all Zarr V3 implementations that support it.
//!
//! ### Specification
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html>
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/gzip>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `gzip`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `gzip`
//!
//! ### Codec `configuration` Example - [`GzipCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "level": 1
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::gzip::GzipCodecConfiguration;
//! # serde_json::from_str::<GzipCodecConfiguration>(JSON).unwrap();

mod gzip_codec;

use std::sync::Arc;

pub use gzip_codec::GzipCodec;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::gzip::{
    GzipCodecConfiguration, GzipCodecConfigurationV1, GzipCompressionLevel,
    GzipCompressionLevelError,
};

zarrs_plugin::impl_extension_aliases!(GzipCodec, v3: "gzip", v2: "gzip");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<GzipCodec>()
}

// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<GzipCodec>()
}

impl CodecTraitsV3 for GzipCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: GzipCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(GzipCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

impl CodecTraitsV2 for GzipCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: GzipCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(GzipCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
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
    use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions};
    use zarrs_storage::byte_range::ByteRange;

    const JSON_VALID: &str = r#"{
        "level": 1
    }"#;

    #[test]
    fn codec_gzip_configuration_valid() {
        assert!(serde_json::from_str::<GzipCodecConfiguration>(JSON_VALID).is_ok());
    }

    #[test]
    fn codec_gzip_configuration_invalid1() {
        const JSON_INVALID1: &str = r#"{
        "level": -1
    }"#;
        assert!(serde_json::from_str::<GzipCodecConfiguration>(JSON_INVALID1).is_err());
    }

    #[test]
    fn codec_gzip_configuration_invalid2() {
        const JSON_INVALID2: &str = r#"{
        "level": 10
    }"#;
        assert!(serde_json::from_str::<GzipCodecConfiguration>(JSON_INVALID2).is_err());
    }

    #[test]
    fn codec_gzip_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: GzipCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = GzipCodec::new_with_configuration(&configuration).unwrap();

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
    fn codec_gzip_decode_into() {
        let codec = GzipCodec::new(1).unwrap();
        crate::array::codec::bytes_to_bytes::check_bytes_decode_into(
            &codec,
            &b"decode directly into this buffer".repeat(32),
            1,
            |encoded| {
                // A truncated stream, and bytes that are not a gzip stream
                vec![encoded[..encoded.len() / 2].to_vec(), vec![0u8; 64]]
            },
        );
    }

    #[test]
    fn codec_gzip_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: GzipCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(GzipCodec::new_with_configuration(&configuration).unwrap());

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
}

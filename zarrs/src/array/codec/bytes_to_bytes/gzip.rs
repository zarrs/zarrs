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
//! # use zarrs_metadata_ext::codec::gzip::GzipCodecConfiguration;
//! # serde_json::from_str::<GzipCodecConfiguration>(JSON).unwrap();

mod gzip_codec;

use std::sync::Arc;

pub use gzip_codec::GzipCodec;
pub use zarrs_metadata_ext::codec::gzip::{
    GzipCodecConfiguration, GzipCodecConfigurationV1, GzipCompressionLevel,
    GzipCompressionLevelError,
};
use zarrs_registry::codec::GZIP;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(GZIP, is_identifier_gzip, create_codec_gzip)
}

fn is_identifier_gzip(identifier: &str) -> bool {
    identifier == GZIP
}

pub(crate) fn create_codec_gzip(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: GzipCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(GZIP, "codec", metadata.to_string()))?;
    let codec = Arc::new(GzipCodec::new_with_configuration(&configuration)?);
    Ok(Codec::BytesToBytes(codec))
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use crate::{
        array::{
            codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions},
            BytesRepresentation,
        },
        storage::byte_range::ByteRange,
    };

    use super::*;

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
            .encode(Cow::Borrowed(&bytes), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &bytes_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded.to_vec());
    }

    #[test]
    fn codec_gzip_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: GzipCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(GzipCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &bytes_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size(), input_handle.size()); // gzip partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode_concat(&decoded_regions, &CodecOptions::default())
            .unwrap()
            .unwrap();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .to_vec()
            .chunks_exact(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_gzip_async_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: GzipCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(GzipCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &bytes_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode_concat(&decoded_regions, &CodecOptions::default())
            .await
            .unwrap()
            .unwrap();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .to_vec()
            .chunks_exact(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }
}

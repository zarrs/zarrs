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

pub use zstd_codec::ZstdCodec;

pub use crate::metadata_ext::codec::zstd::{
    ZstdCodecConfiguration, ZstdCodecConfigurationV1, ZstdCompressionLevel,
};
use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(ZstdCodec::IDENTIFIER, ZstdCodec::matches_name, ZstdCodec::default_name, create_codec_zstd)
}

pub(crate) fn create_codec_zstd(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ZstdCodecConfiguration = metadata.to_configuration().map_err(|_| {
        PluginMetadataInvalidError::new(ZstdCodec::IDENTIFIER, "codec", metadata.to_string())
    })?;
    let codec = Arc::new(ZstdCodec::new_with_configuration(&configuration)?);
    Ok(Codec::BytesToBytes(codec))
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use super::*;
    use crate::{
        array::{
            BytesRepresentation,
            codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions},
        },
        storage::byte_range::ByteRange,
    };

    const JSON_VALID: &str = r#"{
    "level": 22,
    "checksum": false
}"#;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zstd_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ZstdCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = ZstdCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(Cow::Borrowed(&bytes), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &bytes_representation, &CodecOptions::default())
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
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // zstd partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .clone()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn codec_zstd_async_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ZstdCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(ZstdCodec::new_with_configuration(&configuration).unwrap());

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
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
            .await
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .clone()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }
}

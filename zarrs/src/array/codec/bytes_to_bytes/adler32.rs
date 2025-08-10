//! The `adler32` bytes to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! Appends an adler32 checksum of the input bytestream.
//!
//! This codec requires the `adler32` feature, which is disabled by default.
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.adler32` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.adler32>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.adler32`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `adler32`
//!
//! ### Codec `configuration` Example - [`Adler32CodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {}
//! # "#;
//! # use zarrs_metadata_ext::codec::adler32::Adler32CodecConfiguration;
//! # serde_json::from_str::<Adler32CodecConfiguration>(JSON).unwrap();
//! ```

mod adler32_codec;

use std::sync::Arc;

pub use adler32_codec::Adler32Codec;
pub use zarrs_metadata_ext::codec::adler32::{
    Adler32CodecConfiguration, Adler32CodecConfigurationV1,
};
use zarrs_registry::codec::ADLER32;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(ADLER32, is_identifier_adler32, create_codec_adler32)
}

fn is_identifier_adler32(identifier: &str) -> bool {
    identifier == ADLER32
}

pub(crate) fn create_codec_adler32(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(ADLER32, "codec", metadata.to_string()))?;
    let codec = Arc::new(Adler32Codec::new_with_configuration(&configuration)?);
    Ok(Codec::BytesToBytes(codec))
}

const CHECKSUM_SIZE: usize = size_of::<u32>();

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use crate::{
        array::{
            codec::{
                BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions, CodecTraits,
            },
            BytesRepresentation,
        },
        storage::byte_range::ByteRange,
    };

    use super::*;

    const JSON1: &str = r#"{}"#;
    const JSON2: &str = r#"{"location":"start"}"#;
    const JSON3: &str = r#"{"location":"end"}"#;

    #[test]
    fn codec_adler32_configuration_none() {
        let codec_configuration: Adler32CodecConfiguration = serde_json::from_str(r#"{}"#).unwrap();
        let codec = Adler32Codec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec.configuration("numcodecs.adler32").unwrap();
        assert_eq!(serde_json::to_string(&configuration).unwrap(), r#"{}"#);
    }

    #[test]
    fn codec_adler32() {
        let elements: Vec<u8> = (0..6).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        for json in [JSON1, JSON2, JSON3] {
            let codec_configuration: Adler32CodecConfiguration =
                serde_json::from_str(json).unwrap();
            let codec = Adler32Codec::new_with_configuration(&codec_configuration).unwrap();

            let encoded = codec
                .encode(Cow::Borrowed(&bytes), &CodecOptions::default())
                .unwrap();
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &bytes_representation,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded.to_vec());

            // Check that the checksum is correct
            let checksum: &[u8; 4] = if json.contains("end") {
                &encoded[encoded.len() - size_of::<u32>()..]
                    .try_into()
                    .unwrap()
            } else {
                &encoded[..size_of::<u32>()].try_into().unwrap()
            };
            println!("checksum {checksum:?}");
            assert_eq!(checksum, &[16, 0, 41, 0]);
        }
    }

    #[test]
    fn codec_adler32_partial_decode() {
        let elements: Vec<u8> = (0..32).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        for json in [JSON1, JSON2, JSON3] {
            let codec_configuration: Adler32CodecConfiguration =
                serde_json::from_str(json).unwrap();
            let codec =
                Arc::new(Adler32Codec::new_with_configuration(&codec_configuration).unwrap());

            let encoded = codec
                .encode(Cow::Owned(bytes.clone()), &CodecOptions::default())
                .unwrap();
            let decoded_regions = [ByteRange::FromStart(3, Some(2))];
            let input_handle = Arc::new(encoded);
            let partial_decoder = codec
                .partial_decoder(
                    input_handle.clone(),
                    &bytes_representation,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(partial_decoder.size(), input_handle.size()); // adler32 partial decoder does not hold bytes
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&mut decoded_regions.into_iter(), &CodecOptions::default())
                .unwrap()
                .unwrap();
            let answer: &[Vec<u8>] = &[vec![3, 4]];
            assert_eq!(
                answer,
                decoded_partial_chunk
                    .into_iter()
                    .map(|v| v.to_vec())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_adler32_async_partial_decode() {
        let elements: Vec<u8> = (0..32).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        for json in [JSON1, JSON2, JSON3] {
            let codec_configuration: Adler32CodecConfiguration =
                serde_json::from_str(json).unwrap();
            let codec =
                Arc::new(Adler32Codec::new_with_configuration(&codec_configuration).unwrap());

            let encoded = codec
                .encode(Cow::Owned(bytes.clone()), &CodecOptions::default())
                .unwrap();
            let decoded_regions = [ByteRange::FromStart(3, Some(2))];
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
                .partial_decode(&mut decoded_regions.into_iter(), &CodecOptions::default())
                .await
                .unwrap()
                .unwrap();
            let answer: &[Vec<u8>] = &[vec![3, 4]];
            assert_eq!(
                answer,
                decoded_partial_chunk
                    .into_iter()
                    .map(|v| v.to_vec())
                    .collect::<Vec<_>>()
            );
        }
    }
}

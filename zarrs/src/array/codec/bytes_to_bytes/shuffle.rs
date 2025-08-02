//! The `shuffle` bytes to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! Shuffles bytes.
//!
//! The byte length of the input to this codec must be an integer multiple of the shuffle `elementsize`.
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.shuffle` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.shuffle>
//! - <https://codec.zarrs.dev/bytes_to_bytes/shuffle>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.shuffle`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `shuffle`
//!
//! ### Codec `configuration` Example - [`ShuffleCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!   "elementsize": 2
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::shuffle::ShuffleCodecConfiguration;
//! # serde_json::from_str::<ShuffleCodecConfiguration>(JSON).unwrap();
//! ```

// FIXME: Could use a real partial decoder

mod shuffle_codec;

use std::sync::Arc;

pub use shuffle_codec::ShuffleCodec;
pub use zarrs_metadata_ext::codec::shuffle::{
    ShuffleCodecConfiguration, ShuffleCodecConfigurationV1,
};
use zarrs_registry::codec::SHUFFLE;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(SHUFFLE, is_identifier_shuffle, create_codec_shuffle)
}

fn is_identifier_shuffle(identifier: &str) -> bool {
    identifier == SHUFFLE
}

pub(crate) fn create_codec_shuffle(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(SHUFFLE, "codec", metadata.to_string()))?;
    let codec = Arc::new(ShuffleCodec::new_with_configuration(&configuration)?);
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
        byte_range::ByteRange,
    };

    use super::*;

    const JSON_VALID: &str = r#"{"elementsize":2}"#;

    #[test]
    fn codec_shuffle() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ShuffleCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = ShuffleCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(Cow::Borrowed(&bytes), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &bytes_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded.to_vec());
    }
    #[test]
    fn codec_shuffle_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ShuffleCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(ShuffleCodec::new_with_configuration(&configuration).unwrap());

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
        assert_eq!(partial_decoder.size(), input_handle.size()); // shuffle partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode_concat(&mut decoded_regions.into_iter(), &CodecOptions::default())
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
    async fn codec_shuffle_async_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ShuffleCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(ShuffleCodec::new_with_configuration(&configuration).unwrap());

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
            .partial_decode_concat(&mut decoded_regions.into_iter(), &CodecOptions::default())
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

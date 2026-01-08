//! The `crc32c` bytes to bytes codec (Core).
//!
//! Appends a CRC32C checksum of the input bytestream.
//!
//! ### Compatible Implementations
//! This is a core codec and should be compatible with all Zarr V3 implementations that support it.
//!
//! ### Specification
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html>
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/crc32c>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `crc32c`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `crc32c`
//!
//! ### Codec `configuration` Example - [`Crc32cCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {}
//! # "#;
//! # use zarrs::metadata_ext::codec::crc32c::Crc32cCodecConfiguration;
//! # serde_json::from_str::<Crc32cCodecConfiguration>(JSON).unwrap();
//! ```

mod crc32c_codec;

use std::sync::Arc;

pub use crc32c_codec::Crc32cCodec;
use zarrs_plugin::ExtensionIdentifier;

use crate::array::codec::{Codec, CodecPlugin};
use crate::metadata::v3::MetadataV3;
pub use crate::metadata_ext::codec::crc32c::{
    Crc32cCodecConfiguration, Crc32cCodecConfigurationV1,
};
use crate::plugin::{PluginCreateError, PluginMetadataInvalidError};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(Crc32cCodec::IDENTIFIER, Crc32cCodec::matches_name, Crc32cCodec::default_name, create_codec_crc32c)
}
zarrs_plugin::impl_extension_aliases!(Crc32cCodec, "crc32c");

pub(crate) fn create_codec_crc32c(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration = metadata.to_configuration().map_err(|_| {
        PluginMetadataInvalidError::new(Crc32cCodec::IDENTIFIER, "codec", metadata.to_string())
    })?;
    let codec = Arc::new(Crc32cCodec::new_with_configuration(&configuration));
    Ok(Codec::BytesToBytes(codec))
}

const CHECKSUM_SIZE: usize = size_of::<u32>();

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use super::*;
    use crate::array::BytesRepresentation;
    use crate::array::codec::{
        BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecMetadataOptions, CodecOptions,
        CodecTraits,
    };
    use crate::storage::byte_range::ByteRange;

    const JSON1: &str = r"{}";

    #[test]
    fn codec_crc32c_configuration_none() {
        let codec_configuration: Crc32cCodecConfiguration = serde_json::from_str(r"{}").unwrap();
        let codec = Crc32cCodec::new_with_configuration(&codec_configuration);
        let metadata = codec
            .configuration("crc32c", &CodecMetadataOptions::default())
            .unwrap();
        assert_eq!(serde_json::to_string(&metadata).unwrap(), r"{}");
    }

    #[test]
    fn codec_crc32c() {
        let elements: Vec<u8> = (0..6).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Crc32cCodecConfiguration = serde_json::from_str(JSON1).unwrap();
        let codec = Crc32cCodec::new_with_configuration(&codec_configuration);

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
        let checksum: &[u8; 4] = &encoded[encoded.len() - size_of::<u32>()..encoded.len()]
            .try_into()
            .unwrap();
        println!("checksum {checksum:?}");
        assert_eq!(checksum, &[20, 133, 9, 65]);
    }

    #[test]
    fn codec_crc32c_partial_decode() {
        let elements: Vec<u8> = (0..32).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Crc32cCodecConfiguration = serde_json::from_str(JSON1).unwrap();
        let codec = Arc::new(Crc32cCodec::new_with_configuration(&codec_configuration));

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
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
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // crc32c partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
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

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_crc32c_async_partial_decode() {
        let elements: Vec<u8> = (0..32).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Crc32cCodecConfiguration = serde_json::from_str(JSON1).unwrap();
        let codec = Arc::new(Crc32cCodec::new_with_configuration(&codec_configuration));

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
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
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
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

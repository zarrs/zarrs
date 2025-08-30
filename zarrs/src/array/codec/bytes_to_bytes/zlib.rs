//! The `zlib` bytes to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! This codec requires the `zlib` feature, which is disabled by default.
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.zlib` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.zlib>
//! - <https://codec.zarrs.dev/bytes_to_bytes/zlib>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.zlib`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `zlib`
//!
//! ### Codec `configuration` Example - [`ZlibCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "level": 9
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::zlib::ZlibCodecConfiguration;
//! # serde_json::from_str::<ZlibCodecConfiguration>(JSON).unwrap();
//! ```

mod zlib_codec;

use std::sync::Arc;

use zarrs_registry::codec::ZLIB;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

pub use zarrs_metadata_ext::codec::zlib::{
    ZlibCodecConfiguration, ZlibCodecConfigurationV1, ZlibCompressionLevel,
};

pub use self::zlib_codec::ZlibCodec;

// Register the codec.
inventory::submit! {
    CodecPlugin::new(ZLIB, is_identifier_zlib, create_codec_zlib)
}

fn is_identifier_zlib(identifier: &str) -> bool {
    identifier == ZLIB
}

pub(crate) fn create_codec_zlib(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ZlibCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(ZLIB, "codec", metadata.to_string()))?;
    let codec = Arc::new(ZlibCodec::new_with_configuration(&configuration)?);
    Ok(Codec::BytesToBytes(codec))
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use crate::{
        array::{
            codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions},
            ArrayRepresentation, BytesRepresentation, DataType,
        },
        array_subset::ArraySubset,
        indexer::Indexer,
    };

    use super::*;

    const JSON_VALID1: &str = r#"
{
    "level": 5
}"#;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_zlib_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: ZlibCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = ZlibCodec::new_with_configuration(&codec_configuration).unwrap();

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
    fn codec_zlib_partial_decode() {
        let array_representation =
            ArrayRepresentation::new(vec![2, 2, 2], DataType::UInt16, 0u16).unwrap();
        let data_type_size = array_representation.data_type().fixed_size().unwrap();
        let array_size = array_representation.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..array_representation.num_elements() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: ZlibCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(ZlibCodec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
            .byte_ranges(array_representation.shape(), data_type_size)
            .unwrap();
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &bytes_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size(), input_handle.size()); // zlib partial decoder does not hold bytes
        let decoded = partial_decoder
            .partial_decode_concat(Box::new(decoded_regions), &CodecOptions::default())
            .unwrap()
            .unwrap();

        let decoded: Vec<u16> = decoded
            .to_vec()
            .chunks_exact(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn codec_zlib_async_partial_decode() {
        let array_representation =
            ArrayRepresentation::new(vec![2, 2, 2], DataType::UInt16, 0u16).unwrap();
        let data_type_size = array_representation.data_type().fixed_size().unwrap();
        let array_size = array_representation.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..array_representation.num_elements() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: ZlibCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(ZlibCodec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
            .byte_ranges(array_representation.shape(), data_type_size)
            .unwrap();
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &bytes_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded = partial_decoder
            .partial_decode_concat(Box::new(decoded_regions), &CodecOptions::default())
            .await
            .unwrap()
            .unwrap();

        let decoded: Vec<u16> = decoded
            .to_vec()
            .chunks_exact(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }
}

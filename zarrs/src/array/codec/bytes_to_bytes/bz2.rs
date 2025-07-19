//! The `bz2` (bzip2) bytes to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! This codec requires the `bz2` feature, which is disabled by default.
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.bz2` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.bz2>
//! - <https://codec.zarrs.dev/bytes_to_bytes/bz2>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.bz2`
//! - `https://codec.zarrs.dev/bytes_to_bytes/bz2`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `bz2`
//!
//! ### Codec `configuration` Example - [`Bz2CodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "level": 9
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::bz2::Bz2CodecConfiguration;
//! # serde_json::from_str::<Bz2CodecConfiguration>(JSON).unwrap();
//! ```

mod bz2_codec;

use std::sync::Arc;

use zarrs_registry::codec::BZ2;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

pub use zarrs_metadata_ext::codec::bz2::{
    Bz2CodecConfiguration, Bz2CodecConfigurationV1, Bz2CompressionLevel,
};

pub use self::bz2_codec::Bz2Codec;

// Register the codec.
inventory::submit! {
    CodecPlugin::new(BZ2, is_identifier_bz2, create_codec_bz2)
}

fn is_identifier_bz2(identifier: &str) -> bool {
    identifier == BZ2
}

pub(crate) fn create_codec_bz2(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: Bz2CodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(BZ2, "codec", metadata.to_string()))?;
    let codec = Arc::new(Bz2Codec::new_with_configuration(&configuration)?);
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
        byte_range::ByteRange,
    };

    use super::*;

    const JSON_VALID1: &str = r#"
{
    "level": 5
}"#;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_bz2_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Bz2Codec::new_with_configuration(&codec_configuration).unwrap();

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
    fn codec_bz2_partial_decode() {
        let array_representation =
            ArrayRepresentation::new(vec![2, 2, 2], DataType::UInt16, 0u16).unwrap();
        let data_type_size = array_representation.data_type().fixed_size().unwrap();
        let array_size = array_representation.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..array_representation.num_elements() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(Bz2Codec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions: Vec<ByteRange> = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
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
        assert_eq!(partial_decoder.size(), input_handle.size()); // bz2 partial decoder does not hold bytes
        let decoded = partial_decoder
            .partial_decode_concat(&decoded_regions, &CodecOptions::default())
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
    async fn codec_bz2_async_partial_decode() {
        let array_representation =
            ArrayRepresentation::new(vec![2, 2, 2], DataType::UInt16, 0u16).unwrap();
        let data_type_size = array_representation.data_type().fixed_size().unwrap();
        let array_size = array_representation.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..array_representation.num_elements() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(Bz2Codec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions: Vec<ByteRange> = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
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
            .partial_decode_concat(&decoded_regions, &CodecOptions::default())
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

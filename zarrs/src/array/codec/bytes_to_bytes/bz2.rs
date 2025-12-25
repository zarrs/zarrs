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
//! # use zarrs::metadata_ext::codec::bz2::Bz2CodecConfiguration;
//! # serde_json::from_str::<Bz2CodecConfiguration>(JSON).unwrap();
//! ```

mod bz2_codec;

use std::sync::Arc;

pub use self::bz2_codec::Bz2Codec;
pub use crate::metadata_ext::codec::bz2::{
    Bz2CodecConfiguration, Bz2CodecConfigurationV1, Bz2CompressionLevel,
};
use crate::registry::codec::BZ2;
use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

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
    use std::{borrow::Cow, num::NonZeroU64, sync::Arc};

    use super::*;
    use crate::storage::byte_range::ByteRange;
    use crate::{
        array::{
            BytesRepresentation, ChunkShapeTraits, DataType,
            codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions},
        },
        array_subset::ArraySubset,
        indexer::Indexer,
    };

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
        let shape = vec![NonZeroU64::new(2).unwrap(); 3];
        let data_type = DataType::UInt16;
        let data_type_size = data_type.fixed_size().unwrap();
        let array_size = shape.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(Bz2Codec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
            .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(&shape), data_type_size)
            .unwrap()
            .map(ByteRange::new);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &bytes_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // bz2 partial decoder does not hold bytes
        let decoded = partial_decoder
            .partial_decode_many(Box::new(decoded_regions), &CodecOptions::default())
            .unwrap()
            .unwrap()
            .concat();

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
        use crate::indexer::Indexer;

        let shape = vec![NonZeroU64::new(2).unwrap(); 3];
        let data_type = DataType::UInt16;
        let data_type_size = data_type.fixed_size().unwrap();
        let array_size = shape.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(Bz2Codec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
            .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(&shape), data_type_size)
            .unwrap()
            .map(ByteRange::new);
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
            .partial_decode_many(Box::new(decoded_regions), &CodecOptions::default())
            .await
            .unwrap()
            .unwrap()
            .concat();

        let decoded: Vec<u16> = decoded
            .to_vec()
            .chunks_exact(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }
}

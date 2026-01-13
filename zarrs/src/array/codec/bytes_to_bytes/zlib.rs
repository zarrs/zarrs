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
//! # use zarrs::metadata_ext::codec::zlib::ZlibCodecConfiguration;
//! # serde_json::from_str::<ZlibCodecConfiguration>(JSON).unwrap();
//! ```

mod zlib_codec;

use std::sync::Arc;

pub use self::zlib_codec::ZlibCodec;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3};
pub use zarrs_metadata_ext::codec::zlib::{
    ZlibCodecConfiguration, ZlibCodecConfigurationV1, ZlibCompressionLevel,
};
use zarrs_plugin::{PluginConfigurationInvalidError, PluginCreateError};

zarrs_plugin::impl_extension_aliases!(ZlibCodec,
    v3: "numcodecs.zlib", [],
    v2: "zlib", []
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<ZlibCodec>(create_codec_zlib_v3)
}

// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<ZlibCodec>(create_codec_zlib_v2)
}

pub(crate) fn create_codec_zlib_v3(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ZlibCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let codec = Arc::new(ZlibCodec::new_with_configuration(&configuration)?);
    Ok(Codec::BytesToBytes(codec))
}

pub(crate) fn create_codec_zlib_v2(metadata: &MetadataV2) -> Result<Codec, PluginCreateError> {
    let configuration: ZlibCodecConfiguration =
        serde_json::from_value(serde_json::to_value(metadata.configuration()).unwrap())
            .map_err(|_| PluginConfigurationInvalidError::new(format!("{metadata:?}")))?;
    let codec = Arc::new(ZlibCodec::new_with_configuration(&configuration)?);
    Ok(Codec::BytesToBytes(codec))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::{ArraySubset, BytesRepresentation, ChunkShapeTraits, Indexer, data_type};
    use zarrs_codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions};
    use zarrs_storage::byte_range::ByteRange;

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
        let shape = vec![NonZeroU64::new(2).unwrap(); 3];
        let data_type = data_type::uint16();
        let data_type_size = data_type.fixed_size().unwrap();
        let array_size = shape.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: ZlibCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(ZlibCodec::new_with_configuration(&codec_configuration).unwrap());

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
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // zlib partial decoder does not hold bytes
        let decoded = partial_decoder
            .partial_decode_many(Box::new(decoded_regions), &CodecOptions::default())
            .unwrap()
            .unwrap()
            .concat();

        let decoded: Vec<u16> = decoded
            .clone()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn codec_zlib_async_partial_decode() {
        let shape = vec![NonZeroU64::new(2).unwrap(); 3];
        let data_type = data_type::uint16();
        let data_type_size = data_type.fixed_size().unwrap();
        let array_size = shape.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: ZlibCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(ZlibCodec::new_with_configuration(&codec_configuration).unwrap());

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
            .clone()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }
}

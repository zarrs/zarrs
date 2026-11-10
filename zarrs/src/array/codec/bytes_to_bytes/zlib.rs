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

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::zlib::{
    ZlibCodecConfiguration, ZlibCodecConfigurationV1, ZlibCompressionLevel,
};

zarrs_plugin::impl_extension_aliases!(ZlibCodec,
    v3: "numcodecs.zlib", [],
    v2: "zlib", []
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<ZlibCodec>()
}

// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<ZlibCodec>()
}

impl CodecTraitsV3 for ZlibCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: ZlibCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(ZlibCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

impl CodecTraitsV2 for ZlibCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: ZlibCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(ZlibCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use crate::array::Resources;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use zarrs_codec::{ComputeResources, CowBytes};

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
    #[cfg_attr(miri, ignore)]
    fn codec_zlib_optional_and_default_level() {
        use zarrs_codec::CodecMetadataOptions;
        use zarrs_plugin::ZarrVersion;

        let bytes: Vec<u8> = (0..64).collect();
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);
        let check = |codec: Codec, version: ZarrVersion, expected_level: &str| {
            let Codec::BytesToBytes(codec) = codec else {
                panic!("expected a bytes to bytes codec")
            };
            let configuration = codec
                .configuration(version, &CodecMetadataOptions::default())
                .unwrap();
            assert_eq!(
                serde_json::to_string(&configuration).unwrap(),
                format!(r#"{{"level":{expected_level}}}"#)
            );
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
        };

        for (json, expected_level) in [
            (r#"{"name":"numcodecs.zlib","configuration":{}}"#, "1"),
            (r#"{"name":"numcodecs.zlib"}"#, "1"),
            (r#""numcodecs.zlib""#, "1"),
            (
                r#"{"name":"numcodecs.zlib","configuration":{"level":-1}}"#,
                "-1",
            ),
        ] {
            let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
            let codec = <ZlibCodec as CodecTraitsV3>::create(&metadata).unwrap();
            check(codec, ZarrVersion::V3, expected_level);
        }
        for (json, expected_level) in [
            (r#"{"id":"zlib"}"#, "1"),
            (r#"{"id":"zlib","level":-1}"#, "-1"),
        ] {
            let metadata: MetadataV2 = serde_json::from_str(json).unwrap();
            let codec = <ZlibCodec as CodecTraitsV2>::create(&metadata).unwrap();
            check(codec, ZarrVersion::V2, expected_level);
        }
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
            .encode(
                CowBytes::from(bytes),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
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
                &Resources::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // zlib partial decoder does not hold bytes
        let decoded = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions),
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap()
            .unwrap()
            .concat();

        let decoded: Vec<u16> = decoded
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
            .encode(
                CowBytes::from(bytes),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
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
                &Resources::default(),
            )
            .await
            .unwrap();
        let decoded = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions),
                &CodecOptions::default(),
                &Resources::default(),
            )
            .await
            .unwrap()
            .unwrap()
            .concat();

        let decoded: Vec<u16> = decoded
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }
}

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
//! - `https://codec.zarrs.dev/bytes_to_bytes/bz2` (written by `zarrs` 0.13 to 0.19)
//! - `bz2` (unregistered, written by `zarrs` 0.11 and 0.12)
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
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::bz2::{
    Bz2CodecConfiguration, Bz2CodecConfigurationV1, Bz2CompressionLevel,
};

zarrs_plugin::impl_extension_aliases!(Bz2Codec,
    v3: "numcodecs.bz2", [
        // Written by zarrs 0.13 to 0.19
        "https://codec.zarrs.dev/bytes_to_bytes/bz2",
        // UNREGISTERED: A name written by zarrs 0.11 and 0.12, read for backwards compatibility
        "bz2",
    ],
    v2: "bz2", []
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<Bz2Codec>()
}

// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<Bz2Codec>()
}

impl CodecTraitsV3 for Bz2Codec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: Bz2CodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(Bz2Codec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

impl CodecTraitsV2 for Bz2Codec {
    fn create(metadata: &MetadataV2) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: Bz2CodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(Bz2Codec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use crate::array::ComputeResources;
    use crate::array::codec::test_util::partial_decode_bytes;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use zarrs_codec::CowBytes;

    use super::*;
    use crate::array::{ArraySubset, BytesRepresentation, ChunkShapeTraits, Indexer, data_type};
    use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions};
    use zarrs_storage::byte_range::ByteRange;

    const JSON_VALID1: &str = r#"
{
    "level": 5
}"#;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_bz2_decode_into() {
        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Bz2Codec::new_with_configuration(&codec_configuration).unwrap();
        crate::array::codec::bytes_to_bytes::check_bytes_decode_into(
            &codec,
            &b"decode directly into this buffer".repeat(32),
            1,
            |encoded| {
                // A truncated stream, and bytes that are not a bz2 stream
                vec![encoded[..encoded.len() / 2].to_vec(), vec![0u8; 64]]
            },
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_bz2_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Bz2Codec::new_with_configuration(&codec_configuration).unwrap();

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
    fn codec_bz2_legacy_names() {
        for name in ["https://codec.zarrs.dev/bytes_to_bytes/bz2", "bz2"] {
            let metadata = MetadataV3::new_with_serializable_configuration(
                name.to_string(),
                &serde_json::json!({"level": 9}),
            )
            .unwrap();
            let codec = Codec::from_metadata(&metadata).unwrap();
            assert!(matches!(codec, Codec::BytesToBytes(_)));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_bz2_partial_decode() {
        let shape = vec![NonZeroU64::new(2).unwrap(); 3];
        let data_type = data_type::uint16();
        let data_type_size = data_type.fixed_size().unwrap();
        let array_size = shape.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: Bz2CodecConfiguration = serde_json::from_str(JSON_VALID1).unwrap();
        let codec = Arc::new(Bz2Codec::new_with_configuration(&codec_configuration).unwrap());

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
        let decoded_regions: Vec<ByteRange> = decoded_regions.collect();
        let decoded = partial_decode_bytes(
            codec.clone(),
            &encoded,
            bytes_representation.size().unwrap(),
            &decoded_regions,
        )
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

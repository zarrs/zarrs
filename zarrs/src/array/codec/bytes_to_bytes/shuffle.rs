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
//! # use zarrs::metadata_ext::codec::shuffle::ShuffleCodecConfiguration;
//! # serde_json::from_str::<ShuffleCodecConfiguration>(JSON).unwrap();
//! ```

// FIXME: Could use a real partial decoder

mod shuffle_codec;

use std::sync::Arc;

pub use shuffle_codec::ShuffleCodec;
pub use zarrs_metadata_ext::codec::shuffle::{
    ShuffleCodecConfiguration, ShuffleCodecConfigurationV1,
};

use zarrs_codec::Codec;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

zarrs_plugin::impl_extension_aliases!(ShuffleCodec,
    v3: "numcodecs.shuffle", [],
    v2: "shuffle", []
);

inventory::submit! {
    zarrs_codec::CodecPluginV3::new::<ShuffleCodec>()
}
inventory::submit! {
    zarrs_codec::CodecPluginV2::new::<ShuffleCodec>()
}

impl zarrs_codec::CodecTraitsV3 for ShuffleCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration = metadata.to_typed_configuration()?;
        let codec = Arc::new(ShuffleCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

impl zarrs_codec::CodecTraitsV2 for ShuffleCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration = metadata.to_typed_configuration()?;
        let codec = Arc::new(ShuffleCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Resources;
    use std::sync::Arc;
    use zarrs_codec::{ComputeResources, CowBytes};

    use super::*;
    use crate::array::BytesRepresentation;
    use zarrs_codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions};
    use zarrs_storage::byte_range::ByteRange;

    const JSON_VALID: &str = r#"{"elementsize":2}"#;

    #[test]
    fn codec_shuffle() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ShuffleCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = ShuffleCodec::new_with_configuration(&configuration).unwrap();

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
    fn codec_shuffle_matches_reference() {
        for elementsize in [1usize, 2, 3, 4, 5, 8, 16, 32] {
            let codec = ShuffleCodec::new(elementsize);
            for count in [0usize, 1, 7, 8, 9, 64, 1003] {
                let data: Vec<u8> = (0..elementsize * count)
                    .map(|i| u8::try_from(i * 7 % 251).unwrap())
                    .collect();
                let mut expected = vec![0u8; data.len()];
                for k in 0..count {
                    for i in 0..elementsize {
                        expected[i * count + k] = data[k * elementsize + i];
                    }
                }
                let encoded = codec
                    .encode(
                        CowBytes::Borrowed(&data),
                        &CodecOptions::default(),
                        &ComputeResources::default(),
                    )
                    .unwrap();
                assert_eq!(*encoded, *expected, "encode {elementsize}x{count}");
                let decoded = codec
                    .decode(
                        encoded,
                        &BytesRepresentation::FixedSize(data.len() as u64),
                        &CodecOptions::default(),
                        &ComputeResources::default(),
                    )
                    .unwrap();
                assert_eq!(*decoded, *data, "decode {elementsize}x{count}");
            }
        }
    }
    #[test]
    fn codec_shuffle_elementsize_zero() {
        let configuration: ShuffleCodecConfiguration =
            serde_json::from_str(r#"{"elementsize":0}"#).unwrap();
        assert!(ShuffleCodec::new_with_configuration(&configuration).is_err());

        let codec = ShuffleCodec::new(0);
        assert!(
            codec
                .encode(
                    CowBytes::Borrowed(&[]),
                    &CodecOptions::default(),
                    &ComputeResources::default(),
                )
                .is_err()
        );
        assert!(
            codec
                .decode(
                    CowBytes::Borrowed(&[]),
                    &BytesRepresentation::FixedSize(0),
                    &CodecOptions::default(),
                    &ComputeResources::default(),
                )
                .is_err()
        );
    }
    #[test]
    fn codec_shuffle_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ShuffleCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(ShuffleCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(
                CowBytes::from(bytes),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
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
                &Resources::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // shuffle partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
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
    async fn codec_shuffle_async_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: ShuffleCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(ShuffleCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(
                CowBytes::from(bytes),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
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
                &Resources::default(),
            )
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
                &Resources::default(),
            )
            .await
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn codec_shuffle_decode_into() {
        let codec = ShuffleCodec::new(2);
        crate::array::codec::bytes_to_bytes::check_bytes_decode_into(
            &codec,
            &crate::array::transmute_to_bytes_vec((0..32u16).collect::<Vec<_>>()),
            2,
            |_| Vec::new(),
        );

        // The input must be a multiple of the element size
        assert!(
            codec
                .decode_into(
                    CowBytes::from(vec![0u8; 3]),
                    &BytesRepresentation::FixedSize(3),
                    &mut [0; 3],
                    &CodecOptions::default(),
                    &ComputeResources::default(),
                )
                .is_err()
        );
    }

    #[test]
    fn codec_shuffle_direct_output_matches_elements() {
        let options = CodecOptions::default();
        let resources = ComputeResources::default();
        for elementsize in [1usize, 2, 3, 4, 5, 8, 16, 32] {
            let codec = ShuffleCodec::new(elementsize);
            for count in [0usize, 1, 7, 8, 9, 64, 1003] {
                let data: Vec<u8> = (0..elementsize * count)
                    .map(|i| u8::try_from(i * 7 % 251).unwrap())
                    .collect();
                let encoded = codec
                    .encode(CowBytes::Borrowed(&data), &options, &resources)
                    .unwrap();
                let mut output = vec![0xa5; data.len()];
                codec
                    .decode_into(
                        encoded,
                        &BytesRepresentation::FixedSize(data.len() as u64),
                        &mut output,
                        &options,
                        &resources,
                    )
                    .unwrap();
                assert_eq!(output, data, "{elementsize}x{count}");
            }
        }
        assert!(
            ShuffleCodec::new(0)
                .decode_into(
                    CowBytes::Borrowed(&[]),
                    &BytesRepresentation::FixedSize(0),
                    &mut [],
                    &options,
                    &resources
                )
                .is_err()
        );
    }
}

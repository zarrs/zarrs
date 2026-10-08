//! The `fletcher32` bytes to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! Appends a fletcher32 checksum of the input bytestream.
//!
//! This codec requires the `fletcher32` feature, which is disabled by default.
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.fletcher32` codec in `zarr-python`.
//!
//! **Non-conformant** checksums written by `zarrs` 0.19 to 0.23 are read for backwards compatibility: they omit the last byte of data with an odd length.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.fletcher32>
//! - <https://codec.zarrs.dev/bytes_to_bytes/fletcher32>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.fletcher32`
//! - `https://codec.zarrs.dev/bytes_to_bytes/fletcher32`
//! - `fletcher32` (**non-conformant**, written by `zarrs` 0.19)
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `fletcher32`
//!
//! ### Codec `configuration` Example - [`Fletcher32CodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {}
//! # "#;
//! # use zarrs::metadata_ext::codec::fletcher32::Fletcher32CodecConfiguration;
//! # serde_json::from_str::<Fletcher32CodecConfiguration>(JSON).unwrap();
//! ```

mod fletcher32_codec;

use std::sync::Arc;

pub use fletcher32_codec::Fletcher32Codec;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::fletcher32::{
    Fletcher32CodecConfiguration, Fletcher32CodecConfigurationV1,
};

zarrs_plugin::impl_extension_aliases!(Fletcher32Codec,
    v3: "numcodecs.fletcher32", [
        "numcodecs.fletcher32",
        "https://codec.zarrs.dev/bytes_to_bytes/fletcher32",
        // NON-CONFORMANT: An unregistered name written by zarrs 0.19, read for backwards compatibility
        "fletcher32",
    ],
    v2: "fletcher32"
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<Fletcher32Codec>()
}
// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<Fletcher32Codec>()
}

impl CodecTraitsV3 for Fletcher32Codec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration = metadata.to_typed_configuration()?;
        let codec = Arc::new(Fletcher32Codec::new_with_configuration(&configuration));
        Ok(Codec::BytesToBytes(codec))
    }
}

impl CodecTraitsV2 for Fletcher32Codec {
    fn create(metadata: &MetadataV2) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: Fletcher32CodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(Fletcher32Codec::new_with_configuration(&configuration));
        Ok(Codec::BytesToBytes(codec))
    }
}

const CHECKSUM_SIZE: usize = size_of::<u32>();

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use zarrs_codec::CowBytes;

    use super::*;
    use crate::array::BytesRepresentation;
    use zarrs_codec::{
        BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecMetadataOptions, CodecOptions,
        CodecTraits,
    };
    use zarrs_storage::byte_range::ByteRange;

    #[test]
    fn codec_fletcher32_checksum_numcodecs() {
        // Checksums of `numcodecs` 0.17.0
        let checksum = |data: &[u8]| fletcher32_codec::h5_checksum_fletcher32(data).to_le_bytes();
        assert_eq!(checksum(b"a"), [0, 97, 0, 97]);
        assert_eq!(checksum(b"abc"), [98, 196, 197, 37]);
        assert_eq!(checksum(b"abcd"), [198, 196, 41, 38]);
        let data: Vec<u8> = (0..255).collect();
        assert_eq!(checksum(&data), [64, 191, 118, 84]);
    }

    #[test]
    fn codec_fletcher32_decode_legacy_odd_checksum() {
        // Written by zarrs 0.19 to 0.23, which omitted the last byte of data with an odd length from the checksum
        let data = b"abc";
        let legacy = fletcher32_codec::h5_checksum_fletcher32(b"ab").to_le_bytes();
        let encoded = [data.as_slice(), &legacy].concat();
        let codec = Fletcher32Codec::new();
        let decoded = codec
            .decode(
                CowBytes::from(encoded),
                &BytesRepresentation::FixedSize(3),
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(decoded.as_ref(), data);
        // Other checksums are invalid
        let encoded = [data.as_slice(), &[0, 0, 0, 0]].concat();
        assert!(matches!(
            codec.decode(
                CowBytes::from(encoded),
                &BytesRepresentation::FixedSize(3),
                &CodecOptions::default(),
            ),
            Err(zarrs_codec::CodecError::InvalidChecksum)
        ));
    }

    const JSON1: &str = r"{}";

    #[test]
    fn codec_fletcher32_configuration_none() {
        let codec_configuration: Fletcher32CodecConfiguration =
            serde_json::from_str(r"{}").unwrap();
        let codec = Fletcher32Codec::new_with_configuration(&codec_configuration);
        let configuration = codec
            .configuration_v3(&CodecMetadataOptions::default())
            .unwrap();
        assert_eq!(serde_json::to_string(&configuration).unwrap(), r"{}");
    }

    #[test]
    fn codec_fletcher32_legacy_name() {
        let codec = Codec::from_metadata(&MetadataV3::new("fletcher32")).unwrap();
        assert!(matches!(codec, Codec::BytesToBytes(_)));
    }

    #[test]
    fn codec_fletcher32() {
        let elements: Vec<u8> = (0..6).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Fletcher32CodecConfiguration =
            serde_json::from_str(JSON1).unwrap();
        let codec = Fletcher32Codec::new_with_configuration(&codec_configuration);

        let encoded = codec
            .encode(CowBytes::Borrowed(&bytes), &CodecOptions::default())
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
        assert_eq!(checksum, &[9, 6, 14, 8]);
    }

    #[test]
    fn codec_fletcher32_partial_decode() {
        let elements: Vec<u8> = (0..32).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Fletcher32CodecConfiguration =
            serde_json::from_str(JSON1).unwrap();
        let codec = Arc::new(Fletcher32Codec::new_with_configuration(
            &codec_configuration,
        ));

        let encoded = codec
            .encode(CowBytes::from(bytes), &CodecOptions::default())
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
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // fletcher32 partial decoder does not hold bytes
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
    async fn codec_fletcher32_async_partial_decode() {
        let elements: Vec<u8> = (0..32).collect();
        let bytes = elements;
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: Fletcher32CodecConfiguration =
            serde_json::from_str(JSON1).unwrap();
        let codec = Arc::new(Fletcher32Codec::new_with_configuration(
            &codec_configuration,
        ));

        let encoded = codec
            .encode(CowBytes::from(bytes), &CodecOptions::default())
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

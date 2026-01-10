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
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.fletcher32>
//! - <https://codec.zarrs.dev/bytes_to_bytes/fletcher32>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.fletcher32`
//! - `https://codec.zarrs.dev/bytes_to_bytes/fletcher32`
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

use crate::array::codec::{Codec, CodecPluginV2, CodecPluginV3};
pub use crate::metadata_ext::codec::fletcher32::{
    Fletcher32CodecConfiguration, Fletcher32CodecConfigurationV1,
};
use crate::plugin::{PluginConfigurationInvalidError, PluginCreateError};

zarrs_plugin::impl_extension_aliases!(Fletcher32Codec,
    v3: "numcodecs.fletcher32", ["numcodecs.fletcher32", "https://codec.zarrs.dev/bytes_to_bytes/fletcher32"],
    v2: "fletcher32"
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<Fletcher32Codec>(create_codec_fletcher32_v3)
}
// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<Fletcher32Codec>(create_codec_fletcher32_v2)
}

pub(crate) fn create_codec_fletcher32_v3(
    metadata: &MetadataV3,
) -> Result<Codec, PluginCreateError> {
    let configuration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let codec = Arc::new(Fletcher32Codec::new_with_configuration(&configuration));
    Ok(Codec::BytesToBytes(codec))
}

pub(crate) fn create_codec_fletcher32_v2(
    metadata: &MetadataV2,
) -> Result<Codec, PluginCreateError> {
    let configuration: Fletcher32CodecConfiguration =
        serde_json::from_value(serde_json::to_value(metadata.configuration()).unwrap())
            .map_err(|_| PluginConfigurationInvalidError::new(format!("{metadata:?}")))?;
    let codec = Arc::new(Fletcher32Codec::new_with_configuration(&configuration));
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
    fn codec_fletcher32_configuration_none() {
        let codec_configuration: Fletcher32CodecConfiguration =
            serde_json::from_str(r"{}").unwrap();
        let codec = Fletcher32Codec::new_with_configuration(&codec_configuration);
        let configuration = codec
            .configuration(&CodecMetadataOptions::default())
            .unwrap();
        assert_eq!(serde_json::to_string(&configuration).unwrap(), r"{}");
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

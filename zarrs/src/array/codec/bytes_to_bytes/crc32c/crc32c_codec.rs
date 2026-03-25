use std::borrow::Cow;
use std::sync::Arc;

use zarrs_metadata_ext::codec::crc32c::Crc32cCodecConfigurationLocation;
use zarrs_plugin::ZarrVersion;

use super::{CHECKSUM_SIZE, Crc32cCodecConfiguration, Crc32cCodecConfigurationV1};
#[cfg(feature = "async")]
use crate::array::codec::bytes_to_bytes::strip_prefix_partial_decoder::AsyncStripPrefixPartialDecoder;
use crate::array::codec::bytes_to_bytes::strip_prefix_partial_decoder::StripPrefixPartialDecoder;
#[cfg(feature = "async")]
use crate::array::codec::bytes_to_bytes::strip_suffix_partial_decoder::AsyncStripSuffixPartialDecoder;
use crate::array::codec::bytes_to_bytes::strip_suffix_partial_decoder::StripSuffixPartialDecoder;
use crate::array::{ArrayBytesRaw, BytesRepresentation};
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{
    BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
};
use zarrs_metadata::Configuration;

/// A `crc32c` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct Crc32cCodec(Crc32cCodecConfigurationLocation);

impl Crc32cCodec {
    /// Create a new `crc32c` codec.
    #[must_use]
    pub const fn new() -> Self {
        Self(Crc32cCodecConfigurationLocation::End)
    }

    /// Create a new `crc32c` codec.
    #[must_use]
    #[allow(clippy::wildcard_enum_match_arm)]
    pub const fn new_with_configuration(configuration: &Crc32cCodecConfiguration) -> Self {
        let location = match configuration {
            Crc32cCodecConfiguration::Numcodecs(cfg) => cfg.location,
            Crc32cCodecConfiguration::V1(_) | _ => Crc32cCodecConfigurationLocation::End,
        };
        Self(location)
    }
}

impl CodecTraits for Crc32cCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = Crc32cCodecConfiguration::V1(Crc32cCodecConfigurationV1 {});
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,   // TODO
            partial_decode: false, // TODO
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for Crc32cCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let checksum = crc32c::crc32c(&decoded_value).to_le_bytes();
        let mut encoded_value: Vec<u8> = Vec::with_capacity(decoded_value.len() + checksum.len());
        match self.0 {
            Crc32cCodecConfigurationLocation::End => {
                encoded_value.extend_from_slice(&decoded_value);
                encoded_value.extend_from_slice(&checksum);
            }
            Crc32cCodecConfigurationLocation::Start => {
                encoded_value.extend_from_slice(&checksum);
                encoded_value.extend_from_slice(&decoded_value);
            }
        }
        Ok(Cow::Owned(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        if encoded_value.len() >= CHECKSUM_SIZE {
            let (data, checksum_stored): (&[u8], [u8; CHECKSUM_SIZE]) = match self.0 {
                Crc32cCodecConfigurationLocation::End => (
                    &encoded_value[..encoded_value.len() - CHECKSUM_SIZE],
                    encoded_value[encoded_value.len() - CHECKSUM_SIZE..]
                        .try_into()
                        .unwrap(),
                ),
                Crc32cCodecConfigurationLocation::Start => (
                    &encoded_value[CHECKSUM_SIZE..],
                    encoded_value[..CHECKSUM_SIZE].try_into().unwrap(),
                ),
            };

            if options.validate_checksums() {
                let checksum = crc32c::crc32c(data).to_le_bytes();
                if checksum != checksum_stored {
                    return Err(CodecError::InvalidChecksum);
                }
            }

            Ok(Cow::Owned(data.to_vec()))
        } else {
            Err(CodecError::Other(
                "crc32c decoder expects a 32 bit input".to_string(),
            ))
        }
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialDecoderTraits>, CodecError> {
        match self.0 {
            Crc32cCodecConfigurationLocation::End => Ok(Arc::new(StripSuffixPartialDecoder::new(
                input_handle,
                CHECKSUM_SIZE,
            ))),
            Crc32cCodecConfigurationLocation::Start => Ok(Arc::new(
                StripPrefixPartialDecoder::new(input_handle, CHECKSUM_SIZE),
            )),
        }
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialDecoderTraits>, CodecError> {
        match self.0 {
            Crc32cCodecConfigurationLocation::End => Ok(Arc::new(
                AsyncStripSuffixPartialDecoder::new(input_handle, CHECKSUM_SIZE),
            )),
            Crc32cCodecConfigurationLocation::Start => Ok(Arc::new(
                AsyncStripPrefixPartialDecoder::new(input_handle, CHECKSUM_SIZE),
            )),
        }
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        match decoded_representation {
            BytesRepresentation::FixedSize(size) => {
                BytesRepresentation::FixedSize(size + CHECKSUM_SIZE as u64)
            }
            BytesRepresentation::BoundedSize(size) => {
                BytesRepresentation::BoundedSize(size + CHECKSUM_SIZE as u64)
            }
            BytesRepresentation::UnboundedSize => BytesRepresentation::UnboundedSize,
        }
    }
}

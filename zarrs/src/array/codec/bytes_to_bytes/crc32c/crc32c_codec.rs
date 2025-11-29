use std::{borrow::Cow, sync::Arc};

use super::{Crc32cCodecConfiguration, Crc32cCodecConfigurationV1, CHECKSUM_SIZE};
#[cfg(feature = "async")]
use crate::array::codec::bytes_to_bytes::strip_suffix_partial_decoder::AsyncStripSuffixPartialDecoder;
#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;
use crate::array::{
    codec::{
        bytes_to_bytes::strip_suffix_partial_decoder::StripSuffixPartialDecoder,
        BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError, CodecMetadataOptions,
        CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
        RecommendedConcurrency,
    },
    ArrayBytesRaw, BytesRepresentation,
};
use crate::metadata::Configuration;
use crate::registry::codec::CRC32C;

/// A `crc32c` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct Crc32cCodec;

impl Crc32cCodec {
    /// Create a new `crc32c` codec.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }

    /// Create a new `crc32c` codec.
    #[must_use]
    pub const fn new_with_configuration(_configuration: &Crc32cCodecConfiguration) -> Self {
        Self {}
    }
}

impl CodecTraits for Crc32cCodec {
    fn identifier(&self) -> &str {
        CRC32C
    }

    fn configuration_opt(
        &self,
        _name: &str,
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
        encoded_value.extend_from_slice(&decoded_value);
        encoded_value.extend_from_slice(&checksum);
        Ok(Cow::Owned(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        if encoded_value.len() >= CHECKSUM_SIZE {
            if options.validate_checksums() {
                let decoded_value = &encoded_value[..encoded_value.len() - CHECKSUM_SIZE];
                let checksum = crc32c::crc32c(decoded_value).to_le_bytes();
                if checksum != encoded_value[encoded_value.len() - CHECKSUM_SIZE..] {
                    return Err(CodecError::InvalidChecksum);
                }
            }
            let decoded_value = encoded_value[..encoded_value.len() - CHECKSUM_SIZE].to_vec();
            Ok(Cow::Owned(decoded_value))
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
        Ok(Arc::new(StripSuffixPartialDecoder::new(
            input_handle,
            CHECKSUM_SIZE,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(AsyncStripSuffixPartialDecoder::new(
            input_handle,
            CHECKSUM_SIZE,
        )))
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

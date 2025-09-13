use std::{borrow::Cow, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::adler32::Adler32CodecConfigurationChecksumLocation;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::ADLER32;

use crate::array::{
    codec::{
        bytes_to_bytes::{
            strip_prefix_partial_decoder::StripPrefixPartialDecoder,
            strip_suffix_partial_decoder::StripSuffixPartialDecoder,
        },
        BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError, CodecMetadataOptions,
        CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
        RecommendedConcurrency,
    },
    BytesRepresentation, RawBytes,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;

#[cfg(feature = "async")]
use crate::array::codec::bytes_to_bytes::{
    strip_prefix_partial_decoder::AsyncStripPrefixPartialDecoder,
    strip_suffix_partial_decoder::AsyncStripSuffixPartialDecoder,
};

use super::{Adler32CodecConfiguration, Adler32CodecConfigurationV1, CHECKSUM_SIZE};

/// A `adler32` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct Adler32Codec {
    location: Adler32CodecConfigurationChecksumLocation,
}

impl Adler32Codec {
    /// Create a new `adler32` codec.
    #[must_use]
    pub const fn new(location: Adler32CodecConfigurationChecksumLocation) -> Self {
        Self { location }
    }

    /// Create a new `adler32` codec.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &Adler32CodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            Adler32CodecConfiguration::V1(configuration) => Ok(Self {
                location: configuration.location,
            }),
            _ => Err(PluginCreateError::Other(
                "this adler32 codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for Adler32Codec {
    fn identifier(&self) -> &str {
        ADLER32
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = Adler32CodecConfiguration::V1(Adler32CodecConfigurationV1 {
            location: self.location,
        });
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
impl BytesToBytesCodecTraits for Adler32Codec {
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
        decoded_value: RawBytes<'a>,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let mut adler = simd_adler32::Adler32::new();
        adler.write(&decoded_value);
        let checksum = adler.finish().to_le_bytes();

        let mut encoded_value: Vec<u8> = Vec::with_capacity(decoded_value.len() + checksum.len());
        match self.location {
            Adler32CodecConfigurationChecksumLocation::Start => {
                encoded_value.extend_from_slice(&checksum);
                encoded_value.extend_from_slice(&decoded_value);
            }
            Adler32CodecConfigurationChecksumLocation::End => {
                encoded_value.extend_from_slice(&decoded_value);
                encoded_value.extend_from_slice(&checksum);
            }
        }
        Ok(Cow::Owned(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: RawBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        if encoded_value.len() >= CHECKSUM_SIZE {
            let (decoded_value, checksum) = match self.location {
                Adler32CodecConfigurationChecksumLocation::Start => {
                    let (checksum, decoded_value) = encoded_value.split_at(CHECKSUM_SIZE);
                    (decoded_value, checksum)
                }
                Adler32CodecConfigurationChecksumLocation::End => {
                    encoded_value.split_at(encoded_value.len() - CHECKSUM_SIZE)
                }
            };

            if options.validate_checksums() {
                let mut adler = simd_adler32::Adler32::new();
                adler.write(decoded_value);
                if adler.finish().to_le_bytes() != checksum {
                    return Err(CodecError::InvalidChecksum);
                }
            }

            Ok(Cow::Owned(decoded_value.to_vec()))
        } else {
            Err(CodecError::Other(
                "adler32 decoder expects a 32 bit input".to_string(),
            ))
        }
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialDecoderTraits>, CodecError> {
        match self.location {
            Adler32CodecConfigurationChecksumLocation::Start => Ok(Arc::new(
                StripPrefixPartialDecoder::new(input_handle, CHECKSUM_SIZE),
            )),
            Adler32CodecConfigurationChecksumLocation::End => Ok(Arc::new(
                StripSuffixPartialDecoder::new(input_handle, CHECKSUM_SIZE),
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
        match self.location {
            Adler32CodecConfigurationChecksumLocation::Start => Ok(Arc::new(
                AsyncStripPrefixPartialDecoder::new(input_handle, CHECKSUM_SIZE),
            )),
            Adler32CodecConfigurationChecksumLocation::End => Ok(Arc::new(
                AsyncStripSuffixPartialDecoder::new(input_handle, CHECKSUM_SIZE),
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

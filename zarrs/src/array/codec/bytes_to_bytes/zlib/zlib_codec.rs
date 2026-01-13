use std::borrow::Cow;
use std::io::{Cursor, Read};
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{ZlibCodecConfiguration, ZlibCodecConfigurationV1, ZlibCompressionLevel};
use crate::array::{ArrayBytesRaw, BytesRepresentation};
use crate::metadata::Configuration;
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};

/// A `zlib` codec implementation.
#[derive(Clone, Debug)]
pub struct ZlibCodec {
    compression: flate2::Compression,
}

impl ZlibCodec {
    /// Create a new `zlib` codec.
    #[must_use]
    pub fn new(level: ZlibCompressionLevel) -> Self {
        let compression = flate2::Compression::new(level.as_u32());
        Self { compression }
    }

    /// Create a new `zlib` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &ZlibCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ZlibCodecConfiguration::V1(configuration) => Ok(Self::new(configuration.level)),
            _ => Err(PluginCreateError::Other(
                "this zlib codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for ZlibCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ZlibCodecConfiguration::V1(ZlibCodecConfigurationV1 {
            level: ZlibCompressionLevel::try_from(self.compression.level())
                .expect("checked on init"),
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
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
impl BytesToBytesCodecTraits for ZlibCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // zlib does not support parallel decode
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let mut encoder =
            flate2::read::ZlibEncoder::new(Cursor::new(decoded_value), self.compression);
        let mut out: Vec<u8> = Vec::new();
        encoder.read_to_end(&mut out)?;
        Ok(Cow::Owned(out))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let mut decoder = flate2::read::ZlibDecoder::new(Cursor::new(encoded_value));
        let mut out: Vec<u8> = Vec::new();
        decoder.read_to_end(&mut out)?;
        Ok(Cow::Owned(out))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        decoded_representation
            .size()
            .map_or(BytesRepresentation::UnboundedSize, |size| {
                // https://github.com/madler/zlib/blob/v1.3.1/compress.c#L72-L75
                BytesRepresentation::BoundedSize(
                    size + (size >> 12) + (size >> 14) + (size >> 25) + 13,
                )
            })
    }
}

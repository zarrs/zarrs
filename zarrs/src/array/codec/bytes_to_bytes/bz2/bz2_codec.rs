use std::{
    borrow::Cow,
    io::{Cursor, Read},
    sync::Arc,
};

use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::BZ2;

use crate::array::{
    codec::{
        BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        RecommendedConcurrency,
    },
    BytesRepresentation, RawBytes,
};

use super::{Bz2CodecConfiguration, Bz2CodecConfigurationV1, Bz2CompressionLevel};

/// A `bz2` codec implementation.
#[derive(Clone, Debug)]
pub struct Bz2Codec {
    compression: bzip2::Compression,
}

impl Bz2Codec {
    /// Create a new `bz2` codec.
    #[must_use]
    pub fn new(level: Bz2CompressionLevel) -> Self {
        let compression = bzip2::Compression::new(level.as_u32());
        Self { compression }
    }

    /// Create a new `bz2` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &Bz2CodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            Bz2CodecConfiguration::V1(configuration) => Ok(Self::new(configuration.level)),
            _ => Err(PluginCreateError::Other(
                "this bz2 codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for Bz2Codec {
    fn identifier(&self) -> &str {
        BZ2
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = Bz2CodecConfiguration::V1(Bz2CodecConfigurationV1 {
            level: Bz2CompressionLevel::try_from(self.compression.level())
                .expect("checked on init"),
        });
        Some(configuration.into())
    }

    fn partial_decoder_should_cache_input(&self) -> bool {
        false
    }

    fn partial_decoder_decodes_all(&self) -> bool {
        true
    }
}

#[cfg_attr(all(feature = "async", not(target_arch = "wasm32")), async_trait::async_trait)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for Bz2Codec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // bz2 does not support parallel decode
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: RawBytes<'a>,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let mut encoder = bzip2::read::BzEncoder::new(Cursor::new(decoded_value), self.compression);
        let mut out: Vec<u8> = Vec::new();
        encoder.read_to_end(&mut out)?;
        Ok(Cow::Owned(out))
    }

    fn decode<'a>(
        &self,
        encoded_value: RawBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let mut decoder = bzip2::read::BzDecoder::new(Cursor::new(encoded_value));
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
                // https://en.wikipedia.org/wiki/Bzip2#Implementation
                // TODO: Below assumes a maximum expansion of 1.25 for the blocks + header (4 byte) + footer (11 byte), but need to read spec
                BytesRepresentation::BoundedSize(4 + 11 + size + size.div_ceil(4))
            })
    }
}

use std::{borrow::Cow, sync::Arc};

use zarrs_plugin::PluginCreateError;

use super::{
    GDEFLATE_STATIC_HEADER_LENGTH, GDeflateCodecConfiguration, GDeflateCodecConfigurationV0,
    GDeflateCompressionLevel, GDeflateCompressionLevelError, GDeflateCompressor, gdeflate_decode,
};
use crate::array::{
    ArrayBytesRaw, BytesRepresentation, RecommendedConcurrency,
    codec::{
        BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        PartialDecoderCapability, PartialEncoderCapability,
    },
};
use crate::metadata::Configuration;
use zarrs_plugin::ExtensionIdentifier;

/// A `gdeflate` codec implementation.
#[derive(Clone, Debug)]
pub struct GDeflateCodec {
    compression_level: GDeflateCompressionLevel,
}

impl GDeflateCodec {
    /// Create a new `gdeflate` codec.
    ///
    /// # Errors
    /// Returns [`GDeflateCompressionLevelError`] if `compression_level` is not valid.
    pub fn new(compression_level: u32) -> Result<Self, GDeflateCompressionLevelError> {
        let compression_level: GDeflateCompressionLevel = compression_level.try_into()?;
        // let compression_level = compression_level.into();
        Ok(Self { compression_level })
    }

    /// Create a new `gdeflate` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &GDeflateCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            GDeflateCodecConfiguration::V0(configuration) => {
                let compression_level = configuration.level;
                Ok(Self { compression_level })
            }
            _ => Err(PluginCreateError::Other(
                "this gdeflate codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for GDeflateCodec {
    fn identifier(&self) -> &'static str {
        Self::IDENTIFIER
    }

    fn configuration(&self, _name: &str, _options: &CodecMetadataOptions) -> Option<Configuration> {
        let configuration = GDeflateCodecConfiguration::V0(GDeflateCodecConfigurationV0 {
            level: self.compression_level,
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
impl BytesToBytesCodecTraits for GDeflateCodec {
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
        let compressor = GDeflateCompressor::new(self.compression_level)
            .map_err(|err| CodecError::Other(err.to_string()))?;
        let (page_sizes, encoded_bytes) = compressor
            .compress(&decoded_value)
            .map_err(|err| CodecError::Other(err.to_string()))?;
        let mut encoded_value = Vec::with_capacity(
            GDEFLATE_STATIC_HEADER_LENGTH
                + page_sizes.len() * size_of::<u64>()
                + encoded_bytes.len(),
        );

        // Header
        let decoded_value_len = u64::try_from(decoded_value.len()).unwrap();
        let num_pages = u64::try_from(page_sizes.len()).unwrap();
        encoded_value.extend_from_slice(&decoded_value_len.to_le_bytes());
        encoded_value.extend_from_slice(&num_pages.to_le_bytes());
        for page_size_compressed in page_sizes {
            let page_size_compressed = u64::try_from(page_size_compressed).unwrap();
            encoded_value.extend_from_slice(&page_size_compressed.to_le_bytes());
        }

        // Data
        encoded_value.extend_from_slice(&encoded_bytes);

        Ok(Cow::Owned(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        Ok(Cow::Owned(gdeflate_decode(&encoded_value)?))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        match decoded_representation {
            BytesRepresentation::BoundedSize(size) | BytesRepresentation::FixedSize(size) => {
                let compressor = GDeflateCompressor::new(self.compression_level).unwrap(); // FIXME: Make encoded_representation fallible?
                let size = usize::try_from(*size).unwrap();
                let (_, compress_bound) = compressor.get_npages_compress_bound(size);
                let compress_bound = u64::try_from(compress_bound).unwrap();
                BytesRepresentation::BoundedSize(compress_bound)
            }
            BytesRepresentation::UnboundedSize => BytesRepresentation::UnboundedSize,
        }
    }
}

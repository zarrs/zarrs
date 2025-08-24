use std::{borrow::Cow, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::ZSTD;
use zstd::zstd_safe;

use crate::array::{
    codec::{
        BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        RecommendedConcurrency,
    },
    BytesRepresentation, RawBytes,
};

use super::{ZstdCodecConfiguration, ZstdCodecConfigurationV1};

/// A `zstd` codec implementation.
#[derive(Clone, Debug)]
pub struct ZstdCodec {
    compression: zstd_safe::CompressionLevel,
    checksum: bool,
}

impl ZstdCodec {
    /// Create a new `Zstd` codec.
    #[must_use]
    pub const fn new(compression: zstd_safe::CompressionLevel, checksum: bool) -> Self {
        Self {
            compression,
            checksum,
        }
    }

    /// Create a new `Zstd` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &ZstdCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        let (compression, checksum) = match configuration {
            ZstdCodecConfiguration::V1(configuration) => {
                (configuration.level, configuration.checksum)
            }
            ZstdCodecConfiguration::Numcodecs(configuration) => (configuration.level, false),
            _ => Err(PluginCreateError::Other(
                "this zstd codec configuration variant is unsupported".to_string(),
            ))?,
        };
        Ok(Self {
            compression: compression.into(),
            checksum,
        })
    }
}

impl CodecTraits for ZstdCodec {
    fn identifier(&self) -> &str {
        ZSTD
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ZstdCodecConfiguration::V1(ZstdCodecConfigurationV1 {
            level: self.compression.into(),
            checksum: self.checksum,
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

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for ZstdCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: zstd supports multithread, but at what point is it good to kick in?
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: RawBytes<'a>,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let mut compressor = zstd::bulk::Compressor::new(self.compression)?;
        compressor.include_checksum(self.checksum)?;
        // compressor.include_contentsize(true);
        // compressor.set_pledged_src_size(Some(decoded_value.len()))?; // unpublished
        let result = compressor.compress(&decoded_value)?;
        Ok(Cow::Owned(result))
    }

    fn decode<'a>(
        &self,
        encoded_value: RawBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let upper_bound = zstd::bulk::Decompressor::upper_bound(&encoded_value); // requires zstd experimental feature
        if let Some(upper_bound) = upper_bound {
            // Bulk decompression
            let mut result = zstd::bulk::decompress(&encoded_value, upper_bound)?;
            result.shrink_to_fit();
            Ok(Cow::Owned(result))
        } else {
            // Streaming decompression (slower)
            zstd::decode_all(std::io::Cursor::new(&encoded_value))
                .map_err(CodecError::from)
                .map(Cow::Owned)
        }
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        decoded_representation
            .size()
            .map_or(BytesRepresentation::UnboundedSize, |size| {
                // https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md
                // TODO: Validate the window/block relationship
                const HEADER_TRAILER_OVERHEAD: u64 = 4 + 14 + 4;
                const MIN_WINDOW_SIZE: u64 = 1000; // 1KB
                const BLOCK_OVERHEAD: u64 = 3;
                let blocks_overhead = BLOCK_OVERHEAD * size.div_ceil(MIN_WINDOW_SIZE);
                BytesRepresentation::BoundedSize(size + HEADER_TRAILER_OVERHEAD + blocks_overhead)
            })
    }
}

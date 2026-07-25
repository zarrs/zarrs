use std::borrow::Cow;
use std::io::{Cursor, Read};
use std::sync::Arc;

use flate2::bufread::{GzDecoder, GzEncoder};

use super::{
    GzipCodecConfiguration, GzipCodecConfigurationV1, GzipCompressionLevel,
    GzipCompressionLevelError,
};
use crate::array::{ArrayBytesRaw, BytesRepresentation};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};
use zarrs_codec::{
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesToBytesCodecPartialDefault,
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_metadata::Configuration;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

/// A `gzip` codec implementation.
#[derive(Clone, Debug)]
pub struct GzipCodec {
    compression_level: GzipCompressionLevel,
}

impl GzipCodec {
    /// Create a new `gzip` codec.
    ///
    /// # Errors
    /// Returns [`GzipCompressionLevelError`] if `compression_level` is not valid.
    pub fn new(compression_level: u32) -> Result<Self, GzipCompressionLevelError> {
        let compression_level: GzipCompressionLevel = compression_level.try_into()?;
        Ok(Self { compression_level })
    }

    /// Create a new `gzip` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &GzipCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            GzipCodecConfiguration::V1(configuration) => Ok(Self {
                compression_level: configuration.level,
            }),
            _ => Err(PluginCreateError::Other(
                "this gzip codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for GzipCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = GzipCodecConfiguration::V1(GzipCodecConfigurationV1 {
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

#[ambisync::paired(
    sync(fns("async_{}"), types("Async{}")),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
impl BytesToBytesCodecTraits for GzipCodec {
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
        let mut encoder = GzEncoder::new(
            Cursor::new(decoded_value),
            flate2::Compression::new(self.compression_level.as_u32()),
        );
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
        let mut decoder = GzDecoder::new(Cursor::new(encoded_value));
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
                // https://www.gnu.org/software/gzip/manual/gzip.pdf
                const HEADER_TRAILER_OVERHEAD: u64 = 10 + 8; // TODO: validate that extra headers are not populated
                const BLOCK_SIZE: u64 = 32768;
                const BLOCK_OVERHEAD: u64 = 5;
                let blocks_overhead = BLOCK_OVERHEAD * size.div_ceil(BLOCK_SIZE);
                BytesRepresentation::BoundedSize(size + HEADER_TRAILER_OVERHEAD + blocks_overhead)
            })
    }

    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
            input_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }

    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
            input_output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }
}

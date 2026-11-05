use crate::array::codec::bytes_to_bytes::{DecompressProgress, decompress_stream_into};
use std::io::{Cursor, Read};
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{ZlibCodecConfiguration, ZlibCodecConfigurationV1, ZlibCompressionLevel};
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::{
    BytesToBytesCodecDecodeInto, BytesToBytesCodecTraits, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, ComputeResources, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_metadata::Configuration;

/// A `zlib` codec implementation.
#[derive(Clone, Debug)]
pub struct ZlibCodec {
    level: ZlibCompressionLevel,
    compression: flate2::Compression,
}

impl ZlibCodec {
    /// Create a new `zlib` codec.
    #[must_use]
    pub fn new(level: ZlibCompressionLevel) -> Self {
        let compression = flate2::Compression::new(level.as_u32());
        Self { level, compression }
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
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration =
            ZlibCodecConfiguration::V1(ZlibCodecConfigurationV1 { level: self.level });
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
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let mut encoder =
            flate2::read::ZlibEncoder::new(Cursor::new(decoded_value), self.compression);
        let mut out: Vec<u8> = Vec::new();
        encoder.read_to_end(&mut out)?;
        Ok(CowBytes::from(out))
    }

    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let mut decoder = flate2::read::ZlibDecoder::new(Cursor::new(encoded_value));
        let mut out: Vec<u8> = Vec::new();
        decoder.read_to_end(&mut out)?;
        Ok(CowBytes::from(out))
    }

    fn as_decode_into(&self) -> Option<&dyn BytesToBytesCodecDecodeInto> {
        Some(self)
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

// SAFETY: `decompress_stream_into` fails unless the decompressor reports that it initialised all of `output`.
unsafe impl BytesToBytesCodecDecodeInto for ZlibCodec {
    fn decode_into(
        &self,
        encoded_value: CowBytes<'_>,
        _decoded_representation: &BytesRepresentation,
        output: &mut [std::mem::MaybeUninit<u8>],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        let mut decompressor = flate2::Decompress::new(true);
        decompress_stream_into(&encoded_value, output, |input, output| {
            let (total_in, total_out) = (decompressor.total_in(), decompressor.total_out());
            // `Finish` would report output that is too short as a decompression error
            let status = decompressor
                .decompress_uninit(input, output, flate2::FlushDecompress::None)
                .map_err(std::io::Error::from)?;
            Ok(DecompressProgress {
                consumed: usize::try_from(decompressor.total_in() - total_in).unwrap(),
                produced: usize::try_from(decompressor.total_out() - total_out).unwrap(),
                stream_end: status == flate2::Status::StreamEnd,
            })
        })
    }
}

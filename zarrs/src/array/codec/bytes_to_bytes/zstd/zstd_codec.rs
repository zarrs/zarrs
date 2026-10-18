use std::cell::RefCell;
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};
use zstd::zstd_safe;

use super::{ZstdCodecConfiguration, ZstdCodecConfigurationV1};
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_metadata::Configuration;

thread_local! {
    /// The decompression context of the thread, reused by decodes since creating one costs more than decompressing a
    /// small frame.
    static DECOMPRESSOR: RefCell<Option<zstd::bulk::Decompressor<'static>>> = const { RefCell::new(None) };
}

/// Run `f` with the decompression context of the thread.
pub(super) fn with_decompressor<R>(
    f: impl FnOnce(&mut zstd::bulk::Decompressor<'static>) -> std::io::Result<R>,
) -> std::io::Result<R> {
    DECOMPRESSOR.with(|decompressor| {
        let Ok(mut decompressor) = decompressor.try_borrow_mut() else {
            // The context is in use (e.g. by a decode reentered from a decode), so use a new one
            return f(&mut zstd::bulk::Decompressor::new()?);
        };
        let decompressor = match &mut *decompressor {
            Some(decompressor) => decompressor,
            none => none.insert(zstd::bulk::Decompressor::new()?),
        };
        f(decompressor)
    })
}

thread_local! {
    /// The compression context of the thread, reused by encodes of small values since creating one costs more than
    /// compressing a small value.
    static COMPRESSOR: RefCell<Option<zstd_safe::CCtx<'static>>> = const { RefCell::new(None) };
}

/// Values larger than this are compressed with a new context rather than the context of the thread, which keeps the
/// memory it allocates for the largest value it compresses.
const COMPRESSOR_REUSE_MAX_LEN: usize = 1024 * 1024;

/// An I/O error of a zstd error code, as the `zstd` crate maps them.
fn map_error_code(code: zstd_safe::ErrorCode) -> std::io::Error {
    std::io::Error::other(zstd_safe::get_error_name(code))
}

/// Compress `data` at compression `level`, with a checksum if `checksum`.
fn compress(
    data: &[u8],
    level: zstd_safe::CompressionLevel,
    checksum: bool,
) -> std::io::Result<Vec<u8>> {
    let compress = |context: &mut zstd_safe::CCtx<'static>| {
        // Every parameter set on a context of the thread is set for each value
        context
            .set_parameter(zstd_safe::CParameter::CompressionLevel(level))
            .map_err(map_error_code)?;
        context
            .set_parameter(zstd_safe::CParameter::ChecksumFlag(checksum))
            .map_err(map_error_code)?;
        let mut buffer = Vec::with_capacity(zstd_safe::compress_bound(data.len()));
        context
            .compress2(&mut buffer, data)
            .map_err(map_error_code)?;
        Ok(buffer)
    };
    if data.len() > COMPRESSOR_REUSE_MAX_LEN {
        return compress(&mut zstd_safe::CCtx::create());
    }
    COMPRESSOR.with(|compressor| {
        let Ok(mut compressor) = compressor.try_borrow_mut() else {
            // The context is in use (e.g. by an encode reentered from an encode), so use a new one
            return compress(&mut zstd_safe::CCtx::create());
        };
        compress(compressor.get_or_insert_with(zstd_safe::CCtx::create))
    })
}

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
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ZstdCodecConfiguration::V1(ZstdCodecConfigurationV1 {
            level: self.compression.into(),
            checksum: self.checksum,
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
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(CowBytes::from(compress(
            &decoded_value,
            self.compression,
            self.checksum,
        )?))
    }

    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        let upper_bound = zstd::bulk::Decompressor::upper_bound(&encoded_value); // requires zstd experimental feature
        if let Some(upper_bound) = upper_bound {
            // Bulk decompression
            let result = with_decompressor(|decompressor| {
                decompressor.decompress(&encoded_value, upper_bound)
            })?;
            Ok(CowBytes::from(result))
        } else {
            // Streaming decompression (slower)
            zstd::decode_all(std::io::Cursor::new(&encoded_value))
                .map_err(CodecError::from)
                .map(CowBytes::from)
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

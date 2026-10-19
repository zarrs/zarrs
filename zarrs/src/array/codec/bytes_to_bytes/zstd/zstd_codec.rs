use std::cell::RefCell;
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};
use zstd::zstd_safe;

use super::{ZstdCodecConfiguration, ZstdCodecConfigurationV1, ZstdCodecOptions};
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
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

/// An I/O error of a zstd error code, as the `zstd` crate maps them.
fn map_error_code(code: zstd_safe::ErrorCode) -> std::io::Error {
    std::io::Error::other(zstd_safe::get_error_name(code))
}

/// Compress `data` at compression `level`, with a checksum if `checksum`.
fn compress(
    data: &[u8],
    level: zstd_safe::CompressionLevel,
    checksum: bool,
    reuse_threshold: usize,
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
    if reuse_threshold == 0 || data.len() > reuse_threshold {
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
    options: ZstdCodecOptions,
}

impl ZstdCodec {
    /// Create a new `Zstd` codec.
    #[must_use]
    pub const fn new(compression: zstd_safe::CompressionLevel, checksum: bool) -> Self {
        Self {
            compression,
            checksum,
            options: ZstdCodecOptions::DEFAULT,
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
        Ok(Self::new(compression.into(), checksum))
    }

    /// Return a version of this codec with the provided runtime options.
    #[must_use]
    pub fn with_options(mut self, options: ZstdCodecOptions) -> Self {
        self.options = options;
        self
    }

    /// Return the runtime options.
    #[must_use]
    pub fn options(&self) -> &ZstdCodecOptions {
        &self.options
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

    fn with_codec_specific_options(
        mut self: Arc<Self>,
        opts: &CodecSpecificOptions,
    ) -> Result<Arc<dyn BytesToBytesCodecTraits>, CodecCreateError> {
        if let Some(options) = opts.get_option::<ZstdCodecOptions>() {
            Arc::make_mut(&mut self).options = *options;
        }
        Ok(self)
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
            self.options.compression_context_reuse_threshold(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::codec::{ShardingCodecBound, ShardingCodecBuilder};
    use crate::array::{ArrayBuilder, data_type};
    use crate::storage::store::MemoryStore;
    use std::num::NonZeroU64;

    fn options(threshold: usize) -> CodecSpecificOptions {
        CodecSpecificOptions::default().with_option(
            ZstdCodecOptions::default().with_compression_context_reuse_threshold(threshold),
        )
    }

    fn check_encode(codec: &dyn BytesToBytesCodecTraits, input: &[u8], reuse: bool) {
        COMPRESSOR.with(|slot| *slot.borrow_mut() = None);
        let options = CodecOptions::default();
        let encoded = codec.encode(CowBytes::Borrowed(input), &options).unwrap();
        COMPRESSOR.with(|slot| assert_eq!(slot.borrow().is_some(), reuse));
        let decoded = codec
            .decode(
                encoded,
                &BytesRepresentation::FixedSize(input.len() as u64),
                &options,
            )
            .unwrap();
        assert_eq!(&*decoded, input);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compression_context_reuse_threshold_boundaries() {
        for threshold in [0, 8, usize::MAX] {
            let codec = Arc::new(ZstdCodec::new(1, true))
                .with_codec_specific_options(&options(threshold))
                .unwrap();
            for len in [0, 7, 8, 9, 1024 * 1024 + 1] {
                check_encode(
                    codec.as_ref(),
                    &vec![42; len],
                    threshold != 0 && len <= threshold,
                );
            }
        }

        // Preserve the existing inclusive 1 MiB cutoff for both constructors.
        let configuration = ZstdCodecConfiguration::V1(ZstdCodecConfigurationV1 {
            level: 1.into(),
            checksum: false,
        });
        for codec in [
            ZstdCodec::new(1, false),
            ZstdCodec::new_with_configuration(&configuration).unwrap(),
        ] {
            for len in [1024 * 1024, 1024 * 1024 + 1] {
                check_encode(&codec, &vec![42; len], len <= 1024 * 1024);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compression_context_options_preserve_original_codec_and_metadata() {
        let codec =
            Arc::new(ZstdCodec::new(3, true).with_options(
                ZstdCodecOptions::default().with_compression_context_reuse_threshold(8),
            ));
        let inherited = codec
            .clone()
            .with_codec_specific_options(&CodecSpecificOptions::default())
            .unwrap();
        let overridden = codec
            .clone()
            .with_codec_specific_options(&options(0))
            .unwrap();
        check_encode(codec.as_ref(), b"12345678", true);
        check_encode(inherited.as_ref(), b"12345678", true);
        check_encode(inherited.as_ref(), b"123456789", false);
        check_encode(overridden.as_ref(), b"12345678", false);
        for version in [ZarrVersion::V2, ZarrVersion::V3] {
            assert_eq!(
                codec.configuration(version, &CodecMetadataOptions::default()),
                overridden.configuration(version, &CodecMetadataOptions::default())
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compression_context_options_reach_nested_codecs_and_rebind() {
        let codec = Arc::new(ZstdCodec::new(1, false));
        let sharding =
            ShardingCodecBuilder::new(vec![NonZeroU64::new(8).unwrap()], &data_type::uint8())
                .bytes_to_bytes_codecs(vec![codec.clone()])
                .index_bytes_to_bytes_codecs(vec![codec.clone()])
                .build_arc();
        let mut array = ArrayBuilder::new(vec![16], vec![16], data_type::uint8(), 0u8)
            .array_to_bytes_codec(sharding)
            .bytes_to_bytes_codecs(vec![codec])
            .codec_specific_options(options(8))
            .build(Arc::new(MemoryStore::new()), "/")
            .unwrap();

        for (opts, threshold) in [
            (None, 8),
            (Some(options(0)), 0),
            (Some(CodecSpecificOptions::default()), 1024 * 1024),
        ] {
            if let Some(opts) = opts {
                array.set_codec_specific_options(&opts).unwrap();
            }
            let chain = array.codecs_bound();
            let sharding = chain
                .array_to_bytes_codec()
                .as_any()
                .downcast_ref::<ShardingCodecBound>()
                .unwrap();
            for chain in [&chain, sharding.inner_codecs(), sharding.index_codecs()] {
                for len in [8, 9] {
                    check_encode(
                        chain.bytes_to_bytes_codecs()[0].as_ref(),
                        &vec![42; len],
                        threshold != 0 && len <= threshold,
                    );
                }
            }
        }
    }
}

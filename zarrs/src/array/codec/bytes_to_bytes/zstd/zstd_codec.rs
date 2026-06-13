use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};
use zstd::zstd_safe;

use super::{ZstdCodecConfiguration, ZstdCodecConfigurationV1};
use crate::array::{ArrayBytesRaw, BytesRepresentation};
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_metadata::Configuration;

thread_local! {
    static ZSTD_DECOMPRESSOR: RefCell<Option<zstd::bulk::Decompressor<'static>>> =
        const { RefCell::new(None) };
}

fn decompress_to_buffer(encoded_value: &[u8], output: &mut [u8]) -> std::io::Result<usize> {
    ZSTD_DECOMPRESSOR.with(|decompressor| {
        if let Ok(mut decompressor) = decompressor.try_borrow_mut() {
            if decompressor.is_none() {
                *decompressor = Some(zstd::bulk::Decompressor::new()?);
            }
            decompressor
                .as_mut()
                .unwrap()
                .decompress_to_buffer(encoded_value, output)
        } else {
            zstd::bulk::decompress_to_buffer(encoded_value, output)
        }
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

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
        decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let mut compressor = zstd::bulk::Compressor::new(self.compression)?;
        compressor.include_checksum(self.checksum)?;
        // compressor.include_contentsize(true);
        // compressor.set_pledged_src_size(Some(decoded_value.len()))?; // unpublished
        let result = compressor.compress(&decoded_value)?;
        Ok(Cow::Owned(result))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let upper_bound = zstd::bulk::Decompressor::upper_bound(&encoded_value); // requires zstd experimental feature
        if let Some(upper_bound) = upper_bound {
            // Bulk decompression
            let result = zstd::bulk::decompress(&encoded_value, upper_bound)?;
            Ok(Cow::Owned(result))
        } else {
            // Streaming decompression (slower)
            zstd::decode_all(std::io::Cursor::new(&encoded_value))
                .map_err(CodecError::from)
                .map(Cow::Owned)
        }
    }

    fn decode_into(
        &self,
        encoded_value: ArrayBytesRaw<'_>,
        decoded_representation: &BytesRepresentation,
        output: &mut [u8],
        _options: &CodecOptions,
    ) -> Result<usize, CodecError> {
        let decoded_len = decompress_to_buffer(&encoded_value, output)?;
        match decoded_representation {
            BytesRepresentation::FixedSize(size)
                if decoded_len != usize::try_from(*size).unwrap() =>
            {
                Err(zarrs_codec::InvalidBytesLengthError::new(
                    decoded_len,
                    usize::try_from(*size).unwrap(),
                )
                .into())
            }
            BytesRepresentation::BoundedSize(size)
                if decoded_len > usize::try_from(*size).unwrap() =>
            {
                Err(zarrs_codec::InvalidBytesLengthError::new(
                    decoded_len,
                    usize::try_from(*size).unwrap(),
                )
                .into())
            }
            BytesRepresentation::FixedSize(_)
            | BytesRepresentation::BoundedSize(_)
            | BytesRepresentation::UnboundedSize => Ok(decoded_len),
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

    #[test]
    fn decode_into() {
        let codec = ZstdCodec::new(1, false);
        let decoded = b"decode directly into this buffer".repeat(32);
        let encoded = codec
            .encode(Cow::Borrowed(&decoded), &CodecOptions::default())
            .unwrap();

        let mut output = vec![0; decoded.len()];
        assert_eq!(
            codec
                .decode_into(
                    encoded.clone(),
                    &BytesRepresentation::FixedSize(decoded.len() as u64),
                    &mut output,
                    &CodecOptions::default(),
                )
                .unwrap(),
            decoded.len()
        );
        assert_eq!(output, decoded);

        assert!(
            codec
                .decode_into(
                    encoded.clone(),
                    &BytesRepresentation::FixedSize(decoded.len() as u64 + 1),
                    &mut output,
                    &CodecOptions::default(),
                )
                .is_err()
        );
        assert!(
            codec
                .decode_into(
                    encoded.clone(),
                    &BytesRepresentation::BoundedSize(decoded.len() as u64 - 1),
                    &mut output,
                    &CodecOptions::default(),
                )
                .is_err()
        );
        assert!(
            codec
                .decode_into(
                    encoded,
                    &BytesRepresentation::BoundedSize(decoded.len() as u64),
                    &mut output[..decoded.len() - 1],
                    &CodecOptions::default(),
                )
                .is_err()
        );
    }

    #[test]
    fn decode_into_parallel() {
        let codec = Arc::new(ZstdCodec::new(1, false));
        let decoded = b"parallel decode into".repeat(128);
        let encoded = codec
            .encode(Cow::Borrowed(&decoded), &CodecOptions::default())
            .unwrap()
            .into_owned();

        std::thread::scope(|scope| {
            for _ in 0..4 {
                let codec = codec.clone();
                let encoded = &encoded;
                let decoded = &decoded;
                scope.spawn(move || {
                    let mut output = vec![0; decoded.len()];
                    codec
                        .decode_into(
                            Cow::Borrowed(encoded),
                            &BytesRepresentation::FixedSize(decoded.len() as u64),
                            &mut output,
                            &CodecOptions::default(),
                        )
                        .unwrap();
                    assert_eq!(&output, decoded);
                });
            }
        });
    }
}

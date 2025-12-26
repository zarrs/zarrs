use std::sync::Arc;

use super::test_unbounded_partial_decoder;
#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;
use crate::array::{
    ArrayBytesRaw, BytesRepresentation,
    codec::{
        BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError, CodecMetadataOptions,
        CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
        RecommendedConcurrency,
    },
};
use crate::metadata::Configuration;

/// A `test_unbounded` codec implementation.
#[derive(Clone, Debug)]
pub struct TestUnboundedCodec {}

impl TestUnboundedCodec {
    /// Create a new `test_unbounded` codec.
    ///
    /// # Errors
    /// Returns [`TestUnboundedCompressionLevelError`] if `compression_level` is not valid.
    pub fn new() -> Self {
        Self {}
    }
}

impl CodecTraits for TestUnboundedCodec {
    fn identifier(&self) -> &'static str {
        "zarrs.test_unbounded"
    }

    fn configuration(&self, _name: &str, _options: &CodecMetadataOptions) -> Option<Configuration> {
        None
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: true,
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
impl BytesToBytesCodecTraits for TestUnboundedCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    /// Return the maximum internal concurrency supported for the requested decoded representation.
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
        Ok(decoded_value)
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        Ok(encoded_value)
    }

    fn partial_decoder(
        self: Arc<Self>,
        r: Arc<dyn BytesPartialDecoderTraits>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            test_unbounded_partial_decoder::TestUnboundedPartialDecoder::new(r),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        r: Arc<dyn AsyncBytesPartialDecoderTraits>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            test_unbounded_partial_decoder::AsyncTestUnboundedPartialDecoder::new(r),
        ))
    }

    fn encoded_representation(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        BytesRepresentation::UnboundedSize
    }
}

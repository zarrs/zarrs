use std::sync::Arc;

use zarrs_plugin::ZarrVersion;

#[cfg(feature = "async")]
use super::test_unbounded_partial_decoder::AsyncTestUnboundedPartialDecoder;
use super::test_unbounded_partial_decoder::TestUnboundedPartialDecoder;
use crate::array::{ArrayBytesRaw, BytesRepresentation};
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialEncoderTraits;
use zarrs_codec::{
    AsyncBytesPartialDecoderTraits, BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError,
    CodecMetadataOptions, CodecOptions, CodecTraits, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use zarrs_codec::{BytesPartialEncoderTraits, BytesToBytesCodecPartialDefault};
use zarrs_metadata::Configuration;

zarrs_plugin::impl_extension_aliases!(TestUnboundedCodec, v3: "zarrs.test_unbounded");

/// A `test_unbounded` codec implementation.
#[derive(Clone, Debug)]
pub struct TestUnboundedCodec {}

impl TestUnboundedCodec {
    /// Create a new `test_unbounded` codec.
    ///
    /// # Errors
    /// Returns [`TestUnboundedCompressionLevelError`] if `compression_level` is not valid.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TestUnboundedCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecTraits for TestUnboundedCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
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

#[ambisync::paired(
    sync(fns("async_{}"), types("Async{}")),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
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

    async fn async_partial_decoder(
        self: Arc<Self>,
        r: Arc<dyn AsyncBytesPartialDecoderTraits>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(AsyncTestUnboundedPartialDecoder::new(r)))
    }

    fn encoded_representation(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        BytesRepresentation::UnboundedSize
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

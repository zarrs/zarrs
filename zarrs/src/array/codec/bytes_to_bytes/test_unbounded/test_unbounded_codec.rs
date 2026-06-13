use std::sync::Arc;

use zarrs_plugin::ZarrVersion;

use super::test_unbounded_partial_decoder;
use crate::array::{ArrayBytesRaw, BytesRepresentation};
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{
    BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
};
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    #[test]
    fn default_decode_into() {
        let codec = TestUnboundedCodec::new();
        let options = CodecOptions::default();
        let encoded = Cow::Borrowed(&b"decoded"[..]);
        let mut output = [0; 7];

        assert_eq!(
            codec
                .decode_into(
                    encoded.clone(),
                    &BytesRepresentation::FixedSize(7),
                    &mut output,
                    &options,
                )
                .unwrap(),
            7
        );
        assert_eq!(&output, b"decoded");
        assert!(
            codec
                .decode_into(
                    encoded.clone(),
                    &BytesRepresentation::FixedSize(8),
                    &mut [0; 8],
                    &options,
                )
                .is_err()
        );
        assert!(
            codec
                .decode_into(
                    encoded.clone(),
                    &BytesRepresentation::BoundedSize(6),
                    &mut output,
                    &options,
                )
                .is_err()
        );
        assert!(
            codec
                .decode_into(
                    encoded.clone(),
                    &BytesRepresentation::BoundedSize(7),
                    &mut [0; 6],
                    &options,
                )
                .is_err()
        );
        assert_eq!(
            codec
                .decode_into(
                    encoded,
                    &BytesRepresentation::UnboundedSize,
                    &mut output,
                    &options,
                )
                .unwrap(),
            7
        );
    }
}

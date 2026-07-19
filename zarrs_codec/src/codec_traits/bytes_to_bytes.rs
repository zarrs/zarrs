use std::sync::Arc;

use crate::{
    ArrayBytesRaw, BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesRepresentation,
    BytesToBytesCodecPartialDefault, CodecCreateError, CodecError, CodecOptions,
    CodecSpecificOptions, CodecTraits, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use crate::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};

/// Traits for bytes to bytes codecs.
#[ambisync::paired(
    sync(fns("async_{}"), types("Async{}")),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
pub trait BytesToBytesCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits>;

    /// Return a version of this codec reconfigured with the provided codec-specific options.
    ///
    /// The default implementation returns the codec unchanged.
    /// Override this to read your codec's options type from [`CodecSpecificOptions`].
    #[expect(unused_variables)]
    fn with_codec_specific_options(
        self: Arc<Self>,
        opts: &CodecSpecificOptions,
    ) -> Result<Arc<dyn BytesToBytesCodecTraits>, CodecCreateError> {
        Ok(self.into_dyn())
    }

    /// Return the maximum internal concurrency supported for the requested decoded representation.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the decoded representation is not valid for the codec.
    fn recommended_concurrency(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError>;

    /// Returns the size of the encoded representation given a size of the decoded representation.
    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation;

    /// Encode chunk bytes.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn encode<'a>(
        &self,
        decoded_value: ArrayBytesRaw<'a>,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError>;

    /// Decode chunk bytes.
    //
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError>;

    /// Initialises a partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
            input_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }

    /// Initialise a partial encoder.
    ///
    /// The default implementation reencodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
            input_output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }
}

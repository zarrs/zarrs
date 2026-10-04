use std::sync::Arc;

#[cfg(feature = "async")]
use crate::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};
use crate::{
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesRepresentation,
    BytesToBytesCodecPartialDefault, CodecCreateError, CodecError, CodecOptions,
    CodecSpecificOptions, CodecTraits, CowBytes, InvalidBytesLengthError, RecommendedConcurrency,
};

/// Traits for bytes to bytes codecs.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait BytesToBytesCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits>;

    /// Return a version of this codec reconfigured with the provided codec-specific options.
    ///
    /// This is applied when a codec chain containing this codec is bound.
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
        decoded_value: CowBytes<'a>,
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError>;

    /// Decode chunk bytes.
    //
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError>;

    /// Decode chunk bytes into a preallocated output buffer.
    ///
    /// The decoded bytes must fill `output` exactly.
    /// On error, `output` may have been partially written.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded length is not the length of `output`.
    fn decode_into(
        &self,
        encoded_value: CowBytes<'_>,
        decoded_representation: &BytesRepresentation,
        output: &mut [u8],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let decoded_value = self.decode(encoded_value, decoded_representation, options)?;
        Ok(copy_decoded_bytes_into(&decoded_value, output)?)
    }

    /// Initialises a partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialDecoderTraits>, CodecError> {
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
    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
            input_output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialises an asynchronous partial decoder.
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

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial encoder.
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

/// Copy `decoded_value` into `output`, which must be the same length.
///
/// This is for implementations of [`BytesToBytesCodecTraits::decode_into`] that decode into an allocation.
///
/// # Errors
/// Returns an [`InvalidBytesLengthError`] if the length of `decoded_value` is not the length of `output`.
pub fn copy_decoded_bytes_into(
    decoded_value: &[u8],
    output: &mut [u8],
) -> Result<(), InvalidBytesLengthError> {
    if decoded_value.len() != output.len() {
        return Err(InvalidBytesLengthError::new(
            decoded_value.len(),
            output.len(),
        ));
    }
    output.copy_from_slice(decoded_value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_decoded_bytes_into_length() {
        let mut output = [0; 3];
        copy_decoded_bytes_into(&[1, 2, 3], &mut output).unwrap();
        assert_eq!(output, [1, 2, 3]);
        assert!(copy_decoded_bytes_into(&[1, 2], &mut output).is_err());
        assert!(copy_decoded_bytes_into(&[1, 2, 3, 4], &mut output).is_err());
        assert_eq!(output, [1, 2, 3]);
    }
}

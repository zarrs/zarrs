use std::mem::MaybeUninit;
use std::sync::Arc;

#[cfg(feature = "async")]
use crate::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};
use crate::{
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesRepresentation,
    BytesToBytesCodecPartialDefault, CodecCreateError, CodecError, CodecOptions,
    CodecSpecificOptions, CodecTraits, ComputeResources, CowBytes, InvalidBytesLengthError,
    RecommendedConcurrency, Resources, write_copy_of_slice,
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
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the codec cannot be reconfigured with `opts`.
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
        resources: &ComputeResources,
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
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError>;

    /// Return the direct decoder of this codec if it can decode into a preallocated output.
    ///
    /// A direct decoder avoids a full-size decoded intermediate allocation and its separate copy.
    /// It does not promise to avoid decoder state or workspace allocations.
    ///
    /// The default implementation returns [`None`].
    /// Implement [`BytesToBytesCodecDecodeInto`] and return `Some(self)` to decode directly.
    fn as_decode_into(&self) -> Option<&dyn BytesToBytesCodecDecodeInto> {
        None
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
        resources: &Resources,
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
        resources: &Resources,
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
        resources: &Resources,
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
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncBytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
            input_output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }
}

/// Direct decoding of a bytes to bytes codec into a preallocated output.
///
/// Return a codec from [`BytesToBytesCodecTraits::as_decode_into`] to use it.
/// Callers usually decode through [`BytesDecodeSource::decode_into`](crate::BytesDecodeSource::decode_into),
/// which falls back to [`BytesToBytesCodecTraits::decode`] for codecs without a direct decoder.
///
/// # Safety
/// Callers may assume that `output` is initialised when [`decode_into`](Self::decode_into) returns `Ok`, so an implementation must:
/// - initialise every byte of `output` before returning `Ok`, and
/// - never write uninitialised values to `output`.
pub unsafe trait BytesToBytesCodecDecodeInto: BytesToBytesCodecTraits {
    /// Decode chunk bytes into a preallocated output buffer.
    ///
    /// `output` may be uninitialised.
    /// The decoded bytes must fill `output` exactly, so every byte of `output` is initialised on success.
    /// On error, `output` may have been partially written.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded length is not the length of `output`.
    fn decode_into(
        &self,
        encoded_value: CowBytes<'_>,
        decoded_representation: &BytesRepresentation,
        output: &mut [MaybeUninit<u8>],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError>;
}

/// Copy `decoded_value` into `output`, which must be the same length.
///
/// This is for implementations of [`BytesToBytesCodecDecodeInto::decode_into`] that decode into an allocation.
///
/// # Errors
/// Returns an [`InvalidBytesLengthError`] if the length of `decoded_value` is not the length of `output`.
pub fn copy_decoded_bytes_into(
    decoded_value: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> Result<(), InvalidBytesLengthError> {
    if decoded_value.len() != output.len() {
        return Err(InvalidBytesLengthError::new(
            decoded_value.len(),
            output.len(),
        ));
    }
    write_copy_of_slice(output, decoded_value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn copy_decoded_bytes_into_length() {
        let mut output = [MaybeUninit::new(0); 3];
        copy_decoded_bytes_into(&[1, 2, 3], &mut output).unwrap();
        // SAFETY: the output is initialised.
        assert_eq!(output.map(|byte| unsafe { byte.assume_init() }), [1, 2, 3]);
        assert!(copy_decoded_bytes_into(&[1, 2], &mut output).is_err());
        assert!(copy_decoded_bytes_into(&[1, 2, 3, 4], &mut output).is_err());
        // SAFETY: the output is initialised.
        assert_eq!(output.map(|byte| unsafe { byte.assume_init() }), [1, 2, 3]);
    }
}

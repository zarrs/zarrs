use std::mem::MaybeUninit;
use std::sync::Arc;

#[cfg(feature = "async")]
use crate::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};
use crate::{
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesRepresentation,
    BytesToBytesCodecPartialDefault, CodecCreateError, CodecError, CodecOptions,
    CodecSpecificOptions, CodecTraits, ComputeResources, CowBytes, InvalidBytesLengthError,
    RecommendedConcurrency, Resources, slice_assume_init_mut, write_copy_of_slice,
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

/// Read the decoded bytes of `decoded` into `output`, which must be exactly the length of the decoded bytes.
///
/// This is for implementations of [`BytesToBytesCodecDecodeInto::decode_into`] that decode through a [`Read`](std::io::Read) decoder.
/// It stops at the first decoded byte past the end of `output` rather than decoding the remainder, so a stream that is much larger than `output` is not expanded.
///
/// # Errors
/// Returns an [`InvalidBytesLengthError`] if the length of the decoded bytes is not the length of `output`, or an IO error if `decoded` fails to read.
/// If the decoded bytes are too long, the reported length is one more than the length of `output`.
pub fn read_decoded_bytes_into(
    decoded: impl std::io::Read,
    output: &mut [MaybeUninit<u8>],
) -> Result<(), CodecError> {
    // `Read` requires initialised bytes
    output.fill(MaybeUninit::new(0));
    // SAFETY: `output` was initialised above.
    let output = unsafe { slice_assume_init_mut(output) };
    read_decoded_bytes_into_init(decoded, output)
}

/// [`read_decoded_bytes_into`] for an initialised `output`.
fn read_decoded_bytes_into_init(
    mut decoded: impl std::io::Read,
    output: &mut [u8],
) -> Result<(), CodecError> {
    let mut filled = 0;
    while filled < output.len() {
        match decoded.read(&mut output[filled..]) {
            Ok(0) => return Err(InvalidBytesLengthError::new(filled, output.len()).into()),
            Ok(len) => filled += len,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err.into()),
        }
    }
    let mut extra = [0u8; 1];
    loop {
        match decoded.read(&mut extra) {
            Ok(0) => return Ok(()),
            Ok(_) => {
                return Err(InvalidBytesLengthError::new(
                    output.len().saturating_add(1),
                    output.len(),
                )
                .into());
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err.into()),
        }
    }
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

    #[test]
    fn read_decoded_bytes_into_length() {
        let mut output = [0; 3];
        read_decoded_bytes_into_init(&[1, 2, 3][..], &mut output).unwrap();
        assert_eq!(output, [1, 2, 3]);

        // The length of the decoded bytes is reported on a mismatch
        let invalid_len =
            |decoded: &[u8], output: &mut [u8]| match read_decoded_bytes_into_init(decoded, output)
            {
                Err(CodecError::UnexpectedChunkDecodedSize(err)) => err.to_string(),
                result => panic!("unexpected result {result:?}"),
            };
        assert_eq!(
            invalid_len(&[1, 2], &mut output),
            InvalidBytesLengthError::new(2, 3).to_string()
        );
        assert_eq!(
            invalid_len(&[1, 2, 3, 4, 5], &mut output),
            InvalidBytesLengthError::new(4, 3).to_string()
        );
        assert_eq!(
            invalid_len(&[], &mut output),
            InvalidBytesLengthError::new(0, 3).to_string()
        );
        read_decoded_bytes_into_init(&[][..], &mut []).unwrap();
    }

    #[test]
    fn read_decoded_bytes_into_does_not_read_past_the_first_extra_byte() {
        // An endless stream is an error rather than being read to the end
        let mut output = [0; 3];
        assert!(matches!(
            read_decoded_bytes_into_init(std::io::repeat(7), &mut output),
            Err(CodecError::UnexpectedChunkDecodedSize(_))
        ));
        assert_eq!(output, [7; 3]);
    }

    #[test]
    fn read_decoded_bytes_into_read_error() {
        struct Failing;
        impl std::io::Read for Failing {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("failed"))
            }
        }
        assert!(matches!(
            read_decoded_bytes_into_init(Failing, &mut [0; 3]),
            Err(CodecError::IOError(_))
        ));
    }

    #[test]
    fn read_decoded_bytes_into_retries_interruptions_in_fill_and_probe() {
        struct Choppy<'a> {
            bytes: &'a [u8],
            calls: usize,
        }
        impl std::io::Read for Choppy<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                let call = self.calls;
                self.calls += 1;
                if call == 0 || call == 4 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if let Some((&byte, rest)) = self.bytes.split_first() {
                    output[0] = byte;
                    self.bytes = rest;
                    Ok(1)
                } else {
                    Ok(0)
                }
            }
        }
        let mut output = [0; 3];
        read_decoded_bytes_into_init(
            Choppy {
                bytes: &[1, 2, 3],
                calls: 0,
            },
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [1, 2, 3]);
    }

    #[test]
    fn read_decoded_bytes_into_uninit() {
        let mut output = [MaybeUninit::uninit(); 3];
        read_decoded_bytes_into(&[1, 2, 3][..], &mut output).unwrap();
        // SAFETY: a successful read initialises the output.
        assert_eq!(output.map(|byte| unsafe { byte.assume_init() }), [1, 2, 3]);
        assert!(read_decoded_bytes_into(&[1, 2][..], &mut output).is_err());
    }
}

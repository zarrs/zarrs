use crate::{BytesRepresentation, BytesToBytesCodecTraits, CodecError, CodecOptions, CowBytes};

/// Immediate or deferred encoded input for an array-to-bytes decoder.
#[derive(Debug)]
pub enum ArrayBytesDecodeIntoInput<'a> {
    /// Already-produced encoded bytes.
    Bytes(CowBytes<'a>),
    /// One pending bytes-to-bytes decode.
    Deferred(BytesDecodeSource<'a>),
}

impl<'a> From<CowBytes<'a>> for ArrayBytesDecodeIntoInput<'a> {
    fn from(bytes: CowBytes<'a>) -> Self {
        Self::Bytes(bytes)
    }
}

impl<'a> ArrayBytesDecodeIntoInput<'a> {
    /// Resolve this input without cloning its bytes.
    ///
    /// # Errors
    /// Returns the producer's decoding error.
    pub fn into_bytes(self, options: &CodecOptions) -> Result<CowBytes<'a>, CodecError> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Deferred(source) => source.decode(options),
        }
    }
}

/// A single-use bytes-to-bytes decode whose output placement is not yet chosen.
///
/// Both decoding routes consume the source, retaining ownership of its encoded
/// bytes and ensuring that only one route can execute.
///
/// Sources cannot be cloned:
/// ```compile_fail
/// use zarrs_codec::BytesDecodeSource;
/// fn duplicate(source: BytesDecodeSource<'_>) {
///     let _ = source.clone();
/// }
/// ```
///
/// Nor can a consumed source be decoded again:
/// ```compile_fail
/// use zarrs_codec::{BytesDecodeSource, CodecOptions};
/// fn twice(source: BytesDecodeSource<'_>, options: &CodecOptions) {
///     let _ = source.decode(options);
///     let _ = source.decode(options);
/// }
/// ```
#[derive(Debug)]
pub struct BytesDecodeSource<'a> {
    codec: &'a dyn BytesToBytesCodecTraits,
    bytes: CowBytes<'a>,
    decoded_representation: BytesRepresentation,
}

impl<'a> BytesDecodeSource<'a> {
    /// Construct a source without cloning the encoded bytes.
    #[must_use]
    pub fn new(
        codec: &'a dyn BytesToBytesCodecTraits,
        bytes: CowBytes<'a>,
        decoded_representation: BytesRepresentation,
    ) -> Self {
        Self {
            codec,
            bytes,
            decoded_representation,
        }
    }

    /// Return the expected decoded representation.
    #[must_use]
    pub fn decoded_representation(&self) -> &BytesRepresentation {
        &self.decoded_representation
    }

    /// Whether the producer avoids a full-size decoded intermediate and copy.
    ///
    /// This does not promise to avoid decoder state or workspace allocations.
    #[must_use]
    pub fn is_decode_into_efficient(&self) -> bool {
        self.codec.is_decode_into_efficient()
    }

    /// Consume this source and decode normally.
    ///
    /// # Errors
    /// Returns the producer's decoding error.
    pub fn decode(self, options: &CodecOptions) -> Result<CowBytes<'a>, CodecError> {
        self.codec
            .decode(self.bytes, &self.decoded_representation, options)
    }

    /// Consume this source and decode into `output`.
    ///
    /// On error, `output` may have been partially written.
    ///
    /// # Errors
    /// Returns a decoding error, including when decoded bytes do not fill `output`.
    pub fn decode_into(self, output: &mut [u8], options: &CodecOptions) -> Result<(), CodecError> {
        self.codec
            .decode_into(self.bytes, &self.decoded_representation, output, options)
    }
}

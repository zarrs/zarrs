use std::borrow::Cow;

use zarrs_metadata::Endianness;

/// Traits for a data type extension supporting the `bytes` codec.
pub trait DataTypeExtensionBytesCodec {
    /// Encode the bytes of a fixed-size data type to a specified endianness for the `bytes` codec.
    ///
    /// Returns the input bytes unmodified for fixed-size data where endianness is not applicable (i.e. the bytes are serialised directly from the in-memory representation).
    ///
    /// # Errors
    /// Returns a [`DataTypeExtensionBytesCodecError`] if `endianness` is [`None`] but must be specified.
    #[allow(unused_variables)]
    fn encode<'a>(
        &self,
        bytes: Cow<'a, [u8]>,
        endianness: Option<Endianness>,
    ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError>;

    /// Decode the bytes of a fixed-size data type from a specified endianness for the `bytes` codec.
    ///
    /// This performs the inverse operation of [`encode`](DataTypeExtensionBytesCodec::encode).
    ///
    /// # Errors
    /// Returns a [`DataTypeExtensionBytesCodecError`] if `endianness` is [`None`] but must be specified.
    #[allow(unused_variables)]
    fn decode<'a>(
        &self,
        bytes: Cow<'a, [u8]>,
        endianness: Option<Endianness>,
    ) -> Result<Cow<'a, [u8]>, DataTypeExtensionBytesCodecError>;
}

/// A data type extension error related to the `bytes` codec.
#[derive(Debug, Clone, thiserror::Error, derive_more::From, derive_more::Display)]
#[non_exhaustive]
pub enum DataTypeExtensionBytesCodecError {
    /// The endianness was not specified, and it is required for this data type extension.
    EndiannessNotSpecified,
}

// Generate the codec support infrastructure using the generic macro
crate::define_codec_support!(Bytes, DataTypeExtensionBytesCodec);

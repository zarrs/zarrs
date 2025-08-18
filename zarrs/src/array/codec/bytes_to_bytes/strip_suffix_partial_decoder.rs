use std::{borrow::Cow, sync::Arc};

use crate::{
    array::{
        codec::{BytesPartialDecoderTraits, CodecError, CodecOptions},
        RawBytes,
    },
    storage::byte_range::ByteRange,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;

/// Partial decoder for stripping a suffix (e.g. checksum).
pub(crate) struct StripSuffixPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    suffix_size: usize,
}

impl StripSuffixPartialDecoder {
    /// Create a new "strip suffix" partial decoder.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        suffix_size: usize,
    ) -> Self {
        Self {
            input_handle,
            suffix_size,
        }
    }
}

impl BytesPartialDecoderTraits for StripSuffixPartialDecoder {
    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let bytes = self.input_handle.partial_decode(decoded_regions, options)?;
        let Some(bytes) = bytes else {
            return Ok(None);
        };

        // Drop suffix of length `suffix_size`
        let mut output = Vec::with_capacity(bytes.len());
        for (bytes, byte_range) in bytes.into_iter().zip(decoded_regions) {
            let bytes = match byte_range {
                ByteRange::FromStart(_, Some(_)) => bytes,
                ByteRange::FromStart(_, None) => {
                    let length = bytes.len() - self.suffix_size;
                    Cow::Owned(bytes[..length].to_vec())
                }
                ByteRange::Suffix(_) => {
                    let length = bytes.len() as u64 - (self.suffix_size as u64);
                    let length = usize::try_from(length).unwrap();
                    Cow::Owned(bytes[..length].to_vec())
                }
            };
            output.push(bytes);
        }

        Ok(Some(output))
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for stripping a suffix (e.g. checksum).
pub(crate) struct AsyncStripSuffixPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    suffix_size: usize,
}

#[cfg(feature = "async")]
impl AsyncStripSuffixPartialDecoder {
    /// Create a new "strip suffix" partial decoder.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        suffix_size: usize,
    ) -> Self {
        Self {
            input_handle,
            suffix_size,
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncBytesPartialDecoderTraits for AsyncStripSuffixPartialDecoder {
    async fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let bytes = self
            .input_handle
            .partial_decode(decoded_regions, options)
            .await?;
        let Some(bytes) = bytes else {
            return Ok(None);
        };

        // Drop trailing checksum
        let mut output = Vec::with_capacity(bytes.len());
        for (bytes, byte_range) in bytes.into_iter().zip(decoded_regions) {
            let bytes = match byte_range {
                ByteRange::FromStart(_, Some(_)) => bytes,
                ByteRange::FromStart(_, None) => {
                    let length = bytes.len() - self.suffix_size;
                    Cow::Owned(bytes[..length].to_vec())
                }
                ByteRange::Suffix(_) => {
                    let length = bytes.len() as u64 - (self.suffix_size as u64);
                    let length = usize::try_from(length).unwrap();
                    Cow::Owned(bytes[..length].to_vec())
                }
            };
            output.push(bytes);
        }

        Ok(Some(output))
    }
}

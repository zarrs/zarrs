use std::sync::Arc;

use crate::{
    array::RawBytes,
    storage::byte_range::{ByteLength, ByteOffset, ByteRange, ByteRangeIterator},
};

use super::{BytesPartialDecoderTraits, CodecError, CodecOptions};

#[cfg(feature = "async")]
use super::AsyncBytesPartialDecoderTraits;

/// A partial decoder for a byte interval of a [`BytesPartialDecoderTraits`] partial decoder.
///
/// Modifies byte range requests to a specific byte interval in an inner bytes partial decoder.
pub struct ByteIntervalPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
}

impl ByteIntervalPartialDecoder {
    /// Create a new byte interval partial decoder.
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        byte_offset: ByteOffset,
        byte_length: ByteLength,
    ) -> Self {
        Self {
            input_handle,
            byte_offset,
            byte_length,
        }
    }
}

impl BytesPartialDecoderTraits for ByteIntervalPartialDecoder {
    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode_many(
        &self,
        byte_ranges: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let byte_ranges = byte_ranges.map(|byte_range| match byte_range {
            ByteRange::FromStart(offset, None) => {
                ByteRange::FromStart(self.byte_offset + offset, Some(self.byte_length))
            }
            ByteRange::FromStart(offset, Some(length)) => {
                ByteRange::FromStart(self.byte_offset + offset, Some(length))
            }
            ByteRange::Suffix(length) => {
                ByteRange::FromStart(self.byte_offset + self.byte_length - length, Some(length))
            }
        });
        self.input_handle
            .partial_decode_many(Box::new(byte_ranges), options)
    }
}

#[cfg(feature = "async")]
/// A partial decoder for a byte interval of a [`AsyncBytesPartialDecoderTraits`] partial decoder.
///
/// Modifies byte range requests to a specific byte interval in an inner bytes partial decoder.
pub struct AsyncByteIntervalPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
}

#[cfg(feature = "async")]
impl AsyncByteIntervalPartialDecoder {
    /// Create a new byte interval partial decoder.
    pub fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        byte_offset: ByteOffset,
        byte_length: ByteLength,
    ) -> Self {
        Self {
            input_handle,
            byte_offset,
            byte_length,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncByteIntervalPartialDecoder {
    async fn partial_decode_many<'a>(
        &'a self,
        byte_ranges: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        let byte_ranges = byte_ranges.map(|byte_range| match byte_range {
            ByteRange::FromStart(offset, None) => {
                ByteRange::FromStart(self.byte_offset + offset, Some(self.byte_length))
            }
            ByteRange::FromStart(offset, Some(length)) => {
                ByteRange::FromStart(self.byte_offset + offset, Some(length))
            }
            ByteRange::Suffix(length) => {
                ByteRange::FromStart(self.byte_offset + self.byte_length - length, Some(length))
            }
        });
        self.input_handle
            .partial_decode_many(Box::new(byte_ranges), options)
            .await
    }
}

use std::sync::Arc;

use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange, ByteRangeIterator};

use crate::{CodecError, CodecOptions, CowBytes};

#[cfg(feature = "async")]
use crate::AsyncBytesPartialDecoderTraits;
use crate::BytesPartialDecoderTraits;

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
    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode_many(
        &self,
        byte_ranges: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        let byte_ranges = byte_ranges.map(|byte_range| match byte_range {
            ByteRange::FromStart(offset, None) => {
                ByteRange::FromStart(self.byte_offset + offset, Some(self.byte_length - offset))
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

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
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
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        byte_ranges: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
        let byte_ranges = byte_ranges.map(|byte_range| match byte_range {
            ByteRange::FromStart(offset, None) => {
                ByteRange::FromStart(self.byte_offset + offset, Some(self.byte_length - offset))
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

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(byte_range: ByteRange) -> Vec<u8> {
        let input: Arc<dyn BytesPartialDecoderTraits> = Arc::new((0..10u8).collect::<Vec<u8>>());
        let decoder = ByteIntervalPartialDecoder::new(input, 2, 5);
        decoder
            .partial_decode(byte_range, &CodecOptions::default())
            .unwrap()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn byte_interval_partial_decoder() {
        assert_eq!(decode(ByteRange::FromStart(0, None)), vec![2, 3, 4, 5, 6]);
        assert_eq!(decode(ByteRange::FromStart(1, None)), vec![3, 4, 5, 6]);
        assert_eq!(decode(ByteRange::FromStart(1, Some(2))), vec![3, 4]);
        assert_eq!(decode(ByteRange::Suffix(2)), vec![5, 6]);
    }

    #[cfg(feature = "async")]
    fn async_decode(byte_range: ByteRange) -> Vec<u8> {
        let input: Arc<dyn AsyncBytesPartialDecoderTraits> =
            Arc::new((0..10u8).collect::<Vec<u8>>());
        let decoder = AsyncByteIntervalPartialDecoder::new(input, 2, 5);
        futures::executor::block_on(decoder.partial_decode(byte_range, &CodecOptions::default()))
            .unwrap()
            .unwrap()
            .to_vec()
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_byte_interval_partial_decoder() {
        assert_eq!(
            async_decode(ByteRange::FromStart(0, None)),
            vec![2, 3, 4, 5, 6]
        );
        assert_eq!(
            async_decode(ByteRange::FromStart(1, None)),
            vec![3, 4, 5, 6]
        );
        assert_eq!(async_decode(ByteRange::FromStart(1, Some(2))), vec![3, 4]);
        assert_eq!(async_decode(ByteRange::Suffix(2)), vec![5, 6]);
    }
}

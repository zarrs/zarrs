use std::sync::Arc;

use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange, ByteRangeIterator};

use crate::{ArrayBytesRaw, CodecError, CodecOptions};

#[cfg(feature = "async")]
use crate::AsyncBytesPartialDecoderTraits;
use crate::BytesPartialDecoderTraits;

ambisync::scoped! {
#![defaults(
    sync(fns("{}"), types("Async{}")),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]

#[ambisync]
/// A partial decoder for a byte interval of a [`AsyncBytesPartialDecoderTraits`] partial decoder.
///
/// Modifies byte range requests to a specific byte interval in an inner bytes partial decoder.
pub struct AsyncByteIntervalPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
}

#[ambisync]
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

#[ambisync]
impl AsyncBytesPartialDecoderTraits for AsyncByteIntervalPartialDecoder {
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        byte_ranges: ByteRangeIterator<'_>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
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

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

}

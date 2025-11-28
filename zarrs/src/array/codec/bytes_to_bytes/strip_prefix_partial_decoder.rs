use std::sync::Arc;

use zarrs_storage::StorageError;

use crate::{
    array::{
        codec::{BytesPartialDecoderTraits, CodecError, CodecOptions},
        ArrayBytesRaw,
    },
    storage::byte_range::{ByteRange, ByteRangeIterator},
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;

/// Partial decoder for stripping a prefix (e.g. checksum).
pub(crate) struct StripPrefixPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    prefix_size: usize,
}

impl StripPrefixPartialDecoder {
    /// Create a new "strip prefix" partial decoder.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        prefix_size: usize,
    ) -> Self {
        Self {
            input_handle,
            prefix_size,
        }
    }
}

impl BytesPartialDecoderTraits for StripPrefixPartialDecoder {
    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        let decoded_regions = decoded_regions.map(|range| match range {
            ByteRange::FromStart(offset, length) => {
                ByteRange::FromStart(offset.checked_add(self.prefix_size as u64).unwrap(), length)
            }
            ByteRange::Suffix(length) => ByteRange::Suffix(length),
        });

        self.input_handle
            .partial_decode_many(Box::new(decoded_regions), options)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for stripping a prefix (e.g. checksum).
pub(crate) struct AsyncStripPrefixPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    prefix_size: usize,
}

#[cfg(feature = "async")]
impl AsyncStripPrefixPartialDecoder {
    /// Create a new "strip prefix" partial decoder.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        prefix_size: usize,
    ) -> Self {
        Self {
            input_handle,
            prefix_size,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncStripPrefixPartialDecoder {
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        let decoded_regions = decoded_regions.map(|range| match range {
            ByteRange::FromStart(offset, length) => {
                ByteRange::FromStart(offset.checked_add(self.prefix_size as u64).unwrap(), length)
            }
            ByteRange::Suffix(length) => ByteRange::Suffix(length),
        });

        self.input_handle
            .partial_decode_many(Box::new(decoded_regions), options)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

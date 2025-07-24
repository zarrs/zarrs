use std::sync::Arc;

use itertools::Itertools;
use zarrs_storage::byte_range::ByteRangeIndexer;

use crate::{
    array::{
        codec::{BytesPartialDecoderTraits, CodecError, CodecOptions},
        RawBytes,
    },
    byte_range::ByteRange,
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
    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        decoded_regions: &dyn ByteRangeIndexer,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let decoded_regions = decoded_regions
            .iter()
            .map(|range| match range {
                ByteRange::FromStart(offset, length) => ByteRange::FromStart(
                    offset.checked_add(self.prefix_size as u64).unwrap(),
                    *length,
                ),
                ByteRange::Suffix(length) => ByteRange::Suffix(*length),
            });

        self.input_handle.partial_decode(&decoded_regions.collect::<Vec<ByteRange>>(), options)
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
#[async_trait::async_trait]
impl AsyncBytesPartialDecoderTraits for AsyncStripPrefixPartialDecoder {
    async fn partial_decode(
        &self,
        decoded_regions: &dyn ByteRangeIndexer,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let decoded_regions = decoded_regions
            .iter()
            .map(|range| match range {
                ByteRange::FromStart(offset, length) => ByteRange::FromStart(
                    offset.checked_add(self.prefix_size as u64).unwrap(),
                    *length,
                ),
                ByteRange::Suffix(length) => ByteRange::Suffix(*length),
            }).collect::<Vec<ByteRange>>();

        self.input_handle
            .partial_decode(&decoded_regions, options)
            .await
    }
}

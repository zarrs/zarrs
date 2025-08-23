use std::sync::Arc;

use zarrs_shared::{MaybeSend, MaybeSync};

use crate::{
    array::{
        codec::{BytesPartialDecoderTraits, CodecError, CodecOptions},
        RawBytes,
    },
    storage::byte_range::ByteRange,
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
        decoded_regions: &mut dyn ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let mut decoded_regions = decoded_regions.map(|range| match range {
            ByteRange::FromStart(offset, length) => {
                ByteRange::FromStart(offset.checked_add(self.prefix_size as u64).unwrap(), length)
            }
            ByteRange::Suffix(length) => ByteRange::Suffix(length),
        });

        self.input_handle
            .partial_decode(&mut decoded_regions, options)
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
        decoded_regions: &mut dyn ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let mut decoded_regions = decoded_regions.map(|range| match range {
            ByteRange::FromStart(offset, length) => {
                ByteRange::FromStart(offset.checked_add(self.prefix_size as u64).unwrap(), length)
            }
            ByteRange::Suffix(length) => ByteRange::Suffix(length),
        });

        self.input_handle
            .partial_decode(&mut decoded_regions, options)
            .await
    }
}

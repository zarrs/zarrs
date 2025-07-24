//! A cache for partial decoders.

use std::borrow::Cow;

use zarrs_storage::byte_range::ByteRangeIndexer;

use crate::{
    array::RawBytes,
    byte_range::{extract_byte_ranges, ByteRange},
};

use super::{BytesPartialDecoderTraits, CodecError, CodecOptions};

#[cfg(feature = "async")]
use super::AsyncBytesPartialDecoderTraits;

/// A cache for a [`BytesPartialDecoderTraits`] partial decoder.
pub(crate) struct BytesPartialDecoderCache {
    cache: Option<Vec<u8>>,
}

impl BytesPartialDecoderCache {
    /// Create a new partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if caching fails.
    pub(crate) fn new(
        input_handle: &dyn BytesPartialDecoderTraits,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let cache = input_handle
            .partial_decode(&[ByteRange::FromStart(0, None)], options)?
            .map(|mut bytes| bytes.remove(0).into_owned());
        Ok(Self { cache })
    }

    #[cfg(feature = "async")]
    /// Create a new asynchronous partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if caching fails.
    pub(crate) async fn async_new(
        input_handle: &dyn AsyncBytesPartialDecoderTraits,
        options: &CodecOptions,
    ) -> Result<BytesPartialDecoderCache, CodecError> {
        let cache = input_handle
            .partial_decode(&[ByteRange::FromStart(0, None)], options)
            .await?
            .map(|mut bytes| bytes.remove(0).into_owned());
        Ok(Self { cache })
    }
}

impl BytesPartialDecoderTraits for BytesPartialDecoderCache {
    fn size(&self) -> usize {
        self.cache.as_ref().map_or(0, Vec::len)
    }

    fn partial_decode(
        &self,
        decoded_regions: &dyn ByteRangeIndexer,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        Ok(match &self.cache {
            Some(bytes) => Some(
                extract_byte_ranges(bytes, decoded_regions)
                    .map_err(CodecError::InvalidByteRangeError)?
                    .into_iter()
                    .map(Cow::Owned)
                    .collect(),
            ),
            None => None,
        })
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncBytesPartialDecoderTraits for BytesPartialDecoderCache {
    async fn partial_decode(
        &self,
        decoded_regions: &dyn ByteRangeIndexer,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        BytesPartialDecoderTraits::partial_decode(self, decoded_regions, options)
    }
}

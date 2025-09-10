//! A cache for partial decoders.

use std::borrow::Cow;

use crate::{
    array::RawBytes,
    storage::byte_range::{extract_byte_ranges, ByteRange, ByteRangeIterator},
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
            .partial_decode(ByteRange::FromStart(0, None), options)?
            .map(Cow::into_owned);
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
            .partial_decode(ByteRange::FromStart(0, None), options)
            .await?
            .map(Cow::into_owned);
        Ok(Self { cache })
    }
}

impl BytesPartialDecoderTraits for BytesPartialDecoderCache {
    fn size_held(&self) -> usize {
        self.cache.as_ref().map_or(0, Vec::len)
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
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
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for BytesPartialDecoderCache {
    fn size_held(&self) -> usize {
        self.cache.as_ref().map_or(0, Vec::len)
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        BytesPartialDecoderTraits::partial_decode_many(self, decoded_regions, options)
    }
}

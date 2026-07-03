use std::any::Any;
use std::borrow::Cow;

use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator, extract_byte_ranges};
use zarrs_storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits, OffsetBytesIterator,
    StorageError, StoreKey,
};

use crate::{ArrayBytesRaw, CodecError, CodecOptions};

/// Asynchronous partial bytes decoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncBytesPartialDecoderTraits: Any + MaybeSend + MaybeSync {
    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    async fn exists(&self) -> Result<bool, StorageError>;

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize;

    /// Partially decode a byte range.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the byte range is invalid.
    async fn partial_decode<'a>(
        &'a self,
        decoded_region: ByteRange,
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
        Ok(self
            .partial_decode_many(Box::new([decoded_region].into_iter()), options)
            .await?
            .map(|mut v| v.pop().expect("single byte range")))
    }

    /// Partially decode byte ranges.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError>;

    /// Decode all bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    async fn decode<'a>(
        &'a self,
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
        self.partial_decode(ByteRange::FromStart(0, None), options)
            .await
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
}

/// Asynhronous partial bytes encoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncBytesPartialEncoderTraits:
    AsyncBytesPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    async fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the byte range is invalid.
    async fn partial_encode(
        &self,
        offset: u64,
        bytes: ArrayBytesRaw<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.partial_encode_many(Box::new([(offset, bytes)].into_iter()), options)
            .await
    }

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, ArrayBytesRaw<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for Cow<'static, [u8]> {
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.as_ref().len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges(self, decoded_regions)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for Vec<u8> {
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges(self, decoded_regions)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: AsyncReadableStorageTraits + 'static> AsyncBytesPartialDecoderTraits
    for (TStorage, StoreKey)
{
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.0.size_key(&self.1).await?.is_some())
    }

    fn size_held(&self) -> usize {
        0
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        let bytes = self.0.get_partial_many(&self.1, decoded_regions).await?;
        Ok(if let Some(bytes) = bytes {
            use futures::{StreamExt, TryStreamExt};
            Some(
                bytes
                    .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.into())))
                    .try_collect()
                    .await?,
            )
        } else {
            None
        })
    }

    fn supports_partial_decode(&self) -> bool {
        self.0.supports_get_partial()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: AsyncReadableWritableStorageTraits + 'static> AsyncBytesPartialEncoderTraits
    for (TStorage, StoreKey)
{
    async fn erase(&self) -> Result<(), CodecError> {
        Ok(self.0.erase(&self.1).await?)
    }

    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, ArrayBytesRaw<'_>>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let offset_values = offset_values
            .into_iter()
            .map(|(offset, bytes)| (offset, bytes.into_owned().into()));
        Ok(self
            .0
            .set_partial_many(&self.1, Box::new(offset_values))
            .await?)
    }

    fn supports_partial_encode(&self) -> bool {
        self.0.supports_set_partial()
    }
}

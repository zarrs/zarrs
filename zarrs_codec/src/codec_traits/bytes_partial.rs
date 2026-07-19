use std::any::Any;
use std::borrow::Cow;
use std::sync::Mutex;

use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::byte_range::{
    ByteRange, ByteRangeIterator, InvalidByteRangeError, extract_byte_ranges,
};
#[cfg(feature = "async")]
use zarrs_storage::{AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits};
use zarrs_storage::{
    OffsetBytesIterator, ReadableStorageTraits, ReadableWritableStorageTraits, StorageError,
    StoreKey,
};

use crate::{ArrayBytesRaw, CodecError, CodecOptions};

ambisync::scoped! {
#![defaults(
    sync(fns("async_{}", "{}"), types("Async{}")),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]

/// Asynchronous partial bytes decoder traits.
#[ambisync]
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
        decoded_regions: ByteRangeIterator<'_>,
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
#[ambisync]
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
    async fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<'_, ArrayBytesRaw<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

#[ambisync]
impl AsyncBytesPartialDecoderTraits for Cow<'static, [u8]> {
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.as_ref().len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'_>,
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

#[ambisync]
impl AsyncBytesPartialDecoderTraits for Vec<u8> {
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'_>,
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

#[ambisync]
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
        decoded_regions: ByteRangeIterator<'_>,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        let bytes = self.0.get_partial_many(&self.1, decoded_regions).await?;
        Ok(if let Some(bytes) = bytes {
            Some(ambisync::alt!(
                sync => bytes
                    .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.into())))
                    .collect::<Result<Vec<_>, _>>()?,
                async => {
                    use futures::{StreamExt, TryStreamExt};
                    bytes
                        .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.into())))
                        .try_collect()
                        .await?
                },
            ))
        } else {
            None
        })
    }

    fn supports_partial_decode(&self) -> bool {
        self.0.supports_get_partial()
    }
}

#[ambisync]
impl<TStorage: AsyncReadableWritableStorageTraits + 'static> AsyncBytesPartialEncoderTraits
    for (TStorage, StoreKey)
{
    async fn erase(&self) -> Result<(), CodecError> {
        Ok(self.0.erase(&self.1).await?)
    }

    async fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<'_, ArrayBytesRaw<'_>>,
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

}

impl BytesPartialDecoderTraits for Mutex<Option<Vec<u8>>> {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.lock().unwrap().is_some())
    }

    fn size_held(&self) -> usize {
        self.lock().unwrap().as_ref().map_or(0, Vec::len)
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator<'_>,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        if let Some(input) = self.lock().unwrap().as_ref() {
            let size = input.len() as u64;
            let mut outputs = vec![];
            for byte_range in decoded_regions {
                if byte_range.end(size) <= size {
                    outputs.push(Cow::Owned(input[byte_range.to_range_usize(size)].into()));
                } else {
                    return Err(InvalidByteRangeError::new(byte_range, size).into());
                }
            }
            Ok(Some(outputs))
        } else {
            Ok(None)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

impl BytesPartialEncoderTraits for Mutex<Option<Vec<u8>>> {
    fn erase(&self) -> Result<(), CodecError> {
        *self.lock().unwrap() = None;
        Ok(())
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<'_, ArrayBytesRaw<'_>>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let mut v = self.lock().unwrap();
        let mut output = v.as_ref().cloned().unwrap_or_default();

        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if output.len() < offset + value.len() {
                output.resize(offset + value.len(), 0);
            }
            output[offset..offset + value.len()].copy_from_slice(&value);
        }
        *v = Some(output);
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_with_short_lived_ranges<'a>(
        decoder: &'a dyn BytesPartialDecoderTraits,
    ) -> Vec<ArrayBytesRaw<'a>> {
        let ranges = [ByteRange::FromStart(1, Some(2))];
        BytesPartialDecoderTraits::partial_decode_many(
            decoder,
            Box::new(ranges.iter().copied()),
            &CodecOptions::default(),
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn decoded_bytes_can_outlive_ranges() {
        let decoder = vec![0, 1, 2, 3];
        let decoded = decode_with_short_lived_ranges(&decoder);
        assert_eq!(decoded[0].as_ref(), &[1, 2]);
    }

    #[cfg(feature = "async")]
    async fn async_decode_with_short_lived_ranges<'a>(
        decoder: &'a dyn AsyncBytesPartialDecoderTraits,
    ) -> Vec<ArrayBytesRaw<'a>> {
        let ranges = [ByteRange::FromStart(1, Some(2))];
        AsyncBytesPartialDecoderTraits::partial_decode_many(
            decoder,
            Box::new(ranges.iter().copied()),
            &CodecOptions::default(),
        )
        .await
        .unwrap()
        .unwrap()
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_decoded_bytes_can_outlive_ranges() {
        let decoder = vec![0, 1, 2, 3];
        let decoded = futures::executor::block_on(async_decode_with_short_lived_ranges(&decoder));
        assert_eq!(decoded[0].as_ref(), &[1, 2]);
    }
}

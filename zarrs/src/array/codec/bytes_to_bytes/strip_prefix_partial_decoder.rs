use std::sync::Arc;

use crate::array::CowBytes;
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{BytesPartialDecoderTraits, CodecError, CodecOptions, Resources};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator};

/// Partial decoder for stripping a prefix (e.g. checksum).
pub(crate) struct StripPrefixPartialDecoder<T: ?Sized> {
    input_handle: Arc<T>,
    prefix_size: usize,
}

impl<T: ?Sized> StripPrefixPartialDecoder<T> {
    /// Create a new "strip prefix" partial decoder.
    pub(crate) fn new(input_handle: Arc<T>, prefix_size: usize) -> Self {
        Self {
            input_handle,
            prefix_size,
        }
    }
}

impl<T: BytesPartialDecoderTraits + ?Sized + 'static> BytesPartialDecoderTraits
    for StripPrefixPartialDecoder<T>
{
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
        resources: &Resources,
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        let decoded_regions = decoded_regions.map(|range| match range {
            ByteRange::FromStart(offset, length) => {
                ByteRange::FromStart(offset.checked_add(self.prefix_size as u64).unwrap(), length)
            }
            ByteRange::Suffix(length) => ByteRange::Suffix(length),
        });

        self.input_handle
            .partial_decode_many(Box::new(decoded_regions), options, resources)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: AsyncBytesPartialDecoderTraits + ?Sized + 'static> AsyncBytesPartialDecoderTraits
    for StripPrefixPartialDecoder<T>
{
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
        resources: &Resources,
    ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
        let decoded_regions = decoded_regions.map(|range| match range {
            ByteRange::FromStart(offset, length) => {
                ByteRange::FromStart(offset.checked_add(self.prefix_size as u64).unwrap(), length)
            }
            ByteRange::Suffix(length) => ByteRange::Suffix(length),
        });

        self.input_handle
            .partial_decode_many(Box::new(decoded_regions), options, resources)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

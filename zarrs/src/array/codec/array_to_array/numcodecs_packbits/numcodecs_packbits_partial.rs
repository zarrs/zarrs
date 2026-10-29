//! Adapt partial access to packed `uint8` arrays to the packbits byte partial decoders and encoders.

use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, BytesPartialDecoderTraits,
    BytesPartialEncoderTraits, CodecError, CodecOptions, CowBytes, InvalidBytesLengthError,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator, InvalidByteRangeError};
use zarrs_storage::{OffsetBytesIterator, StorageError};

use crate::array::ArraySubset;

pub(super) struct NumcodecsPackBitsPartial<T: ?Sized> {
    input_handle: Arc<T>,
    encoded_size: u64,
}

impl<T: ?Sized> NumcodecsPackBitsPartial<T> {
    pub(super) fn new(input_handle: Arc<T>, encoded_size: u64) -> Self {
        Self {
            input_handle,
            encoded_size,
        }
    }

    fn subset(&self, range: ByteRange) -> Result<ArraySubset, CodecError> {
        let range = range
            .checked_range(self.encoded_size)
            .ok_or_else(|| InvalidByteRangeError::new(range, self.encoded_size))?;
        Ok(ArraySubset::new_with_ranges(&[range]))
    }

    /// The subsets of the packed `uint8` array with `ranges`, which are retrieved together.
    fn subsets(&self, ranges: ByteRangeIterator) -> Result<Vec<ArraySubset>, CodecError> {
        ranges.map(|range| self.subset(range)).collect()
    }
}

/// Whether `subsets` include the padding encoding byte.
///
/// The packed bytes of a missing chunk are retrieved as the encoded fill value, which unpacks to the fill value.
/// So whether the chunk exists only needs to be checked if the padding encoding byte is retrieved (e.g. when decoding).
fn includes_padding_byte(subsets: &[ArraySubset]) -> bool {
    subsets
        .iter()
        .any(|subset| subset.num_elements() > 0 && subset.start()[0] == 0)
}

/// Split the concatenated bytes of `subsets`.
fn split_subsets<'a>(
    bytes: ArrayBytes<'a>,
    subsets: &[ArraySubset],
) -> Result<Vec<CowBytes<'a>>, CodecError> {
    let mut bytes = bytes.into_fixed()?;
    let expected_len = usize::try_from(subsets.iter().map(ArraySubset::num_elements).sum::<u64>())
        .map_err(|_| CodecError::from("the packed byte ranges overflow usize"))?;
    if bytes.len() != expected_len {
        return Err(InvalidBytesLengthError::new(bytes.len(), expected_len).into());
    }
    Ok(subsets
        .iter()
        .map(|subset| bytes.split_to(subset.num_elements_usize()))
        .collect())
}

impl<T: ArrayPartialDecoderTraits + ?Sized> BytesPartialDecoderTraits
    for NumcodecsPackBitsPartial<T>
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
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        let subsets = self.subsets(decoded_regions)?;
        if includes_padding_byte(&subsets) && !self.input_handle.exists()? {
            return Ok(None);
        }
        if subsets.is_empty() {
            return Ok(Some(vec![]));
        }
        let bytes = self.input_handle.partial_decode(&subsets, options)?;
        split_subsets(bytes, &subsets).map(Some)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

impl<T: ArrayPartialEncoderTraits + ?Sized> BytesPartialEncoderTraits
    for NumcodecsPackBitsPartial<T>
{
    fn erase(&self) -> Result<(), CodecError> {
        self.input_handle.erase()
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<CowBytes<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        for (offset, bytes) in offset_values {
            let subset = self.subset(ByteRange::FromStart(offset, Some(bytes.len() as u64)))?;
            self.input_handle
                .partial_encode(&subset, &ArrayBytes::Fixed(bytes), options)?;
        }
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: AsyncArrayPartialDecoderTraits + ?Sized> AsyncBytesPartialDecoderTraits
    for NumcodecsPackBitsPartial<T>
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
    ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
        let subsets = self.subsets(decoded_regions)?;
        if subsets.is_empty() {
            return Ok(Some(vec![]));
        }
        let bytes = if includes_padding_byte(&subsets) {
            // Check that the chunk exists while retrieving it
            let (exists, bytes) = futures::join!(
                self.input_handle.exists(),
                self.input_handle.partial_decode(&subsets, options)
            );
            if !exists? {
                return Ok(None);
            }
            bytes?
        } else {
            self.input_handle.partial_decode(&subsets, options).await?
        };
        split_subsets(bytes, &subsets).map(Some)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: AsyncArrayPartialEncoderTraits + ?Sized> AsyncBytesPartialEncoderTraits
    for NumcodecsPackBitsPartial<T>
{
    async fn erase(&self) -> Result<(), CodecError> {
        self.input_handle.erase().await
    }

    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, CowBytes<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        for (offset, bytes) in offset_values {
            let subset = self.subset(ByteRange::FromStart(offset, Some(bytes.len() as u64)))?;
            self.input_handle
                .partial_encode(&subset, &ArrayBytes::Fixed(bytes), options)
                .await?;
        }
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_handle.supports_partial_encode()
    }
}

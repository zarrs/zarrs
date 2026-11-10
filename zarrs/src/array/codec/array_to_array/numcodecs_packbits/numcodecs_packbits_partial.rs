//! Adapt partial access to packed `uint8` arrays to the packbits byte partial decoders and encoders.

use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayPartialDecoderSubchunkingTraits,
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecGeometryTraits,
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecError, CodecOptions, CowBytes,
    InvalidBytesLengthError, Resources,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits,
    AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits,
};
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator, InvalidByteRangeError};
use zarrs_storage::{OffsetBytesIterator, StorageError};

use super::NumcodecsPackBitsCodecBound;
use crate::array::{ArraySubset, ChunkGrid, ChunkShape, DataType, Indexer};

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

    /// The subset of the entire packed `uint8` array, validating that `bytes` fill it.
    fn subset_all(&self, bytes: &CowBytes<'_>) -> Result<ArraySubset, CodecError> {
        let expected_len = usize::try_from(self.encoded_size)
            .map_err(|_| CodecError::from("the packed chunk size overflows usize"))?;
        if bytes.len() != expected_len {
            return Err(InvalidBytesLengthError::new(bytes.len(), expected_len).into());
        }
        Ok(ArraySubset::new_with_ranges(&[0..self.encoded_size]))
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
        resources: &Resources,
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        let subsets = self.subsets(decoded_regions)?;
        if includes_padding_byte(&subsets) && !self.input_handle.exists()? {
            return Ok(None);
        }
        if subsets.is_empty() {
            return Ok(Some(vec![]));
        }
        let bytes = self
            .input_handle
            .partial_decode(&subsets, options, resources)?;
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

    fn encode(
        &self,
        bytes: CowBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        // The whole-chunk subset replaces every element, so the existing chunk is not read
        let subset = self.subset_all(&bytes)?;
        self.input_handle
            .partial_encode(&subset, &ArrayBytes::Fixed(bytes), options, resources)
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<CowBytes<'_>>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        for (offset, bytes) in offset_values {
            let subset = self.subset(ByteRange::FromStart(offset, Some(bytes.len() as u64)))?;
            self.input_handle.partial_encode(
                &subset,
                &ArrayBytes::Fixed(bytes),
                options,
                resources,
            )?;
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
        resources: &Resources,
    ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
        let subsets = self.subsets(decoded_regions)?;
        if subsets.is_empty() {
            return Ok(Some(vec![]));
        }
        let bytes = if includes_padding_byte(&subsets) {
            // Check that the chunk exists while retrieving it
            let (exists, bytes) = futures::join!(
                self.input_handle.exists(),
                self.input_handle
                    .partial_decode(&subsets, options, resources)
            );
            if !exists? {
                return Ok(None);
            }
            bytes?
        } else {
            self.input_handle
                .partial_decode(&subsets, options, resources)
                .await?
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

    async fn encode(
        &self,
        bytes: CowBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        // The whole-chunk subset replaces every element, so the existing chunk is not read
        let subset = self.subset_all(&bytes)?;
        self.input_handle
            .partial_encode(&subset, &ArrayBytes::Fixed(bytes), options, resources)
            .await
    }

    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, CowBytes<'_>>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        for (offset, bytes) in offset_values {
            let subset = self.subset(ByteRange::FromStart(offset, Some(bytes.len() as u64)))?;
            self.input_handle
                .partial_encode(&subset, &ArrayBytes::Fixed(bytes), options, resources)
                .await?;
        }
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_handle.supports_partial_encode()
    }
}

pub(super) struct NumcodecsPackBitsPartialDecoder<T: ?Sized, D: ?Sized> {
    input_handle: Arc<T>,
    decoded_shape: ChunkShape,
    codec: Arc<NumcodecsPackBitsCodecBound>,
    inner: Arc<D>,
}

impl<T: ?Sized, D: ?Sized> NumcodecsPackBitsPartialDecoder<T, D> {
    pub(super) fn new(
        input_handle: Arc<T>,
        decoded_shape: ChunkShape,
        codec: Arc<NumcodecsPackBitsCodecBound>,
        inner: Arc<D>,
    ) -> Self {
        Self {
            input_handle,
            decoded_shape,
            codec,
            inner,
        }
    }
}

impl<T: ArrayPartialDecoderSubchunkingTraits + ?Sized, D: ArrayPartialDecoderTraits + ?Sized>
    ArrayPartialDecoderSubchunkingTraits for NumcodecsPackBitsPartialDecoder<T, D>
{
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.codec.decoded_local_subchunk_grids(
            &self.decoded_shape,
            self.input_handle.local_subchunk_grids(options, resources)?,
        )
    }
}

impl<T: ArrayPartialDecoderTraits + ?Sized, D: ArrayPartialDecoderTraits + ?Sized>
    ArrayPartialDecoderTraits for NumcodecsPackBitsPartialDecoder<T, D>
{
    fn data_type(&self) -> &DataType {
        self.inner.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.inner.exists()
    }

    fn size_held(&self) -> usize {
        self.inner.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        self.inner.partial_decode(indexer, options, resources)
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        self.inner
            .partial_decode_into(indexer, output_target, options, resources)
    }

    fn supports_partial_decode(&self) -> bool {
        self.inner.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<
    T: AsyncArrayPartialDecoderSubchunkingTraits + ?Sized,
    D: AsyncArrayPartialDecoderTraits + ?Sized,
> AsyncArrayPartialDecoderSubchunkingTraits for NumcodecsPackBitsPartialDecoder<T, D>
{
    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.codec.decoded_local_subchunk_grids(
            &self.decoded_shape,
            self.input_handle
                .local_subchunk_grids(options, resources)
                .await?,
        )
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: AsyncArrayPartialDecoderTraits + ?Sized, D: AsyncArrayPartialDecoderTraits + ?Sized>
    AsyncArrayPartialDecoderTraits for NumcodecsPackBitsPartialDecoder<T, D>
{
    fn data_type(&self) -> &DataType {
        self.inner.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.inner.exists().await
    }

    fn size_held(&self) -> usize {
        self.inner.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.inner.partial_decode(indexer, options, resources).await
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        self.inner
            .partial_decode_into(indexer, output_target, options, resources)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.inner.supports_partial_decode()
    }
}

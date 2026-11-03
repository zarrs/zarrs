use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::CowBytes;

use super::bytes_codec::{decoded_len, fixed_size};
use super::{BytesDataTypeExt, Endianness};
use crate::array::{ArrayBytes, DataType, FillValue, IndexerError, update_array_bytes};
use zarrs_codec::{
    ArrayPartialDecoderNoSubchunkingTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

/// Partial decoder for the `bytes` codec.
pub(crate) struct BytesCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
    endian: Option<Endianness>,
}

impl<T: ?Sized> BytesCodecPartial<T> {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        endian: Option<Endianness>,
    ) -> Self {
        Self {
            input_output_handle,
            shape: shape.to_vec(),
            data_type: data_type.clone(),
            fill_value: fill_value.clone(),
            endian,
        }
    }

    /// Decode bytes, applying endianness conversion if required.
    fn decode_bytes<'a>(&self, bytes: CowBytes<'a>) -> Result<CowBytes<'a>, CodecError> {
        Ok(self.data_type.codec_bytes()?.decode(bytes, self.endian)?)
    }

    /// Encode bytes, applying endianness conversion if required.
    fn encode_bytes<'a>(&self, bytes: CowBytes<'a>) -> Result<CowBytes<'a>, CodecError> {
        Ok(self.data_type.codec_bytes()?.encode(bytes, self.endian)?)
    }

    /// Validate `indexer` against the chunk, returning the data type size.
    fn validate_indexer(&self, indexer: &dyn crate::array::Indexer) -> Result<usize, CodecError> {
        let data_type_size = fixed_size(&self.data_type)?;

        if indexer.dimensionality() != self.shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indexer.dimensionality(),
                self.shape.len(),
            )
            .into());
        }

        Ok(data_type_size)
    }

    /// Pair the `encoded` elements at `indexer` with their byte offsets, for writing in place into an existing chunk.
    fn offset_bytes_in_place<'b>(
        &self,
        indexer: &dyn crate::array::Indexer,
        data_type_size: usize,
        encoded: &'b [u8],
    ) -> Result<Vec<(u64, CowBytes<'b>)>, CodecError> {
        let chunk_shape = bytemuck::must_cast_slice(&self.shape);
        let byte_ranges = indexer.iter_contiguous_byte_ranges(chunk_shape, data_type_size)?;
        Ok(byte_ranges
            .scan(0usize, |offset_in, range_out| {
                let len = usize::try_from(range_out.end - range_out.start).unwrap();
                let value = CowBytes::from(&encoded[*offset_in..*offset_in + len]);
                *offset_in += len;
                Some((range_out.start, value))
            })
            .collect())
    }

    /// Encode a missing chunk initialised with the fill value and updated with `bytes` at `indexer`.
    fn encode_missing<'a>(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
    ) -> Result<CowBytes<'a>, CodecError> {
        let (num_elements, _) = decoded_len(&self.data_type, &self.shape)?;
        let chunk_bytes =
            ArrayBytes::new_fill_value(&self.data_type, num_elements, &self.fill_value)?;
        let chunk_shape = bytemuck::must_cast_slice(&self.shape);
        let chunk_bytes = update_array_bytes(
            chunk_bytes,
            chunk_shape,
            indexer,
            bytes,
            self.data_type.size(),
        )?;
        let chunk_bytes = chunk_bytes.into_fixed().expect("fixed data type");
        self.encode_bytes(chunk_bytes)
    }
}

/// The `bytes` codec encodes a chunk as a whole, so it has no subchunks.
impl<T: ?Sized> ArrayPartialDecoderNoSubchunkingTraits for BytesCodecPartial<T> {}

impl<T: ?Sized> ArrayPartialDecoderTraits for BytesCodecPartial<T>
where
    T: BytesPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let data_type_size = self.validate_indexer(indexer)?;

        let chunk_shape = bytemuck::must_cast_slice(&self.shape);
        // Get byte ranges
        let byte_ranges = indexer
            .iter_contiguous_byte_ranges(chunk_shape, data_type_size)?
            .map(ByteRange::new);

        // Decode
        let decoded = self
            .input_output_handle
            .partial_decode_many(Box::new(byte_ranges), options)?;

        let decoded = if let Some(decoded) = decoded {
            ArrayBytes::from(self.decode_bytes(CowBytes::from(decoded.concat()))?)
        } else {
            ArrayBytes::new_fill_value(&self.data_type, indexer.len(), &self.fill_value)?
        };

        Ok(decoded)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for BytesCodecPartial<T>
where
    T: AsyncBytesPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let data_type_size = self.validate_indexer(indexer)?;

        let chunk_shape = bytemuck::must_cast_slice(&self.shape);

        // Get byte ranges
        let byte_ranges = indexer
            .iter_contiguous_byte_ranges(chunk_shape, data_type_size)?
            .map(ByteRange::new);

        // Decode
        let decoded = self
            .input_output_handle
            .partial_decode_many(Box::new(byte_ranges), options)
            .await?;

        let decoded = if let Some(decoded) = decoded {
            ArrayBytes::from(self.decode_bytes(CowBytes::from(decoded.concat()))?)
        } else {
            ArrayBytes::new_fill_value(&self.data_type, indexer.len(), &self.fill_value)?
        };

        Ok(decoded)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for BytesCodecPartial<T>
where
    T: BytesPartialEncoderTraits,
{
    fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let data_type_size = self.validate_indexer(indexer)?;
        bytes.validate(indexer.len(), &self.data_type)?;

        // Only the written elements are encoded, so an existing chunk is never read.
        // A chunk left holding only the fill value is therefore not erased.
        if self.input_output_handle.exists()? {
            let encoded = self.encode_bytes(bytes.clone().into_fixed()?)?;
            let offset_bytes = self.offset_bytes_in_place(indexer, data_type_size, &encoded)?;
            self.input_output_handle
                .partial_encode_many(Box::new(offset_bytes.into_iter()), options)
        } else {
            let chunk_bytes = self.encode_missing(indexer, bytes)?;
            self.input_output_handle.encode(chunk_bytes, options)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for BytesCodecPartial<T>
where
    T: AsyncBytesPartialEncoderTraits,
{
    async fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let data_type_size = self.validate_indexer(indexer)?;
        bytes.validate(indexer.len(), &self.data_type)?;

        // Only the written elements are encoded, so an existing chunk is never read.
        // A chunk left holding only the fill value is therefore not erased.
        if self.input_output_handle.exists().await? {
            let encoded = self.encode_bytes(bytes.clone().into_fixed()?)?;
            let offset_bytes = self.offset_bytes_in_place(indexer, data_type_size, &encoded)?;
            self.input_output_handle
                .partial_encode_many(Box::new(offset_bytes.into_iter()), options)
                .await
        } else {
            let chunk_bytes = self.encode_missing(indexer, bytes)?;
            self.input_output_handle.encode(chunk_bytes, options).await
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::Indexer;
use zarrs_data_type::FillValue;

use super::{
    ArrayBytes, ArrayBytesOffsets, ArrayPartialDecoderNoSubchunkingTraits,
    ArrayPartialDecoderSubchunkingTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArraySubset, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits,
    BytesPartialEncoderTraits, BytesRepresentation, BytesToBytesCodecTraits, ChunkShape,
    CodecError, CodecOptions, CowBytes, DataType,
};
use crate::array_bytes::update_array_bytes;
#[cfg(feature = "async")]
use crate::{
    AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits,
    AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits,
};
use zarrs_metadata::DataTypeSize;
use zarrs_storage::byte_range::{ByteRangeIterator, extract_byte_ranges};
use zarrs_storage::{OffsetBytesIterator, StorageError};

/// Decoded representation for array codecs shape, data type, and fill value.
pub(super) struct ArrayDecodedRepresentation {
    shape: ChunkShape,
    data_type: DataType,
    fill_value: Option<FillValue>,
}

impl ArrayDecodedRepresentation {
    /// Create a new array decoded representation.
    pub(super) fn new(
        shape: ChunkShape,
        data_type: DataType,
        fill_value: Option<FillValue>,
    ) -> Self {
        Self {
            shape,
            data_type,
            fill_value,
        }
    }

    /// Return the shape.
    pub(super) fn shape(&self) -> &[NonZeroU64] {
        &self.shape
    }

    /// Return the shape as u64.
    pub(super) fn shape_u64(&self) -> &[u64] {
        bytemuck::must_cast_slice(&self.shape)
    }

    /// Return the data type.
    pub(super) fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Return the fill value.
    ///
    /// # Errors
    /// Returns [`CodecError::UndefinedFillValue`] if the fill value is undefined.
    pub(super) fn fill_value(&self) -> Result<&FillValue, CodecError> {
        self.fill_value
            .as_ref()
            .ok_or(CodecError::UndefinedFillValue)
    }

    /// Returns true if `bytes` is entirely the fill value and the chunk can be omitted.
    ///
    /// Always false if the fill value is undefined.
    pub(super) fn is_omittable(&self, bytes: &ArrayBytes<'_>, options: &CodecOptions) -> bool {
        !options.store_empty_chunks()
            && self
                .fill_value
                .as_ref()
                .is_some_and(|fill_value| bytes.is_fill_value(fill_value))
    }

    /// Returns true if `indexer` covers the entire chunk, so a write through it replaces every element.
    pub(super) fn is_whole(&self, indexer: &dyn Indexer) -> bool {
        indexer.covers_array(self.shape_u64())
    }

    /// Return the number of elements.
    pub(super) fn num_elements(&self) -> u64 {
        self.shape.iter().map(|d| d.get()).product()
    }
}

/// Generic partial codec for all codec operations with default behavior.
///
/// The default behaviour for partial encoding/decoding involves reading/writing the entire chunk.
///
/// This struct is generic over the input/output handle type, representation type, and codec type.
pub struct CodecPartialDefault<T: ?Sized, R, C: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: R,
    codec: Arc<C>,
}

/// Type alias for the default array-to-array partial codec.
pub(super) type ArrayToArrayCodecPartialDefault<T> =
    CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits>;

/// Type alias for the default array-to-bytes partial codec.
pub(super) type ArrayToBytesCodecPartialDefault<T> =
    CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToBytesCodecTraits>;

/// Type alias for the default bytes-to-bytes partial codec.
pub(super) type BytesToBytesCodecPartialDefault<T> =
    CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>;

impl<T: ?Sized, C: ?Sized> CodecPartialDefault<T, ArrayDecodedRepresentation, C> {
    /// Create a new [`CodecPartialDefault`] for array codecs.
    #[must_use]
    pub fn new(
        input_output_handle: Arc<T>,
        shape: ChunkShape,
        data_type: DataType,
        fill_value: Option<FillValue>,
        codec: Arc<C>,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation: ArrayDecodedRepresentation::new(shape, data_type, fill_value),
            codec,
        }
    }
}

impl<T: ?Sized, C: ?Sized> CodecPartialDefault<T, BytesRepresentation, C> {
    /// Create a new [`CodecPartialDefault`] for bytes codecs.
    #[must_use]
    pub fn new_bytes(
        input_output_handle: Arc<T>,
        decoded_representation: BytesRepresentation,
        codec: Arc<C>,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl<T: ?Sized> CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits> {
    /// Map the chunk-local subchunk grids of the encoded chunk to the decoded chunk.
    fn decoded_local_subchunk_grids(
        &self,
        encoded_subchunk_grids: Vec<Option<super::ChunkGrid>>,
    ) -> Result<Vec<Option<super::ChunkGrid>>, CodecError> {
        let shape = self.decoded_representation.shape();
        encoded_subchunk_grids
            .into_iter()
            .map(|grid| {
                grid.map_or(Ok(None), |grid| {
                    self.codec.decoded_local_subchunk_grid(shape, &grid)
                })
            })
            .collect()
    }
}

impl<T: ?Sized> ArrayPartialDecoderSubchunkingTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: ArrayPartialDecoderTraits,
{
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<super::ChunkGrid>>, CodecError> {
        self.decoded_local_subchunk_grids(self.input_output_handle.local_subchunk_grids(options)?)
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'_>, super::CodecError> {
        let output_shape = (|| {
            let output_shape = indexer.output_shape();
            let mut shape = Vec::with_capacity(output_shape.len());
            for &size in &output_shape {
                shape.push(NonZeroU64::new(size)?);
            }
            Some(shape)
        })();

        // Read the subsets
        let chunk_bytes = self.input_output_handle.partial_decode(indexer, options)?;

        // Decode the subsets
        if let Some(shape) = output_shape {
            let shape = ChunkShape::from(shape);
            self.codec
                .decode(chunk_bytes, &shape, options)
                .map(ArrayBytes::into_owned)
        } else {
            Ok(match self.decoded_representation.data_type().size() {
                DataTypeSize::Fixed(_) => ArrayBytes::new_flen(vec![]),
                DataTypeSize::Variable => {
                    ArrayBytes::new_vlen(vec![], ArrayBytesOffsets::new(vec![0u32]).unwrap())
                        .unwrap()
                }
            })
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: ArrayPartialEncoderTraits,
{
    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let chunk_shape = self.decoded_representation.shape_u64();
        let array_subset_all = ArraySubset::new_with_shape(chunk_shape);
        bytes.validate(indexer.len(), self.decoded_representation.data_type())?;

        let decoded_value = if self.decoded_representation.is_whole(indexer) {
            // The entire chunk is replaced, so the existing chunk is not read
            bytes.clone()
        } else {
            // Read the entire chunk
            let encoded_value = self
                .input_output_handle
                .partial_decode(&array_subset_all, options)?;
            let decoded_value =
                self.codec
                    .decode(encoded_value, self.decoded_representation.shape(), options)?;

            // Validate the bytes
            decoded_value.validate(
                self.decoded_representation.num_elements(),
                self.decoded_representation.data_type(),
            )?;

            update_array_bytes(
                decoded_value,
                chunk_shape,
                indexer,
                bytes,
                self.decoded_representation.data_type().size(),
            )?
        };

        let is_fill_value = self
            .decoded_representation
            .is_omittable(&decoded_value, options);
        if is_fill_value {
            self.input_output_handle.erase()
        } else {
            // Overwrite the entire chunk rather than erasing it first, so a failure leaves the existing chunk intact
            let encoded_value =
                self.codec
                    .encode(decoded_value, self.decoded_representation.shape(), options)?;
            self.input_output_handle
                .partial_encode(&array_subset_all, &encoded_value, options)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}

// The default array-to-bytes partial codec decodes an entire chunk, so it exposes no subchunks.
// A subchunking codec must implement its own partial decoder to expose them.
impl<T: ?Sized> ArrayPartialDecoderNoSubchunkingTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToBytesCodecTraits>
{
}

impl<T: ?Sized> ArrayPartialDecoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: BytesPartialDecoderTraits,
{
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'_>, super::CodecError> {
        // Read the entire chunk
        let bytes_enc = self.input_output_handle.decode(options)?;

        if let Some(bytes_enc) = bytes_enc {
            // Decode the entire chunk
            let bytes_dec =
                self.codec
                    .decode(bytes_enc, self.decoded_representation.shape(), options)?;

            // Extract the subsets
            let chunk_shape = self.decoded_representation.shape_u64();
            bytes_dec
                .extract_array_subset(
                    indexer,
                    chunk_shape,
                    self.decoded_representation.data_type(),
                )
                .map(ArrayBytes::into_owned)
        } else {
            // The fill value does not touch the indexer, so validate it here.
            indexer.validate(self.decoded_representation.shape_u64())?;
            ArrayBytes::new_fill_value(
                self.decoded_representation.data_type(),
                indexer.len(),
                self.decoded_representation.fill_value()?,
            )
            .map_err(CodecError::from)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: BytesPartialEncoderTraits,
{
    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let chunk_shape = self.decoded_representation.shape_u64();
        bytes.validate(indexer.len(), self.decoded_representation.data_type())?;

        let chunk_bytes = if self.decoded_representation.is_whole(indexer) {
            // The entire chunk is replaced, so the existing chunk is not read
            bytes.clone()
        } else {
            // Read the entire chunk
            let chunk_bytes = self.input_output_handle.decode(options)?;

            // Handle a missing chunk
            let chunk_bytes = if let Some(chunk_bytes) = chunk_bytes {
                self.codec
                    .decode(chunk_bytes, self.decoded_representation.shape(), options)?
            } else {
                ArrayBytes::new_fill_value(
                    self.decoded_representation.data_type(),
                    self.decoded_representation.num_elements(),
                    self.decoded_representation.fill_value()?,
                )?
            };

            // Validate the bytes
            chunk_bytes.validate(
                self.decoded_representation.num_elements(),
                self.decoded_representation.data_type(),
            )?;

            // Update the chunk
            update_array_bytes(
                chunk_bytes,
                chunk_shape,
                indexer,
                bytes,
                self.decoded_representation.data_type().size(),
            )?
        };

        let is_fill_value = self
            .decoded_representation
            .is_omittable(&chunk_bytes, options);
        if is_fill_value {
            self.input_output_handle.erase()
        } else {
            // Replace the existing chunk, rather than erasing it first, so a failure leaves it intact
            let chunk_bytes =
                self.codec
                    .encode(chunk_bytes, self.decoded_representation.shape(), options)?;
            self.input_output_handle.encode(chunk_bytes, options)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}

impl<T: ?Sized> BytesPartialDecoderTraits
    for CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>
where
    T: BytesPartialDecoderTraits,
{
    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        let encoded_value = self.input_output_handle.decode(options)?;

        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        let decoded_value =
            self.codec
                .decode(encoded_value, &self.decoded_representation, options)?;

        Ok(Some(
            extract_byte_ranges(&decoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(CowBytes::from)
                .collect(),
        ))
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}

impl<T: ?Sized> BytesPartialEncoderTraits
    for CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>
where
    T: BytesPartialEncoderTraits,
{
    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn encode(&self, bytes: CowBytes<'_>, options: &CodecOptions) -> Result<(), CodecError> {
        let bytes_encoded = self.codec.encode(bytes, options)?;
        self.input_output_handle.encode(bytes_encoded, options)
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<CowBytes<'_>>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let encoded_value = self.input_output_handle.decode(options)?;

        let mut decoded_value = if let Some(encoded_value) = encoded_value {
            self.codec
                .decode(encoded_value, &self.decoded_representation, options)?
                .into_vec()
        } else {
            vec![]
        };

        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if decoded_value.len() < offset + value.len() {
                decoded_value.resize(offset + value.len(), 0);
            }
            decoded_value[offset..offset + value.len()].copy_from_slice(&value);
        }

        // Replace the entire chunk, since the re-encoded chunk may be shorter than the existing chunk
        let bytes_encoded = self.codec.encode(CowBytes::from(decoded_value), options)?;
        self.input_output_handle.encode(bytes_encoded, options)
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderSubchunkingTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: AsyncArrayPartialDecoderTraits,
{
    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<zarrs_chunk_grid::ChunkGrid>>, CodecError> {
        self.decoded_local_subchunk_grids(
            self.input_output_handle
                .local_subchunk_grids(options)
                .await?,
        )
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: AsyncArrayPartialDecoderTraits,
{
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'a>, super::CodecError> {
        let output_shape = (|| {
            let output_shape = indexer.output_shape();
            let mut shape = Vec::with_capacity(output_shape.len());
            for &size in &output_shape {
                shape.push(NonZeroU64::new(size)?);
            }
            Some(shape)
        })();

        // Read the subsets
        let chunk_bytes = self
            .input_output_handle
            .partial_decode(indexer, options)
            .await?;

        // Decode the subsets
        if let Some(shape) = output_shape {
            let shape = ChunkShape::from(shape);
            self.codec
                .decode(chunk_bytes, &shape, options)
                .map(ArrayBytes::into_owned)
        } else {
            Ok(match self.decoded_representation.data_type().size() {
                DataTypeSize::Fixed(_) => ArrayBytes::new_flen(vec![]),
                DataTypeSize::Variable => {
                    ArrayBytes::new_vlen(vec![], ArrayBytesOffsets::new(vec![0u32]).unwrap())
                        .unwrap()
                }
            })
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: AsyncArrayPartialEncoderTraits,
{
    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let chunk_shape = self.decoded_representation.shape_u64();
        let array_subset_all = ArraySubset::new_with_shape(chunk_shape);
        bytes.validate(indexer.len(), self.decoded_representation.data_type())?;

        let decoded_value = if self.decoded_representation.is_whole(indexer) {
            // The entire chunk is replaced, so the existing chunk is not read
            bytes.clone()
        } else {
            // Read the entire chunk
            let encoded_value = self
                .input_output_handle
                .partial_decode(&array_subset_all, options)
                .await?;
            let decoded_value =
                self.codec
                    .decode(encoded_value, self.decoded_representation.shape(), options)?;

            // Validate the bytes
            decoded_value.validate(
                self.decoded_representation.num_elements(),
                self.decoded_representation.data_type(),
            )?;

            update_array_bytes(
                decoded_value,
                chunk_shape,
                indexer,
                bytes,
                self.decoded_representation.data_type().size(),
            )?
        };

        let is_fill_value = self
            .decoded_representation
            .is_omittable(&decoded_value, options);
        if is_fill_value {
            self.input_output_handle.erase().await
        } else {
            // Overwrite the entire chunk rather than erasing it first, so a failure leaves the existing chunk intact
            let encoded_value =
                self.codec
                    .encode(decoded_value, self.decoded_representation.shape(), options)?;
            self.input_output_handle
                .partial_encode(&array_subset_all, &encoded_value, options)
                .await
        }
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: AsyncBytesPartialDecoderTraits,
{
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'a>, super::CodecError> {
        // Read the entire chunk
        let bytes_enc = self.input_output_handle.decode(options).await?;

        if let Some(bytes_enc) = bytes_enc {
            // Decode the entire chunk
            let bytes_dec =
                self.codec
                    .decode(bytes_enc, self.decoded_representation.shape(), options)?;

            // Extract the subsets
            let chunk_shape = self.decoded_representation.shape_u64();
            bytes_dec
                .extract_array_subset(
                    indexer,
                    chunk_shape,
                    self.decoded_representation.data_type(),
                )
                .map(ArrayBytes::into_owned)
        } else {
            // The fill value does not touch the indexer, so validate it here.
            indexer.validate(self.decoded_representation.shape_u64())?;
            ArrayBytes::new_fill_value(
                self.decoded_representation.data_type(),
                indexer.len(),
                self.decoded_representation.fill_value()?,
            )
            .map_err(CodecError::from)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits
    for CodecPartialDefault<T, ArrayDecodedRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: AsyncBytesPartialEncoderTraits,
{
    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let chunk_shape = self.decoded_representation.shape_u64();
        bytes.validate(indexer.len(), self.decoded_representation.data_type())?;

        let chunk_bytes = if self.decoded_representation.is_whole(indexer) {
            // The entire chunk is replaced, so the existing chunk is not read
            bytes.clone()
        } else {
            // Read the entire chunk
            let chunk_bytes = self.input_output_handle.decode(options).await?;

            // Handle a missing chunk
            let chunk_bytes = if let Some(chunk_bytes) = chunk_bytes {
                self.codec
                    .decode(chunk_bytes, self.decoded_representation.shape(), options)?
            } else {
                ArrayBytes::new_fill_value(
                    self.decoded_representation.data_type(),
                    self.decoded_representation.num_elements(),
                    self.decoded_representation.fill_value()?,
                )?
            };

            // Validate the bytes
            chunk_bytes.validate(
                self.decoded_representation.num_elements(),
                self.decoded_representation.data_type(),
            )?;

            // Update the chunk
            update_array_bytes(
                chunk_bytes,
                chunk_shape,
                indexer,
                bytes,
                self.decoded_representation.data_type().size(),
            )?
        };

        let is_fill_value = self
            .decoded_representation
            .is_omittable(&chunk_bytes, options);
        if is_fill_value {
            self.input_output_handle.erase().await
        } else {
            // Replace the existing chunk, rather than erasing it first, so a failure leaves it intact
            let chunk_bytes =
                self.codec
                    .encode(chunk_bytes, self.decoded_representation.shape(), options)?;
            self.input_output_handle.encode(chunk_bytes, options).await
        }
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncBytesPartialDecoderTraits
    for CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>
where
    T: AsyncBytesPartialDecoderTraits,
{
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
        let encoded_value = self.input_output_handle.decode(options).await?;

        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        let decoded_value =
            self.codec
                .decode(encoded_value, &self.decoded_representation, options)?;

        Ok(Some(
            extract_byte_ranges(&decoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(CowBytes::from)
                .collect(),
        ))
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncBytesPartialEncoderTraits
    for CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>
where
    T: AsyncBytesPartialEncoderTraits,
{
    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn encode(&self, bytes: CowBytes<'_>, options: &CodecOptions) -> Result<(), CodecError> {
        let bytes_encoded = self.codec.encode(bytes, options)?;
        self.input_output_handle
            .encode(bytes_encoded, options)
            .await
    }

    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, CowBytes<'_>>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let encoded_value = self.input_output_handle.decode(options).await?;

        let mut decoded_value = if let Some(encoded_value) = encoded_value {
            self.codec
                .decode(encoded_value, &self.decoded_representation, options)?
                .into_vec()
        } else {
            vec![]
        };

        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if decoded_value.len() < offset + value.len() {
                decoded_value.resize(offset + value.len(), 0);
            }
            decoded_value[offset..offset + value.len()].copy_from_slice(&value);
        }

        // Replace the entire chunk, since the re-encoded chunk may be shorter than the existing chunk
        let bytes_encoded = self.codec.encode(CowBytes::from(decoded_value), options)?;
        self.input_output_handle
            .encode(bytes_encoded, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}

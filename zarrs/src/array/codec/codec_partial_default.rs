use std::{borrow::Cow, num::NonZero, sync::Arc};

use zarrs_metadata::DataTypeSize;
use zarrs_storage::{
    byte_range::{extract_byte_ranges, ByteRangeIterator},
    OffsetBytesIterator,
};

use crate::{
    array::{
        array_bytes::update_array_bytes, ArrayBytes, ArraySize, BytesRepresentation,
        ChunkRepresentation, RawBytes, RawBytesOffsets,
    },
    array_subset::ArraySubset,
};

use super::{
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecTraits,
    ArrayToBytesCodecTraits, BytesPartialDecoderTraits, BytesPartialEncoderTraits,
    BytesToBytesCodecTraits, CodecError, CodecOptions,
};

#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};

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

/// Type alias for the default array-to-bytes partial codec.
pub(super) type ArrayToArrayCodecPartialDefault<T> =
    CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToArrayCodecTraits>;

/// Type alias for the default array-to-bytes partial codec.
pub(super) type ArrayToBytesCodecPartialDefault<T> =
    CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToBytesCodecTraits>;

/// Type alias for the default bytes-to-bytes partial codec.
pub(super) type BytesToBytesCodecPartialDefault<T> =
    CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>;

impl<T: ?Sized, R, C: ?Sized> CodecPartialDefault<T, R, C> {
    /// Create a new [`CodecPartialDefault`].
    #[must_use]
    pub fn new(input_output_handle: Arc<T>, decoded_representation: R, codec: Arc<C>) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'_>, super::CodecError> {
        let output_shape = indexer
            .output_shape()
            .iter()
            .map(|f| NonZero::try_from(*f))
            .collect();

        // Read the subsets
        let chunk_bytes = self.input_output_handle.partial_decode(indexer, options)?;

        // Decode the subsets
        if let Ok(shape) = output_shape {
            self.codec
                .decode(
                    chunk_bytes,
                    &ChunkRepresentation::new(
                        shape,
                        self.decoded_representation.data_type().clone(),
                        self.decoded_representation.fill_value().clone(),
                    )
                    .expect("data type and fill value are compatible"),
                    options,
                )
                .map(ArrayBytes::into_owned)
        } else {
            Ok(match self.decoded_representation.data_type().size() {
                DataTypeSize::Fixed(_) => ArrayBytes::new_flen(vec![]),
                DataTypeSize::Variable => {
                    ArrayBytes::new_vlen(vec![], RawBytesOffsets::new(vec![0]).unwrap()).unwrap()
                }
            })
        }
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: ArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        // Read the entire chunk
        let chunk_shape = self.decoded_representation.shape_u64();
        let array_subset_all = ArraySubset::new_with_shape(chunk_shape.clone());
        let encoded_value = self
            .input_output_handle
            .partial_decode(&array_subset_all, options)?;
        let mut decoded_value =
            self.codec
                .decode(encoded_value, &self.decoded_representation, options)?;

        // Validate the bytes
        decoded_value.validate(
            self.decoded_representation.num_elements(),
            self.decoded_representation.data_type().size(),
        )?;

        bytes.validate(
            indexer.len(),
            self.decoded_representation.data_type().size(),
        )?;

        decoded_value = update_array_bytes(
            decoded_value,
            &chunk_shape,
            indexer,
            bytes,
            self.decoded_representation.data_type().size(),
        )?;

        // Erase existing data
        self.input_output_handle.erase()?;

        let is_fill_value = !options.store_empty_chunks()
            && decoded_value.is_fill_value(self.decoded_representation.fill_value());
        if is_fill_value {
            Ok(())
        } else {
            // Store the updated chunk
            let encoded_value =
                self.codec
                    .encode(decoded_value, &self.decoded_representation, options)?;
            self.input_output_handle
                .partial_encode(&array_subset_all, &encoded_value, options)
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: BytesPartialDecoderTraits,
{
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'_>, super::CodecError> {
        // Read the entire chunk
        let bytes_enc = self.input_output_handle.decode(options)?;

        if let Some(bytes_enc) = bytes_enc {
            // Decode the entire chunk
            let bytes_dec = self
                .codec
                .decode(bytes_enc, &self.decoded_representation, options)?;

            // Extract the subsets
            let chunk_shape = self.decoded_representation.shape_u64();
            bytes_dec
                .extract_array_subset(
                    indexer,
                    &chunk_shape,
                    self.decoded_representation.data_type(),
                )
                .map(ArrayBytes::into_owned)
        } else {
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                indexer.len(),
            );
            Ok(ArrayBytes::new_fill_value(
                array_size,
                self.decoded_representation.fill_value(),
            ))
        }
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: BytesPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        // Read the entire chunk
        let chunk_shape = self.decoded_representation.shape_u64();
        let chunk_bytes = self.input_output_handle.decode(options)?;

        // Handle a missing chunk
        let mut chunk_bytes = if let Some(chunk_bytes) = chunk_bytes {
            self.codec
                .decode(chunk_bytes, &self.decoded_representation, options)?
        } else {
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                self.decoded_representation.num_elements(),
            );
            ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value())
        };

        // Validate the bytes
        chunk_bytes.validate(
            self.decoded_representation.num_elements(),
            self.decoded_representation.data_type().size(),
        )?;

        // Update the chunk
        bytes.validate(
            indexer.len(),
            self.decoded_representation.data_type().size(),
        )?;

        chunk_bytes = update_array_bytes(
            chunk_bytes,
            &chunk_shape,
            indexer,
            bytes,
            self.decoded_representation.data_type().size(),
        )?;

        // Erase existing data
        self.input_output_handle.erase()?;

        let is_fill_value = !options.store_empty_chunks()
            && chunk_bytes.is_fill_value(self.decoded_representation.fill_value());
        if is_fill_value {
            Ok(())
        } else {
            // Store the updated chunk
            let chunk_bytes =
                self.codec
                    .encode(chunk_bytes, &self.decoded_representation, options)?;
            self.input_output_handle
                .partial_encode(0, chunk_bytes, options)
        }
    }
}

impl<T: ?Sized> BytesPartialDecoderTraits
    for CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>
where
    T: BytesPartialDecoderTraits,
{
    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let encoded_value = self.input_output_handle.decode(options)?;

        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        let decoded_value = self
            .codec
            .decode(encoded_value, &self.decoded_representation, options)?
            .into_owned();

        Ok(Some(
            extract_byte_ranges(&decoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}

impl<T: ?Sized> BytesPartialEncoderTraits
    for CodecPartialDefault<T, BytesRepresentation, dyn BytesToBytesCodecTraits>
where
    T: BytesPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<crate::array::RawBytes<'_>>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let encoded_value = self
            .input_output_handle
            .decode(options)?
            .map(Cow::into_owned);

        let mut decoded_value = if let Some(encoded_value) = encoded_value {
            self.codec
                .decode(
                    Cow::Owned(encoded_value),
                    &self.decoded_representation,
                    options,
                )?
                .into_owned()
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

        let bytes_encoded = self
            .codec
            .encode(Cow::Owned(decoded_value), options)?
            .into_owned();

        self.input_output_handle
            .partial_encode(0, Cow::Owned(bytes_encoded), options)
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: AsyncArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'a>, super::CodecError> {
        let output_shape = indexer
            .output_shape()
            .iter()
            .map(|f| NonZero::try_from(*f))
            .collect();

        // Read the subsets
        let chunk_bytes = self
            .input_output_handle
            .partial_decode(indexer, options)
            .await?;

        // Decode the subsets
        if let Ok(shape) = output_shape {
            self.codec
                .decode(
                    chunk_bytes,
                    &ChunkRepresentation::new(
                        shape,
                        self.decoded_representation.data_type().clone(),
                        self.decoded_representation.fill_value().clone(),
                    )
                    .expect("data type and fill value are compatible"),
                    options,
                )
                .map(ArrayBytes::into_owned)
        } else {
            Ok(match self.decoded_representation.data_type().size() {
                DataTypeSize::Fixed(_) => ArrayBytes::new_flen(vec![]),
                DataTypeSize::Variable => {
                    ArrayBytes::new_vlen(vec![], RawBytesOffsets::new(vec![0]).unwrap()).unwrap()
                }
            })
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToArrayCodecTraits>
where
    T: AsyncArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        // Read the entire chunk
        let chunk_shape = self.decoded_representation.shape_u64();
        let array_subset_all = ArraySubset::new_with_shape(chunk_shape.clone());
        let encoded_value = self
            .input_output_handle
            .partial_decode(&array_subset_all, options)
            .await?;
        let mut decoded_value =
            self.codec
                .decode(encoded_value, &self.decoded_representation, options)?;

        // Validate the bytes
        decoded_value.validate(
            self.decoded_representation.num_elements(),
            self.decoded_representation.data_type().size(),
        )?;

        bytes.validate(
            indexer.len(),
            self.decoded_representation.data_type().size(),
        )?;

        decoded_value = update_array_bytes(
            decoded_value,
            &chunk_shape,
            indexer,
            bytes,
            self.decoded_representation.data_type().size(),
        )?;

        // Erase existing data
        self.input_output_handle.erase().await?;

        let is_fill_value = !options.store_empty_chunks()
            && decoded_value.is_fill_value(self.decoded_representation.fill_value());
        if is_fill_value {
            Ok(())
        } else {
            // Store the updated chunk
            let encoded_value =
                self.codec
                    .encode(decoded_value, &self.decoded_representation, options)?;
            self.input_output_handle
                .partial_encode(&array_subset_all, &encoded_value, options)
                .await
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: AsyncBytesPartialDecoderTraits,
{
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'a>, super::CodecError> {
        // Read the entire chunk
        let bytes_enc = self.input_output_handle.decode(options).await?;

        if let Some(bytes_enc) = bytes_enc {
            // Decode the entire chunk
            let bytes_dec = self
                .codec
                .decode(bytes_enc, &self.decoded_representation, options)?;

            // Extract the subsets
            let chunk_shape = self.decoded_representation.shape_u64();
            bytes_dec
                .extract_array_subset(
                    indexer,
                    &chunk_shape,
                    self.decoded_representation.data_type(),
                )
                .map(ArrayBytes::into_owned)
        } else {
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                indexer.len(),
            );
            Ok(ArrayBytes::new_fill_value(
                array_size,
                self.decoded_representation.fill_value(),
            ))
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits
    for CodecPartialDefault<T, ChunkRepresentation, dyn ArrayToBytesCodecTraits>
where
    T: AsyncBytesPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        // Read the entire chunk
        let chunk_shape = self.decoded_representation.shape_u64();
        let chunk_bytes = self.input_output_handle.decode(options).await?;

        // Handle a missing chunk
        let mut chunk_bytes = if let Some(chunk_bytes) = chunk_bytes {
            self.codec
                .decode(chunk_bytes, &self.decoded_representation, options)?
        } else {
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                self.decoded_representation.num_elements(),
            );
            ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value())
        };

        // Validate the bytes
        chunk_bytes.validate(
            self.decoded_representation.num_elements(),
            self.decoded_representation.data_type().size(),
        )?;

        // Update the chunk
        bytes.validate(
            indexer.len(),
            self.decoded_representation.data_type().size(),
        )?;

        chunk_bytes = update_array_bytes(
            chunk_bytes,
            &chunk_shape,
            indexer,
            bytes,
            self.decoded_representation.data_type().size(),
        )?;

        // Erase existing data
        self.input_output_handle.erase().await?;

        let is_fill_value = !options.store_empty_chunks()
            && chunk_bytes.is_fill_value(self.decoded_representation.fill_value());
        if is_fill_value {
            Ok(())
        } else {
            // Store the updated chunk
            let chunk_bytes =
                self.codec
                    .encode(chunk_bytes, &self.decoded_representation, options)?;
            self.input_output_handle
                .partial_encode(0, chunk_bytes, options)
                .await
        }
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
    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        let encoded_value = self.input_output_handle.decode(options).await?;

        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        let decoded_value = self
            .codec
            .decode(encoded_value, &self.decoded_representation, options)?
            .into_owned();

        Ok(Some(
            extract_byte_ranges(&decoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
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
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncBytesPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, crate::array::RawBytes<'_>>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let encoded_value = self
            .input_output_handle
            .decode(options)
            .await?
            .map(Cow::into_owned);

        let mut decoded_value = if let Some(encoded_value) = encoded_value {
            self.codec
                .decode(
                    Cow::Owned(encoded_value),
                    &self.decoded_representation,
                    options,
                )?
                .into_owned()
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

        let bytes_encoded = self
            .codec
            .encode(Cow::Owned(decoded_value), options)?
            .into_owned();

        self.input_output_handle
            .partial_encode(0, Cow::Owned(bytes_encoded), options)
            .await
    }
}

use std::{num::NonZero, sync::Arc};

use zarrs_metadata::DataTypeSize;

use crate::{
    array::{array_bytes::update_array_bytes, ArrayBytes, ChunkRepresentation, RawBytesOffsets},
    array_subset::ArraySubset,
};

use super::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecTraits};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};

/// Generic partial codec for array-to-array operations with default behavior.
pub struct ArrayToArrayCodecPartialDefault<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

impl<T: ?Sized> ArrayToArrayCodecPartialDefault<T> {
    /// Create a new [`ArrayToArrayCodecPartialDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<T>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToArrayCodecTraits>,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for ArrayToArrayCodecPartialDefault<T>
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

impl<T: ?Sized> ArrayPartialEncoderTraits for ArrayToArrayCodecPartialDefault<T>
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for ArrayToArrayCodecPartialDefault<T>
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
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for ArrayToArrayCodecPartialDefault<T>
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

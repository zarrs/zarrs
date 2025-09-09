use std::sync::Arc;

use crate::array::{
    array_bytes::update_array_bytes, ArrayBytes, ArraySize, ChunkRepresentation,
};

use super::{
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, BytesPartialEncoderTraits,
};

#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};

/// Generic partial codec for array-to-bytes operations with default behavior.
pub struct ArrayToBytesCodecPartialDefault<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToBytesCodecTraits>,
}

impl<T: ?Sized> ArrayToBytesCodecPartialDefault<T> {
    /// Create a new [`ArrayToBytesCodecPartialDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<T>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for ArrayToBytesCodecPartialDefault<T>
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
            let bytes_dec = self.codec.decode(bytes_enc, &self.decoded_representation, options)?;

            // Extract the subsets
            let chunk_shape = self.decoded_representation.shape_u64();
            bytes_dec
                .extract_array_subset(indexer, &chunk_shape, self.decoded_representation.data_type())
                .map(ArrayBytes::into_owned)
        } else {
            let array_size = ArraySize::new(self.decoded_representation.data_type().size(), indexer.len());
            Ok(ArrayBytes::new_fill_value(
                array_size,
                self.decoded_representation.fill_value(),
            ))
        }
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for ArrayToBytesCodecPartialDefault<T>
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
            self.codec.decode(chunk_bytes, &self.decoded_representation, options)?
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
            let chunk_bytes = self.codec.encode(chunk_bytes, &self.decoded_representation, options)?;
            self.input_output_handle.partial_encode(0, chunk_bytes, options)
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for ArrayToBytesCodecPartialDefault<T>
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
            let bytes_dec = self.codec.decode(bytes_enc, &self.decoded_representation, options)?;

            // Extract the subsets
            let chunk_shape = self.decoded_representation.shape_u64();
            bytes_dec
                .extract_array_subset(indexer, &chunk_shape, self.decoded_representation.data_type())
                .map(ArrayBytes::into_owned)
        } else {
            let array_size = ArraySize::new(self.decoded_representation.data_type().size(), indexer.len());
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
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for ArrayToBytesCodecPartialDefault<T>
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
            self.codec.decode(chunk_bytes, &self.decoded_representation, options)?
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
            let chunk_bytes = self.codec.encode(chunk_bytes, &self.decoded_representation, options)?;
            self.input_output_handle.partial_encode(0, chunk_bytes, options).await
        }
    }
}
use std::sync::Arc;

use crate::array::{
    array_bytes::update_array_bytes, codec::ArrayPartialDecoderTraits, ArrayBytes, ArraySize,
    ChunkRepresentation,
};

use super::{ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, BytesPartialEncoderTraits};

#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialEncoderTraits,
};

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
    input_output_handle: &Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToBytesCodecTraits>,
    chunk_subset_indexer: &dyn crate::indexer::Indexer,
    chunk_subset_bytes: &ArrayBytes<'_>,
    options: &super::CodecOptions,
)))]
fn partial_encode(
    input_output_handle: &Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToBytesCodecTraits>,
    chunk_subset_indexer: &dyn crate::indexer::Indexer,
    chunk_subset_bytes: &ArrayBytes<'_>,
    options: &super::CodecOptions,
) -> Result<(), super::CodecError> {
    // Read the entire chunk
    let chunk_shape = decoded_representation.shape_u64();
    #[cfg(feature = "async")]
    let chunk_bytes = if _async {
        input_output_handle.decode(options).await
    } else {
        input_output_handle.decode(options)
    }?;
    #[cfg(not(feature = "async"))]
    let chunk_bytes = input_output_handle.decode(options)?;

    // Handle a missing chunk
    let mut chunk_bytes = if let Some(chunk_bytes) = chunk_bytes {
        codec.decode(chunk_bytes, decoded_representation, options)?
    } else {
        let array_size = ArraySize::new(
            decoded_representation.data_type().size(),
            decoded_representation.num_elements(),
        );
        ArrayBytes::new_fill_value(array_size, decoded_representation.fill_value())
    };

    // Validate the bytes
    chunk_bytes.validate(
        decoded_representation.num_elements(),
        decoded_representation.data_type().size(),
    )?;

    // Update the chunk
    chunk_subset_bytes.validate(
        chunk_subset_indexer.len(),
        decoded_representation.data_type().size(),
    )?;

    chunk_bytes = update_array_bytes(
        chunk_bytes,
        &chunk_shape,
        chunk_subset_indexer,
        chunk_subset_bytes,
        decoded_representation.data_type().size(),
    )?;

    // Erase existing data
    #[cfg(feature = "async")]
    if _async {
        input_output_handle.erase().await?;
    } else {
        input_output_handle.erase()?;
    }
    #[cfg(not(feature = "async"))]
    input_output_handle.erase()?;

    let is_fill_value = !options.store_empty_chunks()
        && chunk_bytes.is_fill_value(decoded_representation.fill_value());
    if is_fill_value {
        Ok(())
    } else {
        // Store the updated chunk
        let chunk_bytes = codec.encode(chunk_bytes, decoded_representation, options)?;
        #[cfg(feature = "async")]
        if _async {
            input_output_handle
                .partial_encode(Box::new([(0, chunk_bytes)].into_iter()), options)
                .await
        } else {
            input_output_handle.partial_encode(Box::new([(0, chunk_bytes)].into_iter()), options)
        }
        #[cfg(not(feature = "async"))]
        input_output_handle.partial_encode(Box::new([(0, chunk_bytes)].into_iter()), options)
    }
}

/// The default array-to-bytes partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct ArrayToBytesPartialEncoderDefault {
    input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToBytesCodecTraits>,
}

impl ArrayToBytesPartialEncoderDefault {
    /// Create a new [`ArrayToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
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

impl ArrayPartialDecoderTraits for ArrayToBytesPartialEncoderDefault {
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
        super::array_to_bytes_partial_decoder_default::partial_decode(
            &self.input_output_handle.clone().into_dyn_decoder(),
            &self.decoded_representation,
            &self.codec,
            indexer,
            options,
        )
    }
}

impl ArrayPartialEncoderTraits for ArrayToBytesPartialEncoderDefault {
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
        partial_encode(
            &self.input_output_handle,
            &self.decoded_representation,
            &self.codec,
            indexer,
            bytes,
            options,
        )
    }
}

#[cfg(feature = "async")]
/// The default asynchronous array-to-bytes partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct AsyncArrayToBytesPartialEncoderDefault {
    input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToBytesCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncArrayToBytesPartialEncoderDefault {
    /// Create a new [`ArrayToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncArrayToBytesPartialEncoderDefault {
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'_>, super::CodecError> {
        super::array_to_bytes_partial_decoder_default::partial_decode_async(
            &self.input_output_handle.clone().into_dyn_decoder(),
            &self.decoded_representation,
            &self.codec,
            indexer,
            options,
        )
        .await
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialEncoderTraits for AsyncArrayToBytesPartialEncoderDefault {
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
        partial_encode_async(
            &self.input_output_handle,
            &self.decoded_representation,
            &self.codec,
            indexer,
            bytes,
            options,
        )
        .await
    }
}

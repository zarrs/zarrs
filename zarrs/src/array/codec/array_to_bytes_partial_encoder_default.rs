use std::sync::Arc;

use crate::{
    array::{array_bytes::update_array_bytes, ArrayBytes, ArraySize, ChunkRepresentation},
    array_subset::ArraySubset,
};

use super::{
    ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits,
    BytesPartialEncoderTraits,
};

#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits,
};

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    output_handle: &Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToBytesCodecTraits>,
    chunk_subset_indexer: &ArraySubset,
    chunk_subset_bytes: &ArrayBytes<'_>,
    options: &super::CodecOptions,
)))]
fn partial_encode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    output_handle: &Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToBytesCodecTraits>,
    chunk_subset_indexer: &ArraySubset,
    chunk_subset_bytes: &ArrayBytes<'_>,
    options: &super::CodecOptions,
) -> Result<(), super::CodecError> {
    // Read the entire chunk
    let chunk_shape = decoded_representation.shape_u64();
    #[cfg(feature = "async")]
    let chunk_bytes = if _async {
        input_handle.decode(options).await
    } else {
        input_handle.decode(options)
    }?;
    #[cfg(not(feature = "async"))]
    let chunk_bytes = input_handle.decode(options)?;

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
        chunk_subset_indexer.num_elements(),
        decoded_representation.data_type().size(),
    )?;

    chunk_bytes = update_array_bytes(
        chunk_bytes,
        &chunk_shape,
        chunk_subset_indexer,
        chunk_subset_bytes,
        decoded_representation.data_type().size(),
    )?;

    let is_fill_value = !options.store_empty_chunks()
        && chunk_bytes.is_fill_value(decoded_representation.fill_value());
    if is_fill_value {
        #[cfg(feature = "async")]
        if _async {
            output_handle.erase().await
        } else {
            output_handle.erase()
        }
        #[cfg(not(feature = "async"))]
        output_handle.erase()
    } else {
        // Store the updated chunk
        let chunk_bytes = codec.encode(chunk_bytes, decoded_representation, options)?;
        #[cfg(feature = "async")]
        if _async {
            output_handle
                .partial_encode(&[(0, chunk_bytes)], options)
                .await
        } else {
            output_handle.partial_encode(&[(0, chunk_bytes)], options)
        }
        #[cfg(not(feature = "async"))]
        output_handle.partial_encode(&[(0, chunk_bytes)], options)
    }
}

/// The default array-to-bytes partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct ArrayToBytesPartialEncoderDefault {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    output_handle: Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToBytesCodecTraits>,
}

impl ArrayToBytesPartialEncoderDefault {
    /// Create a new [`ArrayToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl ArrayPartialEncoderTraits for ArrayToBytesPartialEncoderDefault {
    fn erase(&self) -> Result<(), super::CodecError> {
        self.output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &ArraySubset,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode(
            &self.input_handle,
            &self.output_handle,
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
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToBytesCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncArrayToBytesPartialEncoderDefault {
    /// Create a new [`ArrayToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            output_handle,
            decoded_representation,
            codec,
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialEncoderTraits for AsyncArrayToBytesPartialEncoderDefault {
    async fn erase(&self) -> Result<(), super::CodecError> {
        self.output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &ArraySubset,
        bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode_async(
            &self.input_handle,
            &self.output_handle,
            &self.decoded_representation,
            &self.codec,
            indexer,
            bytes,
            options,
        )
        .await
    }
}

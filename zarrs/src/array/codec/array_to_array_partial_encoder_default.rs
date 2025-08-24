use std::sync::Arc;

use crate::{
    array::{array_bytes::update_array_bytes, ArrayBytes, ChunkRepresentation},
    array_subset::ArraySubset,
};

use super::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecTraits};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
        input_output_handle: &Arc<dyn AsyncArrayPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        codec: &Arc<dyn ArrayToArrayCodecTraits>,
        chunk_subset_indexer: &dyn crate::indexer::Indexer,
        chunk_subset_bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
)))]
pub(crate) fn partial_encode_default(
    input_output_handle: &Arc<dyn ArrayPartialEncoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToArrayCodecTraits>,
    chunk_subset_indexer: &dyn crate::indexer::Indexer,
    chunk_subset_bytes: &ArrayBytes<'_>,
    options: &super::CodecOptions,
) -> Result<(), super::CodecError> {
    // Read the entire chunk
    let chunk_shape = decoded_representation.shape_u64();
    let array_subset_all = ArraySubset::new_with_shape(chunk_shape.clone());
    #[cfg(feature = "async")]
    let encoded_value = if _async {
        input_output_handle
            .partial_decode(&array_subset_all, options)
            .await
    } else {
        input_output_handle.partial_decode(&array_subset_all, options)
    }?;
    #[cfg(not(feature = "async"))]
    let encoded_value = input_output_handle.partial_decode(&array_subset_all, options)?;
    let mut decoded_value = codec.decode(encoded_value, decoded_representation, options)?;

    // Validate the bytes
    decoded_value.validate(
        decoded_representation.num_elements(),
        decoded_representation.data_type().size(),
    )?;

    chunk_subset_bytes.validate(
        chunk_subset_indexer.len(),
        decoded_representation.data_type().size(),
    )?;

    decoded_value = update_array_bytes(
        decoded_value,
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
        && decoded_value.is_fill_value(decoded_representation.fill_value());
    if is_fill_value {
        Ok(())
    } else {
        // Store the updated chunk
        let encoded_value = codec.encode(decoded_value, decoded_representation, options)?;
        #[cfg(feature = "async")]
        if _async {
            input_output_handle
                .partial_encode(&array_subset_all, &encoded_value, options)
                .await
        } else {
            input_output_handle.partial_encode(&array_subset_all, &encoded_value, options)
        }
        #[cfg(not(feature = "async"))]
        input_output_handle.partial_encode(&array_subset_all, &encoded_value, options)
    }
}

/// The default array-to-array partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct ArrayToArrayPartialEncoderDefault {
    input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

impl ArrayToArrayPartialEncoderDefault {
    /// Create a new [`ArrayToArrayPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
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

impl ArrayPartialDecoderTraits for ArrayToArrayPartialEncoderDefault {
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
        super::array_to_array_partial_decoder_default::partial_decode(
            &self.input_output_handle.clone().into_dyn_decoder(),
            &self.decoded_representation,
            &self.codec,
            indexer,
            options,
        )
    }
}

impl ArrayPartialEncoderTraits for ArrayToArrayPartialEncoderDefault {
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
        partial_encode_default(
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
/// The default asynchronous array-to-array partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct AsyncArrayToArrayPartialEncoderDefault {
    input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncArrayToArrayPartialEncoderDefault {
    /// Create a new [`AsyncArrayToArrayPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncArrayToArrayPartialEncoderDefault {
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'_>, super::CodecError> {
        super::array_to_array_partial_decoder_default::partial_decode_async(
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
impl AsyncArrayPartialEncoderTraits for AsyncArrayToArrayPartialEncoderDefault {
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
        partial_encode_default_async(
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

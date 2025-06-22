use std::sync::Arc;

use crate::{
    array::{array_bytes::update_array_bytes, ArrayBytes, ChunkRepresentation},
    array_subset::{ArraySubset, IncompatibleArraySubsetAndShapeError},
};

use super::{
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecTraits, CodecError,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
        input_handle: &Arc<dyn AsyncArrayPartialDecoderTraits>,
        output_handle: &Arc<dyn AsyncArrayPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        codec: &Arc<dyn ArrayToArrayCodecTraits>,
        subsets_and_bytes: &[(&ArraySubset, ArrayBytes<'_>)],
        options: &super::CodecOptions,
)))]
fn partial_encode(
    input_handle: &Arc<dyn ArrayPartialDecoderTraits>,
    output_handle: &Arc<dyn ArrayPartialEncoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToArrayCodecTraits>,
    subsets_and_bytes: &[(&ArraySubset, ArrayBytes<'_>)],
    options: &super::CodecOptions,
) -> Result<(), super::CodecError> {
    // Read the entire chunk
    let chunk_shape = decoded_representation.shape_u64();
    let array_subset_all = ArraySubset::new_with_shape(chunk_shape.clone());
    #[cfg(feature = "async")]
    let encoded_value = if _async {
        input_handle
            .partial_decode(std::slice::from_ref(&array_subset_all), options)
            .await
    } else {
        input_handle.partial_decode(std::slice::from_ref(&array_subset_all), options)
    }?
    .pop()
    .unwrap();
    #[cfg(not(feature = "async"))]
    let encoded_value = input_handle
        .partial_decode(&[array_subset_all.clone()], options)?
        .pop()
        .unwrap();
    let mut decoded_value = codec.decode(encoded_value, decoded_representation, options)?;

    // Validate the bytes
    decoded_value.validate(
        decoded_representation.num_elements(),
        decoded_representation.data_type().size(),
    )?;

    // Update the chunk
    // TODO: More efficient update for multiple chunk subsets?
    for (chunk_subset, chunk_subset_bytes) in subsets_and_bytes {
        // Check the subset is within the chunk shape
        if chunk_subset
            .end_exc()
            .iter()
            .zip(decoded_representation.shape())
            .any(|(a, b)| *a > b.get())
        {
            return Err(CodecError::InvalidArraySubsetError(
                IncompatibleArraySubsetAndShapeError::new(
                    (*chunk_subset).clone(),
                    decoded_representation.shape_u64(),
                ),
            ));
        }

        chunk_subset_bytes.validate(
            chunk_subset.num_elements(),
            decoded_representation.data_type().size(),
        )?;

        decoded_value = update_array_bytes(
            decoded_value,
            &chunk_shape,
            chunk_subset,
            chunk_subset_bytes,
            decoded_representation.data_type().size(),
        )?;
    }

    let is_fill_value = !options.store_empty_chunks()
        && decoded_value.is_fill_value(decoded_representation.fill_value());
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
        let encoded_value = codec.encode(decoded_value, decoded_representation, options)?;
        #[cfg(feature = "async")]
        if _async {
            output_handle
                .partial_encode(&[(&array_subset_all, encoded_value)], options)
                .await
        } else {
            output_handle.partial_encode(&[(&array_subset_all, encoded_value)], options)
        }
        #[cfg(not(feature = "async"))]
        output_handle.partial_encode(&[(&array_subset_all, encoded_value)], options)
    }
}

/// The default array-to-array partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct ArrayToArrayPartialEncoderDefault {
    input_handle: Arc<dyn ArrayPartialDecoderTraits>,
    output_handle: Arc<dyn ArrayPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

impl ArrayToArrayPartialEncoderDefault {
    /// Create a new [`ArrayToArrayPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToArrayCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl ArrayPartialEncoderTraits for ArrayToArrayPartialEncoderDefault {
    fn erase(&self) -> Result<(), super::CodecError> {
        self.output_handle.erase()
    }

    fn partial_encode(
        &self,
        subsets_and_bytes: &[(&ArraySubset, ArrayBytes<'_>)],
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode(
            &self.input_handle,
            &self.output_handle,
            &self.decoded_representation,
            &self.codec,
            subsets_and_bytes,
            options,
        )
    }
}

#[cfg(feature = "async")]
/// The default asynchronous array-to-array partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct AsyncArrayToArrayPartialEncoderDefault {
    input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
    output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncArrayToArrayPartialEncoderDefault {
    /// Create a new [`AsyncArrayToArrayPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToArrayCodecTraits>,
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
impl AsyncArrayPartialEncoderTraits for AsyncArrayToArrayPartialEncoderDefault {
    async fn erase(&self) -> Result<(), super::CodecError> {
        self.output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        subsets_and_bytes: &[(&ArraySubset, ArrayBytes<'_>)],
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode_async(
            &self.input_handle,
            &self.output_handle,
            &self.decoded_representation,
            &self.codec,
            subsets_and_bytes,
            options,
        )
        .await
    }
}

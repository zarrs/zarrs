use std::{num::NonZero, sync::Arc};

use zarrs_metadata::DataTypeSize;

use crate::array::{ArrayBytes, ChunkRepresentation, RawBytesOffsets};

use super::{ArrayPartialDecoderTraits, ArrayToArrayCodecTraits, CodecError, CodecOptions};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
    input_handle: &Arc<dyn AsyncArrayPartialDecoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToArrayCodecTraits>,
    indexer: &dyn crate::indexer::Indexer,
    options: &CodecOptions,
)))]
pub(crate) fn partial_decode<'a>(
    input_handle: &Arc<dyn ArrayPartialDecoderTraits>,
    decoded_representation: &ChunkRepresentation,
    codec: &Arc<dyn ArrayToArrayCodecTraits>,
    indexer: &dyn crate::indexer::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'a>, CodecError> {
    let output_shape = indexer
        .output_shape()
        .iter()
        .map(|f| NonZero::try_from(*f))
        .collect();

    // Read the subsets
    #[cfg(feature = "async")]
    let chunk_bytes: ArrayBytes = if _async {
        input_handle.partial_decode(indexer, options).await
    } else {
        input_handle.partial_decode(indexer, options)
    }?;
    #[cfg(not(feature = "async"))]
    let chunk_bytes = input_handle.partial_decode(indexer, options)?;

    // Decode the subsets
    if let Ok(shape) = output_shape {
        codec
            .decode(
                chunk_bytes,
                &ChunkRepresentation::new(
                    shape,
                    decoded_representation.data_type().clone(),
                    decoded_representation.fill_value().clone(),
                )
                .expect("data type and fill value are compatible"),
                options,
            )
            .map(ArrayBytes::into_owned)
    } else {
        Ok(match decoded_representation.data_type().size() {
            DataTypeSize::Fixed(_) => ArrayBytes::new_flen(vec![]),
            DataTypeSize::Variable => {
                ArrayBytes::new_vlen(vec![], RawBytesOffsets::new(vec![0]).unwrap()).unwrap()
            }
        })
    }
}

/// The default array to array partial decoder. Decodes the entire chunk, and decodes the regions of interest.
/// This cannot be applied on a codec reorganises elements (e.g. transpose).
pub struct ArrayToArrayPartialDecoderDefault {
    input_handle: Arc<dyn ArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

impl ArrayToArrayPartialDecoderDefault {
    /// Create a new [`ArrayToArrayPartialDecoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToArrayCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            codec,
        }
    }
}

impl ArrayPartialDecoderTraits for ArrayToArrayPartialDecoderDefault {
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'_>, super::CodecError> {
        partial_decode(
            &self.input_handle,
            &self.decoded_representation,
            &self.codec,
            indexer,
            options,
        )
    }
}

#[cfg(feature = "async")]
/// The default asynchronous array to array partial decoder. Applies a codec to the regions of interest.
/// This cannot be applied on a codec reorganises elements (e.g. transpose).
pub struct AsyncArrayToArrayPartialDecoderDefault {
    input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncArrayToArrayPartialDecoderDefault {
    /// Create a new [`AsyncArrayToArrayPartialDecoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        codec: Arc<dyn ArrayToArrayCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            codec,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncArrayToArrayPartialDecoderDefault {
    fn data_type(&self) -> &super::DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &super::CodecOptions,
    ) -> Result<ArrayBytes<'a>, super::CodecError> {
        partial_decode_async(
            &self.input_handle,
            &self.decoded_representation,
            &self.codec,
            indexer,
            options,
        )
        .await
    }
}

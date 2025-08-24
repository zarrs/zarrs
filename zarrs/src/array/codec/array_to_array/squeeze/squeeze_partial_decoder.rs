use std::sync::Arc;

use super::{get_squeezed_array_subset, get_squeezed_indexer};

use crate::array::{
    codec::{ArrayBytes, ArrayPartialDecoderTraits, CodecError, CodecOptions},
    ChunkRepresentation, DataType,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// Partial decoder for the Squeeze codec.
pub(crate) struct SqueezePartialDecoder {
    input_handle: Arc<dyn ArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
}

impl SqueezePartialDecoder {
    /// Create a new partial decoder for the Squeeze codec.
    pub(crate) fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
        }
    }
}

impl ArrayPartialDecoderTraits for SqueezePartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_handle
                .partial_decode(&array_subset_squeezed, options)
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_handle.partial_decode(&indexer_squeezed, options)
        }
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the Squeeze codec.
pub(crate) struct AsyncSqueezePartialDecoder {
    input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
}

#[cfg(feature = "async")]
impl AsyncSqueezePartialDecoder {
    /// Create a new partial decoder for the Squeeze codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncSqueezePartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_handle
                .partial_decode(&array_subset_squeezed, options)
                .await
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_handle
                .partial_decode(&indexer_squeezed, options)
                .await
        }
    }
}

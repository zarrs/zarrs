use std::sync::Arc;

use super::{do_transpose, get_transposed_array_subset, get_transposed_indexer, TransposeOrder};
use crate::array::{
    codec::{ArrayBytes, ArrayPartialDecoderTraits, CodecError, CodecOptions},
    ChunkRepresentation, DataType,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// Partial decoder for the Transpose codec.
pub(crate) struct TransposePartialDecoder {
    input_handle: Arc<dyn ArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    order: TransposeOrder,
}

impl TransposePartialDecoder {
    /// Create a new partial decoder for the Transpose codec.
    pub(crate) fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        order: TransposeOrder,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            order,
        }
    }
}

impl ArrayPartialDecoderTraits for TransposePartialDecoder {
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
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            let encoded_value = self
                .input_handle
                .partial_decode(&array_subset_transposed, options)?;
            do_transpose(
                encoded_value,
                array_subset,
                &self.order,
                &self.decoded_representation,
            )
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_handle
                .partial_decode(&indexer_transposed, options)
        }
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the Transpose codec.
pub(crate) struct AsyncTransposePartialDecoder {
    input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    order: TransposeOrder,
}

#[cfg(feature = "async")]
impl AsyncTransposePartialDecoder {
    /// Create a new partial decoder for the Transpose codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        order: TransposeOrder,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            order,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncTransposePartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            let encoded_value = self
                .input_handle
                .partial_decode(&array_subset_transposed, options)
                .await?;
            do_transpose(
                encoded_value,
                array_subset,
                &self.order,
                &self.decoded_representation,
            )
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_handle
                .partial_decode(&indexer_transposed, options)
                .await
        }
    }
}

use std::sync::Arc;

use super::{do_transpose, get_transposed_array_subset, get_transposed_indexer, TransposeOrder};

use crate::array::{
    codec::{CodecError, CodecOptions},
    ArrayBytes, ChunkRepresentation, DataType,
};

use crate::array::codec::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits};

/// The `transpose` partial encoder.
pub(crate) struct TransposePartialEncoder {
    input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
    order: TransposeOrder,
}

impl TransposePartialEncoder {
    /// Create a new [`TransposePartialEncoder`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        decoded_representation: ChunkRepresentation,
        order: TransposeOrder,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            order,
        }
    }
}

impl ArrayPartialDecoderTraits for TransposePartialEncoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            let encoded_value = self
                .input_output_handle
                .partial_decode(&array_subset_transposed, options)?;
            do_transpose(
                encoded_value,
                array_subset,
                &self.order,
                &self.decoded_representation,
            )
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_output_handle
                .partial_decode(&indexer_transposed, options)
        }
    }
}

impl ArrayPartialEncoderTraits for TransposePartialEncoder {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            self.input_output_handle
                .partial_encode(&array_subset_transposed, bytes, options)
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_output_handle
                .partial_encode(&indexer_transposed, bytes, options)
        }
    }
}

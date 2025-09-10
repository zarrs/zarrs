use std::sync::Arc;

use super::{do_transpose, get_transposed_array_subset, get_transposed_indexer, TransposeOrder};

use crate::array::{
    codec::{CodecError, CodecOptions},
    ArrayBytes, ChunkRepresentation, DataType,
};

use crate::array::codec::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// Generic partial codec for the Transpose codec.
pub(crate) struct TransposeCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: ChunkRepresentation,
    order: TransposeOrder,
}

impl<T: ?Sized> TransposeCodecPartial<T> {
    /// Create a new [`TransposeCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<T>,
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

impl<T: ?Sized> ArrayPartialDecoderTraits for TransposeCodecPartial<T>
where
    T: ArrayPartialDecoderTraits,
{
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

impl<T: ?Sized> ArrayPartialEncoderTraits for TransposeCodecPartial<T>
where
    T: ArrayPartialEncoderTraits,
{
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for TransposeCodecPartial<T>
where
    T: AsyncArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            let encoded_value = self
                .input_output_handle
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
            self.input_output_handle
                .partial_decode(&indexer_transposed, options)
                .await
        }
    }
}

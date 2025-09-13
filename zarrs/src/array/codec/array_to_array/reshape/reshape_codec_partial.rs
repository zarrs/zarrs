use std::sync::Arc;

use super::{get_reshaped_array_subset, get_reshaped_indexer};

use crate::array::{
    codec::{
        ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions,
    },
    ChunkRepresentation, DataType,
};
use zarrs_metadata_ext::codec::reshape::ReshapeShape;

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// Generic partial codec for the Reshape codec.
pub(crate) struct ReshapeCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: ChunkRepresentation,
    shape: ReshapeShape,
}

impl<T: ?Sized> ReshapeCodecPartial<T> {
    /// Create a new [`ReshapeCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        decoded_representation: ChunkRepresentation,
        shape: ReshapeShape,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            shape,
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for ReshapeCodecPartial<T>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_reshaped = get_reshaped_array_subset(
                array_subset,
                self.decoded_representation.shape(),
                &self.shape,
            )?;
            self.input_output_handle
                .partial_decode(&array_subset_reshaped, options)
        } else {
            let indexer_reshaped = get_reshaped_indexer(
                indexer,
                self.decoded_representation.shape(),
                &self.shape,
            )?;
            self.input_output_handle
                .partial_decode(&indexer_reshaped, options)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for ReshapeCodecPartial<T>
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
            let array_subset_reshaped = get_reshaped_array_subset(
                array_subset,
                self.decoded_representation.shape(),
                &self.shape,
            )?;
            self.input_output_handle
                .partial_encode(&array_subset_reshaped, bytes, options)
        } else {
            let indexer_reshaped = get_reshaped_indexer(
                indexer,
                self.decoded_representation.shape(),
                &self.shape,
            )?;
            self.input_output_handle
                .partial_encode(&indexer_reshaped, bytes, options)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for ReshapeCodecPartial<T>
where
    T: AsyncArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_reshaped = get_reshaped_array_subset(
                array_subset,
                self.decoded_representation.shape(),
                &self.shape,
            )?;
            self.input_output_handle
                .partial_decode(&array_subset_reshaped, options)
                .await
        } else {
            let indexer_reshaped = get_reshaped_indexer(
                indexer,
                self.decoded_representation.shape(),
                &self.shape,
            )?;
            self.input_output_handle
                .partial_decode(&indexer_reshaped, options)
                .await
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}
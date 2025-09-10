use std::sync::Arc;

use super::{get_squeezed_array_subset, get_squeezed_indexer};

use crate::array::{
    codec::{
        ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions,
    },
    ChunkRepresentation, DataType,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// Generic partial codec for the Squeeze codec.
pub(crate) struct SqueezeCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: ChunkRepresentation,
}

impl<T: ?Sized> SqueezeCodecPartial<T> {
    /// Create a new [`SqueezeCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for SqueezeCodecPartial<T>
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
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_decode(&array_subset_squeezed, options)
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_decode(&indexer_squeezed, options)
        }
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for SqueezeCodecPartial<T>
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
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_encode(&array_subset_squeezed, bytes, options)
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_encode(&indexer_squeezed, bytes, options)
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for SqueezeCodecPartial<T>
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
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_decode(&array_subset_squeezed, options)
                .await
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_decode(&indexer_squeezed, options)
                .await
        }
    }
}

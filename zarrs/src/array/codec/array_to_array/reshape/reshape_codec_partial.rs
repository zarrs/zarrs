use std::sync::Arc;

use super::get_reshaped_indexer;
use crate::array::{DataType, FillValue};
use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::ChunkShape;
use zarrs_storage::StorageError;

/// Partial codec for the Reshape codec.
pub(crate) struct ReshapeCodecPartial<T: ?Sized> {
    input_handle: Arc<T>,
    decoded_shape: ChunkShape,
    encoded_shape: ChunkShape,
    data_type: DataType,
}

impl<T: ?Sized> ArrayPartialEncoderTraits for ReshapeCodecPartial<T>
where
    T: ArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        self.input_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let reshaped_indexer =
            get_reshaped_indexer(indexer, &self.decoded_shape, &self.encoded_shape)?;
        self.input_handle
            .partial_encode(&reshaped_indexer, bytes, options)
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_handle.supports_partial_encode()
    }
}

impl<T: ?Sized> ReshapeCodecPartial<T> {
    /// Create a new [`ReshapeCodecPartial`].
    pub(crate) fn new(
        input_handle: Arc<T>,
        decoded_shape: &[u64],
        data_type: &DataType,
        _fill_value: &FillValue,
        encoded_shape: ChunkShape,
    ) -> Self {
        Self {
            input_handle,
            decoded_shape: decoded_shape.to_vec(),
            encoded_shape,
            data_type: data_type.clone(),
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for ReshapeCodecPartial<T>
where
    T: AsyncArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), CodecError> {
        self.input_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let reshaped_indexer =
            get_reshaped_indexer(indexer, &self.decoded_shape, &self.encoded_shape)?;
        self.input_handle
            .partial_encode(&reshaped_indexer, bytes, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_handle.supports_partial_encode()
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for ReshapeCodecPartial<T>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let reshaped_indexer =
            get_reshaped_indexer(indexer, &self.decoded_shape, &self.encoded_shape)?;
        self.input_handle.partial_decode(&reshaped_indexer, options)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
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
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let reshaped_indexer =
            get_reshaped_indexer(indexer, &self.decoded_shape, &self.encoded_shape)?;
        self.input_handle
            .partial_decode(&reshaped_indexer, options)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

use std::num::NonZeroU64;
use std::sync::Arc;

use super::get_reshaped_indexer;
use super::reshape_codec_grid_mapping::reshape_rectilinear_grid;
use crate::array::{ChunkGrid, DataType};
use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_storage::StorageError;

/// Partial codec for the Reshape codec.
pub(crate) struct ReshapeCodecPartial<T: ?Sized> {
    input_handle: Arc<T>,
    decoded_shape: Vec<NonZeroU64>,
    encoded_shape: Vec<NonZeroU64>,
    data_type: DataType,
}

impl<T: ?Sized> ArrayPartialEncoderTraits for ReshapeCodecPartial<T>
where
    T: ArrayPartialEncoderTraits,
{
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
        decoded_shape: &[NonZeroU64],
        data_type: &DataType,
        encoded_shape: Vec<NonZeroU64>,
    ) -> Self {
        Self {
            input_handle,
            decoded_shape: decoded_shape.to_vec(),
            encoded_shape,
            data_type: data_type.clone(),
        }
    }

    fn map_local_subchunk_grid(
        &self,
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        reshape_rectilinear_grid(
            &self.encoded_shape,
            &self.decoded_shape,
            encoded_subchunk_grid,
        )
        .map_err(|err| CodecError::Other(err.to_string()))
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for ReshapeCodecPartial<T>
where
    T: AsyncArrayPartialEncoderTraits,
{
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

    fn local_subchunk_grid(&self, options: &CodecOptions) -> Result<Option<ChunkGrid>, CodecError> {
        let Some(encoded_subchunk_grid) = self.input_handle.local_subchunk_grid(options)? else {
            return Ok(None);
        };
        self.map_local_subchunk_grid(&encoded_subchunk_grid)
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

    async fn local_subchunk_grid(
        &self,
        options: &CodecOptions,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        let Some(encoded_subchunk_grid) = self.input_handle.local_subchunk_grid(options).await?
        else {
            return Ok(None);
        };
        self.map_local_subchunk_grid(&encoded_subchunk_grid)
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

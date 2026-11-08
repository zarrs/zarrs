use std::sync::Arc;

use super::{get_squeezed_array_subset, get_squeezed_indexer};
use crate::array::{ChunkGrid, DataType};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderSubchunkingTraits, ArrayPartialDecoderTraits,
    ArrayPartialEncoderTraits, ArrayToArrayCodecGeometryTraits, CodecError, CodecOptions,
    Resources,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits,
    AsyncArrayPartialEncoderTraits,
};
use zarrs_storage::StorageError;

use super::squeeze_codec::SqueezeCodecBound;

/// Generic partial codec for the Squeeze codec.
pub(crate) struct SqueezeCodecPartial<T: ?Sized> {
    codec: Arc<SqueezeCodecBound>,
    input_output_handle: Arc<T>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
}

impl<T: ?Sized> SqueezeCodecPartial<T> {
    /// Create a new [`SqueezeCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        codec: Arc<SqueezeCodecBound>,
        input_output_handle: Arc<T>,
        shape: &[NonZeroU64],
        data_type: &DataType,
    ) -> Self {
        Self {
            codec,
            input_output_handle,
            shape: shape.to_vec(),
            data_type: data_type.clone(),
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderSubchunkingTraits for SqueezeCodecPartial<T>
where
    T: ArrayPartialDecoderSubchunkingTraits,
{
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.codec.decoded_local_subchunk_grids(
            &self.shape,
            self.input_output_handle
                .local_subchunk_grids(options, resources)?,
        )
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for SqueezeCodecPartial<T>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle
                .partial_decode(&array_subset_squeezed, options, resources)
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_decode(&indexer_squeezed, options, resources)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for SqueezeCodecPartial<T>
where
    T: ArrayPartialEncoderTraits,
{
    fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle.partial_encode(
                &array_subset_squeezed,
                bytes,
                options,
                resources,
            )
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_encode(&indexer_squeezed, bytes, options, resources)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderSubchunkingTraits for SqueezeCodecPartial<T>
where
    T: AsyncArrayPartialDecoderSubchunkingTraits,
{
    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.codec.decoded_local_subchunk_grids(
            &self.shape,
            self.input_output_handle
                .local_subchunk_grids(options, resources)
                .await?,
        )
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
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle
                .partial_decode(&array_subset_squeezed, options, resources)
                .await
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_decode(&indexer_squeezed, options, resources)
                .await
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for SqueezeCodecPartial<T>
where
    T: AsyncArrayPartialEncoderTraits,
{
    async fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle
                .partial_encode(&array_subset_squeezed, bytes, options, resources)
                .await
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_encode(&indexer_squeezed, bytes, options, resources)
                .await
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

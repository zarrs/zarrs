use std::sync::Arc;

use super::{get_squeezed_array_subset, get_squeezed_indexer};
use crate::array::DataType;
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecTraits,
    CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_storage::StorageError;

/// Generic partial codec for the Squeeze codec.
pub(crate) struct SqueezeCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    codec: Arc<dyn ArrayToArrayCodecTraits>,
}

impl<T: ?Sized> SqueezeCodecPartial<T> {
    /// Create a new [`SqueezeCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        codec: Arc<dyn ArrayToArrayCodecTraits>,
    ) -> Self {
        Self {
            input_output_handle,
            shape: shape.to_vec(),
            data_type: data_type.clone(),
            codec,
        }
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
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle
                .partial_decode(&array_subset_squeezed, options)
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_decode(&indexer_squeezed, options)
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
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle
                .partial_encode(&array_subset_squeezed, bytes, options)
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_encode(&indexer_squeezed, bytes, options)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }

    super::super::subchunk_forwarding::encoder_subchunk_forwarding!(input_output_handle);
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
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle
                .partial_decode(&array_subset_squeezed, options)
                .await
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_decode(&indexer_squeezed, options)
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
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed = get_squeezed_array_subset(array_subset, &self.shape)?;
            self.input_output_handle
                .partial_encode(&array_subset_squeezed, bytes, options)
                .await
        } else {
            let indexer_squeezed = get_squeezed_indexer(indexer, &self.shape)?;
            self.input_output_handle
                .partial_encode(&indexer_squeezed, bytes, options)
                .await
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }

    async unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: zarrs_codec::CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let subset =
            zarrs_codec::async_local_subchunk_subset(self, level, subchunk_indices, options)
                .await?;
        // SAFETY: The caller upholds the safety requirements.
        unsafe {
            self.store_encoded_subchunk_by_subset(level, &subset, bytes, options)
                .await
        }
    }

    async unsafe fn store_encoded_subchunk_by_subset(
        &self,
        level: usize,
        subset: &crate::array::ArraySubset,
        bytes: zarrs_codec::CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let encoded_subset = self.encoded_subset(subset)?;
        // SAFETY: The subchunk is unchanged, only its subset is mapped.
        unsafe {
            self.input_output_handle
                .store_encoded_subchunk_by_subset(level, &encoded_subset, bytes, options)
                .await
        }
    }

    async fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let subset =
            zarrs_codec::async_local_subchunk_subset(self, level, subchunk_indices, options)
                .await?;
        self.erase_subchunk_by_subset(level, &subset, options).await
    }

    async fn erase_subchunk_by_subset(
        &self,
        level: usize,
        subset: &crate::array::ArraySubset,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let encoded_subset = self.encoded_subset(subset)?;
        self.input_output_handle
            .erase_subchunk_by_subset(level, &encoded_subset, options)
            .await
    }
}

super::super::subchunk_forwarding::impl_subchunk_forwarding!(
    SqueezeCodecPartial,
    input_output_handle,
    codec,
    shape
);

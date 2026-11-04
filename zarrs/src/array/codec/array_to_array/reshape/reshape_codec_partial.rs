use std::num::NonZeroU64;
use std::sync::Arc;

use super::get_reshaped_indexer;
use crate::array::DataType;
use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecTraits,
    CodecError, CodecOptions,
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
    codec: Arc<dyn ArrayToArrayCodecTraits>,
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
            .partial_encode(reshaped_indexer.as_indexer(), bytes, options)
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_handle.supports_partial_encode()
    }

    super::super::subchunk_forwarding::encoder_subchunk_forwarding!(input_handle);
}

impl<T: ?Sized> ReshapeCodecPartial<T> {
    /// Create a new [`ReshapeCodecPartial`].
    pub(crate) fn new(
        input_handle: Arc<T>,
        decoded_shape: &[NonZeroU64],
        data_type: &DataType,
        encoded_shape: Vec<NonZeroU64>,
        codec: Arc<dyn ArrayToArrayCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            decoded_shape: decoded_shape.to_vec(),
            encoded_shape,
            data_type: data_type.clone(),
            codec,
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
            .partial_encode(reshaped_indexer.as_indexer(), bytes, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_handle.supports_partial_encode()
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
            self.input_handle
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
        self.input_handle
            .erase_subchunk_by_subset(level, &encoded_subset, options)
            .await
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
        self.input_handle
            .partial_decode(reshaped_indexer.as_indexer(), options)
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
            .partial_decode(reshaped_indexer.as_indexer(), options)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

super::super::subchunk_forwarding::impl_subchunk_forwarding!(
    ReshapeCodecPartial,
    input_handle,
    codec,
    decoded_shape
);

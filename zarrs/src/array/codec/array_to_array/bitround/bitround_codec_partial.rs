use std::sync::Arc;

#[cfg(feature = "async")]
use crate::array::codec::array_to_array::subchunk_forwarding::AsyncSubchunkRemap;
use crate::array::codec::array_to_array::subchunk_forwarding::{
    SubchunkMapping, SubchunkRemap, impl_subchunk_forwarding,
};

use super::{BitroundDataTypeExt, round_bytes};
use crate::array::DataType;
use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_storage::StorageError;

/// Generic partial codec for the bitround codec.
pub(crate) struct BitroundCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    data_type: DataType,
    keepbits: u32,
}

impl<T: ?Sized> BitroundCodecPartial<T> {
    /// Create a new [`BitroundCodecPartial`].
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        data_type: &DataType,
        keepbits: u32,
    ) -> Result<Self, CodecError> {
        data_type.codec_bitround()?;
        Ok(Self {
            input_output_handle,
            data_type: data_type.clone(),
            keepbits,
        })
    }
}
impl<T: ?Sized> ArrayPartialDecoderTraits for BitroundCodecPartial<T>
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
        // Bytes codec does pass-through decoding
        self.input_output_handle.partial_decode(indexer, options)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for BitroundCodecPartial<T>
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
        // For bitround codec, we need to apply the rounding to the input bytes before encoding
        let mut bytes_copy = bytes.clone().into_fixed()?;
        bytes_copy.with_mut(|bytes| round_bytes(bytes, &self.data_type, self.keepbits))?;
        let rounded_bytes = ArrayBytes::from(bytes_copy);

        self.input_output_handle
            .partial_encode(indexer, &rounded_bytes, options)
    }

    unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: zarrs_codec::CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self.remap_subchunk_indices(0, subchunk_indices, options)?;
        // SAFETY: Forwarding preserves the target codec and encoded domain.
        unsafe {
            self.inner()
                .store_encoded_subchunk(&indices, bytes, options)
        }
    }

    unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: zarrs_codec::CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self.remap_subchunk_indices(level, subchunk_indices, options)?;
        // SAFETY: Forwarding preserves the target codec and encoded domain.
        unsafe {
            self.inner()
                .store_encoded_subchunk_at_level(level, &indices, bytes, options)
        }
    }

    fn erase_subchunk(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self.remap_subchunk_indices(0, subchunk_indices, options)?;
        self.inner().erase_subchunk(&indices, options)
    }

    fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self.remap_subchunk_indices(level, subchunk_indices, options)?;
        self.inner()
            .erase_subchunk_at_level(level, &indices, options)
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for BitroundCodecPartial<T>
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
        // Bytes codec does pass-through decoding
        self.input_output_handle
            .partial_decode(indexer, options)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for BitroundCodecPartial<T>
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
        // For bitround codec, we need to apply the rounding to the input bytes before encoding
        let mut bytes_copy = bytes.clone().into_fixed()?;
        bytes_copy.with_mut(|bytes| round_bytes(bytes, &self.data_type, self.keepbits))?;
        let rounded_bytes = ArrayBytes::from(bytes_copy);

        self.input_output_handle
            .partial_encode(indexer, &rounded_bytes, options)
            .await
    }

    async unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: zarrs_codec::CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self
            .async_remap_subchunk_indices(0, subchunk_indices, options)
            .await?;
        // SAFETY: Forwarding preserves the target codec and encoded domain.
        unsafe {
            self.inner()
                .store_encoded_subchunk(&indices, bytes, options)
                .await
        }
    }

    async unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: zarrs_codec::CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self
            .async_remap_subchunk_indices(level, subchunk_indices, options)
            .await?;
        // SAFETY: Forwarding preserves the target codec and encoded domain.
        unsafe {
            self.inner()
                .store_encoded_subchunk_at_level(level, &indices, bytes, options)
                .await
        }
    }

    async fn erase_subchunk(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self
            .async_remap_subchunk_indices(0, subchunk_indices, options)
            .await?;
        self.inner().erase_subchunk(&indices, options).await
    }

    async fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let indices = self
            .async_remap_subchunk_indices(level, subchunk_indices, options)
            .await?;
        self.inner()
            .erase_subchunk_at_level(level, &indices, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

// The codec alters values but not positions, so subchunk grids and indices are the same in its
// decoded and encoded domains.
impl_subchunk_forwarding!(BitroundCodecPartial, identity);

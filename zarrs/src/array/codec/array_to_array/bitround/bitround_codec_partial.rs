use std::sync::Arc;

use super::{BitroundCodec, round_bytes};
#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use crate::array::{
    DataType,
    codec::{
        ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions,
    },
};
use crate::storage::StorageError;
use zarrs_plugin::ExtensionIdentifier;

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
        match data_type {
            super::supported_dtypes!() => Ok(Self {
                input_output_handle,
                data_type: data_type.clone(),
                keepbits,
            }),
            super::unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                BitroundCodec::IDENTIFIER.to_string(),
            )),
        }
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
        indexer: &dyn crate::indexer::Indexer,
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
        // For bitround codec, we need to apply the rounding to the input bytes before encoding
        let mut bytes_copy = bytes.clone().into_fixed()?;
        round_bytes(bytes_copy.to_mut(), &self.data_type, self.keepbits)?;
        let rounded_bytes = ArrayBytes::from(bytes_copy);

        self.input_output_handle
            .partial_encode(indexer, &rounded_bytes, options)
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
        indexer: &dyn crate::indexer::Indexer,
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
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        // For bitround codec, we need to apply the rounding to the input bytes before encoding
        let mut bytes_copy = bytes.clone().into_fixed()?;
        round_bytes(bytes_copy.to_mut(), &self.data_type, self.keepbits)?;
        let rounded_bytes = ArrayBytes::from(bytes_copy);

        self.input_output_handle
            .partial_encode(indexer, &rounded_bytes, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

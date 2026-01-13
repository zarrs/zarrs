use std::sync::Arc;

use super::{
    apply_permutation, get_transposed_array_subset, get_transposed_indexer, inverse_permutation,
    permute,
};
use crate::array::{ArrayBytes, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_storage::StorageError;

/// Generic partial codec for the Transpose codec.
pub(crate) struct TransposeCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    data_type: DataType,
    /// Forward permutation order (for encoding).
    order: Vec<usize>,
    /// Inverse permutation order (for decoding).
    order_inverse: Vec<usize>,
}

impl<T: ?Sized> TransposeCodecPartial<T> {
    /// Create a new [`TransposeCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        order: Vec<usize>,
    ) -> Self {
        let order_inverse = inverse_permutation(&order);
        Self {
            input_output_handle,
            data_type: data_type.clone(),
            order,
            order_inverse,
        }
    }

    /// Encode: apply forward permutation to bytes in decoded shape.
    fn encode<'a>(
        &self,
        bytes: &ArrayBytes<'a>,
        shape: &[u64],
    ) -> Result<ArrayBytes<'a>, CodecError> {
        apply_permutation(bytes, shape, &self.order, &self.data_type)
    }

    /// Decode: apply inverse permutation to bytes in encoded (transposed) shape.
    fn decode<'a>(
        &self,
        bytes: &ArrayBytes<'a>,
        shape: &[u64],
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let transposed_shape: Vec<u64> =
            permute(shape, &self.order).expect("matching dimensionality");
        apply_permutation(
            bytes,
            &transposed_shape,
            &self.order_inverse,
            &self.data_type,
        )
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for TransposeCodecPartial<T>
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
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            let encoded_value = self
                .input_output_handle
                .partial_decode(&array_subset_transposed, options)?;
            self.decode(&encoded_value, &array_subset.shape())
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_output_handle
                .partial_decode(&indexer_transposed, options)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
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
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let encoded_value = self.encode(bytes, &array_subset.shape())?;
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            self.input_output_handle.partial_encode(
                &array_subset_transposed,
                &encoded_value,
                options,
            )
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_output_handle
                .partial_encode(&indexer_transposed, bytes, options)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
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
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            let encoded_value = self
                .input_output_handle
                .partial_decode(&array_subset_transposed, options)
                .await?;
            self.decode(&encoded_value, &array_subset.shape())
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_output_handle
                .partial_decode(&indexer_transposed, options)
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
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for TransposeCodecPartial<T>
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
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let encoded_value = self.encode(bytes, &array_subset.shape())?;
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset)?;
            self.input_output_handle
                .partial_encode(&array_subset_transposed, &encoded_value, options)
                .await
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer)?;
            self.input_output_handle
                .partial_encode(&indexer_transposed, bytes, options)
                .await
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

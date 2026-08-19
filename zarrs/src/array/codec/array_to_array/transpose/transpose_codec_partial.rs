use std::sync::Arc;

#[cfg(feature = "async")]
use crate::array::codec::array_to_array::subchunk_forwarding::AsyncSubchunkRemap;
use crate::array::codec::array_to_array::subchunk_forwarding::{
    SubchunkMapping, SubchunkRemap, impl_subchunk_forwarding,
};

use super::{
    apply_permutation, get_transposed_array_subset, get_transposed_indexer, inverse_permutation,
    permute,
};
use crate::array::{ArrayBytes, ChunkGrid, ChunkShape, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_storage::StorageError;

/// Generic partial codec for the Transpose codec.
pub(crate) struct TransposeCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    shape: ChunkShape,
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        order: Vec<usize>,
    ) -> Self {
        let order_inverse = inverse_permutation(&order);
        Self {
            input_output_handle,
            shape: shape.to_vec(),
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

    fn map_local_subchunk_grid_impl(
        &self,
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<ChunkGrid, CodecError> {
        if self.order_inverse.len() != encoded_subchunk_grid.dimensionality() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match local subchunk grid dimensionality"
                    .to_string(),
            ));
        }

        let array_shape = bytemuck::must_cast_slice(&self.shape).to_vec();
        super::transpose_rectilinear_grid(&self.order_inverse, array_shape, encoded_subchunk_grid)
            .map_err(|err| CodecError::Other(err.to_string()))
    }

    /// Map subchunk indices from the outwardly mapped grid into the grid of the inner decoder.
    ///
    /// This inverts the axis permutation that [`map_local_subchunk_grid`] applies to the grid.
    /// That maps outward as `outward[i] = encoded[order_inverse[i]]`, so mapping back inward is
    /// `encoded[i] = outward[order[i]]`.
    ///
    /// [`map_local_subchunk_grid`]: Self::map_local_subchunk_grid_impl
    fn inner_subchunk_indices(&self, subchunk_indices: &[u64]) -> Result<Vec<u64>, CodecError> {
        permute(subchunk_indices, &self.order).ok_or_else(|| {
            CodecError::Other(format!(
                "subchunk indices {subchunk_indices:?} do not match the dimensionality of the transpose codec `order`"
            ))
        })
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

impl<T: ?Sized> SubchunkMapping for TransposeCodecPartial<T> {
    type Inner = T;

    fn inner(&self) -> &Arc<T> {
        &self.input_output_handle
    }

    fn map_local_subchunk_grid(&self, grid: &ChunkGrid) -> Result<Option<ChunkGrid>, CodecError> {
        self.map_local_subchunk_grid_impl(grid).map(Some)
    }
}

impl<T: ?Sized> SubchunkRemap for TransposeCodecPartial<T> {
    fn remap_subchunk_indices(
        &self,
        _level: usize,
        subchunk_indices: &[u64],
        _options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError> {
        self.inner_subchunk_indices(subchunk_indices)
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncSubchunkRemap for TransposeCodecPartial<T>
where
    T: zarrs_storage::MaybeSend + zarrs_storage::MaybeSync,
{
    async fn async_remap_subchunk_indices(
        &self,
        _level: usize,
        subchunk_indices: &[u64],
        _options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError> {
        self.inner_subchunk_indices(subchunk_indices)
    }
}

impl_subchunk_forwarding!(TransposeCodecPartial);

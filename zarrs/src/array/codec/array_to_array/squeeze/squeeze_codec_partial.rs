use std::sync::Arc;

use super::{get_squeezed_array_subset, get_squeezed_indexer};
use crate::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid};
use crate::array::{ChunkGrid, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderSubchunkingTraits, ArrayPartialDecoderTraits,
    ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits,
    AsyncArrayPartialEncoderTraits,
};
use zarrs_storage::StorageError;

/// Generic partial codec for the Squeeze codec.
pub(crate) struct SqueezeCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
}

impl<T: ?Sized> SqueezeCodecPartial<T> {
    /// Create a new [`SqueezeCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
    ) -> Self {
        Self {
            input_output_handle,
            shape: shape.to_vec(),
            data_type: data_type.clone(),
        }
    }

    fn map_local_subchunk_grid(
        &self,
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<ChunkGrid, CodecError> {
        let expected_dimensionality = self.shape.iter().filter(|dim| dim.get() > 1).count().max(1);
        if encoded_subchunk_grid.dimensionality() != expected_dimensionality {
            return Err(CodecError::Other(
                "local subchunk grid dimensionality is incompatible with squeeze encoded dimensionality"
                    .to_string(),
            ));
        }

        let mut encoded_dim = 0;
        let chunk_shapes = self
            .shape
            .iter()
            .map(|dim| {
                if dim.get() == 1 {
                    Ok(ChunkEdgeLengths::Scalar(NonZeroU64::new(1).unwrap()))
                } else {
                    let edge_lengths = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
                    encoded_dim += 1;
                    Ok(ChunkEdgeLengths::encode(&edge_lengths))
                }
            })
            .collect::<Result<Vec<_>, zarrs_chunk_grid::ChunkGridCreateError>>()
            .map_err(|err| CodecError::Other(err.to_string()))?;
        let array_shape = bytemuck::must_cast_slice(&self.shape).to_vec();
        Ok(ChunkGrid::new(
            RectilinearChunkGrid::new(array_shape, &chunk_shapes)
                .map_err(|err| CodecError::Other(err.to_string()))?,
        ))
    }

    /// Map subchunk indices from the outwardly mapped grid into the grid of the inner decoder.
    ///
    /// This inverts what [`map_local_subchunk_grid`] does to the grid: it splices a unit edge back
    /// in at every squeezed axis, so mapping inward drops the indices at those axes. A squeezed
    /// axis has extent one, so its index must be zero.
    ///
    /// [`map_local_subchunk_grid`]: Self::map_local_subchunk_grid
    fn inner_subchunk_indices(&self, subchunk_indices: &[u64]) -> Result<Vec<u64>, CodecError> {
        if subchunk_indices.len() != self.shape.len() {
            return Err(CodecError::Other(format!(
                "subchunk indices {subchunk_indices:?} do not match the dimensionality of the squeeze codec decoded shape {:?}",
                self.shape
            )));
        }
        let inner_indices: Vec<u64> = std::iter::zip(subchunk_indices, &self.shape)
            .filter_map(|(indices, dim)| (dim.get() > 1).then_some(*indices))
            .collect();
        // Squeeze always leaves at least one axis, matching `map_local_subchunk_grid`
        if inner_indices.is_empty() {
            return Ok(vec![0]);
        }
        Ok(inner_indices)
    }
}

impl<T: ?Sized> ArrayPartialDecoderSubchunkingTraits for SqueezeCodecPartial<T>
where
    T: ArrayPartialDecoderSubchunkingTraits,
{
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.input_output_handle
            .local_subchunk_grids(options)?
            .into_iter()
            .map(|grid| {
                grid.map(|grid| self.map_local_subchunk_grid(&grid))
                    .transpose()
            })
            .collect()
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        self.input_output_handle.subchunk_codecs()
    }

    fn encoded_subchunk_shape_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<zarrs_metadata::ChunkShape, CodecError> {
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .encoded_subchunk_shape_at_level(level, &inner_indices, options)
    }

    fn retrieve_encoded_subchunk_bytes(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<zarrs_codec::ArrayBytesRaw<'_>>, CodecError> {
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .retrieve_encoded_subchunk_bytes(&inner_indices, options)
    }

    fn encoded_subchunk_partial_decoder(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<dyn zarrs_codec::BytesPartialDecoderTraits>>, CodecError> {
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .encoded_subchunk_partial_decoder(&inner_indices, options)
    }

    fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<zarrs_codec::EncodedSubchunk<'static>>, CodecError> {
        // Delegate the whole descent so the inner decoder descends in one consistent domain
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .retrieve_encoded_subchunk_at_level(level, &inner_indices, options)
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
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.input_output_handle
            .local_subchunk_grids(options)
            .await?
            .into_iter()
            .map(|grid| {
                grid.map(|grid| self.map_local_subchunk_grid(&grid))
                    .transpose()
            })
            .collect()
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        self.input_output_handle.subchunk_codecs()
    }

    async fn encoded_subchunk_shape_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<zarrs_metadata::ChunkShape, CodecError> {
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .encoded_subchunk_shape_at_level(level, &inner_indices, options)
            .await
    }

    async fn retrieve_encoded_subchunk_bytes<'a>(
        &'a self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<zarrs_codec::ArrayBytesRaw<'a>>, CodecError> {
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .retrieve_encoded_subchunk_bytes(&inner_indices, options)
            .await
    }

    async fn encoded_subchunk_partial_decoder(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<dyn zarrs_codec::AsyncBytesPartialDecoderTraits>>, CodecError> {
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .encoded_subchunk_partial_decoder(&inner_indices, options)
            .await
    }

    async fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<zarrs_codec::EncodedSubchunk<'static>>, CodecError> {
        // Delegate the whole descent so the inner decoder descends in one consistent domain
        let inner_indices = self.inner_subchunk_indices(subchunk_indices)?;
        self.input_output_handle
            .retrieve_encoded_subchunk_at_level(level, &inner_indices, options)
            .await
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
}

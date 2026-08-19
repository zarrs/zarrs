use std::num::NonZeroU64;
use std::sync::Arc;

use super::get_reshaped_indexer;
use super::reshape_codec_grid_mapping::reshape_rectilinear_grid;
use crate::array::{ChunkGrid, DataType};
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

    /// Map subchunk indices from the outwardly mapped grid into the grid of the inner decoder.
    ///
    /// A reshape never moves elements, it only re-splits the row-major element sequence. A subchunk
    /// therefore occupies the same contiguous interval of that sequence in both domains, so the
    /// outward subchunk is located in the encoded grid by mapping its first element through the
    /// linear index.
    ///
    /// This errors if the subchunk grid cannot be mapped between the domains at all, and if an
    /// outward subchunk does not correspond to exactly one encoded subchunk.
    fn inner_subchunk_indices(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError>
    where
        T: ArrayPartialDecoderSubchunkingTraits,
    {
        let encoded_grid = self
            .input_handle
            .local_subchunk_grid_at_level(level, options)?
            .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
        let decoded_grid = self
            .map_local_subchunk_grid(&encoded_grid)?
            .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
        self.map_subchunk_indices(&decoded_grid, &encoded_grid, subchunk_indices)
    }

    /// Locate `subchunk_indices` of `decoded_grid` in `encoded_grid` through the linear index.
    fn map_subchunk_indices(
        &self,
        decoded_grid: &ChunkGrid,
        encoded_grid: &ChunkGrid,
        subchunk_indices: &[u64],
    ) -> Result<Vec<u64>, CodecError> {
        let decoded_subset = decoded_grid.subset(subchunk_indices)?.ok_or_else(|| {
            CodecError::Other(format!("invalid subchunk indices {subchunk_indices:?}"))
        })?;

        // A reshape preserves the row-major element order, so the subchunk starts at the same
        // linear element index in both domains
        let decoded_shape = bytemuck::must_cast_slice(&self.decoded_shape);
        let encoded_shape = bytemuck::must_cast_slice(&self.encoded_shape);
        let linear_index = zarrs_chunk_grid::ravel_indices(decoded_subset.start(), decoded_shape)
            .ok_or_else(|| {
                CodecError::Other(format!(
                    "subchunk indices {subchunk_indices:?} are out of bounds of the reshape codec decoded shape {decoded_shape:?}"
                ))
            })?;
        let encoded_start = crate::array::unravel_index(linear_index, encoded_shape).ok_or_else(|| {
            CodecError::Other(format!(
                "subchunk indices {subchunk_indices:?} do not map into the reshape codec encoded shape {encoded_shape:?}"
            ))
        })?;

        let inner_indices = encoded_grid.chunk_indices(&encoded_start)?.ok_or_else(|| {
            CodecError::Other(format!("invalid subchunk indices {subchunk_indices:?}"))
        })?;

        // The subchunk must occupy exactly one encoded subchunk, or its bytes are not a single
        // encoded subchunk. `reshape_rectilinear_grid` only maps a grid whose subchunk boundaries
        // align in both domains, so this is not reachable through the outwardly mapped grid, but it
        // guards the invariant that the byte interval of a subchunk is the same in both domains.
        let encoded_subset = encoded_grid.subset(&inner_indices)?.ok_or_else(|| {
            CodecError::Other(format!("invalid subchunk indices {inner_indices:?}"))
        })?;
        if encoded_subset.num_elements() != decoded_subset.num_elements() {
            return Err(CodecError::Other(format!(
                "reshaped subchunk {subchunk_indices:?} of {} elements does not map to a single encoded subchunk of {} elements",
                decoded_subset.num_elements(),
                encoded_subset.num_elements(),
            )));
        }
        Ok(inner_indices)
    }
}

#[cfg(feature = "async")]
impl<T: ?Sized> ReshapeCodecPartial<T>
where
    T: AsyncArrayPartialDecoderSubchunkingTraits,
{
    /// Async variant of [`inner_subchunk_indices`](Self::inner_subchunk_indices).
    async fn async_inner_subchunk_indices(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError> {
        let encoded_grid = self
            .input_handle
            .local_subchunk_grid_at_level(level, options)
            .await?
            .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
        let decoded_grid = self
            .map_local_subchunk_grid(&encoded_grid)?
            .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
        self.map_subchunk_indices(&decoded_grid, &encoded_grid, subchunk_indices)
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

impl<T: ?Sized> ArrayPartialDecoderSubchunkingTraits for ReshapeCodecPartial<T>
where
    T: ArrayPartialDecoderSubchunkingTraits,
{
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.input_handle
            .local_subchunk_grids(options)?
            .into_iter()
            .map(|grid| grid.map_or(Ok(None), |grid| self.map_local_subchunk_grid(&grid)))
            .collect()
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        self.input_handle.subchunk_codecs()
    }

    fn encoded_subchunk_shape_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<zarrs_metadata::ChunkShape, CodecError> {
        let inner_indices = self.inner_subchunk_indices(level, subchunk_indices, options)?;
        self.input_handle
            .encoded_subchunk_shape_at_level(level, &inner_indices, options)
    }

    fn retrieve_encoded_subchunk_bytes(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<zarrs_codec::ArrayBytesRaw<'_>>, CodecError> {
        let inner_indices = self.inner_subchunk_indices(0, subchunk_indices, options)?;
        self.input_handle
            .retrieve_encoded_subchunk_bytes(&inner_indices, options)
    }

    fn encoded_subchunk_partial_decoder(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<dyn zarrs_codec::BytesPartialDecoderTraits>>, CodecError> {
        let inner_indices = self.inner_subchunk_indices(0, subchunk_indices, options)?;
        self.input_handle
            .encoded_subchunk_partial_decoder(&inner_indices, options)
    }

    fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<zarrs_codec::EncodedSubchunk<'static>>, CodecError> {
        // Delegate the whole descent so the inner decoder descends in one consistent domain
        let inner_indices = self.inner_subchunk_indices(level, subchunk_indices, options)?;
        self.input_handle
            .retrieve_encoded_subchunk_at_level(level, &inner_indices, options)
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
impl<T: ?Sized> AsyncArrayPartialDecoderSubchunkingTraits for ReshapeCodecPartial<T>
where
    T: AsyncArrayPartialDecoderSubchunkingTraits,
{
    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.input_handle
            .local_subchunk_grids(options)
            .await?
            .into_iter()
            .map(|grid| grid.map_or(Ok(None), |grid| self.map_local_subchunk_grid(&grid)))
            .collect()
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        self.input_handle.subchunk_codecs()
    }

    async fn encoded_subchunk_shape_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<zarrs_metadata::ChunkShape, CodecError> {
        let inner_indices = self
            .async_inner_subchunk_indices(level, subchunk_indices, options)
            .await?;
        self.input_handle
            .encoded_subchunk_shape_at_level(level, &inner_indices, options)
            .await
    }

    async fn retrieve_encoded_subchunk_bytes<'a>(
        &'a self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<zarrs_codec::ArrayBytesRaw<'a>>, CodecError> {
        let inner_indices = self
            .async_inner_subchunk_indices(0, subchunk_indices, options)
            .await?;
        self.input_handle
            .retrieve_encoded_subchunk_bytes(&inner_indices, options)
            .await
    }

    async fn encoded_subchunk_partial_decoder(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<dyn zarrs_codec::AsyncBytesPartialDecoderTraits>>, CodecError> {
        let inner_indices = self
            .async_inner_subchunk_indices(0, subchunk_indices, options)
            .await?;
        self.input_handle
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
        let inner_indices = self
            .async_inner_subchunk_indices(level, subchunk_indices, options)
            .await?;
        self.input_handle
            .retrieve_encoded_subchunk_at_level(level, &inner_indices, options)
            .await
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

use std::sync::Arc;

#[cfg(feature = "async")]
use super::{AsyncArrayCache, AsyncCachedArray};
use super::{CachedArrayMeta, SyncArrayCache, SyncCachedArray};
use crate::array::{
    ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, ChunkShape, CodecOptions, Indexer,
    Resources,
};
use zarrs_codec::{
    ArrayBytesTarget, CodecError, InvalidNumberOfElementsError, copy_fill_value_into,
};
use zarrs_storage::StorageError;

mod decoded;
mod encoded;
mod partial_decoder;
#[cfg(feature = "async")]
mod partial_decoder_async;

pub(super) fn cache_error(error: Arc<ArrayError>) -> ArrayError {
    Arc::try_unwrap(error)
        .unwrap_or_else(|error| ArrayError::StorageError(StorageError::from(error.to_string())))
}

/// Validate that `chunk_indices` are within the chunk grid of `array`.
pub(super) fn validate_chunk_grid_indices(
    array: &(impl CachedArrayMeta + ?Sized),
    chunk_indices: &[u64],
) -> Result<(), ArrayError> {
    if chunk_indices.len() != array.dimensionality()
        || chunk_indices
            .iter()
            .zip(array.chunk_grid_shape())
            .any(|(&index, &size)| index >= size)
    {
        return Err(ArrayError::InvalidChunkGridIndicesError(
            chunk_indices.to_vec(),
        ));
    }
    Ok(())
}

/// Return an error if `output_target` does not have `expected` elements.
pub(super) fn validate_target_num_elements(
    output_target: &ArrayBytesTarget<'_>,
    expected: u64,
) -> Result<(), CodecError> {
    if output_target.num_elements() != expected {
        return Err(
            InvalidNumberOfElementsError::new(output_target.num_elements(), expected).into(),
        );
    }
    Ok(())
}

pub(super) fn validate_partial_chunk_indexer(
    array: &(impl CachedArrayMeta + ?Sized),
    chunk_indices: &[u64],
    indexer: &dyn Indexer,
    output_target: &ArrayBytesTarget<'_>,
) -> Result<ChunkShape, ArrayError> {
    let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
    let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
    crate::array::array_ops::validate_partial_chunk_subset(indexer, chunk_shape_u64)?;
    indexer
        .validate(chunk_shape_u64)
        .map_err(CodecError::from)?;
    validate_target_num_elements(output_target, indexer.len())?;
    Ok(chunk_shape)
}

pub(super) fn copy_array_fill_value_into(
    array: &(impl CachedArrayMeta + ?Sized),
    output_target: &mut ArrayBytesTarget<'_>,
) -> Result<(), ArrayError> {
    copy_fill_value_into(array.data_type(), array.fill_value(), output_target)
        .map_err(ArrayError::CodecError)
}

pub(super) fn validate_chunk_indices(
    array: &(impl CachedArrayMeta + ?Sized),
    chunk_indices: &[u64],
) -> Result<ChunkShape, ArrayError> {
    validate_chunk_grid_indices(array, chunk_indices)?;
    array.chunk_shape(chunk_indices)
}

/// Retrieve `overlap`, a subset of the chunk at `chunk_indices` with array subset `chunk_subset`.
///
/// The complete chunk is retrieved without subsetting if `overlap` covers it.
pub(crate) fn retrieve_chunk_overlap_bytes<C>(
    cache: &C,
    array: &dyn SyncCachedArray,
    chunk_indices: &[u64],
    chunk_subset: &ArraySubset,
    overlap: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    C: SyncArrayCache + ?Sized,
{
    if *chunk_subset == overlap {
        cache.cached_chunk_bytes(array, chunk_indices, options, resources)
    } else {
        cache.cached_partial_chunk_bytes(
            array,
            chunk_indices,
            &overlap.relative_to(chunk_subset.start())?,
            options,
            resources,
        )
    }
}

/// Retrieve `overlap`, a subset of the chunk at `chunk_indices` with array subset `chunk_subset`,
/// directly into `output_target`.
#[expect(clippy::too_many_arguments)]
pub(crate) fn retrieve_chunk_overlap_into<C>(
    cache: &C,
    array: &dyn SyncCachedArray,
    chunk_indices: &[u64],
    chunk_subset: &ArraySubset,
    overlap: &dyn ArraySubsetTraits,
    output_target: &mut ArrayBytesTarget<'_>,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<(), ArrayError>
where
    C: SyncArrayCache + ?Sized,
{
    if *chunk_subset == overlap {
        cache.cached_chunk_into(array, chunk_indices, output_target, options, resources)
    } else {
        cache.cached_partial_chunk_into(
            array,
            chunk_indices,
            &overlap.relative_to(chunk_subset.start())?,
            output_target,
            options,
            resources,
        )
    }
}

/// Asynchronous version of [`retrieve_chunk_overlap_bytes`].
#[cfg(feature = "async")]
pub(crate) async fn async_retrieve_chunk_overlap_bytes<C>(
    cache: &C,
    array: &dyn AsyncCachedArray,
    chunk_indices: &[u64],
    chunk_subset: &ArraySubset,
    overlap: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    C: AsyncArrayCache + ?Sized,
{
    if *chunk_subset == overlap {
        cache
            .async_cached_chunk_bytes(array, chunk_indices, options, resources)
            .await
    } else {
        cache
            .async_cached_partial_chunk_bytes(
                array,
                chunk_indices,
                &overlap.relative_to(chunk_subset.start())?,
                options,
                resources,
            )
            .await
    }
}

/// Asynchronous version of [`retrieve_chunk_overlap_into`].
#[cfg(feature = "async")]
#[expect(clippy::too_many_arguments)]
pub(crate) async fn async_retrieve_chunk_overlap_into<C>(
    cache: &C,
    array: &dyn AsyncCachedArray,
    chunk_indices: &[u64],
    chunk_subset: &ArraySubset,
    overlap: &dyn ArraySubsetTraits,
    output_target: &mut ArrayBytesTarget<'_>,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<(), ArrayError>
where
    C: AsyncArrayCache + ?Sized,
{
    if *chunk_subset == overlap {
        cache
            .async_cached_chunk_into(array, chunk_indices, output_target, options, resources)
            .await
    } else {
        cache
            .async_cached_partial_chunk_into(
                array,
                chunk_indices,
                &overlap.relative_to(chunk_subset.start())?,
                output_target,
                options,
                resources,
            )
            .await
    }
}

/// Expose an in-memory synchronous partial decoder as an asynchronous partial decoder.
///
/// The wrapped decoder must not perform storage operations (i.e. it must be backed by
/// in-memory chunk data), since its methods are called directly from an asynchronous context.
#[cfg(feature = "async")]
pub(super) struct SyncPartialDecoderAsAsync(Arc<dyn zarrs_codec::SyncArrayPartialDecoderTraits>);

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl zarrs_codec::AsyncArrayPartialDecoderSubchunkingTraits for SyncPartialDecoderAsAsync {
    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<zarrs_chunk_grid::ChunkGrid>>, CodecError> {
        self.0.local_subchunk_grids(options, resources)
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl zarrs_codec::AsyncArrayPartialDecoderTraits for SyncPartialDecoderAsAsync {
    fn data_type(&self) -> &zarrs_data_type::DataType {
        self.0.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.0.exists()
    }

    fn size_held(&self) -> usize {
        self.0.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.0.partial_decode(indexer, options, resources)
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn crate::array::Indexer,
        output_target: &mut zarrs_codec::ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        self.0
            .partial_decode_into(indexer, output_target, options, resources)
    }

    fn supports_partial_decode(&self) -> bool {
        self.0.supports_partial_decode()
    }
}

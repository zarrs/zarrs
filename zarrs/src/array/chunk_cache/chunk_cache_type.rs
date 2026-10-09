use std::sync::Arc;

#[cfg(feature = "async")]
use super::{AsyncChunkCache, SealedAsync};
use super::{ChunkCache, SealedSync};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, ChunkShape, ChunkShapeTraits,
    CodecOptions, Resources,
};
use zarrs_codec::CodecError;
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{ReadableStorageTraits, StorageError};

mod decoded;
mod encoded;
mod partial_decoder;
#[cfg(feature = "async")]
mod partial_decoder_async;

pub(super) fn cache_error(error: Arc<ArrayError>) -> ArrayError {
    Arc::try_unwrap(error)
        .unwrap_or_else(|error| ArrayError::StorageError(StorageError::from(error.to_string())))
}

pub(super) fn validate_chunk_indices<TStorage: ?Sized>(
    array: &Array<TStorage>,
    chunk_indices: &[u64],
) -> Result<ChunkShape, ArrayError> {
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
    array.chunk_shape(chunk_indices)
}

pub(crate) fn fill_value_bytes(
    array: &Array<impl ?Sized>,
    num_elements: u64,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
    Ok(
        ArrayBytes::new_fill_value(array.data_type(), num_elements, array.fill_value())
            .map_err(CodecError::from)
            .map_err(ArrayError::from)?
            .into(),
    )
}

pub(crate) fn retrieve_chunk_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
{
    if let Some(bytes) =
        C::Value::retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options, resources)?
    {
        Ok(bytes)
    } else {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        fill_value_bytes(array, chunk_shape.num_elements_u64())
    }
}

/// Retrieve `overlap`, a subset of the chunk at `chunk_indices` with array subset `chunk_subset`.
///
/// The complete chunk is retrieved without subsetting if `overlap` covers it.
pub(crate) fn retrieve_chunk_overlap_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    chunk_subset: &ArraySubset,
    overlap: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
{
    if *chunk_subset == overlap {
        retrieve_chunk_bytes(cache, array, chunk_indices, options, resources)
    } else {
        C::Value::retrieve_partial_chunk_bytes(
            cache,
            array,
            chunk_indices,
            &overlap.relative_to(chunk_subset.start())?,
            options,
            resources,
        )
    }
}

#[cfg(feature = "async")]
pub(crate) async fn async_retrieve_chunk_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    if let Some(bytes) = C::Value::async_retrieve_chunk_bytes_if_exists(
        cache,
        array,
        chunk_indices,
        options,
        resources,
    )
    .await?
    {
        Ok(bytes)
    } else {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        fill_value_bytes(array, chunk_shape.num_elements_u64())
    }
}

/// Asynchronous version of [`retrieve_chunk_overlap_bytes`].
#[cfg(feature = "async")]
pub(crate) async fn async_retrieve_chunk_overlap_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    chunk_subset: &ArraySubset,
    overlap: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    if *chunk_subset == overlap {
        async_retrieve_chunk_bytes(cache, array, chunk_indices, options, resources).await
    } else {
        C::Value::async_retrieve_partial_chunk_bytes(
            cache,
            array,
            chunk_indices,
            &overlap.relative_to(chunk_subset.start())?,
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
pub(super) struct SyncPartialDecoderAsAsync(Arc<dyn zarrs_codec::ArrayPartialDecoderTraits>);

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
        output_target: zarrs_codec::ArrayBytesDecodeIntoTarget<'_>,
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

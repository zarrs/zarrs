use std::sync::Arc;

#[cfg(feature = "async")]
use super::AsyncChunkCacheType;
use super::{ChunkCache, SyncChunkCacheType};
use crate::array::{Array, ArrayBytes, ArrayError, ChunkShape, ChunkShapeTraits, CodecOptions};
use zarrs_codec::CodecError;
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{ReadableStorageTraits, StorageError};

mod decoded;
mod encoded;
mod partial_decoder;

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

#[ambisync::ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncReadableStorageTraits => ReadableStorageTraits,
            AsyncChunkCacheType => SyncChunkCacheType,
        ),
    ),
    async(feature = "async"),
)]
pub(crate) async fn async_retrieve_chunk_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
    C::Value: AsyncChunkCacheType,
{
    if let Some(bytes) =
        C::Value::async_retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options).await?
    {
        Ok(bytes)
    } else {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        fill_value_bytes(array, chunk_shape.num_elements_u64())
    }
}

/// Return a cached value or insert the value returned by `f`.
///
/// A free-function counterpart of [`ChunkCache::try_get_or_insert_with`] with the same call
/// shape as `async_try_get_or_insert_with`, so cached retrieval can be written once in async
/// form and derived synchronously.
pub(super) fn try_get_or_insert_with<C, F>(
    cache: &C,
    chunk_indices: Vec<u64>,
    f: F,
) -> Result<C::Value, Arc<ArrayError>>
where
    C: ChunkCache + ?Sized,
    F: FnOnce() -> Result<C::Value, ArrayError>,
{
    cache.try_get_or_insert_with(chunk_indices, f)
}

/// Return a cached value or insert the value returned by the asynchronous `f`.
///
/// Unlike [`ChunkCache::try_get_or_insert_with`], concurrent calls for an uncached chunk may
/// each invoke `f`; only one of the returned values is retained by the cache.
#[cfg(feature = "async")]
pub(super) async fn async_try_get_or_insert_with<C, F>(
    cache: &C,
    chunk_indices: Vec<u64>,
    f: F,
) -> Result<C::Value, Arc<ArrayError>>
where
    C: ChunkCache + ?Sized,
    F: AsyncFnOnce() -> Result<C::Value, ArrayError>,
{
    if let Some(value) = cache.get(&chunk_indices) {
        return Ok(value);
    }
    let value = f().await.map_err(Arc::new)?;
    cache.try_get_or_insert_with(chunk_indices, move || Ok(value))
}

/// Expose an in-memory synchronous partial decoder as an asynchronous partial decoder.
///
/// The wrapped decoder must not perform storage operations (i.e. it must be backed by
/// in-memory chunk data), since its methods are called directly from an asynchronous context.
#[cfg(feature = "async")]
pub(super) struct SyncPartialDecoderAsAsync(pub Arc<dyn zarrs_codec::ArrayPartialDecoderTraits>);

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

    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<zarrs_chunk_grid::ChunkGrid>>, CodecError> {
        self.0.local_subchunk_grids(options)
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.0.partial_decode(indexer, options)
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn crate::array::Indexer,
        output_target: zarrs_codec::ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.0.partial_decode_into(indexer, output_target, options)
    }

    fn supports_partial_decode(&self) -> bool {
        self.0.supports_partial_decode()
    }
}

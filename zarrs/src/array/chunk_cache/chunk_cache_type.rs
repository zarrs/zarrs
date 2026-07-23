use std::sync::Arc;

use super::{ChunkCache, ChunkCacheType};
use crate::array::{Array, ArrayBytes, ArrayError, ChunkShape, ChunkShapeTraits, CodecOptions};
use zarrs_codec::CodecError;
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

pub(crate) fn retrieve_chunk_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
{
    if let Some(bytes) =
        C::Value::retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)?
    {
        Ok(bytes)
    } else {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        fill_value_bytes(array, chunk_shape.num_elements())
    }
}

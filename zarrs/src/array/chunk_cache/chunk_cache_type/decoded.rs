use std::sync::Arc;

use super::{cache_error, fill_value_bytes, validate_chunk_indices};
use crate::array::chunk_cache::{ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubsetTraits, ChunkShape, CodecOptions, DataType,
    FillValue, Indexer,
};
use zarrs_codec::{ArrayPartialDecoderTraits, CodecError};
use zarrs_storage::{ReadableStorageTraits, StorageError};

struct CachedArrayBytesPartialDecoder {
    bytes: ChunkCacheTypeDecoded,
    shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
}

impl ArrayPartialDecoderTraits for CachedArrayBytesPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.bytes.is_some())
    }

    fn size_held(&self) -> usize {
        self.bytes.size()
    }

    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<zarrs_chunk_grid::ChunkGrid>>, CodecError> {
        Ok(Vec::new())
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(bytes) = &self.bytes {
            Ok(bytes.extract_array_subset(
                indexer,
                bytemuck::must_cast_slice(&self.shape),
                &self.data_type,
            )?)
        } else {
            Ok(ArrayBytes::new_fill_value(
                &self.data_type,
                indexer.len(),
                &self.fill_value,
            )?)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

impl ChunkCacheType for ChunkCacheTypeDecoded {
    fn size(&self) -> usize {
        self.as_ref().map_or(0, |value| value.size())
    }

    fn partial_decoder<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let bytes = Self::retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)?;
        Ok(Arc::new(CachedArrayBytesPartialDecoder {
            bytes,
            shape: validate_chunk_indices(array, chunk_indices)?,
            data_type: array.data_type().clone(),
            fill_value: array.fill_value().clone(),
        }))
    }

    fn retrieve_chunk_bytes_if_exists<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        validate_chunk_indices(array, chunk_indices)?;
        cache
            .try_get_or_insert_with(chunk_indices.to_vec(), || {
                Ok(array
                    .retrieve_chunk_if_exists_opt::<ArrayBytes<'static>>(chunk_indices, options)?
                    .map(Arc::new))
            })
            .map_err(cache_error)
    }

    fn retrieve_chunk_subset_bytes<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        if let Some(chunk) =
            Self::retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)?
        {
            let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
            Ok(chunk
                .extract_array_subset(
                    chunk_subset,
                    bytemuck::must_cast_slice(&chunk_shape),
                    array.data_type(),
                )?
                .into_owned()
                .into())
        } else {
            fill_value_bytes(array, chunk_subset.num_elements())
        }
    }
}

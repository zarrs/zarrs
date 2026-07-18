use std::sync::Arc;

use super::{cache_error, validate_chunk_indices};
use crate::array::chunk_cache::{
    ChunkCache, ChunkCacheType, ChunkCacheTypePartialDecoder, SyncChunkCacheType,
};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, CodecOptions,
    chunk_shape_to_array_shape,
};
use zarrs_codec::ArrayPartialDecoderTraits;
use zarrs_storage::ReadableStorageTraits;

impl ChunkCacheType for ChunkCacheTypePartialDecoder {
    fn size(&self) -> usize {
        self.as_ref().size_held()
    }
}

impl SyncChunkCacheType for ChunkCacheTypePartialDecoder {
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
        validate_chunk_indices(array, chunk_indices)?;
        cache
            .try_get_or_insert_with(chunk_indices.to_vec(), || {
                array.partial_decoder_opt(chunk_indices, options)
            })
            .map_err(cache_error)
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
        let shape = chunk_shape_to_array_shape(&validate_chunk_indices(array, chunk_indices)?);
        let decoder = Self::partial_decoder(cache, array, chunk_indices, options)?;
        if decoder.exists()? {
            Ok(Some(
                decoder
                    .partial_decode(&ArraySubset::new_with_shape(shape), options)?
                    .into_owned()
                    .into(),
            ))
        } else {
            Ok(None)
        }
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
        Ok(Self::partial_decoder(cache, array, chunk_indices, options)?
            .partial_decode(chunk_subset, options)?
            .into_owned()
            .into())
    }
}

use std::sync::Arc;

use ambisync::ambisync;

#[cfg(feature = "async")]
use super::async_try_get_or_insert_with;
use super::{cache_error, try_get_or_insert_with, validate_chunk_indices};
#[cfg(feature = "async")]
use crate::array::chunk_cache::{AsyncChunkCacheType, ChunkCacheTypeAsyncPartialDecoder};
use crate::array::chunk_cache::{
    ChunkCache, ChunkCacheType, ChunkCacheTypePartialDecoder, SyncChunkCacheType,
};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, CodecOptions,
    chunk_shape_to_array_shape,
};
use zarrs_codec::ArrayPartialDecoderTraits;
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::ReadableStorageTraits;

#[ambisync(
    sync(types(ChunkCacheTypeAsyncPartialDecoder => ChunkCacheTypePartialDecoder)),
    async(feature = "async"),
)]
impl ChunkCacheType for ChunkCacheTypeAsyncPartialDecoder {
    fn size(&self) -> usize {
        self.as_ref().size_held()
    }
}

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncChunkCacheType => SyncChunkCacheType,
            ChunkCacheTypeAsyncPartialDecoder => ChunkCacheTypePartialDecoder,
            AsyncReadableStorageTraits => ReadableStorageTraits,
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
impl AsyncChunkCacheType for ChunkCacheTypeAsyncPartialDecoder {
    async fn async_partial_decoder<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        validate_chunk_indices(array, chunk_indices)?;
        async_try_get_or_insert_with(cache, chunk_indices.to_vec(), async || {
            array
                .async_partial_decoder_opt(chunk_indices, options)
                .await
        })
        .await
        .map_err(cache_error)
    }

    async fn async_retrieve_chunk_bytes_if_exists<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let shape = chunk_shape_to_array_shape(&validate_chunk_indices(array, chunk_indices)?);
        let decoder = Self::async_partial_decoder(cache, array, chunk_indices, options).await?;
        if decoder.exists().await? {
            Ok(Some(
                decoder
                    .partial_decode(&ArraySubset::new_with_shape(shape), options)
                    .await?
                    .into_owned()
                    .into(),
            ))
        } else {
            Ok(None)
        }
    }

    async fn async_retrieve_chunk_subset_bytes<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        Ok(
            Self::async_partial_decoder(cache, array, chunk_indices, options)
                .await?
                .partial_decode(chunk_subset, options)
                .await?
                .into_owned()
                .into(),
        )
    }
}

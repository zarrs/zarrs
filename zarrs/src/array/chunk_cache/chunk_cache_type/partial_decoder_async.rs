use std::sync::Arc;

use super::{cache_error, validate_chunk_indices};
use crate::array::chunk_cache::{
    AsyncChunkCache, AsyncChunkCacheType, ChunkCacheType, ChunkCacheTypeAsyncPartialDecoder,
};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, CodecOptions,
    chunk_shape_to_array_shape,
};
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_storage::AsyncReadableStorageTraits;

impl ChunkCacheType for ChunkCacheTypeAsyncPartialDecoder {
    fn size(&self) -> usize {
        self.as_ref().size_held()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncChunkCacheType for ChunkCacheTypeAsyncPartialDecoder {
    async fn async_partial_decoder<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        validate_chunk_indices(array, chunk_indices)?;
        cache
            .try_get_or_insert_with(chunk_indices.to_vec(), async move {
                array
                    .async_partial_decoder_with_options(chunk_indices, options)
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
        C: AsyncChunkCache<Value = Self> + ?Sized,
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
        C: AsyncChunkCache<Value = Self> + ?Sized,
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

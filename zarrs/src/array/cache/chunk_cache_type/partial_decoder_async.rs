use std::sync::Arc;

use super::{
    cache_error, copy_array_fill_value_into, validate_chunk_indices,
    validate_partial_chunk_indexer, validate_target_num_elements,
};
use crate::array::cache::{
    AsyncCachedArray, AsyncChunkCacheTraits, AsyncChunkCacheTypePartialDecoder, ChunkCacheType,
    SealedAsync,
};
use crate::array::{
    ArrayBytes, ArrayError, ArraySubset, ChunkShapeTraits, CodecOptions, Indexer, Resources,
    chunk_shape_to_array_shape,
};
use zarrs_codec::{ArrayBytesTarget, AsyncArrayPartialDecoderTraits};

impl ChunkCacheType for AsyncChunkCacheTypePartialDecoder {
    fn size(&self) -> usize {
        self.as_ref().size_held()
    }
}

/// Retrieve the partial decoder of a chunk from the cache, or create it on a miss.
///
/// The chunk indices must be validated with [`validate_chunk_indices`] first, so that an out-of-bounds chunk is never inserted.
async fn async_retrieve_partial_decoder<C>(
    cache: &C,
    array: &dyn AsyncCachedArray,
    chunk_indices: &[u64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<AsyncChunkCacheTypePartialDecoder, ArrayError>
where
    C: AsyncChunkCacheTraits<Value = AsyncChunkCacheTypePartialDecoder> + ?Sized,
{
    cache
        .try_get_or_insert_with(
            chunk_indices.into(),
            std::pin::pin!(async move {
                array
                    .async_partial_decoder_with_options(chunk_indices, options, resources)
                    .await
            }),
        )
        .await
        .map_err(cache_error)
}

impl SealedAsync for AsyncChunkCacheTypePartialDecoder {
    async fn async_partial_decoder<C>(
        cache: &C,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
    where
        C: AsyncChunkCacheTraits<Value = Self> + ?Sized,
    {
        validate_chunk_indices(array, chunk_indices)?;
        async_retrieve_partial_decoder(cache, array, chunk_indices, options, resources).await
    }

    async fn async_retrieve_chunk_bytes_if_exists<C>(
        cache: &C,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        C: AsyncChunkCacheTraits<Value = Self> + ?Sized,
    {
        let shape = chunk_shape_to_array_shape(&validate_chunk_indices(array, chunk_indices)?);
        let decoder =
            async_retrieve_partial_decoder(cache, array, chunk_indices, options, resources).await?;
        if decoder.exists().await? {
            Ok(Some(
                decoder
                    .partial_decode(&ArraySubset::new_with_shape(shape), options, resources)
                    .await?
                    .into_owned()
                    .into(),
            ))
        } else {
            Ok(None)
        }
    }

    async fn async_retrieve_partial_chunk_bytes<C>(
        cache: &C,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        C: AsyncChunkCacheTraits<Value = Self> + ?Sized,
    {
        Ok(
            Self::async_partial_decoder(cache, array, chunk_indices, options, resources)
                .await?
                .partial_decode(indexer, options, resources)
                .await?
                .into_owned()
                .into(),
        )
    }

    async fn async_retrieve_chunk_into<C>(
        cache: &C,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        C: AsyncChunkCacheTraits<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        validate_target_num_elements(output_target, chunk_shape.num_elements_u64())?;
        let decoder =
            async_retrieve_partial_decoder(cache, array, chunk_indices, options, resources).await?;
        if !decoder.exists().await? {
            return copy_array_fill_value_into(array, output_target);
        }
        let shape = chunk_shape_to_array_shape(&chunk_shape);
        decoder
            .partial_decode_into(
                &ArraySubset::new_with_shape(shape),
                output_target,
                options,
                resources,
            )
            .await
            .map_err(ArrayError::from)
    }

    async fn async_retrieve_partial_chunk_into<C>(
        cache: &C,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        C: AsyncChunkCacheTraits<Value = Self> + ?Sized,
    {
        validate_partial_chunk_indexer(array, chunk_indices, indexer, output_target)?;
        async_retrieve_partial_decoder(cache, array, chunk_indices, options, resources)
            .await?
            .partial_decode_into(indexer, output_target, options, resources)
            .await
            .map_err(ArrayError::from)
    }
}

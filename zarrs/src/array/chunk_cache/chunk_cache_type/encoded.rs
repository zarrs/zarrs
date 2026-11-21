use std::sync::{Arc, Mutex};
use zarrs_codec::CowBytes;

#[cfg(feature = "async")]
use super::SyncPartialDecoderAsAsync;
use super::{
    cache_error, copy_array_fill_value_into, validate_chunk_indices,
    validate_partial_chunk_indexer, validate_target_num_elements,
};
#[cfg(feature = "async")]
use crate::array::chunk_cache::{AsyncCachedArray, AsyncChunkCache, SealedAsync};
use crate::array::chunk_cache::{
    CachedArrayMeta, ChunkCache, ChunkCacheType, ChunkCacheTypeEncoded, SealedSync, SyncCachedArray,
};
use crate::array::{
    ArrayBytes, ArrayError, ChunkShape, ChunkShapeTraits, CodecOptions, Indexer, Resources,
};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{ArrayBytesTarget, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits};

impl ChunkCacheType for ChunkCacheTypeEncoded {
    fn size(&self) -> usize {
        self.as_ref().map_or(0, bytes::Bytes::len)
    }
}

/// Create a partial decoder over an in-memory encoded chunk, or over the fill value if absent.
///
/// `chunk_shape` must come from [`validate_chunk_indices`].
fn partial_decoder_over_encoded(
    array: &(impl CachedArrayMeta + ?Sized),
    encoded: ChunkCacheTypeEncoded,
    chunk_shape: &ChunkShape,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
    let input: Arc<dyn zarrs_codec::BytesPartialDecoderTraits> = match encoded {
        Some(encoded) => Arc::new(CowBytes::Shared(encoded)),
        None => Arc::new(Mutex::new(None)),
    };
    Ok(array
        .codecs_bound()
        .clone()
        .partial_decoder(input, chunk_shape, options, resources)?)
}

/// Decode an in-memory encoded chunk, or return [`None`] if it is absent.
///
/// `chunk_shape` must come from [`validate_chunk_indices`], which callers run *before* populating
/// the cache so that an out-of-bounds chunk is never inserted.
fn decode_encoded(
    array: &(impl CachedArrayMeta + ?Sized),
    encoded: &ChunkCacheTypeEncoded,
    chunk_shape: &ChunkShape,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
    encoded
        .as_ref()
        .map(|encoded| {
            let bytes = array
                .codecs_bound()
                .decode(
                    CowBytes::Shared(encoded.clone()),
                    chunk_shape,
                    options,
                    resources.compute(),
                )
                .map_err(ArrayError::CodecError)?;
            bytes.validate(chunk_shape.num_elements_u64(), array.data_type())?;
            Ok(Arc::new(bytes.into_owned()))
        })
        .transpose()
}

/// Retrieve the encoded chunk from the cache, or fetch it synchronously on a miss.
///
/// The chunk indices must be validated with [`validate_chunk_indices`] first, so that an out-of-bounds chunk is never inserted.
fn retrieve_encoded<C>(
    cache: &C,
    array: &dyn SyncCachedArray,
    chunk_indices: &[u64],
) -> Result<ChunkCacheTypeEncoded, ArrayError>
where
    C: ChunkCache<Value = ChunkCacheTypeEncoded> + ?Sized,
{
    cache
        .try_get_or_insert_with(chunk_indices.into(), &mut || {
            array.retrieve_encoded_chunk(chunk_indices)
        })
        .map_err(cache_error)
}

/// Decode an in-memory encoded chunk directly into `output_target`, or copy the fill value if absent.
fn decode_encoded_into(
    array: &(impl CachedArrayMeta + ?Sized),
    encoded: &ChunkCacheTypeEncoded,
    chunk_shape: &ChunkShape,
    output_target: &mut ArrayBytesTarget<'_>,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<(), ArrayError> {
    if let Some(encoded) = encoded {
        array
            .codecs_bound()
            .decode_into(
                CowBytes::Shared(encoded.clone()).into(),
                chunk_shape,
                output_target,
                options,
                resources.compute(),
            )
            .map_err(ArrayError::CodecError)
    } else {
        copy_array_fill_value_into(array, output_target)
    }
}

/// Partially decode an in-memory encoded chunk directly into `output_target`, or copy the fill value if absent.
fn partial_decode_encoded_into(
    array: &(impl CachedArrayMeta + ?Sized),
    encoded: ChunkCacheTypeEncoded,
    chunk_shape: &ChunkShape,
    indexer: &dyn Indexer,
    output_target: &mut ArrayBytesTarget<'_>,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<(), ArrayError> {
    if indexer.covers_array(bytemuck::must_cast_slice(chunk_shape)) {
        return decode_encoded_into(
            array,
            &encoded,
            chunk_shape,
            output_target,
            options,
            resources,
        );
    }
    if encoded.is_none() {
        return copy_array_fill_value_into(array, output_target);
    }
    partial_decoder_over_encoded(array, encoded, chunk_shape, options, resources)?
        .partial_decode_into(indexer, output_target, options, resources)
        .map_err(ArrayError::from)
}

impl SealedSync for ChunkCacheTypeEncoded {
    fn partial_decoder<C>(
        cache: &C,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
    where
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let encoded = retrieve_encoded(cache, array, chunk_indices)?;
        partial_decoder_over_encoded(array, encoded, &chunk_shape, options, resources)
    }

    fn retrieve_chunk_bytes_if_exists<C>(
        cache: &C,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let encoded = retrieve_encoded(cache, array, chunk_indices)?;
        decode_encoded(array, &encoded, &chunk_shape, options, resources)
    }

    fn retrieve_partial_chunk_bytes<C>(
        cache: &C,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        C: ChunkCache<Value = Self> + ?Sized,
    {
        Self::partial_decoder(cache, array, chunk_indices, options, resources)?
            .partial_decode(indexer, options, resources)
            .map(|bytes| bytes.into_owned().into())
            .map_err(ArrayError::from)
    }

    fn retrieve_chunk_into<C>(
        cache: &C,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        validate_target_num_elements(output_target, chunk_shape.num_elements_u64())?;
        let encoded = retrieve_encoded(cache, array, chunk_indices)?;
        decode_encoded_into(
            array,
            &encoded,
            &chunk_shape,
            output_target,
            options,
            resources,
        )
    }

    fn retrieve_partial_chunk_into<C>(
        cache: &C,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape =
            validate_partial_chunk_indexer(array, chunk_indices, indexer, output_target)?;
        let encoded = retrieve_encoded(cache, array, chunk_indices)?;
        partial_decode_encoded_into(
            array,
            encoded,
            &chunk_shape,
            indexer,
            output_target,
            options,
            resources,
        )
    }
}

/// Retrieve the encoded chunk from the cache, or fetch it asynchronously on a miss.
///
/// The chunk indices must be validated with [`validate_chunk_indices`] first, so that an out-of-bounds chunk is never inserted.
#[cfg(feature = "async")]
async fn async_retrieve_encoded<C>(
    cache: &C,
    array: &dyn AsyncCachedArray,
    chunk_indices: &[u64],
) -> Result<ChunkCacheTypeEncoded, ArrayError>
where
    C: AsyncChunkCache<Value = ChunkCacheTypeEncoded> + ?Sized,
{
    cache
        .try_get_or_insert_with(
            chunk_indices.into(),
            std::pin::pin!(async move { array.async_retrieve_encoded_chunk(chunk_indices).await }),
        )
        .await
        .map_err(cache_error)
}

/// Create a synchronous partial decoder over the in-memory encoded chunk.
#[cfg(feature = "async")]
async fn async_partial_decoder_sync<C>(
    cache: &C,
    array: &dyn AsyncCachedArray,
    chunk_indices: &[u64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
where
    C: AsyncChunkCache<Value = ChunkCacheTypeEncoded> + ?Sized,
{
    let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
    let encoded = async_retrieve_encoded(cache, array, chunk_indices).await?;
    partial_decoder_over_encoded(array, encoded, &chunk_shape, options, resources)
}

#[cfg(feature = "async")]
impl SealedAsync for ChunkCacheTypeEncoded {
    async fn async_partial_decoder<C>(
        cache: &C,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
    where
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let decoder = SyncPartialDecoderAsAsync(
            async_partial_decoder_sync(cache, array, chunk_indices, options, resources).await?,
        );
        Ok(Arc::new(decoder) as Arc<dyn AsyncArrayPartialDecoderTraits>)
    }

    async fn async_retrieve_chunk_bytes_if_exists<C>(
        cache: &C,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let encoded = async_retrieve_encoded(cache, array, chunk_indices).await?;
        decode_encoded(array, &encoded, &chunk_shape, options, resources)
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
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        async_partial_decoder_sync(cache, array, chunk_indices, options, resources)
            .await?
            .partial_decode(indexer, options, resources)
            .map(|bytes| bytes.into_owned().into())
            .map_err(ArrayError::from)
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
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        validate_target_num_elements(output_target, chunk_shape.num_elements_u64())?;
        let encoded = async_retrieve_encoded(cache, array, chunk_indices).await?;
        decode_encoded_into(
            array,
            &encoded,
            &chunk_shape,
            output_target,
            options,
            resources,
        )
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
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape =
            validate_partial_chunk_indexer(array, chunk_indices, indexer, output_target)?;
        let encoded = async_retrieve_encoded(cache, array, chunk_indices).await?;
        partial_decode_encoded_into(
            array,
            encoded,
            &chunk_shape,
            indexer,
            output_target,
            options,
            resources,
        )
    }
}

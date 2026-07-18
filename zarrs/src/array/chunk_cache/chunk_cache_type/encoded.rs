use std::borrow::Cow;
use std::sync::{Arc, Mutex};

#[cfg(feature = "async")]
use super::{SyncPartialDecoderAsAsync, async_try_get_or_insert_with};
use super::{cache_error, validate_chunk_indices};
#[cfg(feature = "async")]
use crate::array::chunk_cache::AsyncChunkCacheType;
use crate::array::chunk_cache::{
    ChunkCache, ChunkCacheType, ChunkCacheTypeEncoded, SyncChunkCacheType,
};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubsetTraits, ChunkShapeTraits, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{ArrayPartialDecoderTraits, ArrayToBytesCodecTraits};
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::ReadableStorageTraits;

impl ChunkCacheType for ChunkCacheTypeEncoded {
    fn size(&self) -> usize {
        self.as_ref().map_or(0, |value| value.len())
    }
}

impl SyncChunkCacheType for ChunkCacheTypeEncoded {
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
        let encoded = cache
            .try_get_or_insert_with(chunk_indices.to_vec(), || {
                Ok(array
                    .retrieve_encoded_chunk(chunk_indices)?
                    .map(|chunk| Arc::new(Cow::Owned(chunk))))
            })
            .map_err(cache_error)?;
        let input: Arc<dyn zarrs_codec::BytesPartialDecoderTraits> = match encoded {
            Some(encoded) => encoded,
            None => Arc::new(Mutex::new(None)),
        };
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        Ok(array
            .codecs_bound()
            .partial_decoder(input, &chunk_shape, options)?)
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
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let encoded = cache
            .try_get_or_insert_with(chunk_indices.to_vec(), || {
                Ok(array
                    .retrieve_encoded_chunk(chunk_indices)?
                    .map(|chunk| Arc::new(Cow::Owned(chunk))))
            })
            .map_err(cache_error)?;
        encoded
            .as_ref()
            .map(|encoded| {
                let bytes = array
                    .codecs_bound()
                    .decode(Cow::Borrowed(encoded), &chunk_shape, options)
                    .map_err(ArrayError::CodecError)?;
                bytes.validate(chunk_shape.num_elements_u64(), array.data_type())?;
                Ok(Arc::new(bytes.into_owned()))
            })
            .transpose()
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
        Self::partial_decoder(cache, array, chunk_indices, options)?
            .partial_decode(chunk_subset, options)
            .map(|bytes| bytes.into_owned().into())
            .map_err(ArrayError::from)
    }
}

/// Retrieve the encoded chunk from the cache, or fetch it asynchronously on a miss.
#[cfg(feature = "async")]
async fn async_retrieve_encoded<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
) -> Result<ChunkCacheTypeEncoded, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: ChunkCache<Value = ChunkCacheTypeEncoded> + ?Sized,
{
    async_try_get_or_insert_with(cache, chunk_indices.to_vec(), async || {
        Ok(array
            .async_retrieve_encoded_chunk(chunk_indices)
            .await?
            .map(|chunk| Arc::new(Cow::Owned(chunk.into()))))
    })
    .await
    .map_err(cache_error)
}

/// Create a synchronous partial decoder over the in-memory encoded chunk.
#[cfg(feature = "async")]
async fn async_partial_decoder_sync<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: ChunkCache<Value = ChunkCacheTypeEncoded> + ?Sized,
{
    let encoded = async_retrieve_encoded(cache, array, chunk_indices).await?;
    let input: Arc<dyn zarrs_codec::BytesPartialDecoderTraits> = match encoded {
        Some(encoded) => encoded,
        None => Arc::new(Mutex::new(None)),
    };
    let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
    Ok(array
        .codecs_bound()
        .partial_decoder(input, &chunk_shape, options)?)
}

#[cfg(feature = "async")]
impl AsyncChunkCacheType for ChunkCacheTypeEncoded {
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
        let decoder = SyncPartialDecoderAsAsync(
            async_partial_decoder_sync(cache, array, chunk_indices, options).await?,
        );
        Ok(Arc::new(decoder) as Arc<dyn AsyncArrayPartialDecoderTraits>)
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
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let encoded = async_retrieve_encoded(cache, array, chunk_indices).await?;
        encoded
            .as_ref()
            .map(|encoded| {
                let bytes = array
                    .codecs_bound()
                    .decode(Cow::Borrowed(encoded), &chunk_shape, options)
                    .map_err(ArrayError::CodecError)?;
                bytes.validate(chunk_shape.num_elements_u64(), array.data_type())?;
                Ok(Arc::new(bytes.into_owned()))
            })
            .transpose()
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
        async_partial_decoder_sync(cache, array, chunk_indices, options)
            .await?
            .partial_decode(chunk_subset, options)
            .map(|bytes| bytes.into_owned().into())
            .map_err(ArrayError::from)
    }
}

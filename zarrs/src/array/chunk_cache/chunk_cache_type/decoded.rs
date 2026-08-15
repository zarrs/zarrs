use std::sync::Arc;

#[cfg(feature = "async")]
use super::SyncPartialDecoderAsAsync;
use super::{cache_error, fill_value_bytes, validate_chunk_indices};
#[cfg(feature = "async")]
use crate::array::chunk_cache::{AsyncChunkCache, SealedAsync};
use crate::array::chunk_cache::{
    ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, SealedSync,
};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubsetTraits, ChunkShape, CodecOptions, DataType,
    FillValue, Indexer,
};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{ArrayPartialDecoderTraits, CodecError};
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
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
}

/// Wrap already-decoded chunk bytes in a partial decoder.
fn cached_partial_decoder<TStorage>(
    array: &Array<TStorage>,
    bytes: ChunkCacheTypeDecoded,
    chunk_indices: &[u64],
) -> Result<CachedArrayBytesPartialDecoder, ArrayError>
where
    TStorage: ?Sized + 'static,
{
    Ok(CachedArrayBytesPartialDecoder {
        bytes,
        shape: validate_chunk_indices(array, chunk_indices)?,
        data_type: array.data_type().clone(),
        fill_value: array.fill_value().clone(),
    })
}

/// Extract a chunk subset from already-decoded chunk bytes, or from the fill value if absent.
fn cached_chunk_subset_bytes<TStorage>(
    array: &Array<TStorage>,
    chunk: ChunkCacheTypeDecoded,
    chunk_indices: &[u64],
    chunk_subset: &dyn ArraySubsetTraits,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + 'static,
{
    if let Some(chunk) = chunk {
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

impl SealedSync for ChunkCacheTypeDecoded {
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
        Ok(Arc::new(cached_partial_decoder(
            array,
            bytes,
            chunk_indices,
        )?))
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
                    .retrieve_chunk_if_exists_with_options::<ArrayBytes<'static>>(
                        chunk_indices,
                        options,
                    )?
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
        let chunk = Self::retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)?;
        cached_chunk_subset_bytes(array, chunk, chunk_indices, chunk_subset)
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl SealedAsync for ChunkCacheTypeDecoded {
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
        let bytes =
            Self::async_retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)
                .await?;
        let decoder = SyncPartialDecoderAsAsync(Arc::new(cached_partial_decoder(
            array,
            bytes,
            chunk_indices,
        )?));
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
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        validate_chunk_indices(array, chunk_indices)?;
        cache
            .try_get_or_insert_with(chunk_indices.to_vec(), async move {
                Ok(array
                    .async_retrieve_chunk_if_exists_with_options::<ArrayBytes<'static>>(
                        chunk_indices,
                        options,
                    )
                    .await?
                    .map(Arc::new))
            })
            .await
            .map_err(cache_error)
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
        let chunk =
            Self::async_retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)
                .await?;
        cached_chunk_subset_bytes(array, chunk, chunk_indices, chunk_subset)
    }
}

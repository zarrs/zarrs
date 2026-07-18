use std::sync::Arc;

use ambisync::ambisync;

#[cfg(feature = "async")]
use super::{SyncPartialDecoderAsAsync, async_try_get_or_insert_with};
use super::{cache_error, fill_value_bytes, try_get_or_insert_with, validate_chunk_indices};
#[cfg(feature = "async")]
use crate::array::chunk_cache::AsyncChunkCacheType;
use crate::array::chunk_cache::{
    ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, SyncChunkCacheType,
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

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncChunkCacheType => SyncChunkCacheType,
            AsyncReadableStorageTraits => ReadableStorageTraits,
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
impl AsyncChunkCacheType for ChunkCacheTypeDecoded {
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
        let bytes =
            Self::async_retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)
                .await?;
        let decoder = CachedArrayBytesPartialDecoder {
            bytes,
            shape: validate_chunk_indices(array, chunk_indices)?,
            data_type: array.data_type().clone(),
            fill_value: array.fill_value().clone(),
        };
        Ok(ambisync::alt!(
            sync => Arc::new(decoder),
            async => Arc::new(SyncPartialDecoderAsAsync(Arc::new(decoder)))
                as Arc<dyn AsyncArrayPartialDecoderTraits>,
        ))
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
        validate_chunk_indices(array, chunk_indices)?;
        async_try_get_or_insert_with(cache, chunk_indices.to_vec(), async || {
            Ok(array
                .async_retrieve_chunk_if_exists_opt::<ArrayBytes<'static>>(chunk_indices, options)
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
        C: ChunkCache<Value = Self> + ?Sized,
    {
        if let Some(chunk) =
            Self::async_retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options).await?
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

use std::sync::Arc;

#[cfg(feature = "async")]
use super::SyncPartialDecoderAsAsync;
use super::{
    cache_error, copy_array_fill_value_into, validate_chunk_indices,
    validate_partial_chunk_indexer, validate_target_num_elements,
};
use crate::array::array_ops::fill_value_bytes;
#[cfg(feature = "async")]
use crate::array::chunk_cache::{AsyncChunkCache, SealedAsync};
use crate::array::chunk_cache::{ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, SealedSync};
use crate::array::{
    Array, ArrayBytes, ArrayError, ChunkShape, ChunkShapeTraits, CodecOptions, DataType, FillValue,
    Indexer, Resources,
};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{
    ArrayBytesTarget, ArrayPartialDecoderNoSubchunkingTraits, ArrayPartialDecoderTraits,
    CodecError, copy_fill_value_into, decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{ReadableStorageTraits, StorageError};

struct CachedArrayBytesPartialDecoder {
    bytes: ChunkCacheTypeDecoded,
    shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
}

/// A cache of decoded chunk bytes cannot expose encoded subchunks.
impl ArrayPartialDecoderNoSubchunkingTraits for CachedArrayBytesPartialDecoder {}

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

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(bytes) = &self.bytes {
            Ok(bytes.extract_array_subset(
                indexer,
                bytemuck::must_cast_slice(&self.shape),
                &self.data_type,
            )?)
        } else {
            indexer.validate(bytemuck::must_cast_slice(&self.shape))?;
            Ok(ArrayBytes::new_fill_value(
                &self.data_type,
                indexer.len(),
                &self.fill_value,
            )?)
        }
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<(), CodecError> {
        let shape = bytemuck::must_cast_slice(&self.shape);
        indexer.validate(shape)?;
        validate_target_num_elements(output_target, indexer.len())?;
        decoded_chunk_subset_into(
            self.bytes.as_deref(),
            shape,
            &self.data_type,
            &self.fill_value,
            indexer,
            output_target,
        )
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
///
/// `chunk_shape` must come from [`validate_chunk_indices`].
fn cached_partial_decoder<TStorage>(
    array: &Array<TStorage>,
    bytes: ChunkCacheTypeDecoded,
    chunk_shape: ChunkShape,
) -> CachedArrayBytesPartialDecoder
where
    TStorage: ?Sized + 'static,
{
    CachedArrayBytesPartialDecoder {
        bytes,
        shape: chunk_shape,
        data_type: array.data_type().clone(),
        fill_value: array.fill_value().clone(),
    }
}

/// Extract a chunk subset from already-decoded chunk bytes, or from the fill value if absent.
///
/// `chunk_shape` must come from [`validate_chunk_indices`].
fn cached_chunk_subset_bytes<TStorage>(
    array: &Array<TStorage>,
    chunk: ChunkCacheTypeDecoded,
    chunk_shape: &ChunkShape,
    indexer: &dyn Indexer,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + 'static,
{
    if let Some(chunk) = chunk {
        Ok(chunk
            .extract_array_subset(
                indexer,
                bytemuck::must_cast_slice(chunk_shape),
                array.data_type(),
            )?
            .into_owned()
            .into())
    } else {
        indexer
            .validate(bytemuck::must_cast_slice(chunk_shape))
            .map_err(CodecError::from)?;
        fill_value_bytes(array, indexer.len()).map(Arc::new)
    }
}

/// Decode already-decoded chunk bytes into `output_target`, or copy the fill value if absent.
fn cached_chunk_into(
    array: &Array<impl ?Sized>,
    chunk: Option<&ArrayBytes<'static>>,
    output_target: &mut ArrayBytesTarget<'_>,
) -> Result<(), ArrayError> {
    if let Some(chunk) = chunk {
        decode_into_array_bytes_target(chunk, output_target).map_err(ArrayError::CodecError)
    } else {
        copy_array_fill_value_into(array, output_target)
    }
}

/// Decode the elements of `indexer` in already-decoded chunk bytes of `chunk_shape` into `output_target`, or copy the fill value if absent.
///
/// `indexer` must be validated against `chunk_shape`, and match the number of elements of `output_target`.
pub(in crate::array::chunk_cache) fn decoded_chunk_subset_into(
    chunk: Option<&ArrayBytes<'static>>,
    chunk_shape: &[u64],
    data_type: &DataType,
    fill_value: &FillValue,
    indexer: &dyn Indexer,
    output_target: &mut ArrayBytesTarget<'_>,
) -> Result<(), CodecError> {
    let Some(chunk) = chunk else {
        return copy_fill_value_into(data_type, fill_value, output_target);
    };
    if indexer.covers_array(chunk_shape) {
        decode_into_array_bytes_target(chunk, output_target)
    } else if let Some(subset) = indexer.as_array_subset() {
        // Copy the runs of the subset directly, without extracting them into a new buffer
        let runs = subset.contiguous_linearised_indices(chunk_shape)?;
        chunk.extract_runs_into(&runs, output_target)
    } else {
        let subset = chunk.extract_array_subset(indexer, chunk_shape, data_type)?;
        decode_into_array_bytes_target(&subset, output_target)
    }
}

/// Retrieve the decoded chunk from the cache, or retrieve and decode it synchronously on a miss.
///
/// The chunk indices must be validated with [`validate_chunk_indices`] first, so that an out-of-bounds chunk is never inserted.
fn retrieve_decoded<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<ChunkCacheTypeDecoded, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
{
    cache
        .try_get_or_insert_with(chunk_indices.into(), || {
            Ok(array
                .retrieve_chunk_if_exists_with_options(chunk_indices, options, resources)?
                .map(Arc::new))
        })
        .map_err(cache_error)
}

/// Retrieve the decoded chunk from the cache, or retrieve and decode it asynchronously on a miss.
///
/// The chunk indices must be validated with [`validate_chunk_indices`] first, so that an out-of-bounds chunk is never inserted.
#[cfg(feature = "async")]
async fn async_retrieve_decoded<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<ChunkCacheTypeDecoded, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
{
    cache
        .try_get_or_insert_with(chunk_indices.into(), async move {
            Ok(array
                .async_retrieve_chunk_if_exists_with_options(chunk_indices, options, resources)
                .await?
                .map(Arc::new))
        })
        .await
        .map_err(cache_error)
}

impl SealedSync for ChunkCacheTypeDecoded {
    fn partial_decoder<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let bytes = retrieve_decoded(cache, array, chunk_indices, options, resources)?;
        Ok(Arc::new(cached_partial_decoder(array, bytes, chunk_shape)))
    }

    fn retrieve_chunk_bytes_if_exists<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        validate_chunk_indices(array, chunk_indices)?;
        retrieve_decoded(cache, array, chunk_indices, options, resources)
    }

    fn retrieve_partial_chunk_bytes<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let chunk = retrieve_decoded(cache, array, chunk_indices, options, resources)?;
        cached_chunk_subset_bytes(array, chunk, &chunk_shape, indexer)
    }

    fn retrieve_chunk_into<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        validate_target_num_elements(output_target, chunk_shape.num_elements_u64())?;
        let chunk = retrieve_decoded(cache, array, chunk_indices, options, resources)?;
        cached_chunk_into(array, chunk.as_deref(), output_target)
    }

    fn retrieve_partial_chunk_into<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape =
            validate_partial_chunk_indexer(array, chunk_indices, indexer, output_target)?;
        let chunk = retrieve_decoded(cache, array, chunk_indices, options, resources)?;
        Ok(decoded_chunk_subset_into(
            chunk.as_deref(),
            bytemuck::must_cast_slice(&chunk_shape),
            array.data_type(),
            array.fill_value(),
            indexer,
            output_target,
        )?)
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
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let bytes = async_retrieve_decoded(cache, array, chunk_indices, options, resources).await?;
        let decoder =
            SyncPartialDecoderAsAsync(Arc::new(cached_partial_decoder(array, bytes, chunk_shape)));
        Ok(Arc::new(decoder) as Arc<dyn AsyncArrayPartialDecoderTraits>)
    }

    async fn async_retrieve_chunk_bytes_if_exists<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        validate_chunk_indices(array, chunk_indices)?;
        async_retrieve_decoded(cache, array, chunk_indices, options, resources).await
    }

    async fn async_retrieve_partial_chunk_bytes<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        let chunk = async_retrieve_decoded(cache, array, chunk_indices, options, resources).await?;
        cached_chunk_subset_bytes(array, chunk, &chunk_shape, indexer)
    }

    async fn async_retrieve_chunk_into<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        validate_target_num_elements(output_target, chunk_shape.num_elements_u64())?;
        let chunk = async_retrieve_decoded(cache, array, chunk_indices, options, resources).await?;
        cached_chunk_into(array, chunk.as_deref(), output_target)
    }

    async fn async_retrieve_partial_chunk_into<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: AsyncChunkCache<Value = Self> + ?Sized,
    {
        let chunk_shape =
            validate_partial_chunk_indexer(array, chunk_indices, indexer, output_target)?;
        let chunk = async_retrieve_decoded(cache, array, chunk_indices, options, resources).await?;
        Ok(decoded_chunk_subset_into(
            chunk.as_deref(),
            bytemuck::must_cast_slice(&chunk_shape),
            array.data_type(),
            array.fill_value(),
            indexer,
            output_target,
        )?)
    }
}

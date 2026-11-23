use std::sync::Arc;

use super::{
    cache_error, copy_array_fill_value_into, validate_chunk_indices,
    validate_partial_chunk_indexer, validate_target_num_elements,
};
use crate::array::cache::{
    ChunkCache, ChunkCacheType, ChunkCacheTypePartialDecoder, SealedSync, SyncCachedArray,
};
use crate::array::{
    ArrayBytes, ArrayError, ArraySubset, ChunkShapeTraits, CodecOptions, Indexer, Resources,
    chunk_shape_to_array_shape,
};
use zarrs_codec::{ArrayBytesTarget, ArrayPartialDecoderTraits};

impl ChunkCacheType for ChunkCacheTypePartialDecoder {
    fn size(&self) -> usize {
        self.as_ref().size_held()
    }
}

/// Retrieve the partial decoder of a chunk from the cache, or create it on a miss.
///
/// The chunk indices must be validated with [`validate_chunk_indices`] first, so that an out-of-bounds chunk is never inserted.
fn retrieve_partial_decoder<C>(
    cache: &C,
    array: &dyn SyncCachedArray,
    chunk_indices: &[u64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<ChunkCacheTypePartialDecoder, ArrayError>
where
    C: ChunkCache<Value = ChunkCacheTypePartialDecoder> + ?Sized,
{
    cache
        .try_get_or_insert_with(chunk_indices.into(), &mut || {
            array.partial_decoder_with_options(chunk_indices, options, resources)
        })
        .map_err(cache_error)
}

impl SealedSync for ChunkCacheTypePartialDecoder {
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
        validate_chunk_indices(array, chunk_indices)?;
        retrieve_partial_decoder(cache, array, chunk_indices, options, resources)
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
        let shape = chunk_shape_to_array_shape(&validate_chunk_indices(array, chunk_indices)?);
        let decoder = retrieve_partial_decoder(cache, array, chunk_indices, options, resources)?;
        if decoder.exists()? {
            Ok(Some(
                decoder
                    .partial_decode(&ArraySubset::new_with_shape(shape), options, resources)?
                    .into_owned()
                    .into(),
            ))
        } else {
            Ok(None)
        }
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
        Ok(
            Self::partial_decoder(cache, array, chunk_indices, options, resources)?
                .partial_decode(indexer, options, resources)?
                .into_owned()
                .into(),
        )
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
        let decoder = retrieve_partial_decoder(cache, array, chunk_indices, options, resources)?;
        if !decoder.exists()? {
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
            .map_err(ArrayError::from)
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
        validate_partial_chunk_indexer(array, chunk_indices, indexer, output_target)?;
        retrieve_partial_decoder(cache, array, chunk_indices, options, resources)?
            .partial_decode_into(indexer, output_target, options, resources)
            .map_err(ArrayError::from)
    }
}

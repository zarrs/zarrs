use inherent::inherent;
use std::sync::Arc;
use zarrs_codec::Resources;

use super::{
    Array, ArrayCached, ArrayError, ArrayIndices, ArraySubsetTraits, ChunkGrid, CodecOptions,
    DataType, IntoArrayBytes, SyncArrayUpdateOps, SyncReadableStorageTraits,
    SyncReadableWritableStorageTraits,
};
use crate::array::cache::SyncArrayCache;
use crate::array::{ArrayBytes, Indexer};
use zarrs_codec::{
    ArrayBytesTarget, CodecError, SyncArrayPartialDecoderSubchunkingTraits,
    SyncArrayPartialDecoderTraits, SyncArrayPartialEncoderTraits,
};
use zarrs_storage::StorageError;

pub(super) struct CachedArrayPartialEncoder<TStorage: ?Sized, C: ?Sized, E: ?Sized> {
    pub(super) cached: ArrayCached<TStorage, C>,
    pub(super) encoder: Arc<E>,
    pub(super) chunk_indices: ArrayIndices,
}

impl<TStorage, C, E> SyncArrayPartialDecoderSubchunkingTraits
    for CachedArrayPartialEncoder<TStorage, C, E>
where
    TStorage: ?Sized + SyncReadableWritableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized + 'static,
    E: SyncArrayPartialEncoderTraits + ?Sized + 'static,
{
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.encoder.local_subchunk_grids(options, resources)
    }
}

impl<TStorage, C, E> SyncArrayPartialDecoderTraits for CachedArrayPartialEncoder<TStorage, C, E>
where
    TStorage: ?Sized + SyncReadableWritableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized + 'static,
    E: SyncArrayPartialEncoderTraits + ?Sized + 'static,
{
    fn data_type(&self) -> &DataType {
        self.encoder.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.encoder.exists()
    }

    fn size_held(&self) -> usize {
        self.encoder.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        self.encoder.partial_decode(indexer, options, resources)
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        self.encoder
            .partial_decode_into(indexer, output_target, options, resources)
    }

    fn supports_partial_decode(&self) -> bool {
        self.encoder.supports_partial_decode()
    }
}

impl<TStorage, C, E> SyncArrayPartialEncoderTraits for CachedArrayPartialEncoder<TStorage, C, E>
where
    TStorage: ?Sized + SyncReadableWritableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized + 'static,
    E: SyncArrayPartialEncoderTraits + ?Sized + 'static,
{
    fn erase(&self) -> Result<(), CodecError> {
        let result = self.encoder.erase();
        self.cached
            .cache()
            .invalidate_cached_chunk(self.cached.array().as_ref(), &self.chunk_indices);
        result
    }

    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        let result = self
            .encoder
            .partial_encode(indexer, bytes, options, resources);
        self.cached
            .cache()
            .invalidate_cached_chunk(self.cached.array().as_ref(), &self.chunk_indices);
        result
    }

    fn supports_partial_encode(&self) -> bool {
        self.encoder.supports_partial_encode()
    }
}

#[inherent]
impl<TStorage, C> SyncArrayUpdateOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + SyncReadableWritableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized + 'static,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn store_partial_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result =
            self.array()
                .store_partial_chunk(chunk_indices, indexer, indexer_data, resources);
        self.cache()
            .invalidate_cached_chunk(self.array().as_ref(), chunk_indices);
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_array_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result = self
            .array()
            .store_array_subset(array_subset, subset_data, resources);
        let invalidated = match self.array().chunks_in_array_subset(array_subset) {
            Ok(Some(chunks)) => {
                self.cache()
                    .invalidate_cached_chunks(self.array().as_ref(), &chunks);
                Ok(())
            }
            Ok(None) => {
                self.cache().invalidate_cached_all(self.array().as_ref());
                Ok(())
            }
            Err(err) => Err(err.into()),
        };
        result.and(invalidated)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn compact_chunk(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<bool, ArrayError> {
        let compacted = self.array().compact_chunk(chunk_indices, resources);
        if !matches!(compacted, Ok(false)) {
            self.cache()
                .invalidate_cached_chunk(self.array().as_ref(), chunk_indices);
        }
        compacted
    }

    #[allow(clippy::must_use_candidate)]
    pub fn readable(&self) -> Array<dyn SyncReadableStorageTraits> {
        self.array().readable()
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_encoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn SyncArrayPartialEncoderTraits>, ArrayError> {
        let encoder = self.array().partial_encoder(chunk_indices, resources)?;
        Ok(Arc::new(CachedArrayPartialEncoder {
            cached: self.clone(),
            encoder,
            chunk_indices: chunk_indices.into(),
        }))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::single_range_in_vec_init)]
    use super::*;
    use crate::array::cache::{
        SyncChunkCacheDecoded, SyncChunkCacheEncoded, SyncChunkCachePartialDecoder,
        SyncChunkCacheTraits,
    };
    use crate::array::{ArrayBuilder, ArraySubset, data_type};
    use zarrs_storage::store::MemoryStore;

    fn test_partial_encoder_invalidates<C>(cache: C)
    where
        C: SyncChunkCacheTraits + 'static,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .store_chunk(&[0], &[1u8, 2], &Resources::default())
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(
            cached
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap(),
            vec![1, 2]
        );
        assert_eq!(cached.cache().len(), 1);

        let encoder = cached.partial_encoder(&[0], &Resources::default()).unwrap();
        encoder
            .partial_encode(
                &ArraySubset::new_with_ranges(&[1..2]),
                &vec![3u8].into(),
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        assert!(cached.cache().is_empty());
        assert_eq!(
            cached
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap(),
            vec![1, 3]
        );

        encoder.erase().unwrap();
        assert!(cached.cache().is_empty());
        assert_eq!(
            cached
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap(),
            vec![0, 0]
        );

        assert!(!encoder.exists().unwrap());
    }

    fn test_failed_partial_encode_invalidates<C>(cache: C)
    where
        C: SyncChunkCacheTraits + 'static,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .store_chunk(&[0], &[1u8, 2], &Resources::default())
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(
            cached
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap(),
            vec![1, 2]
        );
        assert_eq!(cached.cache().len(), 1);

        let encoder = cached.partial_encoder(&[0], &Resources::default()).unwrap();
        assert!(
            encoder
                .partial_encode(
                    &ArraySubset::new_with_ranges(&[2..3]),
                    &vec![3u8].into(),
                    &CodecOptions::default(),
                    &Resources::default(),
                )
                .is_err()
        );
        assert!(cached.cache().is_empty());
    }

    #[test]
    fn partial_encoder_invalidates_all_cache_value_types() {
        test_partial_encoder_invalidates(SyncChunkCacheEncoded::new_chunk_limit(1));
        test_partial_encoder_invalidates(SyncChunkCacheDecoded::new_chunk_limit(1));
        test_partial_encoder_invalidates(SyncChunkCachePartialDecoder::new_chunk_limit(1));
    }

    #[test]
    fn failed_partial_encode_invalidates_all_cache_value_types() {
        test_failed_partial_encode_invalidates(SyncChunkCacheEncoded::new_chunk_limit(1));
        test_failed_partial_encode_invalidates(SyncChunkCacheDecoded::new_chunk_limit(1));
        test_failed_partial_encode_invalidates(SyncChunkCachePartialDecoder::new_chunk_limit(1));
    }
}

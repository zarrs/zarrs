use inherent::inherent;
use std::sync::Arc;

use super::super::*;
use super::ArrayUpdateOps;
use crate::array::{ArrayBytes, Indexer};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError,
};
use zarrs_storage::StorageError;

struct CachedArrayPartialEncoder<C> {
    encoder: Arc<dyn ArrayPartialEncoderTraits>,
    cache: Arc<C>,
    chunk_indices: ArrayIndices,
}

impl<C> ArrayPartialDecoderTraits for CachedArrayPartialEncoder<C>
where
    C: ChunkCache + 'static,
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

    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.encoder.local_subchunk_grids(options)
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        self.encoder.partial_decode(indexer, options)
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.encoder
            .partial_decode_into(indexer, output_target, options)
    }

    fn supports_partial_decode(&self) -> bool {
        self.encoder.supports_partial_decode()
    }
}

impl<C> ArrayPartialEncoderTraits for CachedArrayPartialEncoder<C>
where
    C: ChunkCache + 'static,
{
    fn erase(&self) -> Result<(), CodecError> {
        let result = self.encoder.erase();
        self.cache.invalidate_chunk(&self.chunk_indices);
        result
    }

    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let result = self.encoder.partial_encode(indexer, bytes, options);
        self.cache.invalidate_chunk(&self.chunk_indices);
        result
    }

    fn supports_partial_encode(&self) -> bool {
        self.encoder.supports_partial_encode()
    }
}

#[inherent]
impl<TStorage, C> ArrayUpdateOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + ReadableWritableStorageTraits + 'static,
    C: ChunkCache + 'static,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
    ) -> Result<(), ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_subset_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array().store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_data,
            options,
        )?;
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_array_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn store_array_subset_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .store_array_subset_opt(array_subset, subset_data, options)?;
        if let Some(chunks) = self.array().chunks_in_array_subset(array_subset)? {
            self.cache().invalidate_chunks(&chunks);
        } else {
            self.cache().invalidate();
        }
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn compact_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<bool, ArrayError> {
        let compacted = self.array().compact_chunk(chunk_indices, options)?;
        if compacted {
            self.cache().invalidate_chunk(chunk_indices);
        }
        Ok(compacted)
    }

    pub fn readable(&self) -> Array<dyn ReadableStorageTraits> {
        self.array().readable()
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_encoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        let encoder = self.array().partial_encoder(chunk_indices, options)?;
        Ok(Arc::new(CachedArrayPartialEncoder {
            encoder,
            cache: self.cache_arc(),
            chunk_indices: chunk_indices.to_vec(),
        }))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::single_range_in_vec_init)]
    use super::*;
    use crate::array::chunk_cache::{
        ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
        ChunkCachePartialDecoderLruChunkLimit,
    };
    use crate::array::{ArrayBuilder, ArraySubset, data_type};
    use zarrs_storage::store::MemoryStore;

    fn test_partial_encoder_invalidates<C>(cache: C)
    where
        C: ChunkCache + 'static,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array.store_chunk(&[0], &[1u8, 2]).unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0]).unwrap(), vec![1, 2]);
        assert_eq!(cached.cache().len(), 1);

        let options = CodecOptions::default();
        let encoder = cached.partial_encoder(&[0], &options).unwrap();
        encoder
            .partial_encode(
                &ArraySubset::new_with_ranges(&[1..2]),
                &vec![3u8].into(),
                &options,
            )
            .unwrap();
        assert!(cached.cache().is_empty());
        assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0]).unwrap(), vec![1, 3]);

        encoder.erase().unwrap();
        assert!(cached.cache().is_empty());
        assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0]).unwrap(), vec![0, 0]);

        assert!(!encoder.exists().unwrap());
    }

    fn test_failed_partial_encode_invalidates<C>(cache: C)
    where
        C: ChunkCache + 'static,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array.store_chunk(&[0], &[1u8, 2]).unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0]).unwrap(), vec![1, 2]);
        assert_eq!(cached.cache().len(), 1);

        let options = CodecOptions::default();
        let encoder = cached.partial_encoder(&[0], &options).unwrap();
        assert!(
            encoder
                .partial_encode(
                    &ArraySubset::new_with_ranges(&[2..3]),
                    &vec![3u8].into(),
                    &options,
                )
                .is_err()
        );
        assert!(cached.cache().is_empty());
    }

    #[test]
    fn partial_encoder_invalidates_all_cache_value_types() {
        test_partial_encoder_invalidates(ChunkCacheEncodedLruChunkLimit::new(1));
        test_partial_encoder_invalidates(ChunkCacheDecodedLruChunkLimit::new(1));
        test_partial_encoder_invalidates(ChunkCachePartialDecoderLruChunkLimit::new(1));
    }

    #[test]
    fn failed_partial_encode_invalidates_all_cache_value_types() {
        test_failed_partial_encode_invalidates(ChunkCacheEncodedLruChunkLimit::new(1));
        test_failed_partial_encode_invalidates(ChunkCacheDecodedLruChunkLimit::new(1));
        test_failed_partial_encode_invalidates(ChunkCachePartialDecoderLruChunkLimit::new(1));
    }
}

use inherent::inherent;
use std::sync::Arc;

use super::{AsyncArrayUpdateOps, *};
use crate::array::chunk_cache::AsyncChunkCacheType;
use crate::array::{ArrayBytes, Indexer};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits,
    CodecError,
};
use zarrs_storage::StorageError;

struct CachedAsyncArrayPartialEncoder<C> {
    encoder: Arc<dyn AsyncArrayPartialEncoderTraits>,
    cache: Arc<C>,
    chunk_indices: ArrayIndices,
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<C> AsyncArrayPartialDecoderTraits for CachedAsyncArrayPartialEncoder<C>
where
    C: ChunkCache + 'static,
{
    fn data_type(&self) -> &DataType {
        self.encoder.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.encoder.exists().await
    }

    fn size_held(&self) -> usize {
        self.encoder.size_held()
    }

    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.encoder.local_subchunk_grids(options).await
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.encoder.partial_decode(indexer, options).await
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.encoder
            .partial_decode_into(indexer, output_target, options)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.encoder.supports_partial_decode()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<C> AsyncArrayPartialEncoderTraits for CachedAsyncArrayPartialEncoder<C>
where
    C: ChunkCache + 'static,
{
    async fn erase(&self) -> Result<(), CodecError> {
        let result = self.encoder.erase().await;
        self.cache.invalidate_chunk(&self.chunk_indices);
        result
    }

    async fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let result = self.encoder.partial_encode(indexer, bytes, options).await;
        self.cache.invalidate_chunk(&self.chunk_indices);
        result
    }

    fn supports_partial_encode(&self) -> bool {
        self.encoder.supports_partial_encode()
    }
}

#[inherent]
impl<TStorage, C> AsyncArrayUpdateOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static,
    C: ChunkCache + 'static,
    C::Value: AsyncChunkCacheType,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_subset<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_data,
            self.codec_options(),
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_subset_opt<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .async_store_chunk_subset_opt(chunk_indices, chunk_subset, chunk_subset_data, options)
            .await?;
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_opt(array_subset, subset_data, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset_opt<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .async_store_array_subset_opt(array_subset, subset_data, options)
            .await?;
        if let Some(chunks) = self.array().chunks_in_array_subset(array_subset)? {
            self.cache().invalidate_chunks(&chunks);
        } else {
            self.cache().invalidate();
        }
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_compact_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<bool, ArrayError> {
        let compacted = self
            .array()
            .async_compact_chunk(chunk_indices, options)
            .await?;
        if compacted {
            self.cache().invalidate_chunk(chunk_indices);
        }
        Ok(compacted)
    }

    pub fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits> {
        self.array().async_readable()
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError> {
        let encoder = self
            .array()
            .async_partial_encoder(chunk_indices, options)
            .await?;
        Ok(Arc::new(CachedAsyncArrayPartialEncoder {
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
        ChunkCacheAsyncPartialDecoderLruChunkLimit, ChunkCacheDecodedLruChunkLimit,
        ChunkCacheEncodedLruChunkLimit,
    };
    use crate::array::{ArrayBuilder, ArraySubset, data_type};
    use object_store::memory::InMemory;
    use zarrs_object_store::AsyncObjectStore;

    async fn test_async_partial_encoder_invalidates<C>(cache: C)
    where
        C: ChunkCache + 'static,
        C::Value: AsyncChunkCacheType,
    {
        let store = Arc::new(AsyncObjectStore::new(InMemory::new()));
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array.async_store_chunk(&[0], &[1u8, 2]).await.unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(
            cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap(),
            vec![1, 2]
        );
        assert_eq!(cached.cache().len(), 1);

        let options = CodecOptions::default();
        let encoder = cached.async_partial_encoder(&[0], &options).await.unwrap();
        encoder
            .partial_encode(
                &ArraySubset::new_with_ranges(&[1..2]),
                &vec![3u8].into(),
                &options,
            )
            .await
            .unwrap();
        assert!(cached.cache().is_empty());
        assert_eq!(
            cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap(),
            vec![1, 3]
        );

        encoder.erase().await.unwrap();
        assert!(cached.cache().is_empty());
        assert_eq!(
            cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap(),
            vec![0, 0]
        );

        assert!(!encoder.exists().await.unwrap());

        // Failed partial encodes also invalidate
        assert_eq!(
            cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap(),
            vec![0, 0]
        );
        assert_eq!(cached.cache().len(), 1);
        assert!(
            encoder
                .partial_encode(
                    &ArraySubset::new_with_ranges(&[2..3]),
                    &vec![3u8].into(),
                    &options,
                )
                .await
                .is_err()
        );
        assert!(cached.cache().is_empty());
    }

    #[tokio::test]
    async fn async_partial_encoder_invalidates_all_cache_value_types() {
        test_async_partial_encoder_invalidates(ChunkCacheEncodedLruChunkLimit::new(1)).await;
        test_async_partial_encoder_invalidates(ChunkCacheDecodedLruChunkLimit::new(1)).await;
        test_async_partial_encoder_invalidates(ChunkCacheAsyncPartialDecoderLruChunkLimit::new(1))
            .await;
    }
}

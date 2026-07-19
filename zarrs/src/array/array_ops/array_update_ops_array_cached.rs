use ambisync::ambisync;
use inherent::inherent;
use std::sync::Arc;

#[cfg(feature = "async")]
use super::AsyncArrayUpdateOps;
use super::{ArrayUpdateOps, *};
#[cfg(feature = "async")]
use crate::array::chunk_cache::AsyncChunkCacheType;
use crate::array::chunk_cache::SyncChunkCacheType;
use crate::array::{ArrayBytes, Indexer};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_storage::StorageError;

#[ambisync(
    sync(types(
        CachedAsyncArrayPartialEncoder => CachedArrayPartialEncoder,
        AsyncArrayPartialEncoderTraits => ArrayPartialEncoderTraits,
    )),
    async(feature = "async"),
)]
struct CachedAsyncArrayPartialEncoder<C> {
    encoder: Arc<dyn AsyncArrayPartialEncoderTraits>,
    cache: Arc<C>,
    chunk_indices: ArrayIndices,
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
            CachedAsyncArrayPartialEncoder<C> => CachedArrayPartialEncoder<C>,
        ),
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
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

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncArrayPartialEncoderTraits => ArrayPartialEncoderTraits,
            CachedAsyncArrayPartialEncoder<C> => CachedArrayPartialEncoder<C>,
        ),
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
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

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncArrayUpdateOps => ArrayUpdateOps,
            AsyncReadableWritableStorageTraits => ReadableWritableStorageTraits,
            AsyncReadableStorageTraits => ReadableStorageTraits,
            AsyncChunkCacheType => SyncChunkCacheType,
            AsyncArrayPartialEncoderTraits => ArrayPartialEncoderTraits,
            CachedAsyncArrayPartialEncoder => CachedArrayPartialEncoder,
        ),
    ),
    async(feature = "async"),
)]
#[inherent]
impl<TStorage, C> AsyncArrayUpdateOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static,
    C: ChunkCache + 'static,
    C::Value: AsyncChunkCacheType,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_subset<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
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
    pub async fn async_store_chunk_subset_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
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
    pub async fn async_store_array_subset<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_opt(array_subset, subset_data, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
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

    #[sync_name(readable)]
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
    #[cfg(feature = "async")]
    use crate::array::chunk_cache::ChunkCacheAsyncPartialDecoderLruChunkLimit;
    use crate::array::chunk_cache::{
        ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
        ChunkCachePartialDecoderLruChunkLimit,
    };
    use crate::array::{ArrayBuilder, ArraySubset, data_type};
    #[cfg(feature = "async")]
    use object_store::memory::InMemory;
    #[cfg(feature = "async")]
    use zarrs_object_store::AsyncObjectStore;
    use zarrs_storage::store::MemoryStore;

    #[ambisync(
        sync(
            fns(
                "async_{}",
                test_async_partial_encoder_invalidates => test_partial_encoder_invalidates,
            ),
            types(AsyncChunkCacheType => SyncChunkCacheType),
        ),
        async(feature = "async"),
    )]
    async fn test_async_partial_encoder_invalidates<C>(cache: C)
    where
        C: ChunkCache + 'static,
        C::Value: AsyncChunkCacheType,
    {
        let store = ambisync::alt!(
            sync => Arc::new(MemoryStore::default()),
            async => Arc::new(AsyncObjectStore::new(InMemory::new())),
        );
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

    #[ambisync::test(
        sync(
            fns(test_async_partial_encoder_invalidates => test_partial_encoder_invalidates),
            types(
                ChunkCacheAsyncPartialDecoderLruChunkLimit => ChunkCachePartialDecoderLruChunkLimit,
            ),
            name = "partial_encoder_invalidates_all_cache_value_types",
        ),
        async(feature = "async", test_attr = #[tokio::test]),
    )]
    async fn async_partial_encoder_invalidates_all_cache_value_types() {
        test_async_partial_encoder_invalidates(ChunkCacheEncodedLruChunkLimit::new(1)).await;
        test_async_partial_encoder_invalidates(ChunkCacheDecodedLruChunkLimit::new(1)).await;
        test_async_partial_encoder_invalidates(ChunkCacheAsyncPartialDecoderLruChunkLimit::new(1))
            .await;
    }
}

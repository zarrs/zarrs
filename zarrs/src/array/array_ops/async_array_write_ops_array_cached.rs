use inherent::inherent;
use zarrs_codec::{CowBytes, Resources};

use super::{
    ArrayCached, ArrayError, ArraySubsetTraits, AsyncArrayWriteOps, AsyncWritableStorageTraits,
    Indexer, IntoArrayBytes, MaybeSend, StorageError,
};
use crate::array::chunk_cache::AsyncArrayCache;

#[inherent]
impl<TStorage, C> AsyncArrayWriteOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncWritableStorageTraits + 'static,
    C: AsyncArrayCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata(&self) -> Result<(), StorageError> {
        let result = self.array().async_store_metadata().await;
        self.async_invalidate().await;
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata(&self) -> Result<(), StorageError> {
        let result = self.array().async_erase_metadata().await;
        self.async_invalidate().await;
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result = self
            .array()
            .async_store_chunk(chunk_indices, chunk_data, resources)
            .await;
        self.async_invalidate_chunk(chunk_indices).await;
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result = self
            .array()
            .async_store_chunks(chunks, chunks_data, resources)
            .await;
        self.async_invalidate_chunks(chunks).await;
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), ArrayError> {
        let result = self.array().async_erase_chunk(chunk_indices).await;
        self.async_invalidate_chunk(chunk_indices).await;
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result = self.array().async_erase_chunks(chunks, resources).await;
        self.async_invalidate_chunks(chunks).await;
        result
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub async unsafe fn async_store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let result = unsafe {
            self.array()
                .async_store_encoded_chunk(chunk_indices, encoded_chunk_bytes)
                .await
        };
        self.async_invalidate_chunk(chunk_indices).await;
        result
    }
}

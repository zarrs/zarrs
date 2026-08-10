use inherent::inherent;

use super::{AsyncArrayWriteOps, *};
use crate::array::chunk_cache::ChunkCacheLocalityShared;

#[inherent]
impl<TStorage, C> AsyncArrayWriteOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncWritableStorageTraits + 'static,
    C: ChunkCache<Locality = ChunkCacheLocalityShared>,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata(&self) -> Result<(), StorageError> {
        self.array().async_store_metadata().await?;
        self.cache().invalidate();
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata(&self) -> Result<(), StorageError> {
        self.array().async_erase_metadata().await?;
        self.cache().invalidate();
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError> {
        self.array()
            .async_store_chunk(chunk_indices, chunk_data)
            .await?;
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError> {
        self.array().async_store_chunks(chunks, chunks_data).await?;
        self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError> {
        self.array().async_erase_chunk(chunk_indices).await?;
        let _ = self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<(), StorageError> {
        self.array().async_erase_chunks(chunks).await?;
        let _ = self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub async unsafe fn async_store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: bytes::Bytes,
    ) -> Result<(), ArrayError> {
        unsafe {
            self.array()
                .async_store_encoded_chunk(chunk_indices, encoded_chunk_bytes)
                .await?;
        }
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }
}

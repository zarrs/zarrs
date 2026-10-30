use inherent::inherent;
use zarrs_codec::CowBytes;

use super::{
    ArrayCached, ArrayError, ArraySubsetTraits, ArrayWriteOps, ChunkCache, Indexer, IntoArrayBytes,
    StorageError, WritableStorageTraits,
};

#[inherent]
impl<TStorage, C> ArrayWriteOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + WritableStorageTraits + 'static,
    C: ChunkCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn store_metadata(&self) -> Result<(), StorageError> {
        self.array().store_metadata()?;
        self.cache().invalidate();
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_metadata(&self) -> Result<(), StorageError> {
        self.array().erase_metadata()?;
        self.cache().invalidate();
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError> {
        self.array().store_chunk(chunk_indices, chunk_data)?;
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunks<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError> {
        self.array().store_chunks(chunks, chunks_data)?;
        self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), ArrayError> {
        self.array().erase_chunk(chunk_indices)?;
        let _ = self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_chunks(&self, chunks: &dyn Indexer) -> Result<(), ArrayError> {
        self.array().erase_chunks(chunks)?;
        let _ = self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        unsafe {
            self.array()
                .store_encoded_chunk(chunk_indices, encoded_chunk_bytes)?;
        }
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }
}

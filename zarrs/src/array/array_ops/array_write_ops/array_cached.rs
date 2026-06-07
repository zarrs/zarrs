use inherent::inherent;

use super::super::*;
use super::ArrayWriteOps;

#[inherent]
impl<TStorage, C> ArrayWriteOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + WritableStorageTraits + 'static,
    C: ChunkCache,
{
    pub fn store_metadata(&self) -> Result<(), StorageError> {
        self.array().store_metadata()?;
        self.cache().invalidate();
        Ok(())
    }

    pub fn store_metadata_opt(&self, options: &ArrayMetadataOptions) -> Result<(), StorageError> {
        self.array().store_metadata_opt(options)?;
        self.cache().invalidate();
        Ok(())
    }

    pub fn erase_metadata(&self) -> Result<(), StorageError> {
        self.array().erase_metadata()?;
        self.cache().invalidate();
        Ok(())
    }

    pub fn erase_metadata_opt(&self, options: MetadataEraseVersion) -> Result<(), StorageError> {
        self.array().erase_metadata_opt(options)?;
        self.cache().invalidate();
        Ok(())
    }

    pub fn store_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError> {
        self.store_chunk_opt(chunk_indices, chunk_data, &CodecOptions::default())
    }

    pub fn store_chunk_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .store_chunk_opt(chunk_indices, chunk_data, options)?;
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    pub fn store_chunks<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError> {
        self.store_chunks_opt(chunks, chunks_data, &CodecOptions::default())
    }

    pub fn store_chunks_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .store_chunks_opt(chunks, chunks_data, options)?;
        self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    pub fn erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError> {
        self.array().erase_chunk(chunk_indices)?;
        let _ = self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    pub fn erase_chunks(&self, chunks: &dyn ArraySubsetTraits) -> Result<(), StorageError> {
        self.array().erase_chunks(chunks)?;
        let _ = self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: bytes::Bytes,
    ) -> Result<(), ArrayError> {
        unsafe {
            self.array()
                .store_encoded_chunk(chunk_indices, encoded_chunk_bytes)?;
        }
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }
}

use crate::array::chunk_cache::SyncArrayCache;
use inherent::inherent;
use zarrs_codec::{CowBytes, Resources};

use super::{
    ArrayCached, ArrayError, ArraySubsetTraits, ArrayWriteOps, Indexer, IntoArrayBytes,
    StorageError, WritableStorageTraits,
};

#[inherent]
impl<TStorage, C> ArrayWriteOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + WritableStorageTraits + 'static,
    C: SyncArrayCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn store_metadata(&self) -> Result<(), StorageError> {
        let result = self.array().store_metadata();
        self.cache().invalidate_cached_all(self.array());
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_metadata(&self) -> Result<(), StorageError> {
        let result = self.array().erase_metadata();
        self.cache().invalidate_cached_all(self.array());
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result = self
            .array()
            .store_chunk(chunk_indices, chunk_data, resources);
        self.cache()
            .invalidate_cached_chunk(self.array(), chunk_indices);
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunks<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result = self.array().store_chunks(chunks, chunks_data, resources);
        self.cache().invalidate_cached_chunks(self.array(), chunks);
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), ArrayError> {
        let result = self.array().erase_chunk(chunk_indices);
        self.cache()
            .invalidate_cached_chunk(self.array(), chunk_indices);
        result
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let result = self.array().erase_chunks(chunks, resources);
        self.cache().invalidate_cached_chunks(self.array(), chunks);
        result
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let result = unsafe {
            self.array()
                .store_encoded_chunk(chunk_indices, encoded_chunk_bytes)
        };
        self.cache()
            .invalidate_cached_chunk(self.array(), chunk_indices);
        result
    }
}

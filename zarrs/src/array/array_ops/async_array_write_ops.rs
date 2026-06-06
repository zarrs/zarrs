use super::*;

mod array;

/// Asynchronous array write operations.
#[cfg(feature = "async")]
#[allow(async_fn_in_trait)]
pub trait AsyncArrayWriteOps: ArrayOps {
    /// Async variant of [`ArrayWriteOps::store_metadata`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_metadata(&self) -> Result<(), StorageError>;

    /// Async variant of [`ArrayWriteOps::store_metadata_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_metadata_opt(
        &self,
        options: &ArrayMetadataOptions,
    ) -> Result<(), StorageError>;

    /// Async variant of [`ArrayWriteOps::erase_metadata`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_metadata(&self) -> Result<(), StorageError>;

    /// Async variant of [`ArrayWriteOps::erase_metadata_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_metadata_opt(
        &self,
        options: MetadataEraseVersion,
    ) -> Result<(), StorageError>;

    /// Async variant of [`ArrayWriteOps::store_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_chunk<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayWriteOps::store_chunk_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_chunk_opt<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayWriteOps::store_chunks`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_chunks<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayWriteOps::store_chunks_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_chunks_opt<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayWriteOps::erase_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError>;

    /// Async variant of [`ArrayWriteOps::erase_chunks`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_chunks(&self, chunks: &dyn ArraySubsetTraits) -> Result<(), StorageError>;

    /// Async variant of [`ArrayWriteOps::store_encoded_chunk`].
    ///
    /// # Safety
    /// The responsibility is on the caller to ensure the chunk is encoded correctly
    #[allow(clippy::missing_errors_doc)]
    async unsafe fn async_store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: bytes::Bytes,
    ) -> Result<(), ArrayError>;
}

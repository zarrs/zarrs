use super::*;

mod array;

/// Synchronous array write operations.
pub trait ArrayWriteOps: ArrayOps {
    /// Store metadata with default [`ArrayMetadataOptions`].
    ///
    /// The metadata is created with [`Array::metadata_opt`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    fn store_metadata(&self) -> Result<(), StorageError>;

    /// Store metadata with non-default [`ArrayMetadataOptions`].
    ///
    /// The metadata is created with [`Array::metadata_opt`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    fn store_metadata_opt(&self, options: &ArrayMetadataOptions) -> Result<(), StorageError>;

    /// Erase the metadata with default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    fn erase_metadata(&self) -> Result<(), StorageError>;

    /// Erase the metadata with non-default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    fn erase_metadata_opt(&self, options: MetadataEraseVersion) -> Result<(), StorageError>;

    /// Encode `chunk_data` and store at `chunk_indices`.
    ///
    /// Use [`store_chunk_opt`](ArrayWriteOps::store_chunk_opt) to control codec options.
    /// A chunk composed entirely of the fill value will not be written to the store.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_indices` are invalid,
    ///  - the length of `chunk_data` is not equal to the expected length (the product of the number of elements in the chunk and the data type size in bytes),
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    fn store_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError>;

    /// Explicit options version of [`store_chunk`](ArrayWriteOps::store_chunk).
    fn store_chunk_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Encode `chunks_data` and store at the chunks with indices represented by the `chunks` array subset.
    ///
    /// Use [`store_chunks_opt`](ArrayWriteOps::store_chunks_opt) to control codec options.
    /// A chunk composed entirely of the fill value will not be written to the store.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunks` are invalid,
    ///  - the length of `chunks_data` is not equal to the expected length (the product of the number of elements in the chunks and the data type size in bytes),
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    fn store_chunks<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError>;

    /// Explicit options version of [`store_chunks`](ArrayWriteOps::store_chunks).
    fn store_chunks_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Erase the chunk at `chunk_indices`.
    ///
    /// Succeeds if the chunk does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    fn erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError>;

    /// Erase the chunks in `chunks`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    fn erase_chunks(&self, chunks: &dyn ArraySubsetTraits) -> Result<(), StorageError>;

    /// Store `encoded_chunk_bytes` at `chunk_indices`.
    ///
    /// # Safety
    /// The responsibility is on the caller to ensure the chunk is encoded correctly.
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    unsafe fn store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: bytes::Bytes,
    ) -> Result<(), ArrayError>;
}

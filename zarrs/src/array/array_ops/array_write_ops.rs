use super::*;

/// Synchronous array write operations.
///
/// These operations encode with the array's [`codec_options`](ArrayOps::codec_options) and write
/// metadata according to its [`metadata_options`](ArrayOps::metadata_options) and
/// [`metadata_erase_version`](ArrayOps::metadata_erase_version).
pub trait ArrayWriteOps: ArrayOps {
    /// Store the array metadata.
    ///
    /// The metadata is created with [`ArrayOps::metadata_opt`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    fn store_metadata(&self) -> Result<(), StorageError>;

    /// Erase the array metadata.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    fn erase_metadata(&self) -> Result<(), StorageError>;

    /// Encode `chunk_data` and store at `chunk_indices`.
    ///
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

    /// Encode `chunks_data` and store at the chunks with indices represented by the `chunks` array subset.
    ///
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

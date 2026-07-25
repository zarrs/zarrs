use super::*;

/// Asynchronous array write operations.
#[ambisync::ambisync(
    sync(
        fns("async_{}"),
        types(AsyncArrayWriteOps => ArrayWriteOps),
        declaration {
            /// Synchronous array write operations.
            pub trait ArrayWriteOps: ArrayOps {}
        },
    ),
    async(feature = "async"),
)]
pub trait AsyncArrayWriteOps: ArrayOps {
    /// Store metadata with default [`ArrayMetadataOptions`].
    ///
    /// The metadata is created with [`Array::metadata_opt`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    async fn async_store_metadata(&self) -> Result<(), StorageError> {
        self.async_store_metadata_opt(self.metadata_options()).await
    }

    /// Store metadata with non-default [`ArrayMetadataOptions`].
    ///
    /// The metadata is created with [`Array::metadata_opt`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    async fn async_store_metadata_opt(
        &self,
        options: &ArrayMetadataOptions,
    ) -> Result<(), StorageError>;

    /// Erase the metadata with default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    async fn async_erase_metadata(&self) -> Result<(), StorageError> {
        self.async_erase_metadata_opt(self.metadata_erase_version())
            .await
    }

    /// Erase the metadata with non-default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    async fn async_erase_metadata_opt(
        &self,
        options: MetadataEraseVersion,
    ) -> Result<(), StorageError>;

    /// Encode `chunk_data` and store at `chunk_indices`.
    ///
    /// Use the explicit-options variant to control codec options.
    /// A chunk composed entirely of the fill value will not be written to the store.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_indices` are invalid,
    ///  - the length of `chunk_data` is not equal to the expected length (the product of the number of elements in the chunk and the data type size in bytes),
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    async fn async_store_chunk<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_opt(chunk_indices, chunk_data, self.codec_options())
            .await
    }

    /// Explicit-options variant of the corresponding default-options method.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if encoding or storage fails.
    async fn async_store_chunk_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Encode `chunks_data` and store at the chunks with indices represented by the `chunks` array subset.
    ///
    /// Use the explicit-options variant to control codec options.
    /// A chunk composed entirely of the fill value will not be written to the store.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunks` are invalid,
    ///  - the length of `chunks_data` is not equal to the expected length (the product of the number of elements in the chunks and the data type size in bytes),
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    async fn async_store_chunks<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_chunks_opt(chunks, chunks_data, self.codec_options())
            .await
    }

    /// Explicit-options variant of the corresponding default-options method.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if encoding or storage fails.
    async fn async_store_chunks_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
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
    async fn async_erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError>;

    /// Erase the chunks in `chunks`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    async fn async_erase_chunks(&self, chunks: &dyn ArraySubsetTraits) -> Result<(), StorageError>;

    /// Store `encoded_chunk_bytes` at `chunk_indices`.
    ///
    /// # Safety
    /// The responsibility is on the caller to ensure the chunk is encoded correctly.
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    async unsafe fn async_store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: bytes::Bytes,
    ) -> Result<(), ArrayError>;
}

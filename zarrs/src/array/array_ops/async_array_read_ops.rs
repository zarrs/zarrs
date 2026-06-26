use super::*;
use std::sync::Arc;
use zarrs_codec::{ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits};
use zarrs_storage::Bytes;

mod array;

/// Asynchronous array read operations.
#[cfg(feature = "async")]
#[allow(async_fn_in_trait)]
pub trait AsyncArrayReadOps: ArrayOps {
    /// Async variant of [`ArrayReadOps::retrieve_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunk_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunks`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunks_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunks_opt<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        self.async_retrieve_array_subset_opt(&array_subset, options)
            .await
    }

    /// Async variant of [`ArrayReadOps::retrieve_chunk_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunk_subset_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_subset_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_array_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_array_subset_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset_opt<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunk_if_exists`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunk_if_exists_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_if_exists_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<T>, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_encoded_chunk`].
    ///
    /// Retrieve the encoded bytes of a chunk.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, StorageError> {
        self.async_retrieve_encoded_chunk_opt(chunk_indices, self.codec_options())
            .await
    }

    /// Async variant of [`ArrayReadOps::retrieve_encoded_chunk_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_chunk_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Bytes>, StorageError>;

    /// Async variant of [`ArrayReadOps::retrieve_encoded_chunks`].
    ///
    /// Retrieve the encoded bytes of the chunks in `chunks`.
    ///
    /// The chunks are in order of the chunk indices returned by `chunks.indices().into_iter()`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<Option<Bytes>>, StorageError> {
        self.async_retrieve_encoded_chunks_opt(chunks, self.codec_options())
            .await
    }

    /// Async variant of [`ArrayReadOps::retrieve_encoded_chunks_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_chunks_opt(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Bytes>>, StorageError>;

    /// Async variant of [`ArrayReadOps::retrieve_encoded_subchunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_subchunk_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunk_opt<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_subchunks_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunks_opt<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_array_subset_into`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_array_subset_into_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset_into_opt(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayReadOps::partial_decoder`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>;

    /// Async variant of [`ArrayReadOps::partial_decoder_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_partial_decoder_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>;

    /// Async variant of [`ArrayReadOps::local_subchunk_grid`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ChunkGrid>, ArrayError> {
        self.async_partial_decoder_opt(chunk_indices, options)
            .await?
            .local_subchunk_grid(options)
            .await
            .map_err(ArrayError::CodecError)
    }
}

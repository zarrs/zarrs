use super::*;
use std::sync::Arc;
use zarrs_codec::{ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits, EncodedSubchunk};
use zarrs_storage::Bytes;

/// Asynchronous array read operations.
///
/// These operations decode with the array's [`codec_options`](ArrayOps::codec_options).
#[cfg(feature = "async")]
#[allow(async_fn_in_trait)]
pub trait AsyncArrayReadOps: ArrayOps {
    /// Async variant of [`ArrayReadOps::retrieve_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunk_into`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunks`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        self.async_retrieve_array_subset(&array_subset).await
    }

    /// Async variant of [`ArrayReadOps::retrieve_chunk_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunk_subset_into`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_array_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    /// Async variant of [`ArrayReadOps::retrieve_chunk_if_exists`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
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
    ) -> Result<Vec<Option<Bytes>>, StorageError>;

    /// Async variant of [`ArrayReadOps::retrieve_encoded_subchunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
    ) -> Result<Option<EncodedSubchunk<'static>>, ArrayError> {
        self.async_retrieve_encoded_subchunk_at_level(0, subchunk_indices)
            .await
    }

    /// Async variant of [`ArrayReadOps::retrieve_encoded_subchunk_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<Option<EncodedSubchunk<'static>>, ArrayError> {
        // Locate the chunk holding the subchunk, then defer to its partial decoder
        let (partial_decoder, local_indices) =
            super::async_array_read_ops_common::subchunk_partial_decoder_and_local_indices(
                self,
                level,
                subchunk_indices,
            )
            .await?;
        partial_decoder
            .retrieve_encoded_subchunk_at_level(level, &local_indices, self.codec_options())
            .await
            .map_err(ArrayError::CodecError)
    }

    /// Async variant of [`ArrayReadOps::encoded_subchunk_shape_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_encoded_subchunk_shape_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<ChunkShape, ArrayError> {
        let (partial_decoder, local_indices) =
            super::async_array_read_ops_common::subchunk_partial_decoder_and_local_indices(
                self,
                level,
                subchunk_indices,
            )
            .await?;
        partial_decoder
            .encoded_subchunk_shape_at_level(level, &local_indices, self.codec_options())
            .await
            .map_err(ArrayError::CodecError)
    }

    /// Async variant of [`ArrayReadOps::retrieve_subchunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunk<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
    ) -> Result<T, ArrayError> {
        self.async_retrieve_subchunk_at_level(0, subchunk_indices)
            .await
    }

    /// Async variant of [`ArrayReadOps::retrieve_subchunk_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunk_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<T, ArrayError> {
        let subchunk_grid = self
            .subchunk_grid_at_level(level)
            .as_chunk_grid()
            .ok_or(ArrayError::MissingSubchunkGrid)?;
        let array_subset = subchunk_grid
            .subset(subchunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
        self.async_retrieve_array_subset(&array_subset).await
    }

    /// Async variant of [`ArrayReadOps::retrieve_subchunks`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunks<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_subchunks_at_level(0, subchunks).await
    }

    /// Async variant of [`ArrayReadOps::retrieve_subchunks_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunks_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        let subchunk_grid = self
            .subchunk_grid_at_level(level)
            .as_chunk_grid()
            .ok_or(ArrayError::MissingSubchunkGrid)?;
        let array_subset = subchunk_grid.chunks_subset(subchunks)?.ok_or_else(|| {
            ArrayError::InvalidArraySubset(
                subchunks.to_array_subset(),
                subchunk_grid.grid_shape().to_vec(),
            )
        })?;
        self.async_retrieve_array_subset(&array_subset).await
    }

    /// Async variant of [`ArrayReadOps::retrieve_array_subset_into`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayReadOps::partial_decoder`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>;

    /// Async variant of [`ArrayReadOps::local_subchunk_grid`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkGrid>, ArrayError> {
        self.async_local_subchunk_grid_at_level(0, chunk_indices)
            .await
    }

    /// Async variant of [`ArrayReadOps::local_subchunk_grid_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_local_subchunk_grid_at_level(
        &self,
        level: usize,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkGrid>, ArrayError> {
        Ok(self
            .async_partial_decoder(chunk_indices)
            .await?
            .local_subchunk_grids(self.codec_options())
            .await
            .map_err(ArrayError::CodecError)?
            .into_iter()
            .nth(level)
            .flatten())
    }
}

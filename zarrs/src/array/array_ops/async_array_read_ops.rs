use super::*;
use crate::array::Tensor;
use std::sync::Arc;
use zarrs_codec::{ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits};
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

    /// Async variant of [`ArrayReadOps::retrieve_chunk_stored_layout`].
    ///
    /// Read the chunk at `chunk_indices` in the element layout it is stored in.
    ///
    /// # Errors
    /// Returns [`ArrayError::NoStoredLayout`] if the array to bytes codec does not declare an
    /// element layout, or an [`ArrayError`] if `chunk_indices` are invalid, a bytes to bytes codec
    /// fails to decode, or there is an underlying store error.
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_chunk_stored_layout(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Tensor>, ArrayError> {
        let codecs = self.codecs_bound();
        let layout = codecs
            .array_to_bytes_codec()
            .encoded_element_layout()
            .ok_or(ArrayError::NoStoredLayout)?;

        let Some(encoded) = self.async_retrieve_encoded_chunk(chunk_indices).await? else {
            return Ok(None);
        };

        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let bytes = codecs.decode_bytes_to_bytes(
            std::borrow::Cow::Owned(encoded.into()),
            &chunk_shape,
            self.codec_options(),
        )?;

        // The encoded chunk is described by the array to bytes codec's context, which is the
        // representation after any array to array codecs
        let mut shape = chunk_shape;
        for codec in codecs.array_to_array_codecs() {
            shape = codec.encoded_shape(&shape)?;
        }
        Ok(Some(Tensor::new_with_layout(
            bytes.into_owned(),
            codecs.array_to_bytes_codec().data_type().clone(),
            shape.iter().map(|s| s.get()).collect(),
            layout,
        )))
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

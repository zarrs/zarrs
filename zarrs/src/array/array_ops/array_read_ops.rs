use super::*;
use crate::iter_concurrent_limit;
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs_codec::{ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits};
use zarrs_storage::MaybeSync;

/// Synchronous array read operations.
///
/// These operations decode with the array's [`codec_options`](ArrayOps::codec_options).
pub trait ArrayReadOps: ArrayOps + MaybeSync {
    /// Read and decode the chunk at `chunk_indices` into its bytes or the fill value if it does not exist.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_indices` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of elements in the chunk exceeds `usize::MAX`.
    fn retrieve_chunk<T: FromArrayBytes>(&self, chunk_indices: &[u64]) -> Result<T, ArrayError> {
        let chunk = self.retrieve_chunk_if_exists::<T>(chunk_indices)?;
        super::chunk_or_fill_value(self, chunk_indices, chunk)
    }

    /// Read and decode the chunk at `chunk_indices` into a preallocated `output_target`.
    ///
    /// Only supports fixed-length data types (including optional types with fixed inner types).
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the chunk indices are invalid,
    ///  - the data type is variable-length,
    ///  - the number of elements in `output_target` does not match the chunk,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    /// Read and decode the chunks at `chunks` into their bytes.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - any chunk indices in `chunks` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of array elements in the chunk exceeds `usize::MAX`.
    fn retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        self.retrieve_array_subset(&array_subset)
    }

    /// Read and decode the `chunk_subset` of the chunk at `chunk_indices` into its bytes.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the chunk indices are invalid,
    ///  - the chunk subset is invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Will panic if the number of elements in `chunk_subset` is `usize::MAX` or larger.
    fn retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    /// Read and decode the `chunk_subset` of the chunk at `chunk_indices` into a preallocated `output_target`.
    ///
    /// Only supports fixed-length data types (including optional types with fixed inner types).
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the chunk indices are invalid,
    ///  - the chunk subset is invalid,
    ///  - the data type is variable-length,
    ///  - the number of elements in `output_target` does not match `chunk_subset`,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    fn retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    /// Read and decode the `array_subset` of array into its bytes.
    ///
    /// Out-of-bounds elements will have the fill value.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the `array_subset` dimensionality does not match the chunk grid dimensionality,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if attempting to reference a byte beyond `usize::MAX`.
    fn retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    /// Read and decode the chunk at `chunk_indices` into its bytes if it exists.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_indices` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of elements in the chunk exceeds `usize::MAX`.
    fn retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError>;

    /// Retrieve the encoded bytes of a chunk.
    ///
    /// # Errors
    /// Returns an [`StorageError`] if there is an underlying store error.
    fn retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, StorageError>;

    /// Retrieve the encoded bytes of the chunks in `chunks`.
    ///
    /// The chunks are in order of the chunk indices returned by `chunks.indices().into_iter()`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    fn retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        iter_concurrent_limit!(
            self.codec_options().concurrent_target(),
            chunks.indices(),
            map,
            |chunk_indices| self.retrieve_encoded_chunk(&chunk_indices)
        )
        .collect()
    }

    /// Read and decode the subchunk at `subchunk_indices`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the array does not have a subchunk grid, the subchunk indices
    /// are invalid, there is a codec decoding error, or there is an underlying store error.
    fn retrieve_subchunk<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
    ) -> Result<T, ArrayError> {
        self.retrieve_subchunk_at_level(0, subchunk_indices)
    }

    /// Read and decode the subchunk at `subchunk_indices` from `level`.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the selected level does not have a globally resolvable grid,
    /// the subchunk indices are invalid, a codec fails, or there is an underlying store error.
    fn retrieve_subchunk_at_level<T: FromArrayBytes>(
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
        self.retrieve_array_subset(&array_subset)
    }

    /// Read and decode the subchunks at `subchunks`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the array does not have a subchunk grid, any subchunk indices
    /// are invalid, there is a codec decoding error, or there is an underlying store error.
    fn retrieve_subchunks<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.retrieve_subchunks_at_level(0, subchunks)
    }

    /// Read and decode subchunks from `level`.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the selected level does not have a globally resolvable grid,
    /// any subchunk indices are invalid, a codec fails, or there is an underlying store error.
    fn retrieve_subchunks_at_level<T: FromArrayBytes>(
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
        self.retrieve_array_subset(&array_subset)
    }

    /// Read and decode the `array_subset` of array into a preallocated `output_target`.
    ///
    /// Only supports fixed-length data types (including optional types with fixed inner types).
    ///
    /// Out-of-bounds elements will have the fill value.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the `array_subset` dimensionality does not match the chunk grid dimensionality,
    ///  - the data type is variable-length,
    ///  - the number of elements in `output_target` does not match `array_subset`,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of chunks intersecting `array_subset` exceeds `usize::MAX`.
    fn retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    /// Initialises a partial decoder for the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if initialisation of the partial decoder fails.
    fn partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>;

    /// Return the chunk-local subchunk grid for a chunk, if available.
    ///
    /// The returned grid is relative to the decoded chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the chunk indices are invalid or the local grid cannot be resolved.
    fn local_subchunk_grid(&self, chunk_indices: &[u64]) -> Result<Option<ChunkGrid>, ArrayError> {
        self.local_subchunk_grid_at_level(0, chunk_indices)
    }

    /// Return the chunk-local subchunk grid at `level` for a chunk, if available.
    ///
    /// The returned grid is relative to the decoded chunk at `chunk_indices`.
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the chunk indices are invalid or the local grid hierarchy cannot be resolved.
    fn local_subchunk_grid_at_level(
        &self,
        level: usize,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkGrid>, ArrayError> {
        Ok(self
            .partial_decoder(chunk_indices)?
            .local_subchunk_grids(self.codec_options())
            .map_err(ArrayError::CodecError)?
            .into_iter()
            .nth(level)
            .flatten())
    }
}

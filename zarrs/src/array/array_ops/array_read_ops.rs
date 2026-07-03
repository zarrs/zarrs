use super::*;
use crate::array::ArrayBytes;
use crate::array::array_sharded_ext::subchunk_shard_index_and_subset;
use crate::iter_concurrent_limit;
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs_codec::{ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits, CodecError};
use zarrs_storage::MaybeSync;

mod array;
mod array_cached;
mod common;

/// Synchronous array read operations.
pub trait ArrayReadOps: ArrayOps + MaybeSync {
    /// Read and decode the chunk at `chunk_indices` into its bytes or the fill value if it does not exist with default codec options.
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
        self.retrieve_chunk_opt(chunk_indices, self.codec_options())
    }

    /// Read and decode the chunk at `chunk_indices` with explicit codec options.
    /// Explicit options version of [`retrieve_chunk`](ArrayReadOps::retrieve_chunk).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if let Some(chunk) = self.retrieve_chunk_if_exists_opt::<T>(chunk_indices, options)? {
            Ok(chunk)
        } else {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = ArrayBytes::new_fill_value(
                self.data_type(),
                chunk_shape.iter().map(|&x| x.get()).product::<u64>(),
                self.fill_value(),
            )
            .map_err(CodecError::from)
            .map_err(ArrayError::from)?;
            T::from_array_bytes(
                bytes,
                bytemuck::must_cast_slice(&chunk_shape),
                self.data_type(),
            )
        }
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
        options: &CodecOptions,
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
        self.retrieve_chunks_opt(chunks, self.codec_options())
    }

    /// Read and decode the chunks in `chunks` with explicit codec options.
    /// Explicit options version of [`retrieve_chunks`](ArrayReadOps::retrieve_chunks).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_opt<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        self.retrieve_array_subset_opt(&array_subset, options)
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
    ) -> Result<T, ArrayError> {
        self.retrieve_chunk_subset_opt(chunk_indices, chunk_subset, self.codec_options())
    }

    /// Read and decode a subset of the chunk at `chunk_indices` with explicit codec options.
    /// Explicit options version of [`retrieve_chunk_subset`](ArrayReadOps::retrieve_chunk_subset).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
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
        options: &CodecOptions,
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
    ) -> Result<T, ArrayError> {
        self.retrieve_array_subset_opt(array_subset, self.codec_options())
    }

    /// Read and decode the array subset with explicit codec options.
    /// Explicit options version of [`retrieve_array_subset`](ArrayReadOps::retrieve_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    fn retrieve_array_subset_opt<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    /// Read and decode the chunk at `chunk_indices` into its bytes if it exists with default codec options.
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
    ) -> Result<Option<T>, ArrayError> {
        self.retrieve_chunk_if_exists_opt(chunk_indices, self.codec_options())
    }

    /// Read and decode the chunk at `chunk_indices` if it exists with explicit codec options.
    /// Explicit options version of [`retrieve_chunk_if_exists`](ArrayReadOps::retrieve_chunk_if_exists).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_if_exists_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<T>, ArrayError>;

    /// Retrieve the encoded bytes of a chunk.
    ///
    /// # Errors
    /// Returns an [`StorageError`] if there is an underlying store error.
    #[allow(clippy::missing_panics_doc)]
    fn retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, StorageError> {
        self.retrieve_encoded_chunk_opt(chunk_indices, self.codec_options())
    }

    /// Retrieve the encoded bytes of a chunk with explicit codec options.
    ///
    /// # Errors
    /// Returns an [`StorageError`] if there is an underlying store error.
    #[allow(clippy::missing_panics_doc)]
    fn retrieve_encoded_chunk_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
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
        self.retrieve_encoded_chunks_opt(chunks, self.codec_options())
    }

    /// Retrieve the encoded bytes of the chunks in `chunks` with explicit codec options.
    ///
    /// The chunks are in order of the chunk indices returned by `chunks.indices().into_iter()`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    fn retrieve_encoded_chunks_opt(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        iter_concurrent_limit!(
            options.concurrent_target(),
            chunks.indices(),
            map,
            |chunk_indices| self.retrieve_encoded_chunk_opt(&chunk_indices, options)
        )
        .collect()
    }

    /// Read and decode the subchunk at `subchunk_indices` with explicit codec options.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the array does not have a subchunk grid, the subchunk indices
    /// are invalid, there is a codec decoding error, or there is an underlying store error.
    fn retrieve_subchunk_opt<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let subchunk_grid = self
            .subchunk_grid()
            .ok_or(ArrayError::MissingSubchunkGrid)?;
        let (chunk_indices, chunk_subset) =
            subchunk_shard_index_and_subset(self, subchunk_grid, subchunk_indices)?;
        self.retrieve_chunk_subset_opt(&chunk_indices, &chunk_subset, options)
    }

    /// Read and decode the subchunks at `subchunks` with explicit codec options.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the array does not have a subchunk grid, any subchunk indices
    /// are invalid, there is a codec decoding error, or there is an underlying store error.
    fn retrieve_subchunks_opt<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let subchunk_grid = self
            .subchunk_grid()
            .ok_or(ArrayError::MissingSubchunkGrid)?;
        let array_subset = subchunk_grid.chunks_subset(subchunks)?.ok_or_else(|| {
            ArrayError::InvalidArraySubset(
                subchunks.to_array_subset(),
                subchunk_grid.grid_shape().to_vec(),
            )
        })?;
        self.retrieve_array_subset_opt(&array_subset, options)
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
    fn retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError> {
        self.retrieve_array_subset_into_opt(array_subset, output_target, self.codec_options())
    }

    /// Read and decode an array subset into a preallocated target with explicit codec options.
    /// Explicit options version of [`retrieve_array_subset_into`](ArrayReadOps::retrieve_array_subset_into).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    fn retrieve_array_subset_into_opt(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Initialises a partial decoder for the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if initialisation of the partial decoder fails.
    fn partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        self.partial_decoder_opt(chunk_indices, self.codec_options())
    }

    /// Initialises a partial decoder for the chunk at `chunk_indices` with explicit codec options.
    /// Explicit options version of [`partial_decoder`](ArrayReadOps::partial_decoder).
    #[allow(clippy::missing_errors_doc)]
    fn partial_decoder_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>;

    /// Return the chunk-local subchunk grid for a chunk, if available.
    ///
    /// The returned grid is relative to the decoded chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the chunk indices are invalid or the local grid cannot be resolved.
    fn local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ChunkGrid>, ArrayError> {
        self.partial_decoder_opt(chunk_indices, options)?
            .local_subchunk_grid(options)
            .map_err(ArrayError::CodecError)
    }
}

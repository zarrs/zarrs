use super::*;
use zarrs_codec::{ArrayPartialEncoderTraits, CowBytes};

/// Synchronous array read/write update operations.
///
/// These operations encode and decode with the array's
/// [`codec_options`](ArrayOps::codec_options).
pub trait ArrayUpdateOps: ArrayReadOps + ArrayWriteOps {
    /// Encode `indexer_data` and store it in the elements selected by `indexer` in the chunk at `chunk_indices`.
    ///
    /// `indexer` is relative to the chunk. It may be an [`ArraySubset`] or any other
    /// [`Indexer`], such as a list of chunk-relative indices.
    ///
    /// Prefer to use [`store_chunk`](crate::array::ArrayWriteOps::store_chunk) where possible, since this function may decode the chunk before updating it and reencoding it.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `indexer` is out-of-bounds of the chunk or has an incompatible dimensionality,
    ///  - `indexer` is not an [`ArraySubset`] and the chunk is encoded with the `sharding_indexed`
    ///    codec while experimental partial encoding is enabled (not yet supported),
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if attempting to reference a byte beyond `usize::MAX`.
    fn store_partial_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
    ) -> Result<(), ArrayError>;

    /// Encode `subset_data` and store in `array_subset`.
    ///
    /// Prefer to use [`store_chunk`](crate::array::ArrayWriteOps::store_chunk) or [`store_chunks`](crate::array::ArrayWriteOps::store_chunks) where possible, since this will decode and encode each chunk intersecting `array_subset`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the dimensionality of `array_subset` does not match the chunk grid dimensionality
    ///  - the length of `subset_data` does not match the expected length governed by the shape of the array subset and the data type size,
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of chunks intersecting `array_subset` exceeds `usize::MAX`, or if
    /// attempting to reference a byte beyond `usize::MAX`.
    fn store_array_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError>;

    /// Retrieve the chunk at `chunk_indices`, compact it if possible, and store the compacted chunk back.
    ///
    /// Compaction removes any extraneous data from the encoded chunk representation.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - there is a codec error, or
    ///  - an underlying store error.
    fn compact_chunk(&self, chunk_indices: &[u64]) -> Result<bool, ArrayError>;

    /// Store encoded bytes at indices of the array's outermost subchunk grid.
    ///
    /// Bytes are stored verbatim, including encoded fill values. Nested writes may read and
    /// rewrite enclosing encoded subchunks. Concurrent updates to the same chunk require
    /// external synchronisation.
    ///
    /// # Safety
    /// The bytes must match the destination subchunk codec, encoded-domain shape, data type,
    /// and fill value. See [`EncodedSubchunk`](zarrs_codec::EncodedSubchunk).
    ///
    /// # Errors
    /// Returns an error for invalid indices, unavailable grids, unsupported encoders, or codec
    /// and storage failures.
    unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        // SAFETY: The caller guarantees the payload matches the destination subchunk.
        unsafe { self.store_encoded_subchunk_at_level(0, subchunk_indices, bytes) }
    }

    /// Store encoded bytes at a selected level of the array's subchunk grid hierarchy.
    ///
    /// Level zero is outermost. Enclosing encoded subchunks may be buffered and rewritten.
    ///
    /// # Safety
    /// The bytes must match the selected subchunk codec, encoded-domain shape, data type, and fill value.
    ///
    /// # Errors
    /// Returns an error for invalid addressing, unsupported encoders, or codec and storage failures.
    unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let (encoder, indices) =
            subchunk_partial_encoder_and_local_indices(self, level, subchunk_indices)?;
        // SAFETY: Address translation preserves the target codec and encoded domain.
        unsafe {
            encoder.store_encoded_subchunk_at_level(level, &indices, bytes, self.codec_options())
        }
        .map_err(ArrayError::CodecError)
    }

    /// Erase a subchunk addressed in the array's outermost subchunk grid.
    ///
    /// Missing subchunks read as fill values. Erasing an absent subchunk succeeds.
    /// Concurrent updates to the same chunk require external synchronisation.
    ///
    /// # Errors
    /// Returns an error for invalid addressing, unsupported encoders, or codec and storage failures.
    fn erase_subchunk(&self, subchunk_indices: &[u64]) -> Result<(), ArrayError> {
        self.erase_subchunk_at_level(0, subchunk_indices)
    }

    /// Erase a subchunk at a selected level of the array's subchunk grid hierarchy.
    ///
    /// Level zero is outermost. Enclosing encoded subchunks may be buffered and rewritten;
    /// empty shards are removed.
    ///
    /// # Errors
    /// Returns an error for invalid addressing, unsupported encoders, or codec and storage failures.
    fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<(), ArrayError> {
        let (encoder, indices) =
            subchunk_partial_encoder_and_local_indices(self, level, subchunk_indices)?;
        encoder
            .erase_subchunk_at_level(level, &indices, self.codec_options())
            .map_err(ArrayError::CodecError)
    }

    /// Return a read-only instantiation of the array.
    fn readable(&self) -> Array<dyn ReadableStorageTraits>;

    /// Initialises a partial encoder for the chunk at `chunk_indices`.
    ///
    /// Only one partial encoder should be created for a chunk at a time because:
    /// - partial encoders can hold internal state that may become out of sync, and
    /// - parallel writing to the same chunk [may result in data loss](#parallel-writing).
    ///
    /// Partial encoding with [`ArrayPartialEncoderTraits::partial_encode`] will use parallelism internally where possible.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if initialisation of the partial encoder fails.
    fn partial_encoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError>;
}

fn subchunk_partial_encoder_and_local_indices<A: ArrayUpdateOps + ?Sized>(
    array: &A,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<(Arc<dyn ArrayPartialEncoderTraits>, ArrayIndices), ArrayError> {
    let (chunk_indices, subset) =
        super::subchunk_chunk_and_local_subset(array, level, subchunk_indices)?;
    let encoder = array.partial_encoder(&chunk_indices)?;
    let grid = encoder
        .local_subchunk_grid_at_level(level, array.codec_options())?
        .ok_or(ArrayError::MissingSubchunkGrid)?;
    let indices = zarrs_codec::enclosing_chunk_indices(&grid, &subset)?;
    Ok((encoder, indices))
}

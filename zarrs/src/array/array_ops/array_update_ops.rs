use super::*;
use zarrs_codec::ArrayPartialEncoderTraits;

/// Synchronous array read/write update operations.
///
/// These operations encode and decode with the array's
/// [`codec_options`](ArrayOps::codec_options).
pub trait ArrayUpdateOps: ArrayReadOps + ArrayWriteOps {
    /// Encode `chunk_subset_data` and store in `chunk_subset` of the chunk at `chunk_indices`.
    ///
    /// Prefer to use [`store_chunk`](crate::array::ArrayWriteOps::store_chunk) where possible, since this function may decode the chunk before updating it and reencoding it.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_subset` is invalid or out of bounds of the chunk,
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if attempting to reference a byte beyond `usize::MAX`.
    fn store_chunk_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
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

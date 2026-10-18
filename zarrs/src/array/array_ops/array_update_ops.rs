use super::*;
use crate::array::CowBytes;
use zarrs_codec::ArrayPartialEncoderTraits;

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
    /// The chunk is partially encoded if its codecs support it and the store supports partial writes: only the parts of
    /// the encoded chunk that change are read and written (e.g. the updated subchunks and the index of a shard).
    /// Partial encoding is supported by the `sharding_indexed` and `bytes` codecs, optionally preceded by array-to-array
    /// codecs such as `transpose`, but not followed by bytes-to-bytes codecs.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `indexer` is out-of-bounds of the chunk or has an incompatible dimensionality,
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
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    /// Encode `subset_data` and store in `array_subset`.
    ///
    /// Prefer to use [`store_chunk`](crate::array::ArrayWriteOps::store_chunk) or [`store_chunks`](crate::array::ArrayWriteOps::store_chunks) where possible, since this will decode and encode each chunk intersecting `array_subset`.
    /// Chunks partially covered by `array_subset` are partially encoded where possible, as in [`store_partial_chunk`](Self::store_partial_chunk).
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
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    /// Retrieve the chunk at `chunk_indices`, compact it if possible, and store the compacted chunk back.
    ///
    /// Compaction removes any extraneous data from the encoded chunk representation.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - there is a codec error, or
    ///  - an underlying store error.
    fn compact_chunk(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<bool, ArrayError>;

    /// Store `encoded`, the encoded bytes of the subchunk at `subchunk_indices` of the level zero
    /// [`subchunk_grid`](ArrayOps::subchunk_grid).
    ///
    /// See [`store_encoded_subchunk_at_level`](ArrayUpdateOps::store_encoded_subchunk_at_level).
    ///
    /// # Safety
    /// See [`store_encoded_subchunk_at_level`](ArrayUpdateOps::store_encoded_subchunk_at_level).
    ///
    /// # Errors
    /// See [`store_encoded_subchunk_at_level`](ArrayUpdateOps::store_encoded_subchunk_at_level).
    unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        encoded: CowBytes<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        // SAFETY: The caller upholds the requirements of `store_encoded_subchunk_at_level`.
        unsafe { self.store_encoded_subchunk_at_level(0, subchunk_indices, encoded, resources) }
    }

    /// Store `encoded`, the encoded bytes of the subchunk at `subchunk_indices` of the
    /// [`subchunk_grid_at_level`](ArrayOps::subchunk_grid_at_level) `level`.
    ///
    /// The subchunk is addressed in the decoded domain of the array, like
    /// [`retrieve_encoded_subchunk_at_level`](ArrayReadOps::retrieve_encoded_subchunk_at_level), and its bytes are
    /// stored verbatim. The indexes leading to the subchunk are read, and the subchunk is appended to its chunk with
    /// the index rewritten (e.g. of a shard). The chunk is instead read and written whole if the store does not support
    /// partial writes or a bytes-to-bytes codec encodes the entire chunk (e.g. a checksum), and a subchunk below level
    /// zero is stored in its enclosing subchunks, which are read and rewritten.
    ///
    /// Concurrent updates of the same chunk require external synchronisation.
    ///
    /// # Safety
    /// `encoded` must be encoded by the subchunk codec at `level` ([`ArrayOps::subchunk_codecs_at_level`]) with the
    /// encoded shape of the subchunk, data type and fill value (see [`EncodedSubchunk`](zarrs_codec::EncodedSubchunk)),
    /// such as an encoded subchunk retrieved from an array with the same codecs. The bytes are not validated.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the selected level does not have a globally resolvable grid, the subchunk indices
    /// are invalid, the codecs do not support storing encoded subchunks at `level`, or there is a codec or underlying
    /// store error.
    unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        encoded: CowBytes<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    /// Erase the subchunk at `subchunk_indices` of the level zero [`subchunk_grid`](ArrayOps::subchunk_grid).
    ///
    /// See [`erase_subchunk_at_level`](ArrayUpdateOps::erase_subchunk_at_level).
    ///
    /// # Errors
    /// See [`erase_subchunk_at_level`](ArrayUpdateOps::erase_subchunk_at_level).
    fn erase_subchunk(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.erase_subchunk_at_level(0, subchunk_indices, resources)
    }

    /// Erase the subchunk at `subchunk_indices` of the [`subchunk_grid_at_level`](ArrayOps::subchunk_grid_at_level)
    /// `level`, so its elements are the fill value.
    ///
    /// The subchunk is removed from the index of its chunk as in
    /// [`store_encoded_subchunk_at_level`](ArrayUpdateOps::store_encoded_subchunk_at_level), and a chunk with no
    /// subchunks left is erased. Erasing a subchunk that is not stored succeeds.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the selected level does not have a globally resolvable grid, the subchunk indices
    /// are invalid, the codecs do not support storing encoded subchunks at `level`, or there is a codec or underlying
    /// store error.
    fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<(), ArrayError>;

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
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError>;
}

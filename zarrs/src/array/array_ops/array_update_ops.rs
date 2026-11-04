use super::{
    Arc, Array, ArrayError, ArrayIndices, ArrayReadOps, ArraySubsetTraits, ArrayWriteOps, Indexer,
    IntoArrayBytes, ReadableStorageTraits,
};
#[cfg(doc)]
use super::{ArrayOps, ArraySubset};
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

    /// Store the encoded subchunk at `subchunk_indices` of the outermost subchunk grid, without decoding it.
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
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        // SAFETY: The caller upholds the safety requirements.
        unsafe { self.store_encoded_subchunk_at_level(0, subchunk_indices, bytes) }
    }

    /// Store the encoded subchunk at `subchunk_indices` of the `level` subchunk grid, without decoding it.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    /// The bytes are stored verbatim, even if they encode the fill value.
    /// Subchunks enclosing a subchunk at a deeper level are read, updated, and rewritten without decoding.
    ///
    /// Only one update of a chunk should be in progress at a time, see
    /// [`partial_encoder`](ArrayUpdateOps::partial_encoder).
    ///
    /// # Safety
    /// The `bytes` must be an encoded subchunk of the subchunk codec at `level` (see
    /// [`subchunk_codecs_at_level`](ArrayOps::subchunk_codecs_at_level)), with the shape, data type, and fill value of
    /// the subchunk in its encoded domain (see [`EncodedSubchunk`](zarrs_codec::EncodedSubchunk)), such as an encoded
    /// subchunk retrieved from an array with the same codecs and subchunk grid.
    /// The bytes are not validated, so a mismatch corrupts the chunk.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the selected level does not have a globally resolvable grid, the codecs of the
    /// array do not support storing encoded subchunks at that level, the subchunk indices are invalid, a codec fails,
    /// or there is an underlying store error.
    unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let (encoder, local_indices) =
            subchunk_partial_encoder_and_local_indices(self, level, subchunk_indices)?;
        // SAFETY: The caller upholds the safety requirements, the subchunk is only located in its chunk.
        unsafe {
            encoder.store_encoded_subchunk_at_level(
                level,
                &local_indices,
                bytes,
                self.codec_options(),
            )?;
        }
        Ok(())
    }

    /// Erase the subchunk at `subchunk_indices` of the outermost subchunk grid.
    ///
    /// See [`erase_subchunk_at_level`](ArrayUpdateOps::erase_subchunk_at_level).
    ///
    /// # Errors
    /// See [`erase_subchunk_at_level`](ArrayUpdateOps::erase_subchunk_at_level).
    fn erase_subchunk(&self, subchunk_indices: &[u64]) -> Result<(), ArrayError> {
        self.erase_subchunk_at_level(0, subchunk_indices)
    }

    /// Erase the subchunk at `subchunk_indices` of the `level` subchunk grid, so that it has the fill value.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    /// Erasing a subchunk that is not stored succeeds, and a chunk or enclosing subchunk left empty is erased.
    ///
    /// Only one update of a chunk should be in progress at a time, see
    /// [`partial_encoder`](ArrayUpdateOps::partial_encoder).
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if the selected level does not have a globally resolvable grid, the codecs of the
    /// array do not support erasing subchunks at that level, the subchunk indices are invalid, a codec fails, or there
    /// is an underlying store error.
    fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<(), ArrayError> {
        let (encoder, local_indices) =
            subchunk_partial_encoder_and_local_indices(self, level, subchunk_indices)?;
        encoder.erase_subchunk_at_level(level, &local_indices, self.codec_options())?;
        Ok(())
    }
}

/// Initialise the partial encoder of the chunk holding a subchunk, and locate the subchunk in its `level` subchunk
/// grid.
fn subchunk_partial_encoder_and_local_indices<A: ArrayUpdateOps + ?Sized>(
    array: &A,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<(Arc<dyn ArrayPartialEncoderTraits>, ArrayIndices), ArrayError> {
    let (chunk_indices, subset) =
        super::subchunk_chunk_and_local_subset(array, level, subchunk_indices)?;
    let encoder = array.partial_encoder(&chunk_indices)?;
    let local_indices =
        super::local_subchunk_indices(encoder.as_ref(), level, &subset, array.codec_options())?;
    Ok((encoder, local_indices))
}

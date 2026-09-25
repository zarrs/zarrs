use super::*;
use std::sync::Arc;
use zarrs_codec::{AsyncArrayPartialEncoderTraits, CowBytes};

/// Asynchronous array read/write update operations.
///
/// These operations encode and decode with the array's
/// [`codec_options`](ArrayOps::codec_options).
#[cfg(feature = "async")]
#[allow(async_fn_in_trait)]
pub trait AsyncArrayUpdateOps: AsyncArrayReadOps + AsyncArrayWriteOps {
    /// Async variant of [`ArrayUpdateOps::store_partial_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_partial_chunk<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_array_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_array_subset<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::compact_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_compact_chunk(&self, chunk_indices: &[u64]) -> Result<bool, ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_encoded_subchunk`].
    ///
    /// # Safety
    /// The bytes must match the destination subchunk codec, encoded-domain shape, data type, and fill value.
    ///
    /// # Errors
    /// Returns an error for invalid addressing, unsupported encoders, or codec and storage failures.
    async unsafe fn async_store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        // SAFETY: The caller guarantees the payload matches the destination subchunk.
        unsafe {
            self.async_store_encoded_subchunk_at_level(0, subchunk_indices, bytes)
                .await
        }
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
    async unsafe fn async_store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let (encoder, indices) =
            subchunk_partial_encoder_and_local_indices(self, level, subchunk_indices).await?;
        // SAFETY: Address translation preserves the target codec and encoded domain.
        unsafe {
            encoder
                .store_encoded_subchunk_at_level(level, &indices, bytes, self.codec_options())
                .await
        }
        .map_err(ArrayError::CodecError)
    }

    /// Return a read-only instantiation of the array.
    fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits>;

    /// Async variant of [`ArrayUpdateOps::partial_encoder`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError>;
}

async fn subchunk_partial_encoder_and_local_indices<A: AsyncArrayUpdateOps + ?Sized>(
    array: &A,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<(Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayIndices), ArrayError> {
    let (chunk_indices, subset) =
        super::subchunk_chunk_and_local_subset(array, level, subchunk_indices)?;
    let encoder = array.async_partial_encoder(&chunk_indices).await?;
    let grid = encoder
        .local_subchunk_grid_at_level(level, array.codec_options())
        .await?
        .ok_or(ArrayError::MissingSubchunkGrid)?;
    let indices = zarrs_codec::enclosing_chunk_indices(&grid, &subset)?;
    Ok((encoder, indices))
}

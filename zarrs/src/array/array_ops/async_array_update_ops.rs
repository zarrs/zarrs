use super::{
    Array, ArrayError, ArrayIndices, ArraySubsetTraits, AsyncArrayReadOps, AsyncArrayWriteOps,
    AsyncReadableStorageTraits, Indexer, IntoArrayBytes, MaybeSend,
};
#[cfg(doc)]
use super::{ArrayOps, ArrayUpdateOps};
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

    /// Return a read-only instantiation of the array.
    fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits>;

    /// Async variant of [`ArrayUpdateOps::partial_encoder`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_encoded_subchunk`].
    ///
    /// # Safety
    /// See [`ArrayUpdateOps::store_encoded_subchunk_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async unsafe fn async_store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        // SAFETY: The caller upholds the safety requirements.
        unsafe {
            self.async_store_encoded_subchunk_at_level(0, subchunk_indices, bytes)
                .await
        }
    }

    /// Async variant of [`ArrayUpdateOps::store_encoded_subchunk_at_level`].
    ///
    /// # Safety
    /// See [`ArrayUpdateOps::store_encoded_subchunk_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async unsafe fn async_store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let (encoder, local_indices) =
            async_subchunk_partial_encoder_and_local_indices(self, level, subchunk_indices).await?;
        // SAFETY: The caller upholds the safety requirements, the subchunk is only located in its chunk.
        unsafe {
            encoder
                .store_encoded_subchunk_at_level(level, &local_indices, bytes, self.codec_options())
                .await?;
        }
        Ok(())
    }

    /// Async variant of [`ArrayUpdateOps::erase_subchunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_subchunk(&self, subchunk_indices: &[u64]) -> Result<(), ArrayError> {
        self.async_erase_subchunk_at_level(0, subchunk_indices)
            .await
    }

    /// Async variant of [`ArrayUpdateOps::erase_subchunk_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<(), ArrayError> {
        let (encoder, local_indices) =
            async_subchunk_partial_encoder_and_local_indices(self, level, subchunk_indices).await?;
        encoder
            .erase_subchunk_at_level(level, &local_indices, self.codec_options())
            .await?;
        Ok(())
    }
}

/// Initialise the partial encoder of the chunk holding a subchunk, and locate the subchunk in its `level` subchunk
/// grid.
#[cfg(feature = "async")]
async fn async_subchunk_partial_encoder_and_local_indices<A: AsyncArrayUpdateOps + ?Sized>(
    array: &A,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<(Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayIndices), ArrayError> {
    let (chunk_indices, subset) =
        super::subchunk_chunk_and_local_subset(array, level, subchunk_indices)?;
    let encoder = array.async_partial_encoder(&chunk_indices).await?;
    let local_indices = super::local_subchunk_indices(
        encoder
            .local_subchunk_grid_at_level(level, array.codec_options())
            .await?,
        &subset,
    )?;
    Ok((encoder, local_indices))
}

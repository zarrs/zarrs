use super::*;
use crate::array::CowBytes;
use std::sync::Arc;
use zarrs_codec::AsyncArrayPartialEncoderTraits;

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
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_array_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_array_subset<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::compact_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_compact_chunk(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<bool, ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_encoded_subchunk`].
    ///
    /// # Safety
    /// See [`ArrayUpdateOps::store_encoded_subchunk_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async unsafe fn async_store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        encoded: CowBytes<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        // SAFETY: The caller upholds the requirements of `store_encoded_subchunk_at_level`.
        unsafe {
            self.async_store_encoded_subchunk_at_level(0, subchunk_indices, encoded, resources)
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
        encoded: CowBytes<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::erase_subchunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_subchunk(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_erase_subchunk_at_level(0, subchunk_indices, resources)
            .await
    }

    /// Async variant of [`ArrayUpdateOps::erase_subchunk_at_level`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    /// Return a read-only instantiation of the array.
    fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits>;

    /// Async variant of [`ArrayUpdateOps::partial_encoder`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError>;
}

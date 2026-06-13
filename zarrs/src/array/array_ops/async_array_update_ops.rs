use super::*;
use std::sync::Arc;
use zarrs_codec::AsyncArrayPartialEncoderTraits;

mod array;

/// Asynchronous array read/write update operations.
#[cfg(feature = "async")]
#[allow(async_fn_in_trait)]
pub trait AsyncArrayUpdateOps: AsyncArrayReadOps + AsyncArrayWriteOps {
    /// Async variant of [`ArrayUpdateOps::store_chunk_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_chunk_subset<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_chunk_subset_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_chunk_subset_opt<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_array_subset`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_array_subset<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::store_array_subset_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_store_array_subset_opt<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Async variant of [`ArrayUpdateOps::compact_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_compact_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<bool, ArrayError>;

    /// Return a read-only instantiation of the array.
    fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits>;

    /// Async variant of [`ArrayUpdateOps::partial_encoder`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError>;
}

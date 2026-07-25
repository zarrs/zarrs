use super::*;
use std::sync::Arc;
use zarrs_codec::ArrayPartialEncoderTraits;
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialEncoderTraits;

/// Asynchronous array read/write update operations.
#[ambisync::ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncArrayUpdateOps => ArrayUpdateOps,
            AsyncArrayReadOps => ArrayReadOps,
            AsyncArrayWriteOps => ArrayWriteOps,
            AsyncReadableStorageTraits => ReadableStorageTraits,
            AsyncArrayPartialEncoderTraits => ArrayPartialEncoderTraits,
        ),
        declaration {
            /// Synchronous array read/write update operations.
            pub trait ArrayUpdateOps: ArrayReadOps + ArrayWriteOps {}
        },
    ),
    async(feature = "async"),
)]
pub trait AsyncArrayUpdateOps: AsyncArrayReadOps + AsyncArrayWriteOps {
    /// Encode `chunk_subset_data` and store in `chunk_subset` of the chunk at `chunk_indices` with default codec options.
    ///
    /// Use the explicit-options variant to control codec options.
    /// Prefer to use the whole-chunk store operation where possible, since this function may decode the chunk before updating it and reencoding it.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_subset` is invalid or out of bounds of the chunk,
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if attempting to reference a byte beyond `usize::MAX`.
    async fn async_store_chunk_subset<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_data,
            self.codec_options(),
        )
        .await
    }

    /// Explicit-options variant of the corresponding default-options method.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if encoding or storage fails.
    async fn async_store_chunk_subset_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Encode `subset_data` and store in `array_subset`.
    ///
    /// Use the explicit-options variant to control codec options.
    /// Prefer to use the whole-chunk store operations where possible, since this will decode and encode each chunk intersecting `array_subset`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the dimensionality of `array_subset` does not match the chunk grid dimensionality
    ///  - the length of `subset_data` does not match the expected length governed by the shape of the array subset and the data type size,
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    async fn async_store_array_subset<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_opt(array_subset, subset_data, self.codec_options())
            .await
    }

    /// Explicit-options variant of the corresponding default-options method.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if encoding or storage fails.
    async fn async_store_array_subset_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    /// Retrieve the chunk at `chunk_indices`, compact it if possible, and store the compacted chunk back.
    ///
    /// Compaction removes any extraneous data from the encoded chunk representation.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - there is a codec error, or
    ///  - an underlying store error.
    async fn async_compact_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<bool, ArrayError>;

    /// Return a read-only instantiation of the array.
    #[sync_name(readable)]
    fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits>;

    /// Initialises a partial encoder for the chunk at `chunk_indices`.
    ///
    /// Only one partial encoder should be created for a chunk at a time because:
    /// - partial encoders can hold internal state that may become out of sync, and
    /// - parallel writing to the same chunk [may result in data loss](#parallel-writing).
    ///
    /// Partial encoding with [`AsyncArrayPartialEncoderTraits::partial_encode`] will use parallelism internally where possible.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if initialisation of the partial encoder fails.
    async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError>;
}

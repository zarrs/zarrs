use std::any::Any;

use zarrs_chunk_grid::{ArraySubset, ChunkGrid, Indexer};
use zarrs_data_type::DataType;
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::StorageError;

use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayPartialDecoderNoSubchunkingTraits, CodecError,
    CodecOptions, CowBytes, EncodedSubchunk, InvalidNumberOfElementsError,
    decode_into_array_bytes_target,
};

/// Subchunking traits for an asynchronous partial array decoder.
///
/// This is the asynchronous equivalent of
/// [`ArrayPartialDecoderSubchunkingTraits`](crate::ArrayPartialDecoderSubchunkingTraits).
///
/// Implement [`ArrayPartialDecoderNoSubchunkingTraits`] instead of this trait for a partial decoder
/// without subchunks.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncArrayPartialDecoderSubchunkingTraits: MaybeSend + MaybeSync {
    /// Return the chunk-local subchunk grid hierarchy for this decoder.
    ///
    /// Grids are ordered from outermost to innermost and are relative to the decoded
    /// chunk handled by this partial decoder, not to the full array. A `None` entry
    /// preserves a level that cannot be resolved in this decoder's local context.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid cannot be resolved.
    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError>;

    /// Return the outermost chunk-local subchunk grid for this decoder, if available.
    ///
    /// This is a compatibility wrapper around [`local_subchunk_grids`](Self::local_subchunk_grids).
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid hierarchy cannot be resolved.
    async fn local_subchunk_grid(
        &self,
        options: &CodecOptions,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        self.local_subchunk_grid_at_level(0, options).await
    }

    /// Return the chunk-local subchunk grid at `level` for this decoder, if available.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid hierarchy cannot be resolved.
    async fn local_subchunk_grid_at_level(
        &self,
        level: usize,
        options: &CodecOptions,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        Ok(self
            .local_subchunk_grids(options)
            .await?
            .into_iter()
            .nth(level)
            .flatten())
    }

    /// Retrieve the encoded subchunk at `subchunk_indices` of the outermost subchunk grid.
    ///
    /// See [`ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk`](crate::ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk).
    ///
    /// # Errors
    /// See [`retrieve_encoded_subchunk_at_level`](Self::retrieve_encoded_subchunk_at_level).
    async fn retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<EncodedSubchunk>, CodecError> {
        self.retrieve_encoded_subchunk_at_level(0, subchunk_indices, options)
            .await
    }

    /// Retrieve the encoded subchunk at `subchunk_indices` of the `level` subchunk grid.
    ///
    /// See [`ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_at_level`](crate::ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_at_level).
    ///
    /// # Errors
    /// Returns [`CodecError::UnsupportedEncodedSubchunk`] if this decoder does not expose encoded subchunks, or a
    /// [`CodecError`] if `level` is beyond the subchunk grid hierarchy, the subchunk indices are invalid, a codec
    /// fails, or there is an underlying store error.
    async fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<EncodedSubchunk>, CodecError> {
        let _ = (level, subchunk_indices, options);
        Err(CodecError::UnsupportedEncodedSubchunk)
    }

    /// Retrieve the encoded subchunk at the chunk-local `subset` of the `level` subchunk grid.
    ///
    /// See [`ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_by_subset`](crate::ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_by_subset).
    ///
    /// # Errors
    /// See [`retrieve_encoded_subchunk_at_level`](Self::retrieve_encoded_subchunk_at_level). Also returns a
    /// [`CodecError`] if `subset` is not exactly a subchunk of the `level` subchunk grid.
    async fn retrieve_encoded_subchunk_by_subset(
        &self,
        level: usize,
        subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Option<EncodedSubchunk>, CodecError> {
        let subchunk_indices =
            crate::async_local_subchunk_indices(self, level, subset, options).await?;
        self.retrieve_encoded_subchunk_at_level(level, &subchunk_indices, options)
            .await
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T> AsyncArrayPartialDecoderSubchunkingTraits for T
where
    T: ArrayPartialDecoderNoSubchunkingTraits + MaybeSend + MaybeSync + ?Sized,
{
    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        Ok(Vec::new())
    }
}

/// Asynchronous partial array decoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncArrayPartialDecoderTraits:
    AsyncArrayPartialDecoderSubchunkingTraits + Any + MaybeSend + MaybeSync
{
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    async fn exists(&self) -> Result<bool, StorageError>;

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize;

    /// Partially decode a chunk.
    ///
    /// # Implementation Requirements
    /// Implementations **must** reject an `indexer` that is out-of-bounds of the shape the partial
    /// codec was created with, or that has an incompatible dimensionality, by returning
    /// [`IndexerError`](zarrs_chunk_grid::IndexerError).
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails, array subset is invalid, or the array subset shape does not match array view subset shape.
    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Partially decode into a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It currently only works for fixed length data types.
    ///
    /// The `indexer` shape and dimensionality does not need to match `output_subset`, but the number of elements must match.
    /// Extracted elements from the `indexer` are written as ordered by the indexer.
    /// For an [`ArraySubset`](zarrs_chunk_grid::ArraySubset), that is C order.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in `indexer` does not match the number of elements in `output_view`,
    #[allow(clippy::missing_safety_doc)]
    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }

        let decoded_value = self.partial_decode(indexer, options).await?;
        decode_into_array_bytes_target(&decoded_value, output_target)
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
}

/// Asynchronous partial array encoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncArrayPartialEncoderTraits:
    AsyncArrayPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    async fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Implementation Requirements
    /// Implementations **must** reject an `indexer` that is out-of-bounds of the shape the partial
    /// codec was created with, or that has an incompatible dimensionality, by returning
    /// [`IndexerError`](zarrs_chunk_grid::IndexerError).
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    async fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Store the encoded subchunk at `subchunk_indices` of the outermost subchunk grid.
    ///
    /// See [`ArrayPartialEncoderTraits::store_encoded_subchunk`](crate::ArrayPartialEncoderTraits::store_encoded_subchunk).
    ///
    /// # Safety
    /// See [`store_encoded_subchunk_at_level`](Self::store_encoded_subchunk_at_level).
    ///
    /// # Errors
    /// See [`store_encoded_subchunk_at_level`](Self::store_encoded_subchunk_at_level).
    async unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        // SAFETY: The caller upholds the safety requirements.
        unsafe {
            self.store_encoded_subchunk_at_level(0, subchunk_indices, bytes, options)
                .await
        }
    }

    /// Store the encoded subchunk at `subchunk_indices` of the `level` subchunk grid, without decoding it.
    ///
    /// See [`ArrayPartialEncoderTraits::store_encoded_subchunk_at_level`](crate::ArrayPartialEncoderTraits::store_encoded_subchunk_at_level).
    ///
    /// # Safety
    /// The `bytes` must be an encoded subchunk of the subchunk codec at `level` (see
    /// [`subchunk_codecs_at_level`](crate::ArrayToBytesCodecSubchunkingTraits::subchunk_codecs_at_level)), with the
    /// shape, data type, and fill value of the subchunk in its encoded domain (see [`EncodedSubchunk`]).
    /// The bytes are stored verbatim, so a mismatch is not detected and corrupts the chunk.
    ///
    /// # Errors
    /// Returns [`CodecError::UnsupportedEncodedSubchunk`] if this encoder does not expose encoded subchunks, or a
    /// [`CodecError`] if `level` is beyond the subchunk grid hierarchy, the subchunk indices are invalid, a codec fails,
    /// or there is an underlying store error.
    async unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let _ = (level, subchunk_indices, bytes, options);
        Err(CodecError::UnsupportedEncodedSubchunk)
    }

    /// Store the encoded subchunk at the chunk-local `subset` of the `level` subchunk grid, without decoding it.
    ///
    /// See [`ArrayPartialEncoderTraits::store_encoded_subchunk_by_subset`](crate::ArrayPartialEncoderTraits::store_encoded_subchunk_by_subset).
    ///
    /// # Safety
    /// See [`store_encoded_subchunk_at_level`](Self::store_encoded_subchunk_at_level).
    ///
    /// # Errors
    /// See [`store_encoded_subchunk_at_level`](Self::store_encoded_subchunk_at_level). Also returns a [`CodecError`]
    /// if `subset` is not exactly a subchunk of the `level` subchunk grid.
    async unsafe fn store_encoded_subchunk_by_subset(
        &self,
        level: usize,
        subset: &ArraySubset,
        bytes: CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let subchunk_indices =
            crate::async_local_subchunk_indices(self, level, subset, options).await?;
        // SAFETY: The caller upholds the safety requirements.
        unsafe {
            self.store_encoded_subchunk_at_level(level, &subchunk_indices, bytes, options)
                .await
        }
    }

    /// Erase the subchunk at `subchunk_indices` of the outermost subchunk grid.
    ///
    /// See [`ArrayPartialEncoderTraits::erase_subchunk`](crate::ArrayPartialEncoderTraits::erase_subchunk).
    ///
    /// # Errors
    /// See [`erase_subchunk_at_level`](Self::erase_subchunk_at_level).
    async fn erase_subchunk(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.erase_subchunk_at_level(0, subchunk_indices, options)
            .await
    }

    /// Erase the subchunk at `subchunk_indices` of the `level` subchunk grid, so that it has the fill value.
    ///
    /// See [`ArrayPartialEncoderTraits::erase_subchunk_at_level`](crate::ArrayPartialEncoderTraits::erase_subchunk_at_level).
    ///
    /// # Errors
    /// Returns [`CodecError::UnsupportedEncodedSubchunk`] if this encoder does not expose encoded subchunks, or a
    /// [`CodecError`] if `level` is beyond the subchunk grid hierarchy, the subchunk indices are invalid, a codec fails,
    /// or there is an underlying store error.
    async fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let _ = (level, subchunk_indices, options);
        Err(CodecError::UnsupportedEncodedSubchunk)
    }

    /// Erase the subchunk at the chunk-local `subset` of the `level` subchunk grid, so that it has the fill value.
    ///
    /// See [`ArrayPartialEncoderTraits::erase_subchunk_by_subset`](crate::ArrayPartialEncoderTraits::erase_subchunk_by_subset).
    ///
    /// # Errors
    /// See [`erase_subchunk_at_level`](Self::erase_subchunk_at_level). Also returns a [`CodecError`] if `subset` is
    /// not exactly a subchunk of the `level` subchunk grid.
    async fn erase_subchunk_by_subset(
        &self,
        level: usize,
        subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let subchunk_indices =
            crate::async_local_subchunk_indices(self, level, subset, options).await?;
        self.erase_subchunk_at_level(level, &subchunk_indices, options)
            .await
    }

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

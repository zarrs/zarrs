use std::any::Any;

use zarrs_chunk_grid::{ChunkGrid, Indexer};
use zarrs_data_type::DataType;
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::StorageError;

use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, CodecError, CodecOptions, CowBytes, EncodedSubchunk,
    InvalidNumberOfElementsError, decode_into_array_bytes_target,
};

/// Subchunking traits for a partial array decoder.
///
/// A partial decoder exposes subchunks if the codec that created it encodes a chunk as
/// independently encoded subchunks (e.g. the `sharding_indexed` codec), or if it forwards the
/// subchunks of an inner partial decoder.
///
/// Implement [`ArrayPartialDecoderNoSubchunkingTraits`] instead of this trait for a partial decoder
/// without subchunks.
pub trait ArrayPartialDecoderSubchunkingTraits: MaybeSend + MaybeSync {
    /// Return the chunk-local subchunk grid hierarchy for this decoder.
    ///
    /// Grids are ordered from outermost to innermost and are relative to the decoded
    /// chunk handled by this partial decoder, not to the full array. A `None` entry
    /// preserves a level that cannot be resolved in this decoder's local context.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid cannot be resolved.
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError>;

    /// Return the outermost chunk-local subchunk grid for this decoder, if available.
    ///
    /// This is a compatibility wrapper around [`local_subchunk_grids`](Self::local_subchunk_grids).
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid hierarchy cannot be resolved.
    fn local_subchunk_grid(&self, options: &CodecOptions) -> Result<Option<ChunkGrid>, CodecError> {
        self.local_subchunk_grid_at_level(0, options)
    }

    /// Return the chunk-local subchunk grid at `level` for this decoder, if available.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid hierarchy cannot be resolved.
    fn local_subchunk_grid_at_level(
        &self,
        level: usize,
        options: &CodecOptions,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        Ok(self
            .local_subchunk_grids(options)?
            .into_iter()
            .nth(level)
            .flatten())
    }

    /// Retrieve the encoded subchunk at `subchunk_indices` of the outermost subchunk grid.
    ///
    /// This is a compatibility wrapper around
    /// [`retrieve_encoded_subchunk_at_level`](Self::retrieve_encoded_subchunk_at_level).
    ///
    /// # Errors
    /// See [`retrieve_encoded_subchunk_at_level`](Self::retrieve_encoded_subchunk_at_level).
    fn retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<EncodedSubchunk>, CodecError> {
        self.retrieve_encoded_subchunk_at_level(0, subchunk_indices, options)
    }

    /// Retrieve the encoded subchunk at `subchunk_indices` of the `level` subchunk grid.
    ///
    /// The `subchunk_indices` index the `level` grid returned by
    /// [`local_subchunk_grids`](Self::local_subchunk_grids), which is local to the chunk handled by this decoder.
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    ///
    /// The returned [`EncodedSubchunk`] holds the shape to decode the bytes with in the encoded domain of the subchunk
    /// codec, which is not necessarily the shape of the subchunk in the grid it was addressed by.
    ///
    /// Returns [`None`] if the subchunk, or any subchunk enclosing it, is not stored, in which case it has the fill
    /// value.
    ///
    /// The default implementation returns [`CodecError::UnsupportedEncodedSubchunk`]. A decoder that exposes subchunks
    /// should implement it, and a decoder that wraps another should forward it.
    ///
    /// # Errors
    /// Returns [`CodecError::UnsupportedEncodedSubchunk`] if this decoder does not expose encoded subchunks, or a
    /// [`CodecError`] if `level` is beyond the subchunk grid hierarchy, the subchunk indices are invalid, a codec
    /// fails, or there is an underlying store error.
    fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<EncodedSubchunk>, CodecError> {
        let _ = (level, subchunk_indices, options);
        Err(CodecError::UnsupportedEncodedSubchunk)
    }
}

/// Marker trait for partial array decoders that do not expose subchunks.
pub trait ArrayPartialDecoderNoSubchunkingTraits {}

impl<T> ArrayPartialDecoderSubchunkingTraits for T
where
    T: ArrayPartialDecoderNoSubchunkingTraits + MaybeSend + MaybeSync + ?Sized,
{
    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        Ok(Vec::new())
    }
}

/// Partial array decoder traits.
pub trait ArrayPartialDecoderTraits:
    ArrayPartialDecoderSubchunkingTraits + Any + MaybeSend + MaybeSync
{
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    fn exists(&self) -> Result<bool, StorageError>;

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize;

    /// Partially decode a chunk.
    ///
    /// If the inner `input_handle` is a bytes decoder and partial decoding returns [`None`], then the array subsets have the fill value.
    ///
    /// # Implementation Requirements
    /// Implementations **must** reject an `indexer` that is out-of-bounds of the shape the partial
    /// codec was created with, or that has an incompatible dimensionality, by returning
    /// [`IndexerError`](zarrs_chunk_grid::IndexerError).
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError>;

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
    fn partial_decode_into(
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

        let decoded_value = self.partial_decode(indexer, options)?;
        decode_into_array_bytes_target(&decoded_value, output_target)
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
}

/// Partial array encoder traits.
pub trait ArrayPartialEncoderTraits:
    ArrayPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Implementation Requirements
    /// Implementations **must** reject an `indexer` that is out-of-bounds of the shape the partial
    /// codec was created with, or that has an incompatible dimensionality, by returning
    /// [`IndexerError`](zarrs_chunk_grid::IndexerError).
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Store the encoded subchunk at `subchunk_indices` of the outermost subchunk grid.
    ///
    /// This is a compatibility wrapper around
    /// [`store_encoded_subchunk_at_level`](Self::store_encoded_subchunk_at_level).
    ///
    /// # Safety
    /// See [`store_encoded_subchunk_at_level`](Self::store_encoded_subchunk_at_level).
    ///
    /// # Errors
    /// See [`store_encoded_subchunk_at_level`](Self::store_encoded_subchunk_at_level).
    unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        // SAFETY: The caller upholds the safety requirements.
        unsafe { self.store_encoded_subchunk_at_level(0, subchunk_indices, bytes, options) }
    }

    /// Store the encoded subchunk at `subchunk_indices` of the `level` subchunk grid, without decoding it.
    ///
    /// The `subchunk_indices` index the `level` grid returned by
    /// [`local_subchunk_grids`](ArrayPartialDecoderSubchunkingTraits::local_subchunk_grids).
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    /// Subchunks enclosing a subchunk at a deeper level are read, updated, and rewritten without decoding.
    ///
    /// The default implementation returns [`CodecError::UnsupportedEncodedSubchunk`]. An encoder that exposes
    /// subchunks should implement it, and an encoder that wraps another should forward it.
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
    unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: CowBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let _ = (level, subchunk_indices, bytes, options);
        Err(CodecError::UnsupportedEncodedSubchunk)
    }

    /// Erase the subchunk at `subchunk_indices` of the outermost subchunk grid.
    ///
    /// This is a compatibility wrapper around [`erase_subchunk_at_level`](Self::erase_subchunk_at_level).
    ///
    /// # Errors
    /// See [`erase_subchunk_at_level`](Self::erase_subchunk_at_level).
    fn erase_subchunk(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.erase_subchunk_at_level(0, subchunk_indices, options)
    }

    /// Erase the subchunk at `subchunk_indices` of the `level` subchunk grid, so that it has the fill value.
    ///
    /// The `subchunk_indices` index the `level` grid returned by
    /// [`local_subchunk_grids`](ArrayPartialDecoderSubchunkingTraits::local_subchunk_grids).
    /// Erasing a subchunk that is not stored succeeds, and a chunk or enclosing subchunk left empty is erased.
    ///
    /// The default implementation returns [`CodecError::UnsupportedEncodedSubchunk`]. An encoder that exposes
    /// subchunks should implement it, and an encoder that wraps another should forward it.
    ///
    /// # Errors
    /// Returns [`CodecError::UnsupportedEncodedSubchunk`] if this encoder does not expose encoded subchunks, or a
    /// [`CodecError`] if `level` is beyond the subchunk grid hierarchy, the subchunk indices are invalid, a codec fails,
    /// or there is an underlying store error.
    fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let _ = (level, subchunk_indices, options);
        Err(CodecError::UnsupportedEncodedSubchunk)
    }

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

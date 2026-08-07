use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

use zarrs_chunk_grid::{ChunkGrid, Indexer};
use zarrs_data_type::DataType;
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::StorageError;

use super::subchunking::{enclosing_chunk_indices, plan_subchunk_descent};
use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesRaw, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, CodecError, CodecOptions, InvalidNumberOfElementsError,
    decode_into_array_bytes_target,
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

    /// Return the codecs that encode the subchunks exposed by this decoder, outermost first.
    ///
    /// The codecs at each level match the grids returned by
    /// [`local_subchunk_grids`](Self::local_subchunk_grids). The level zero codecs decode the
    /// bytes returned by [`retrieve_encoded_subchunk`](Self::retrieve_encoded_subchunk) into the
    /// array bytes of a subchunk with the shape given by the level zero grid.
    /// Deeper levels apply to subchunks nested inside subchunks.
    ///
    /// An empty vector indicates that this decoder does not expose encoded subchunks.
    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        Vec::new()
    }

    /// Return the codecs that encode the subchunks at `level`, if any.
    ///
    /// This is a compatibility wrapper around [`subchunk_codecs`](Self::subchunk_codecs).
    fn subchunk_codecs_at_level(&self, level: usize) -> Option<Arc<dyn ArrayToBytesCodecTraits>> {
        self.subchunk_codecs().into_iter().nth(level)
    }

    /// Retrieve the encoded bytes of the subchunk at `subchunk_indices`.
    ///
    /// The `subchunk_indices` are relative to the chunk handled by this partial decoder and index
    /// the level zero grid returned by [`local_subchunk_grids`](Self::local_subchunk_grids).
    /// The returned bytes are decodable with the level zero codecs returned by
    /// [`subchunk_codecs`](Self::subchunk_codecs).
    ///
    /// Returns [`None`] if the subchunk is not stored, in which case it has the fill value.
    ///
    /// # Errors
    /// Returns [`CodecError::UnsupportedEncodedSubchunk`] if this decoder does not expose encoded
    /// subchunks, or a [`CodecError`] if the subchunk indices are invalid, a codec fails, or there
    /// is an underlying store error.
    fn retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        _ = (subchunk_indices, options);
        Err(CodecError::UnsupportedEncodedSubchunk)
    }

    /// Initialise a partial decoder over the encoded bytes of the subchunk at `subchunk_indices`.
    ///
    /// This is the lazy equivalent of
    /// [`retrieve_encoded_subchunk`](Self::retrieve_encoded_subchunk), and it is how
    /// [`retrieve_encoded_subchunk_at_level`](Self::retrieve_encoded_subchunk_at_level) descends
    /// into nested subchunks.
    /// The default implementation retrieves the entire encoded subchunk. Override it if the
    /// encoded subchunk can be read lazily (e.g. it occupies a byte range of the input handle),
    /// so that nested subchunks can be read without retrieving their enclosing subchunk in full.
    ///
    /// Returns [`None`] if the subchunk is not stored.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if [`retrieve_encoded_subchunk`](Self::retrieve_encoded_subchunk)
    /// fails, or an override fails to initialise a partial decoder.
    fn encoded_subchunk_partial_decoder(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<dyn BytesPartialDecoderTraits>>, CodecError> {
        Ok(self
            .retrieve_encoded_subchunk(subchunk_indices, options)?
            .map(|bytes| Arc::new(bytes.into_owned()) as Arc<dyn BytesPartialDecoderTraits>))
    }

    /// Retrieve the encoded bytes of the subchunk at `subchunk_indices` of `level`.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    /// The `subchunk_indices` index the `level` grid returned by
    /// [`local_subchunk_grids`](Self::local_subchunk_grids), and the returned bytes are decodable
    /// with [`subchunk_codecs_at_level`](Self::subchunk_codecs_at_level) at the same level.
    ///
    /// The default implementation descends through the subchunk hierarchy with
    /// [`encoded_subchunk_partial_decoder`](Self::encoded_subchunk_partial_decoder) and
    /// [`subchunk_codecs`](Self::subchunk_codecs), so a codec only has to implement level zero
    /// access.
    ///
    /// Returns [`None`] if the subchunk, or any subchunk enclosing it, is not stored.
    ///
    /// # Errors
    /// Returns [`CodecError::UnsupportedEncodedSubchunk`] if a level does not expose encoded
    /// subchunks, or a [`CodecError`] if `level` is beyond the subchunk grid hierarchy, the
    /// subchunk indices are invalid, a codec fails, or there is an underlying store error.
    fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'static>>, CodecError> {
        if level == 0 {
            return Ok(self
                .retrieve_encoded_subchunk(subchunk_indices, options)?
                .map(|bytes| Cow::Owned(bytes.into_owned())));
        }

        // Descend into the level zero subchunk enclosing the requested subchunk
        let descent = plan_subchunk_descent(
            &self.local_subchunk_grids(options)?,
            level,
            subchunk_indices,
        )?;
        let Some(input_handle) =
            self.encoded_subchunk_partial_decoder(&descent.subchunk_indices, options)?
        else {
            return Ok(None);
        };
        let codecs = self
            .subchunk_codecs_at_level(0)
            .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
        let partial_decoder =
            codecs.partial_decoder(input_handle, &descent.subchunk_shape, options)?;

        // Repeat with the subchunk grids of the subchunk
        let subchunk_grid = partial_decoder
            .local_subchunk_grid_at_level(level - 1, options)?
            .ok_or_else(|| {
                CodecError::Other(format!(
                    "the subchunk grid at level {level} is not resolvable for this subchunk"
                ))
            })?;
        let subchunk_indices = enclosing_chunk_indices(&subchunk_grid, &descent.subset)?;
        partial_decoder.retrieve_encoded_subchunk_at_level(level - 1, &subchunk_indices, options)
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
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

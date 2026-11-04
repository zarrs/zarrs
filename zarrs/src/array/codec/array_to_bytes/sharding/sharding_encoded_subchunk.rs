//! Encoded subchunk access for the sharding codec.

use std::sync::Arc;

use bytes::Bytes;

use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_codec::{
    ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, ByteIntervalPartialDecoder,
    BytesPartialDecoderTraits, CodecError, CodecOptions, EncodedSubchunk,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

use super::{nested_local_subchunk_grids, subchunk_byte_range};
use crate::array::array_ops::{enclosing_chunk_indices, subset_chunk_indices};
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::codec::{InMemoryBytesEncoder, InMemoryBytesUpdate};
use crate::array::{
    ArrayIndices, ArraySubset, ChunkGrid, ChunkShape, CodecChainBound, ravel_indices,
};

/// Return the shape of the subchunk at `subchunk_indices` of a shard.
fn subchunk_shape(
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_indices: &[u64],
) -> Result<ChunkShape, CodecError> {
    subchunk_grid.chunk_shape(subchunk_indices)?.ok_or_else(|| {
        CodecError::Other(format!(
            "subchunk indices {subchunk_indices:?} are out of bounds of a subchunk grid with shape {:?}",
            subchunk_grid.grid_shape()
        ))
    })
}

/// Return the linear index of the subchunk at `subchunk_indices` in the shard index.
pub(crate) fn subchunk_linear_index(
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_indices: &[u64],
) -> Result<u64, CodecError> {
    ravel_indices(subchunk_indices, subchunk_grid.grid_shape()).ok_or_else(|| {
        CodecError::Other(format!(
            "subchunk indices {subchunk_indices:?} are out of bounds of a subchunk grid with shape {:?}",
            subchunk_grid.grid_shape()
        ))
    })
}

/// The level zero subchunk of a shard enclosing a subchunk at a deeper level of its subchunk grid hierarchy.
pub(crate) struct EnclosingSubchunk {
    /// The indices of the level zero subchunk.
    pub(crate) indices: ArrayIndices,
    /// The shape of the level zero subchunk.
    pub(crate) shape: ChunkShape,
    /// The subset of the nested subchunk, relative to the level zero subchunk.
    pub(crate) subset: ArraySubset,
}

/// Locate the level zero subchunk enclosing the subchunk at `subchunk_indices` of the `level` subchunk grid of a shard.
///
/// # Errors
/// Returns a [`CodecError`] if `level` is zero or beyond the subchunk grid hierarchy, the grid at `level` is not
/// resolvable, or `subchunk_indices` are invalid.
pub(crate) fn enclosing_subchunk(
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &CodecChainBound,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<EnclosingSubchunk, CodecError> {
    debug_assert!(level > 0, "level zero does not require a descent");
    let outer_grid = ChunkGrid::new(subchunk_grid.clone());
    let local_subchunk_grids = nested_local_subchunk_grids(outer_grid.clone(), inner_codecs)?;
    let target_grid = match local_subchunk_grids.get(level) {
        Some(Some(grid)) => grid,
        Some(None) => {
            return Err(CodecError::Other(format!(
                "the subchunk grid at level {level} is not resolvable for this shard"
            )));
        }
        None => {
            return Err(CodecError::Other(format!(
                "there is no subchunk grid at level {level}"
            )));
        }
    };
    let subset = target_grid.subset(subchunk_indices)?.ok_or_else(|| {
        CodecError::Other(format!(
            "subchunk indices {subchunk_indices:?} are out of bounds of a subchunk grid with shape {:?}",
            target_grid.grid_shape()
        ))
    })?;
    let indices = enclosing_chunk_indices(&outer_grid, &subset)?;
    let outer_subset = outer_grid
        .subset(&indices)?
        .expect("enclosing subchunk indices are in bounds");
    Ok(EnclosingSubchunk {
        shape: subchunk_shape(subchunk_grid, &indices)?,
        subset: subset.relative_to(outer_subset.start())?,
        indices,
    })
}

/// An in-memory partial encoder of the level zero subchunk of a shard enclosing a subchunk at a deeper level.
pub(crate) struct EnclosingSubchunkEncoder {
    /// The linear index of the level zero subchunk in the shard index.
    pub(crate) index: u64,
    /// The encoded bytes of the level zero subchunk, updated in place by [`encoder`](Self::encoder).
    buffer: Arc<InMemoryBytesEncoder>,
    /// The partial encoder of the level zero subchunk.
    pub(crate) encoder: Arc<dyn ArrayPartialEncoderTraits>,
    /// The indices of the subchunk in the subchunk grid one level below of [`encoder`](Self::encoder).
    pub(crate) nested_indices: ArrayIndices,
}

impl EnclosingSubchunkEncoder {
    /// Create a partial encoder of the `existing` encoded bytes of the level zero subchunk `enclosing`.
    pub(crate) fn new(
        inner_codecs: &Arc<CodecChainBound>,
        index: u64,
        existing: Option<Bytes>,
        enclosing: &EnclosingSubchunk,
        level: usize,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let buffer = Arc::new(InMemoryBytesEncoder::new(existing));
        let encoder =
            inner_codecs
                .clone()
                .partial_encoder(buffer.clone(), &enclosing.shape, options)?;
        let nested_grid = encoder
            .local_subchunk_grid_at_level(level - 1, options)?
            .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
        let nested_indices = subset_chunk_indices(&nested_grid, &enclosing.subset)?;
        Ok(Self {
            index,
            buffer,
            encoder,
            nested_indices,
        })
    }

    /// Returns whether the level zero subchunk is stored.
    pub(crate) fn exists(&self) -> Result<bool, StorageError> {
        self.buffer.exists()
    }

    /// Take the update of the encoded bytes of the level zero subchunk by [`encoder`](Self::encoder).
    pub(crate) fn take_update(&self) -> InMemoryBytesUpdate {
        self.buffer.take_update()
    }
}

/// Return the byte offset and length of a non-empty encoded subchunk byte range.
fn offset_length(byte_range: ByteRange) -> (u64, u64) {
    match byte_range {
        ByteRange::FromStart(offset, Some(length)) => (offset, length),
        ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => {
            unreachable!("subchunk byte ranges are bounded")
        }
    }
}

/// Retrieve the encoded subchunk at `subchunk_indices` of the `level` subchunk grid of a shard.
pub(crate) fn retrieve_encoded_subchunk_at_level(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    level: usize,
    subchunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<Option<EncodedSubchunk>, CodecError> {
    if level == 0 {
        let shape = subchunk_shape(subchunk_grid, subchunk_indices)?;
        let Some(byte_range) = subchunk_byte_range(shard_index, subchunk_grid, subchunk_indices)?
        else {
            return Ok(None);
        };
        return Ok(input_handle
            .partial_decode(byte_range, options)?
            .map(|bytes| EncodedSubchunk::new(bytes.into_bytes(), shape)));
    }

    // Descend into the level zero subchunk enclosing the subchunk, which is read lazily
    let enclosing = enclosing_subchunk(subchunk_grid, inner_codecs, level, subchunk_indices)?;
    let Some(byte_range) = subchunk_byte_range(shard_index, subchunk_grid, &enclosing.indices)?
    else {
        return Ok(None);
    };
    let (offset, length) = offset_length(byte_range);
    let subchunk_handle: Arc<dyn BytesPartialDecoderTraits> = Arc::new(
        ByteIntervalPartialDecoder::new(input_handle.clone(), offset, length),
    );
    let partial_decoder = inner_codecs.clone().encoded_subchunk_partial_decoder(
        subchunk_handle,
        &enclosing.shape,
        options,
    )?;
    let nested_grid = partial_decoder
        .local_subchunk_grid_at_level(level - 1, options)?
        .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
    let nested_indices = subset_chunk_indices(&nested_grid, &enclosing.subset)?;
    partial_decoder.retrieve_encoded_subchunk_at_level(level - 1, &nested_indices, options)
}

/// Retrieve the encoded subchunk at `subchunk_indices` of the `level` subchunk grid of a shard.
#[cfg(feature = "async")]
pub(crate) async fn async_retrieve_encoded_subchunk_at_level(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    level: usize,
    subchunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<Option<EncodedSubchunk>, CodecError> {
    if level == 0 {
        let shape = subchunk_shape(subchunk_grid, subchunk_indices)?;
        let Some(byte_range) = subchunk_byte_range(shard_index, subchunk_grid, subchunk_indices)?
        else {
            return Ok(None);
        };
        return Ok(input_handle
            .partial_decode(byte_range, options)
            .await?
            .map(|bytes| EncodedSubchunk::new(bytes.into_bytes(), shape)));
    }

    // Descend into the level zero subchunk enclosing the subchunk, which is read lazily
    let enclosing = enclosing_subchunk(subchunk_grid, inner_codecs, level, subchunk_indices)?;
    let Some(byte_range) = subchunk_byte_range(shard_index, subchunk_grid, &enclosing.indices)?
    else {
        return Ok(None);
    };
    let (offset, length) = offset_length(byte_range);
    let subchunk_handle: Arc<dyn AsyncBytesPartialDecoderTraits> = Arc::new(
        AsyncByteIntervalPartialDecoder::new(input_handle.clone(), offset, length),
    );
    let partial_decoder = inner_codecs
        .clone()
        .async_encoded_subchunk_partial_decoder(subchunk_handle, &enclosing.shape, options)
        .await?;
    let nested_grid = partial_decoder
        .local_subchunk_grid_at_level(level - 1, options)
        .await?
        .ok_or(CodecError::UnsupportedEncodedSubchunk)?;
    let nested_indices = subset_chunk_indices(&nested_grid, &enclosing.subset)?;
    partial_decoder
        .retrieve_encoded_subchunk_at_level(level - 1, &nested_indices, options)
        .await
}

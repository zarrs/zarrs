//! Encoded subchunk access for the sharding codec.
//!
//! Subchunks are addressed by their subset of a shard as they descend into nested subchunks (see
//! [`ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_by_subset`]). Each level locates the enclosing
//! subchunk in its own subchunk grid, so the subchunk grid hierarchy is not resolved.
//!
//! [`ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_by_subset`]: zarrs_codec::ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_by_subset

use std::sync::Arc;

use bytes::Bytes;

use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_codec::{
    ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, ByteIntervalPartialDecoder,
    BytesPartialDecoderTraits, CodecError, CodecOptions, EncodedSubchunk, subset_subchunk_indices,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

use super::{nested_local_subchunk_grids, subchunk_byte_range};
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

/// Return the subset of the subchunk at `subchunk_indices` of the `level` subchunk grid of a shard.
///
/// Level zero is located in the subchunk grid of the shard. A deeper level resolves the subchunk grid hierarchy of the
/// shard, which is needed only to address a subchunk by indices.
///
/// # Errors
/// Returns a [`CodecError`] if the grid at `level` is not resolvable, or `subchunk_indices` are out of bounds.
pub(crate) fn subchunk_subset_at_level(
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &CodecChainBound,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<ArraySubset, CodecError> {
    let subset = if level == 0 {
        subchunk_grid.subset(subchunk_indices)?
    } else {
        let local_subchunk_grids =
            nested_local_subchunk_grids(ChunkGrid::new(subchunk_grid.clone()), inner_codecs)?;
        match local_subchunk_grids.get(level) {
            Some(Some(grid)) => grid.subset(subchunk_indices)?,
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
        }
    };
    subset.ok_or_else(|| {
        CodecError::Other(format!(
            "subchunk indices {subchunk_indices:?} are out of bounds of the subchunk grid at level {level}"
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

/// Locate the level zero subchunk of a shard enclosing the shard-local `subset` of a subchunk at a deeper level.
///
/// # Errors
/// Returns a [`CodecError`] if `subset` is out of bounds of the shard or spans several level zero subchunks.
pub(crate) fn enclosing_subchunk(
    subchunk_grid: &RegularBoundedChunkGrid,
    subset: &ArraySubset,
) -> Result<EnclosingSubchunk, CodecError> {
    let chunks = subchunk_grid
        .chunks_in_array_subset(subset)?
        .ok_or_else(|| {
            CodecError::Other(format!(
                "subchunk subset {subset:?} is out of bounds of a subchunk grid with shape {:?}",
                subchunk_grid.grid_shape()
            ))
        })?;
    if chunks.num_elements() != 1 {
        return Err(CodecError::Other(format!(
            "subchunk subset {subset:?} spans multiple level zero subchunks of a shard"
        )));
    }
    let indices: ArrayIndices = chunks.start().into();
    let enclosing_subset = subchunk_grid
        .subset(&indices)?
        .expect("enclosing subchunk indices are in bounds");
    Ok(EnclosingSubchunk {
        shape: subchunk_shape(subchunk_grid, &indices)?,
        subset: subset.relative_to(enclosing_subset.start())?,
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
    /// The subset of the subchunk, relative to the level zero subchunk.
    pub(crate) nested_subset: ArraySubset,
}

impl EnclosingSubchunkEncoder {
    /// Create a partial encoder of the `existing` encoded bytes of the level zero subchunk `enclosing`.
    pub(crate) fn new(
        inner_codecs: &Arc<CodecChainBound>,
        index: u64,
        existing: Option<Bytes>,
        enclosing: EnclosingSubchunk,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let buffer = Arc::new(InMemoryBytesEncoder::new(existing));
        let encoder =
            inner_codecs
                .clone()
                .partial_encoder(buffer.clone(), &enclosing.shape, options)?;
        Ok(Self {
            index,
            buffer,
            encoder,
            nested_subset: enclosing.subset,
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

/// Retrieve the encoded subchunk at the shard-local `subset` of the `level` subchunk grid of a shard.
pub(crate) fn retrieve_encoded_subchunk_by_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    level: usize,
    subset: &ArraySubset,
    options: &CodecOptions,
) -> Result<Option<EncodedSubchunk>, CodecError> {
    if level == 0 {
        let subchunk_indices = subset_subchunk_indices(subchunk_grid, subset)?;
        let shape = subchunk_shape(subchunk_grid, &subchunk_indices)?;
        let Some(byte_range) = subchunk_byte_range(shard_index, subchunk_grid, &subchunk_indices)?
        else {
            return Ok(None);
        };
        return Ok(input_handle
            .partial_decode(byte_range, options)?
            .map(|bytes| EncodedSubchunk::new(bytes.into_bytes(), shape)));
    }

    // Descend into the level zero subchunk enclosing the subchunk, which is read lazily
    let enclosing = enclosing_subchunk(subchunk_grid, subset)?;
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
    partial_decoder.retrieve_encoded_subchunk_by_subset(level - 1, &enclosing.subset, options)
}

/// Retrieve the encoded subchunk at the shard-local `subset` of the `level` subchunk grid of a shard.
#[cfg(feature = "async")]
pub(crate) async fn async_retrieve_encoded_subchunk_by_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    level: usize,
    subset: &ArraySubset,
    options: &CodecOptions,
) -> Result<Option<EncodedSubchunk>, CodecError> {
    if level == 0 {
        let subchunk_indices = subset_subchunk_indices(subchunk_grid, subset)?;
        let shape = subchunk_shape(subchunk_grid, &subchunk_indices)?;
        let Some(byte_range) = subchunk_byte_range(shard_index, subchunk_grid, &subchunk_indices)?
        else {
            return Ok(None);
        };
        return Ok(input_handle
            .partial_decode(byte_range, options)
            .await?
            .map(|bytes| EncodedSubchunk::new(bytes.into_bytes(), shape)));
    }

    // Descend into the level zero subchunk enclosing the subchunk, which is read lazily
    let enclosing = enclosing_subchunk(subchunk_grid, subset)?;
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
    partial_decoder
        .retrieve_encoded_subchunk_by_subset(level - 1, &enclosing.subset, options)
        .await
}

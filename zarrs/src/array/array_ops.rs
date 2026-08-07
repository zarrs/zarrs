use std::sync::Arc;

use super::chunk_cache::ChunkCache;
use super::{
    Array, ArrayBuilder, ArrayCached, ArrayCreateError, ArrayError, ArrayIndices, ArrayMetadata,
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkGridDecoded,
    ChunkGridDecodedRef, ChunkKeyEncoding, ChunkShape, CodecChain, CodecChainBound,
    CodecCreateError, CodecOptions, CodecSpecificOptions, DataType, DimensionName, FillValue,
    FromArrayBytes, IncompatibleDimensionalityError, IntoArrayBytes, NodePath,
    StorageTransformerChain,
};
use crate::config::MetadataEraseVersion;
use zarrs_codec::{ArrayToBytesCodecSubchunkingTraits, ArrayToBytesCodecTraits};
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits, AsyncWritableStorageTraits,
    MaybeSend,
};
use zarrs_storage::{
    ReadableStorageTraits, ReadableWritableStorageTraits, StorageError, StoreKey,
    WritableStorageTraits,
};

/// Locate the chunk holding the subchunk at `subchunk_indices` of the `level` subchunk grid.
///
/// Returns the chunk indices and the subchunk subset relative to the origin of that chunk.
fn subchunk_chunk_and_local_subset<A: ArrayOps + ?Sized>(
    array: &A,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<(ArrayIndices, ArraySubset), ArrayError> {
    let subchunk_grid = array
        .subchunk_grid_at_level(level)
        .as_chunk_grid()
        .ok_or(ArrayError::MissingSubchunkGrid)?;
    if subchunk_indices.len() != subchunk_grid.dimensionality()
        || std::iter::zip(subchunk_indices, subchunk_grid.grid_shape())
            .any(|(indices, shape)| indices >= shape)
    {
        return Err(ArrayError::InvalidChunkGridIndicesError(
            subchunk_indices.to_vec(),
        ));
    }
    let subchunk_subset = subchunk_grid
        .subset(subchunk_indices)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    let chunks = array
        .chunks_in_array_subset(&subchunk_subset)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    if chunks.num_elements() != 1 {
        // This should not happen, as a subchunk grid must refine the chunk grid
        return Err(ArrayError::UnsupportedMethod(
            "a subchunk spanning multiple chunks cannot be retrieved".to_string(),
        ));
    }
    let chunk_indices = chunks.start().to_vec();
    let chunk_origin = array.chunk_origin(&chunk_indices)?;
    let local_subset = subchunk_subset.relative_to(&chunk_origin)?;
    Ok((chunk_indices, local_subset))
}

/// Return the indices of the chunk of `chunk_grid` which encloses `subset`.
fn enclosing_subchunk_indices(
    chunk_grid: &ChunkGrid,
    subset: &ArraySubset,
) -> Result<ArrayIndices, ArrayError> {
    let chunks = chunk_grid.chunks_in_array_subset(subset)?.ok_or_else(|| {
        ArrayError::InvalidArraySubset(subset.clone(), chunk_grid.grid_shape().to_vec())
    })?;
    if chunks.num_elements() != 1 {
        return Err(ArrayError::UnsupportedMethod(
            "a subchunk spanning multiple subchunks of an inner grid cannot be retrieved"
                .to_string(),
        ));
    }
    Ok(chunks.start().to_vec())
}

mod array_mut_ops;
mod array_mut_ops_array;
#[allow(clippy::module_inception)]
mod array_ops;
mod array_ops_array;
mod array_ops_array_cached;
mod array_read_ops;
mod array_read_ops_array;
mod array_read_ops_array_cached;
mod array_read_ops_common;
mod array_update_ops;
mod array_update_ops_array;
mod array_update_ops_array_cached;
mod array_write_ops;
mod array_write_ops_array;
mod array_write_ops_array_cached;
#[cfg(feature = "async")]
mod async_array_read_ops;
#[cfg(feature = "async")]
mod async_array_read_ops_array;
#[cfg(feature = "async")]
mod async_array_update_ops;
#[cfg(feature = "async")]
mod async_array_update_ops_array;
#[cfg(feature = "async")]
mod async_array_write_ops;
#[cfg(feature = "async")]
mod async_array_write_ops_array;

pub use array_mut_ops::ArrayMutOps;
pub use array_ops::ArrayOps;
pub use array_read_ops::ArrayReadOps;
pub use array_update_ops::ArrayUpdateOps;
pub use array_write_ops::ArrayWriteOps;
#[cfg(feature = "async")]
pub use async_array_read_ops::AsyncArrayReadOps;
#[cfg(feature = "async")]
pub use async_array_update_ops::AsyncArrayUpdateOps;
#[cfg(feature = "async")]
pub use async_array_write_ops::AsyncArrayWriteOps;

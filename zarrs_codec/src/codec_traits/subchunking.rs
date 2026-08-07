//! Shared logic for the subchunking traits of partial decoders.

use zarrs_chunk_grid::{ArrayIndices, ArraySubset, ChunkGrid, ChunkShape};

use crate::CodecError;

/// Return the level `level` chunk-local subchunk grid, if it is resolvable.
///
/// # Errors
/// Returns [`CodecError`] if `level` is beyond the subchunk grid hierarchy or the grid at `level`
/// cannot be resolved in the local context of the partial decoder.
pub(crate) fn subchunk_grid_at_level(
    local_subchunk_grids: &[Option<ChunkGrid>],
    level: usize,
) -> Result<&ChunkGrid, CodecError> {
    match local_subchunk_grids.get(level) {
        Some(Some(chunk_grid)) => Ok(chunk_grid),
        Some(None) => Err(CodecError::Other(format!(
            "the subchunk grid at level {level} is not resolvable for this chunk"
        ))),
        None => Err(CodecError::Other(format!(
            "there is no subchunk grid at level {level}"
        ))),
    }
}

/// Return the indices of the chunk of `chunk_grid` which encloses `subset`.
///
/// # Errors
/// Returns [`CodecError`] if `subset` is out of bounds of `chunk_grid` or spans multiple chunks.
pub(crate) fn enclosing_chunk_indices(
    chunk_grid: &ChunkGrid,
    subset: &ArraySubset,
) -> Result<ArrayIndices, CodecError> {
    let chunks = chunk_grid.chunks_in_array_subset(subset)?.ok_or_else(|| {
        CodecError::Other(format!(
            "subchunk subset {subset:?} is out of bounds of a subchunk grid with shape {:?}",
            chunk_grid.grid_shape()
        ))
    })?;
    if chunks.num_elements() != 1 {
        return Err(CodecError::Other(format!(
            "subchunk subset {subset:?} spans multiple subchunks of an inner subchunk grid"
        )));
    }
    Ok(chunks.start().to_vec())
}

/// One step of a descent from a chunk into the level zero subchunk holding a nested subchunk.
pub(crate) struct SubchunkDescent {
    /// The level zero subchunk indices of the subchunk holding the target subchunk.
    pub subchunk_indices: ArrayIndices,
    /// The shape of the level zero subchunk at [`subchunk_indices`](Self::subchunk_indices).
    pub subchunk_shape: ChunkShape,
    /// The target subchunk subset, relative to the origin of that level zero subchunk.
    pub subset: ArraySubset,
}

/// Plan a descent into the level zero subchunk holding the subchunk at `subchunk_indices` of `level`.
///
/// `local_subchunk_grids` are the grids of the partial decoder being descended from, and `level`
/// must be greater than zero.
///
/// # Errors
/// Returns [`CodecError`] if a required grid level is unavailable or `subchunk_indices` are invalid.
pub(crate) fn plan_subchunk_descent(
    local_subchunk_grids: &[Option<ChunkGrid>],
    level: usize,
    subchunk_indices: &[u64],
) -> Result<SubchunkDescent, CodecError> {
    debug_assert!(level > 0, "level zero does not require a descent");
    let target_grid = subchunk_grid_at_level(local_subchunk_grids, level)?;
    if subchunk_indices.len() != target_grid.dimensionality()
        || std::iter::zip(subchunk_indices, target_grid.grid_shape())
            .any(|(indices, shape)| indices >= shape)
    {
        return Err(CodecError::Other(format!(
            "subchunk indices {subchunk_indices:?} are out of bounds of a subchunk grid with shape {:?}",
            target_grid.grid_shape()
        )));
    }
    let subset = target_grid.subset(subchunk_indices)?.ok_or_else(|| {
        CodecError::Other(format!("invalid subchunk indices {subchunk_indices:?}"))
    })?;

    let outer_grid = subchunk_grid_at_level(local_subchunk_grids, 0)?;
    let outer_indices = enclosing_chunk_indices(outer_grid, &subset)?;
    let outer_subset = outer_grid
        .subset(&outer_indices)?
        .ok_or_else(|| CodecError::Other(format!("invalid subchunk indices {outer_indices:?}")))?;
    let outer_shape = outer_grid
        .chunk_shape(&outer_indices)?
        .ok_or_else(|| CodecError::Other(format!("invalid subchunk indices {outer_indices:?}")))?;

    Ok(SubchunkDescent {
        subchunk_indices: outer_indices,
        subchunk_shape: outer_shape,
        subset: subset.relative_to(outer_subset.start())?,
    })
}

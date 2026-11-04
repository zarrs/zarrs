//! Pure conversion from exact grid subsets to subchunk indices.

use crate::CodecError;
use zarrs_chunk_grid::{ArrayIndices, ArraySubset, ChunkGridTraits};

/// Return the indices of the subchunk of `subchunk_grid` whose subset is exactly `subset`.
///
/// A subchunk addressed by its subset must be exactly a subchunk of the grid of the codec that encodes it for its
/// encoded bytes to be those of the addressed subchunk, so a subset that is only part of a subchunk is an error.
/// The subset must also be nonempty and in bounds of the grid's array shape. If a nominal regular
/// edge chunk overhangs that shape, neither its full subset nor an arbitrary clipping is an exact,
/// in-bounds subchunk. Represent clipped chunks with a grid whose own edge lengths match them.
///
/// # Errors
/// Returns a [`CodecError`] if `subset` is empty, out of bounds of `subchunk_grid` or is not one of its subchunks.
pub fn subset_subchunk_indices(
    subchunk_grid: &dyn ChunkGridTraits,
    subset: &ArraySubset,
) -> Result<ArrayIndices, CodecError> {
    if subset.is_empty() || !subset.inbounds_shape(subchunk_grid.array_shape()) {
        return Err(CodecError::Other(format!(
            "subchunk subset {subset:?} is empty or out of bounds of the subchunk grid"
        )));
    }
    match subchunk_grid.chunk_indices(subset.start())? {
        Some(indices) if subchunk_grid.subset(&indices)?.as_ref() == Some(subset) => Ok(indices),
        _ => Err(CodecError::Other(format!(
            "subchunk subset {subset:?} is not a subchunk of the subchunk grid"
        ))),
    }
}

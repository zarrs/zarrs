//! Addressing encoded subchunks by indices or by subset.
//!
//! Partial codecs address an encoded subchunk by its chunk-local subset as they descend through a codec chain (see
//! [`ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_by_subset`]). A subset is mapped through each
//! codec without resolving the subchunk grid hierarchy of the codecs below it. These functions convert between the
//! indices and the subset of a subchunk at the boundaries of that descent.

use zarrs_chunk_grid::{ArrayIndices, ArraySubset, ChunkGridTraits};

#[cfg(feature = "async")]
use crate::AsyncArrayPartialDecoderSubchunkingTraits;
use crate::{ArrayPartialDecoderSubchunkingTraits, CodecError, CodecOptions};

/// The error for a `level` without a resolvable chunk-local subchunk grid.
fn unresolvable_subchunk_grid(level: usize) -> CodecError {
    CodecError::Other(format!(
        "there is no resolvable subchunk grid at level {level}"
    ))
}

/// Return the indices of the subchunk of `subchunk_grid` whose subset is exactly `subset`.
///
/// A subchunk addressed by its subset must be exactly a subchunk of the grid of the codec that encodes it for its
/// encoded bytes to be those of the addressed subchunk, so a subset that is only part of a subchunk is an error.
///
/// # Errors
/// Returns a [`CodecError`] if `subset` is out of bounds of `subchunk_grid` or is not one of its subchunks.
pub fn subset_subchunk_indices(
    subchunk_grid: &dyn ChunkGridTraits,
    subset: &ArraySubset,
) -> Result<ArrayIndices, CodecError> {
    let indices = subchunk_grid
        .chunk_indices(subset.start())?
        .ok_or_else(|| {
            CodecError::Other(format!(
                "subchunk subset {subset:?} is out of bounds of a subchunk grid with shape {:?}",
                subchunk_grid.grid_shape()
            ))
        })?;
    let subchunk_subset = subchunk_grid.subset(&indices)?;
    if subchunk_subset.as_ref() != Some(subset) {
        return Err(CodecError::Other(format!(
            "subchunk subset {subset:?} is not a subchunk of the subchunk grid, the enclosing subchunk is {subchunk_subset:?}"
        )));
    }
    Ok(indices)
}

/// Return the chunk-local subset of the subchunk at `subchunk_indices` of the `level` subchunk grid of `partial`.
///
/// # Errors
/// Returns a [`CodecError`] if there is no resolvable subchunk grid at `level`, or `subchunk_indices` are out of
/// bounds.
pub fn local_subchunk_subset(
    partial: &(impl ArrayPartialDecoderSubchunkingTraits + ?Sized),
    level: usize,
    subchunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<ArraySubset, CodecError> {
    let subchunk_grid = partial
        .local_subchunk_grid_at_level(level, options)?
        .ok_or_else(|| unresolvable_subchunk_grid(level))?;
    subchunk_subset(&**subchunk_grid, subchunk_indices)
}

/// Return the indices of the subchunk at the chunk-local `subset` of the `level` subchunk grid of `partial`.
///
/// # Errors
/// Returns a [`CodecError`] if there is no resolvable subchunk grid at `level`, or `subset` is not one of its
/// subchunks.
pub fn local_subchunk_indices(
    partial: &(impl ArrayPartialDecoderSubchunkingTraits + ?Sized),
    level: usize,
    subset: &ArraySubset,
    options: &CodecOptions,
) -> Result<ArrayIndices, CodecError> {
    let subchunk_grid = partial
        .local_subchunk_grid_at_level(level, options)?
        .ok_or_else(|| unresolvable_subchunk_grid(level))?;
    subset_subchunk_indices(&**subchunk_grid, subset)
}

/// Return the chunk-local subset of the subchunk at `subchunk_indices` of the `level` subchunk grid of the
/// asynchronous `partial`.
///
/// # Errors
/// See [`local_subchunk_subset`].
#[cfg(feature = "async")]
pub async fn async_local_subchunk_subset(
    partial: &(impl AsyncArrayPartialDecoderSubchunkingTraits + ?Sized),
    level: usize,
    subchunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<ArraySubset, CodecError> {
    let subchunk_grid = partial
        .local_subchunk_grid_at_level(level, options)
        .await?
        .ok_or_else(|| unresolvable_subchunk_grid(level))?;
    subchunk_subset(&**subchunk_grid, subchunk_indices)
}

/// Return the indices of the subchunk at the chunk-local `subset` of the `level` subchunk grid of the asynchronous
/// `partial`.
///
/// # Errors
/// See [`local_subchunk_indices`].
#[cfg(feature = "async")]
pub async fn async_local_subchunk_indices(
    partial: &(impl AsyncArrayPartialDecoderSubchunkingTraits + ?Sized),
    level: usize,
    subset: &ArraySubset,
    options: &CodecOptions,
) -> Result<ArrayIndices, CodecError> {
    let subchunk_grid = partial
        .local_subchunk_grid_at_level(level, options)
        .await?
        .ok_or_else(|| unresolvable_subchunk_grid(level))?;
    subset_subchunk_indices(&**subchunk_grid, subset)
}

/// Return the subset of the subchunk at `subchunk_indices` of `subchunk_grid`.
fn subchunk_subset(
    subchunk_grid: &dyn ChunkGridTraits,
    subchunk_indices: &[u64],
) -> Result<ArraySubset, CodecError> {
    subchunk_grid.subset(subchunk_indices)?.ok_or_else(|| {
        CodecError::Other(format!(
            "subchunk indices {subchunk_indices:?} are out of bounds of a subchunk grid with shape {:?}",
            subchunk_grid.grid_shape()
        ))
    })
}

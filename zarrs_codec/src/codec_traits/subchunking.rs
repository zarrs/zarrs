//! Shared logic for the subchunking traits of partial decoders.

use std::sync::Arc;

use zarrs_chunk_grid::{ArrayIndices, ArraySubset, ChunkGrid, ChunkShape};
use zarrs_data_type::DataType;

use crate::{ArrayBytes, ArrayBytesRaw, ArrayToBytesCodecTraits, CodecError, CodecOptions};

/// The encoded bytes of a subchunk, with the codec and shape needed to decode them.
///
/// The bytes are in the *encoded domain* of the codec that created the subchunk, which is not
/// necessarily the domain of the array they were retrieved from. If the array has array-to-array
/// codecs, then [`decode`](Self::decode) yields array bytes that those codecs have yet to decode:
///
/// - the values are in [`data_type`](Self::data_type), which may differ from the data type of the
///   array (e.g. with a `cast` codec), and
/// - the elements are ordered by [`shape`](Self::shape), which may differ in extent and even in
///   dimensionality from the shape of the subchunk in the subchunk grid of the array (e.g. with a
///   `transpose`, `reshape`, or `squeeze` codec).
///
/// Use [`retrieve_subchunk`](https://docs.rs/zarrs/latest/zarrs/array/trait.ArrayReadOps.html#method.retrieve_subchunk)
/// instead to read a subchunk decoded into the domain of the array.
#[derive(Clone, Debug)]
pub struct EncodedSubchunk<'a> {
    bytes: ArrayBytesRaw<'a>,
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    shape: ChunkShape,
}

impl<'a> EncodedSubchunk<'a> {
    /// Create a new [`EncodedSubchunk`].
    ///
    /// The `shape` must be the shape of the subchunk in the encoded domain of `codec`.
    #[must_use]
    pub fn new(
        bytes: ArrayBytesRaw<'a>,
        codec: Arc<dyn ArrayToBytesCodecTraits>,
        shape: ChunkShape,
    ) -> Self {
        Self {
            bytes,
            codec,
            shape,
        }
    }

    /// Return the encoded bytes of the subchunk.
    ///
    /// These bytes are suitable for passthrough without decoding, such as copying a subchunk into
    /// another array with matching subchunk codecs.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the [`EncodedSubchunk`] and return the encoded bytes of the subchunk.
    #[must_use]
    pub fn into_bytes(self) -> ArrayBytesRaw<'a> {
        self.bytes
    }

    /// Return the shape of the subchunk in the encoded domain of [`codec`](Self::codec).
    ///
    /// This may differ in extent and dimensionality from the shape of the subchunk in the subchunk
    /// grid it was addressed by.
    #[must_use]
    pub fn shape(&self) -> &[std::num::NonZeroU64] {
        &self.shape
    }

    /// Return the data type of the bytes decoded by [`decode`](Self::decode).
    ///
    /// This may differ from the data type of the array the subchunk was retrieved from.
    #[must_use]
    pub fn data_type(&self) -> &DataType {
        self.codec.data_type()
    }

    /// Return the codec that decodes the [`bytes`](Self::bytes).
    #[must_use]
    pub fn codec(&self) -> &Arc<dyn ArrayToBytesCodecTraits> {
        &self.codec
    }

    /// Decode the subchunk into encoded domain array bytes.
    ///
    /// The decoded values are in [`data_type`](Self::data_type) with elements ordered by
    /// [`shape`](Self::shape). Any array-to-array codecs of the array are **not** applied.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the codec fails.
    pub fn decode(&self, options: &CodecOptions) -> Result<ArrayBytes<'_>, CodecError> {
        self.codec.decode(self.bytes.clone(), &self.shape, options)
    }

    /// Convert into an [`EncodedSubchunk`] with a `'static` lifetime, cloning the bytes if borrowed.
    #[must_use]
    pub fn into_owned(self) -> EncodedSubchunk<'static> {
        EncodedSubchunk {
            bytes: ArrayBytesRaw::Owned(self.bytes.into_owned()),
            codec: self.codec,
            shape: self.shape,
        }
    }
}

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

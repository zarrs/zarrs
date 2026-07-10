//! Grid mapping for the `reshape` codec.
//!
//! A reshape does not move elements. It keeps the row-major linear element order and only changes
//! how that linear sequence is split into dimensions. Chunk and subchunk boundaries can therefore
//! be mapped through a reshape when they describe compatible pieces of that same linear sequence.
//!
//! ## Mapping a decoded chunk grid to an encoded chunk grid
//!
//! [`ReshapeCodecBound::grid_mapping`] chooses one of three useful mappings:
//!
//! 1. **Identity.** If every output dimension is the same single input dimension, the grid is
//!    unchanged.
//! 2. **Input-dimension partitions.** A shape such as `[[0], [1, 2]]` keeps dimension 0 and flattens
//!    dimensions 1 and 2. [`input_dim_partitions`] recognises this form when every decoded
//!    dimension appears exactly once. [`encoded_chunk_grid_for_input_dim_partitions`] combines the
//!    chunk edge lengths of each group using their Cartesian product. This also works for varying
//!    rectilinear chunk sizes.
//! 3. **Repeated per-chunk reshape.** Other shapes, including those with fixed sizes or an automatic
//!    dimension, are resolved separately for each decoded chunk. This is only possible globally
//!    when every decoded chunk has the same shape. [`repeated_chunk_grid`] reshapes one chunk and
//!    uses [`RepeatChunkGrid`] to repeat that encoded tile for all decoded chunks. The tiles keep
//!    their row-major chunk order and are repeated along the last encoded dimension.
//!
//! ## Mapping encoded subchunks back to decoded subchunks
//!
//! A downstream codec supplies an encoded subchunk grid. The reshape codec maps it back in two
//! steps:
//!
//! - [`split_edge_refinements`] separates the downstream edges belonging to each encoded chunk.
//!   A downstream edge may not cross an encoded chunk boundary.
//! - The edges inside one chunk are treated as contiguous row-major intervals.
//!   [`reshape_rectilinear_grid`] converts those intervals from the encoded shape to the decoded
//!   shape. The result is accepted only if the intervals form a rectilinear grid in both shapes.
//!
//! The partition mapping performs this operation independently for each input-dimension group.
//! The repeated mapping resolves one local decoded grid and repeats it with [`RepeatChunkGrid`].
//!
//! ## Limitations
//!
//! A global encoded or decoded subchunk grid is not available in these cases:
//!
//! - A reshape that is not an input-dimension partition is used with decoded chunks of different
//!   shapes. Such a reshape may still be resolved locally for an individual chunk.
//! - A downstream subchunk crosses an encoded chunk boundary.
//! - A subchunk's linear span is not an axis-aligned rectangular box after reshaping. Many ordinary
//!   n-dimensional boxes are not contiguous in row-major order and therefore cannot be represented
//!   by this mapping.
//! - Repeated occurrences of the same outer chunk shape use different local refinements. A single
//!   global rectilinear grid cannot describe that variation.
//! - The reshape is invalid for the decoded chunk shape, dimensionalities disagree, a shape is
//!   empty, or shape/count arithmetic overflows.
//!
//! Returning no global grid does not disable reshape partial reads or writes. It only means that a
//! useful global subchunk grid cannot be advertised. A partial decoder may still expose a local
//! subchunk grid for a particular chunk.

use std::num::NonZeroU64;

use zarrs_chunk_grid::{ChunkGrid, ChunkGridCreateError};
use zarrs_codec::CodecError;
use zarrs_metadata::ChunkShape;
use zarrs_metadata_ext::chunk_grid::rectilinear::ChunkEdgeLengths;
use zarrs_metadata_ext::codec::reshape::{ReshapeDim, ReshapeShape};

use crate::array::chunk_grid::repeat::{RepeatChunkGrid, RepeatChunkGridCreateError};
use crate::array::chunk_grid::{RectilinearChunkGrid, RegularChunkGrid};

pub(super) enum GridMapping {
    Identity,
    InputDimPartitions(Vec<Vec<usize>>),
    Repeated {
        decoded_chunk_shape: ChunkShape,
        encoded_chunk_shape: ChunkShape,
        repeats: Vec<u64>,
    },
    ChunkLocal,
    InvalidShape(CodecError),
    Unrepresentable,
}

/// Return the chunk shape if every chunk has the same edge lengths.
///
/// Returns `None` if any dimension is empty or has varying edge lengths.
pub(super) fn regular_chunk_shape(
    chunk_grid: &ChunkGrid,
) -> Result<Option<ChunkShape>, ChunkGridCreateError> {
    let mut chunk_shape = Vec::with_capacity(chunk_grid.dimensionality());
    for dim in 0..chunk_grid.dimensionality() {
        let edge_lengths = chunk_grid
            .chunk_edge_lengths(dim)
            .map_err(ChunkGridCreateError::from)?;
        let Some(first) = edge_lengths.first().copied() else {
            return Ok(None);
        };
        if edge_lengths.iter().any(|&edge_length| edge_length != first) {
            return Ok(None);
        }
        chunk_shape.push(first);
    }
    Ok(Some(chunk_shape))
}

pub(super) fn input_dim_partitions(
    reshape_shape: &ReshapeShape,
    decoded_dimensionality: usize,
) -> Option<Vec<Vec<usize>>> {
    let mut seen = vec![false; decoded_dimensionality];
    let mut partitions = Vec::with_capacity(reshape_shape.0.len());
    for reshape_dim in &reshape_shape.0 {
        let ReshapeDim::InputDims(input_dims) = reshape_dim else {
            return None;
        };
        if input_dims.is_empty() {
            return None;
        }
        let input_dims = input_dims
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        for &input_dim in &input_dims {
            let seen = seen.get_mut(input_dim)?;
            if *seen {
                return None;
            }
            *seen = true;
        }
        partitions.push(input_dims);
    }
    seen.into_iter().all(|seen| seen).then_some(partitions)
}

pub(super) fn input_dim_partitions_are_identity(partitions: &[Vec<usize>]) -> bool {
    partitions
        .iter()
        .enumerate()
        .all(|(dim, partition)| partition.as_slice() == [dim])
}

fn cartesian_product_edge_lengths(edge_lengths: &[Vec<NonZeroU64>]) -> Option<Vec<NonZeroU64>> {
    let mut products = vec![NonZeroU64::new(1).unwrap()];
    for edges in edge_lengths {
        let capacity = products.len().checked_mul(edges.len())?;
        let mut next = Vec::with_capacity(capacity);
        for product in products {
            for edge in edges {
                next.push(NonZeroU64::new(product.get().checked_mul(edge.get())?)?);
            }
        }
        products = next;
    }
    Some(products)
}

fn rectilinear_chunk_grid(
    array_shape: Vec<u64>,
    edge_lengths: &[Vec<NonZeroU64>],
) -> Result<ChunkGrid, ChunkGridCreateError> {
    let chunk_shapes = edge_lengths
        .iter()
        .map(|edges| ChunkEdgeLengths::encode(edges))
        .collect::<Vec<_>>();
    Ok(ChunkGrid::new(RectilinearChunkGrid::new(
        array_shape,
        &chunk_shapes,
    )?))
}

pub(super) fn encoded_chunk_grid_for_input_dim_partitions(
    partitions: &[Vec<usize>],
    decoded_chunk_grid: &ChunkGrid,
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    let mut array_shape = Vec::with_capacity(partitions.len());
    let mut edge_lengths = Vec::with_capacity(partitions.len());
    for partition in partitions {
        let partition_edge_lengths = partition
            .iter()
            .map(|&dim| decoded_chunk_grid.chunk_edge_lengths(dim))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(encoded_edge_lengths) = cartesian_product_edge_lengths(&partition_edge_lengths)
        else {
            return Ok(None);
        };
        let Some(encoded_dim_shape) = encoded_edge_lengths
            .iter()
            .try_fold(0u64, |sum, edge| sum.checked_add(edge.get()))
        else {
            return Ok(None);
        };
        array_shape.push(encoded_dim_shape);
        edge_lengths.push(encoded_edge_lengths);
    }

    rectilinear_chunk_grid(array_shape, &edge_lengths).map(Some)
}

pub(super) fn repeat_shape(grid_shape: &[u64], dimensionality: usize) -> Option<Vec<u64>> {
    if dimensionality == 0 {
        return None;
    }
    let num_chunks = grid_shape
        .iter()
        .try_fold(1u64, |product, count| product.checked_mul(*count))?;
    let mut repeats = vec![1; dimensionality];
    *repeats.last_mut().unwrap() = num_chunks;
    Some(repeats)
}

pub(super) fn repeated_chunk_grid(
    chunk_shape: &[NonZeroU64],
    repeats: &[u64],
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    let inner_chunk_grid = ChunkGrid::new(RegularChunkGrid::new(
        chunk_shape.iter().map(|dim| dim.get()).collect(),
        chunk_shape.to_vec(),
    )?);
    repeat_chunk_grid(repeats, inner_chunk_grid)
}

fn repeat_chunk_grid(
    repeats: &[u64],
    inner_chunk_grid: ChunkGrid,
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    match RepeatChunkGrid::new(repeats.to_vec(), inner_chunk_grid) {
        Ok(chunk_grid) => Ok(Some(ChunkGrid::new(chunk_grid))),
        Err(
            RepeatChunkGridCreateError::ZeroRepeat(_) | RepeatChunkGridCreateError::ShapeOverflow,
        ) => Ok(None),
        Err(err) => Err(err.into()),
    }
}

fn split_edge_refinements(
    outer_edges: &[NonZeroU64],
    refined_edges: &[NonZeroU64],
) -> Option<Vec<Vec<NonZeroU64>>> {
    let mut refined = refined_edges.iter().copied();
    let mut groups = Vec::with_capacity(outer_edges.len());
    for outer_edge in outer_edges {
        let mut group = Vec::new();
        let mut sum = 0u64;
        while sum < outer_edge.get() {
            let edge = refined.next()?;
            sum = sum.checked_add(edge.get())?;
            if sum > outer_edge.get() {
                return None;
            }
            group.push(edge);
        }
        groups.push(group);
    }
    refined.next().is_none().then_some(groups)
}

fn unravel_flat_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for (index_out, size) in indices.iter_mut().zip(shape).rev() {
        *index_out = index % size;
        index /= size;
    }
    indices
}

/// Map a sequence of contiguous row-major intervals to a rectilinear grid.
fn rectilinear_edge_lengths_from_intervals(
    shape: &[NonZeroU64],
    intervals: &[NonZeroU64],
) -> Option<Vec<Vec<NonZeroU64>>> {
    let num_elements = shape
        .iter()
        .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))?;
    if intervals
        .iter()
        .try_fold(0u64, |sum, edge| sum.checked_add(edge.get()))?
        != num_elements
    {
        return None;
    }

    for pivot in 0..shape.len() {
        let trailing = shape[pivot + 1..]
            .iter()
            .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))?;
        let outer_repeats = shape[..pivot]
            .iter()
            .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))?;
        let outer_repeats = usize::try_from(outer_repeats).ok()?;

        let mut pivot_edges = Vec::new();
        let mut pivot_sum = 0u64;
        for interval in intervals {
            if !interval.get().is_multiple_of(trailing) {
                break;
            }
            let edge = interval.get() / trailing;
            pivot_sum = pivot_sum.checked_add(edge)?;
            if pivot_sum > shape[pivot].get() {
                break;
            }
            pivot_edges.push(NonZeroU64::new(edge)?);
            if pivot_sum == shape[pivot].get() {
                break;
            }
        }
        if pivot_sum != shape[pivot].get() {
            continue;
        }
        let expected_len = pivot_edges.len().checked_mul(outer_repeats)?;
        if intervals.len() != expected_len {
            continue;
        }
        let expected_intervals = (0..outer_repeats)
            .flat_map(|_| pivot_edges.iter())
            .map(|edge| NonZeroU64::new(edge.get().checked_mul(trailing)?))
            .collect::<Option<Vec<_>>>()?;
        if intervals != expected_intervals {
            continue;
        }

        let mut edge_lengths = Vec::with_capacity(shape.len());
        for dim in &shape[..pivot] {
            edge_lengths.push(vec![
                NonZeroU64::new(1).unwrap();
                usize::try_from(dim.get()).ok()?
            ]);
        }
        edge_lengths.push(pivot_edges);
        for dim in &shape[pivot + 1..] {
            edge_lengths.push(vec![*dim]);
        }
        return Some(edge_lengths);
    }
    None
}

fn rectilinear_grid_to_intervals(
    shape: &[NonZeroU64],
    grid: &ChunkGrid,
) -> Result<Option<Vec<NonZeroU64>>, ChunkGridCreateError> {
    if grid.dimensionality() != shape.len()
        || !grid
            .array_shape()
            .iter()
            .zip(shape)
            .all(|(grid, shape)| *grid == shape.get())
    {
        return Ok(None);
    }
    let edge_lengths = (0..grid.dimensionality())
        .map(|dim| grid.chunk_edge_lengths(dim))
        .collect::<Result<Vec<_>, _>>()?;
    for pivot in 0..shape.len() {
        let leading_is_unit = edge_lengths[..pivot]
            .iter()
            .all(|edges| edges.iter().all(|edge| edge.get() == 1));
        let trailing_is_full = edge_lengths[pivot + 1..]
            .iter()
            .zip(&shape[pivot + 1..])
            .all(|(edges, shape)| edges.as_slice() == [*shape]);
        if leading_is_unit && trailing_is_full {
            return Ok(cartesian_product_edge_lengths(&edge_lengths));
        }
    }
    Ok(None)
}

fn rectilinear_grid_from_intervals(
    shape: &[NonZeroU64],
    intervals: &[NonZeroU64],
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    let Some(edge_lengths) = rectilinear_edge_lengths_from_intervals(shape, intervals) else {
        return Ok(None);
    };
    rectilinear_chunk_grid(shape.iter().map(|dim| dim.get()).collect(), &edge_lengths).map(Some)
}

pub(super) fn reshape_rectilinear_grid(
    source_shape: &[NonZeroU64],
    target_shape: &[NonZeroU64],
    source_grid: &ChunkGrid,
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    if source_grid.dimensionality() != source_shape.len() {
        return Err(ChunkGridCreateError::new(format!(
            "source grid dimensionality {} is incompatible with reshape source dimensionality {}",
            source_grid.dimensionality(),
            source_shape.len()
        )));
    }
    let Some(intervals) = rectilinear_grid_to_intervals(source_shape, source_grid)? else {
        return Ok(None);
    };
    rectilinear_grid_from_intervals(target_shape, &intervals)
}

pub(super) fn decoded_subchunk_grid_for_input_dim_partitions(
    partitions: &[Vec<usize>],
    decoded_chunk_grid: &ChunkGrid,
    encoded_subchunk_grid: &ChunkGrid,
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    if encoded_subchunk_grid.dimensionality() != partitions.len() {
        return Err(ChunkGridCreateError::new(format!(
            "encoded subchunk grid dimensionality {} is incompatible with encoded dimensionality {}",
            encoded_subchunk_grid.dimensionality(),
            partitions.len()
        )));
    }

    let decoded_edge_lengths = (0..decoded_chunk_grid.dimensionality())
        .map(|dim| decoded_chunk_grid.chunk_edge_lengths(dim))
        .collect::<Result<Vec<_>, _>>()?;
    let mut local_edges = decoded_edge_lengths
        .iter()
        .map(|edges| vec![None; edges.len()])
        .collect::<Vec<Vec<Option<Vec<NonZeroU64>>>>>();

    for (encoded_dim, partition) in partitions.iter().enumerate() {
        let partition_edges = partition
            .iter()
            .map(|&dim| decoded_edge_lengths[dim].clone())
            .collect::<Vec<_>>();
        let Some(encoded_outer_edges) = cartesian_product_edge_lengths(&partition_edges) else {
            return Ok(None);
        };
        let refined_edges = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
        let Some(refined_by_outer_chunk) =
            split_edge_refinements(&encoded_outer_edges, &refined_edges)
        else {
            return Ok(None);
        };
        let partition_grid_shape = partition_edges.iter().map(Vec::len).collect::<Vec<_>>();

        for (flat_index, intervals) in refined_by_outer_chunk.iter().enumerate() {
            let indices = unravel_flat_index(flat_index, &partition_grid_shape);
            let decoded_local_shape = partition
                .iter()
                .zip(&indices)
                .map(|(&dim, &index)| decoded_edge_lengths[dim][index])
                .collect::<Vec<_>>();
            let Some(decoded_local_edges) =
                rectilinear_edge_lengths_from_intervals(&decoded_local_shape, intervals)
            else {
                return Ok(None);
            };

            for ((&decoded_dim, &outer_index), edges) in
                partition.iter().zip(&indices).zip(decoded_local_edges)
            {
                let candidate = &mut local_edges[decoded_dim][outer_index];
                if candidate
                    .as_ref()
                    .is_some_and(|candidate| candidate != &edges)
                {
                    return Ok(None);
                }
                *candidate = Some(edges);
            }
        }
    }

    let edge_lengths = local_edges
        .into_iter()
        .map(|per_outer_chunk| {
            per_outer_chunk
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(Some(rectilinear_chunk_grid(
        decoded_chunk_grid.array_shape().to_vec(),
        &edge_lengths,
    )?))
}

pub(super) fn decoded_subchunk_grid_for_repeated_chunks(
    decoded_chunk_grid: &ChunkGrid,
    encoded_subchunk_grid: &ChunkGrid,
    decoded_chunk_shape: &[NonZeroU64],
    encoded_chunk_shape: &[NonZeroU64],
    repeats: &[u64],
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    if encoded_subchunk_grid.dimensionality() != encoded_chunk_shape.len() {
        return Err(ChunkGridCreateError::new(format!(
            "encoded subchunk grid dimensionality {} is incompatible with encoded dimensionality {}",
            encoded_subchunk_grid.dimensionality(),
            encoded_chunk_shape.len()
        )));
    }

    let mut local_encoded_edges = Vec::with_capacity(encoded_chunk_shape.len());
    for (dim, (chunk_edge, repeat)) in encoded_chunk_shape.iter().zip(repeats).enumerate() {
        let outer_edges = vec![
            *chunk_edge;
            usize::try_from(*repeat)
                .map_err(|err| ChunkGridCreateError::new(err.to_string()))?
        ];
        let refined_edges = encoded_subchunk_grid.chunk_edge_lengths(dim)?;
        let Some(refined_tiles) = split_edge_refinements(&outer_edges, &refined_edges) else {
            return Ok(None);
        };
        let Some(first) = refined_tiles.first() else {
            return Ok(None);
        };
        if refined_tiles.iter().any(|tile| tile != first) {
            return Ok(None);
        }
        local_encoded_edges.push(first.clone());
    }
    let local_encoded_grid = rectilinear_chunk_grid(
        encoded_chunk_shape.iter().map(|dim| dim.get()).collect(),
        &local_encoded_edges,
    )?;
    let Some(local_decoded_grid) = reshape_rectilinear_grid(
        encoded_chunk_shape,
        decoded_chunk_shape,
        &local_encoded_grid,
    )?
    else {
        return Ok(None);
    };
    repeat_chunk_grid(decoded_chunk_grid.grid_shape(), local_decoded_grid)
}

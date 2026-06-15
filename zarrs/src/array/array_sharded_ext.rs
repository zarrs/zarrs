use std::num::NonZeroU64;

use super::chunk_grid::{RectilinearChunkGrid, RegularBoundedChunkGrid, RegularChunkGrid};
use super::{ArrayError, ArrayOps, ArrayShape, ArraySubset, ChunkGrid, ChunkShape, CodecChain};
use crate::array::chunk_grid::ChunkEdgeLengths;
use crate::array::chunk_grid::repeat::RepeatChunkGrid;
use crate::array::codec::array_to_bytes::sharding::ShardingCodec;
use zarrs_codec::UnboundArrayToBytesCodecTraits;
use zarrs_metadata_ext::chunk_grid::rectilinear::RunLengthElement;
use zarrs_plugin::ExtensionAliasesV3;

/// Iterate over subchunk sizes for a parent chunk along a single dimension.
fn subchunk_sizes(chunk_size: u64, subchunk_size: NonZeroU64) -> impl Iterator<Item = NonZeroU64> {
    let subchunk_size = subchunk_size.get();
    (0..chunk_size.div_ceil(subchunk_size)).map(move |i| {
        let start = i * subchunk_size;
        let remaining = chunk_size - start;
        NonZeroU64::new(remaining.min(subchunk_size)).expect("size is non-zero")
    })
}

/// Compute the subchunk grid shape and edge lengths for a single dimension.
///
/// Returns `None` if the dimension has zero grid shape and the decoded chunk
/// shape is not evenly divisible by the subchunk size (caller should return
/// the original chunk grid unchanged).
fn compute_dimension_subchunk_info(
    dim: usize,
    chunk_edge_lengths: &[NonZeroU64],
    decoded_chunk_shape: &ChunkShape,
    subchunk_shape: &ChunkShape,
) -> Option<(u64, ChunkEdgeLengths)> {
    let subchunk_size = subchunk_shape[dim];

    if chunk_edge_lengths.is_empty() {
        if decoded_chunk_shape[dim]
            .get()
            .is_multiple_of(subchunk_size.get())
        {
            return Some((0, ChunkEdgeLengths::Scalar(subchunk_shape[dim])));
        }
        return None;
    }

    let mut dimension_shape = 0;
    let mut sizes: Option<Vec<RunLengthElement>> = None;
    let mut regular_subchunk_count = 0;

    for chunk_size in chunk_edge_lengths.iter().map(|chunk_size| chunk_size.get()) {
        dimension_shape += chunk_size;

        if chunk_size % subchunk_size.get() == 0 {
            let count = chunk_size / subchunk_size.get();
            if let Some(sizes) = &mut sizes {
                sizes.push(RunLengthElement::Repeated([
                    subchunk_size,
                    NonZeroU64::new(count).expect("chunk size is non-zero"),
                ]));
            } else {
                regular_subchunk_count += count;
            }
        } else {
            let sizes = sizes.get_or_insert_with(|| {
                if let Some(count) = NonZeroU64::new(regular_subchunk_count) {
                    vec![RunLengthElement::Repeated([subchunk_size, count])]
                } else {
                    Vec::new()
                }
            });
            sizes.extend(subchunk_sizes(chunk_size, subchunk_size).map(RunLengthElement::Single));
        }
    }

    let edge_lengths = if let Some(sizes) = sizes {
        ChunkEdgeLengths::Varying(sizes)
    } else {
        ChunkEdgeLengths::Scalar(subchunk_size)
    };

    Some((dimension_shape, edge_lengths))
}

fn repeated_subchunk_grid_for_regular_shards(
    chunk_grid: &ChunkGrid,
    decoded_chunk_shape: &ChunkShape,
    subchunk_shape: &ChunkShape,
) -> Option<ChunkGrid> {
    if !chunk_grid
        .name_v3()
        .is_some_and(|name| RegularChunkGrid::matches_name_v3(name.as_ref()))
    {
        return None;
    }

    let tile_shape: ArrayShape = decoded_chunk_shape
        .iter()
        .map(|edge_length| edge_length.get())
        .collect();
    let repeats = chunk_grid.grid_shape().to_vec();
    let inner_chunk_grid = if std::iter::zip(&tile_shape, subchunk_shape)
        .all(|(tile_size, subchunk_size)| tile_size % subchunk_size.get() == 0)
    {
        RegularChunkGrid::new(tile_shape.clone(), subchunk_shape.clone())
            .ok()?
            .into()
    } else {
        RegularBoundedChunkGrid::new(tile_shape, subchunk_shape.clone())
            .ok()?
            .into()
    };

    RepeatChunkGrid::new(repeats, inner_chunk_grid)
        .ok()
        .map(ChunkGrid::new)
}

pub(crate) fn create_subchunk_grid(
    chunk_grid: &ChunkGrid,
    codecs: &CodecChain,
) -> Option<ChunkGrid> {
    if !codecs.array_to_bytes_codec().as_any().is::<ShardingCodec>() {
        return None;
    }

    let dimensionality = chunk_grid.dimensionality();
    let origin_chunk = vec![0; dimensionality];
    let decoded_chunk_shape = chunk_grid.chunk_shape(&origin_chunk).ok().flatten()?;
    let subchunk_shape = codecs
        .partial_decode_granularity(&decoded_chunk_shape)
        .ok()?;
    if subchunk_shape == decoded_chunk_shape {
        return Some(chunk_grid.clone());
    }

    if let Some(subchunk_grid) =
        repeated_subchunk_grid_for_regular_shards(chunk_grid, &decoded_chunk_shape, &subchunk_shape)
    {
        return Some(subchunk_grid);
    }

    let mut needs_rectilinear = false;
    let mut subchunk_grid_shape = Vec::with_capacity(dimensionality);
    let mut subchunk_edge_lengths = Vec::with_capacity(dimensionality);

    for dim in 0..dimensionality {
        let chunk_edge_lengths = chunk_grid.chunk_edge_lengths(dim).ok()?;
        let Some((dimension_shape, edge_lengths)) = compute_dimension_subchunk_info(
            dim,
            &chunk_edge_lengths,
            &decoded_chunk_shape,
            &subchunk_shape,
        ) else {
            return Some(chunk_grid.clone());
        };

        if let ChunkEdgeLengths::Varying(_) = &edge_lengths {
            needs_rectilinear = true;
        }
        subchunk_grid_shape.push(dimension_shape);
        subchunk_edge_lengths.push(edge_lengths);
    }

    if needs_rectilinear {
        Some(ChunkGrid::new(
            RectilinearChunkGrid::new(subchunk_grid_shape, &subchunk_edge_lengths).ok()?,
        ))
    } else {
        Some(ChunkGrid::new(
            RegularChunkGrid::new(subchunk_grid_shape, subchunk_shape).ok()?,
        ))
    }
}

struct SubchunkShardLocation {
    shard_indices: Vec<u64>,
    shard_origin: Vec<u64>,
    array_subset: ArraySubset,
}

fn subchunk_shard_location<A: ArrayOps + ?Sized>(
    array: &A,
    subchunk_grid: &ChunkGrid,
    subchunk_indices: &[u64],
) -> Result<SubchunkShardLocation, ArrayError> {
    let array_subset = subchunk_grid
        .subset(subchunk_indices)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    let shards = array
        .chunks_in_array_subset(&array_subset)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    if shards.num_elements() != 1 {
        // This should not happen, but it is checked just in case.
        return Err(ArrayError::InvalidChunkGridIndicesError(
            subchunk_indices.to_vec(),
        ));
    }
    let shard_indices = shards.start();
    let shard_origin = array.chunk_origin(shard_indices)?;
    Ok(SubchunkShardLocation {
        shard_indices: shard_indices.to_vec(),
        shard_origin,
        array_subset,
    })
}

pub(super) fn subchunk_shard_index_and_subset<A: ArrayOps + ?Sized>(
    array: &A,
    subchunk_grid: &ChunkGrid,
    subchunk_indices: &[u64],
) -> Result<(Vec<u64>, ArraySubset), ArrayError> {
    let location = subchunk_shard_location(array, subchunk_grid, subchunk_indices)?;
    let shard_subset = location.array_subset.relative_to(&location.shard_origin)?;
    Ok((location.shard_indices, shard_subset))
}

pub(super) fn subchunk_shard_index_and_chunk_index<A: ArrayOps + ?Sized>(
    array: &A,
    subchunk_grid: &ChunkGrid,
    subchunk_indices: &[u64],
) -> Result<(Vec<u64>, Vec<u64>), ArrayError> {
    let location = subchunk_shard_location(array, subchunk_grid, subchunk_indices)?;
    let first_subchunk_indices = subchunk_grid
        .chunk_indices(&location.shard_origin)
        .map_err(|_| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    let chunk_indices: Vec<u64> = subchunk_indices
        .iter()
        .zip(first_subchunk_indices)
        .map(|(subchunk_index, first_subchunk_index)| {
            subchunk_index.checked_sub(first_subchunk_index)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    Ok((location.shard_indices, chunk_indices))
}
#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::chunk_grid::{RectangularChunkGrid, RectilinearChunkGrid};
    use crate::array::{Array, ArrayBuilder, data_type};
    use zarrs_metadata_ext::chunk_grid::rectangular::RectangularChunkGridDimensionConfiguration;
    use zarrs_metadata_ext::chunk_grid::rectilinear::{ChunkEdgeLengths, RunLengthElement};
    use zarrs_plugin::ExtensionName;
    use zarrs_storage::store::MemoryStore;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn build_array_with_chunk_grid(
        chunk_grid: impl Into<ChunkGrid>,
        subchunk_shape: Vec<u64>,
    ) -> Result<Array<MemoryStore>, Box<dyn std::error::Error>> {
        let store = Arc::new(MemoryStore::default());
        let mut builder = ArrayBuilder::new_with_chunk_grid(chunk_grid, data_type::uint16(), 0u16);
        builder.subchunk_shape(subchunk_shape);
        Ok(builder.build(store, "/array")?)
    }

    fn assert_subchunk_grid(
        array: &Array<MemoryStore>,
        expected_array_shape: &[u64],
        expected_grid_shape: &[u64],
        expected_edge_lengths: &[NonZeroU64],
    ) -> Result<ChunkGrid, Box<dyn std::error::Error>> {
        let subchunk_grid = array.subchunk_grid().clone();
        assert_eq!(subchunk_grid.array_shape(), expected_array_shape);
        assert_eq!(subchunk_grid.grid_shape(), expected_grid_shape);
        assert_eq!(subchunk_grid.chunk_edge_lengths(0)?, expected_edge_lengths);
        Ok(subchunk_grid)
    }

    #[test]
    fn subchunk_grid_regular_outer_uses_repeat_chunk_grid() -> Result<(), Box<dyn std::error::Error>>
    {
        let store = Arc::new(MemoryStore::default());
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![2, 2]);
        let array = builder.build(store, "/array")?;

        let subchunk_grid = array.subchunk_grid();
        assert_eq!(subchunk_grid.name_v3(), None);
        assert_eq!(subchunk_grid.array_shape(), &[8, 8]);
        assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);
        assert_eq!(
            subchunk_grid.subset(&[2, 3])?,
            Some(ArraySubset::new_with_ranges(&[4..6, 6..8]))
        );
        assert_eq!(
            subchunk_shard_index_and_chunk_index(&array, subchunk_grid, &[2, 3])?,
            (vec![1, 1], vec![0, 1])
        );

        Ok(())
    }

    #[test]
    fn subchunk_grid_regular_outer_covers_full_repeated_shard_extent()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(MemoryStore::default());
        let mut builder = ArrayBuilder::new(vec![7, 7], vec![4, 4], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![2, 2]);
        let array = builder.build(store, "/array")?;

        let subchunk_grid = array.subchunk_grid();
        assert_eq!(array.shape(), &[7, 7]);
        assert_eq!(array.chunk_grid_shape(), &[2, 2]);
        assert_eq!(subchunk_grid.name_v3(), None);
        assert_eq!(subchunk_grid.array_shape(), &[8, 8]);
        assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);
        assert_eq!(
            subchunk_grid.subset(&[3, 3])?,
            Some(ArraySubset::new_with_ranges(&[6..8, 6..8]))
        );

        Ok(())
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn subchunk_grid_regular_outer_uses_bounded_inner_grid()
    -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(MemoryStore::default());
        let mut builder = ArrayBuilder::new(vec![10], vec![5], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![3]);
        let array = builder.build(store, "/array")?;

        let subchunk_grid = array.subchunk_grid();
        assert_eq!(subchunk_grid.name_v3(), None);
        assert_eq!(subchunk_grid.array_shape(), &[10]);
        assert_eq!(subchunk_grid.grid_shape(), &[4]);
        assert_eq!(
            subchunk_grid.chunk_edge_lengths(0)?,
            vec![nz(3), nz(2), nz(3), nz(2)]
        );
        assert_eq!(
            subchunk_grid.subset(&[1])?,
            Some(ArraySubset::new_with_ranges(&[3..5]))
        );
        assert_eq!(
            subchunk_grid.subset(&[3])?,
            Some(ArraySubset::new_with_ranges(&[8..10]))
        );
        assert_eq!(
            subchunk_shard_index_and_chunk_index(&array, subchunk_grid, &[3])?,
            (vec![1], vec![1])
        );

        Ok(())
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn subchunk_grid_from_varying_shard_edges_uses_rectilinear_subchunk_grid()
    -> Result<(), Box<dyn std::error::Error>> {
        let arrays = [
            build_array_with_chunk_grid(
                RectilinearChunkGrid::new(
                    vec![12],
                    &[ChunkEdgeLengths::Varying(vec![
                        RunLengthElement::Single(nz(5)),
                        RunLengthElement::Single(nz(7)),
                    ])],
                )?,
                vec![3],
            )?,
            build_array_with_chunk_grid(
                RectangularChunkGrid::new(
                    vec![12],
                    &[RectangularChunkGridDimensionConfiguration::Varying(vec![
                        nz(5),
                        nz(7),
                    ])],
                )?,
                vec![3],
            )?,
        ];

        for array in arrays {
            let subchunk_grid =
                assert_subchunk_grid(&array, &[12], &[5], &[nz(3), nz(2), nz(3), nz(3), nz(1)])?;
            let subchunk_grid_name = subchunk_grid.name_v3().unwrap();
            assert!(RectilinearChunkGrid::matches_name_v3(
                subchunk_grid_name.as_ref()
            ));
            assert_eq!(
                subchunk_grid.subset(&[1])?,
                Some(ArraySubset::new_with_ranges(&[3..5]))
            );
            assert_eq!(
                subchunk_grid.subset(&[2])?,
                Some(ArraySubset::new_with_ranges(&[5..8]))
            );
            assert_eq!(
                subchunk_shard_index_and_chunk_index(&array, &subchunk_grid, &[2])?,
                (vec![1], vec![0])
            );
            assert_eq!(
                subchunk_shard_index_and_chunk_index(&array, &subchunk_grid, &[4])?,
                (vec![1], vec![2])
            );
        }

        Ok(())
    }
}

#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_grid::{RectangularChunkGrid, RectilinearChunkGrid};
use zarrs::array::codec::array_to_array::transpose::{TransposeCodec, TransposeOrder};
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecBuilder, ShardingIndexLocation};
use zarrs::array::{Array, ArrayBuilder, ArrayCreateError, data_type};
use zarrs_chunk_grid::{ArraySubset, ChunkGrid, ChunkGridCreateError};
use zarrs_metadata_ext::chunk_grid::rectangular::RectangularChunkGridDimensionConfiguration;
use zarrs_metadata_ext::chunk_grid::rectilinear::{ChunkEdgeLengths, RunLengthElement};
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
    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap().clone();
    assert_eq!(subchunk_grid.array_shape(), expected_array_shape);
    assert_eq!(subchunk_grid.grid_shape(), expected_grid_shape);
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(0)?,
        Some(expected_edge_lengths.to_vec())
    );
    Ok(subchunk_grid)
}

#[test]
fn subchunk_grid_regular_outer_uses_repeat_chunk_grid() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
    builder.subchunk_shape(vec![2, 2]);
    let array = builder.build(store, "/array")?;

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    assert_eq!(array.subchunk_shape(), Some(vec![2; 2]));
    assert_eq!(subchunk_grid.name_v3(), None);
    assert_eq!(subchunk_grid.array_shape(), &[8, 8]);
    assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);
    assert_eq!(
        subchunk_grid.subset(&[2, 3])?,
        Some(ArraySubset::new_with_ranges(&[4..6, 6..8]))
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

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    assert_eq!(array.subchunk_shape(), Some(vec![2; 2]));
    assert_eq!(array.shape(), &[7, 7]);
    assert_eq!(array.chunk_grid_shape(), &[2, 2]);
    assert_eq!(subchunk_grid.name_v3(), None);
    assert_eq!(subchunk_grid.array_shape(), &[7, 7]);
    assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);
    assert_eq!(
        subchunk_grid.subset(&[3, 3])?,
        Some(ArraySubset::new_with_ranges(&[6..8, 6..8]))
    );

    Ok(())
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn subchunk_grid_rejects_non_even_sharding_chunk_shape() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![10], vec![5], data_type::uint16(), 0u16);
    builder.subchunk_shape(vec![3]);
    let err = builder.build(store, "/array").unwrap_err();

    assert!(matches!(
        err,
        ArrayCreateError::ChunkGridCreateError(ChunkGridCreateError::Other(ref str)) if str.contains("must evenly divide shard shape")
    ));
    assert!(err.to_string().contains("must evenly divide shard shape"));

    Ok(())
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn subchunk_grid_from_varying_shard_edges_requires_even_division()
-> Result<(), Box<dyn std::error::Error>> {
    let arrays = [
        build_array_with_chunk_grid(
            RectilinearChunkGrid::new(
                vec![15],
                &[ChunkEdgeLengths::Varying(vec![
                    RunLengthElement::Single(nz(6)),
                    RunLengthElement::Single(nz(9)),
                ])],
            )?,
            vec![3],
        )?,
        build_array_with_chunk_grid(
            RectangularChunkGrid::new(
                vec![15],
                &[RectangularChunkGridDimensionConfiguration::Varying(vec![
                    nz(6),
                    nz(9),
                ])],
            )?,
            vec![3],
        )?,
    ];

    for array in arrays {
        let subchunk_grid =
            assert_subchunk_grid(&array, &[15], &[5], &[nz(3), nz(3), nz(3), nz(3), nz(3)])?;
        assert_eq!(array.subchunk_shape(), Some(vec![3]));
        assert_eq!(
            subchunk_grid.subset(&[1])?,
            Some(ArraySubset::new_with_ranges(&[3..6]))
        );
        assert_eq!(
            subchunk_grid.subset(&[2])?,
            Some(ArraySubset::new_with_ranges(&[6..9]))
        );
    }

    Ok(())
}

#[test]
fn subchunk_grid_accounts_for_transpose_before_sharding() -> Result<(), Box<dyn std::error::Error>>
{
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let sharding_codec = ShardingCodecBuilder::new(vec![nz(3), nz(2)], &data_type)
        .index_location(ShardingIndexLocation::End)
        .build();
    let mut builder = ArrayBuilder::new(vec![8, 6], vec![4, 6], data_type, 0u16);
    builder
        .array_to_array_codecs(vec![Arc::new(TransposeCodec::new(TransposeOrder::new(
            &[1, 0],
        )?))])
        .array_to_bytes_codec(Arc::new(sharding_codec));
    let array = builder.build(store, "/array")?;

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    assert_eq!(array.subchunk_shape(), Some(vec![2, 3]));
    assert_eq!(subchunk_grid.array_shape(), &[8, 6]);
    assert_eq!(subchunk_grid.grid_shape(), &[4, 2]);
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(0)?,
        Some(vec![nz(2), nz(2), nz(2), nz(2)])
    );
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(1)?,
        Some(vec![nz(3), nz(3)])
    );

    Ok(())
}

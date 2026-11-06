#![allow(missing_docs)]

use std::num::NonZeroU64;
use zarrs::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid, RegularChunkGrid};
use zarrs::array::{ArrayIndices, ArraySubset, ChunkGrid};

fn shape(s: &[u64]) -> Vec<NonZeroU64> {
    s.iter().map(|&n| NonZeroU64::new(n).unwrap()).collect()
}
fn check(grid: &ChunkGrid) {
    for indices in ArraySubset::new_with_shape(grid.grid_shape()).indices() {
        let subset = grid.subset(&indices).unwrap().unwrap();
        assert_eq!(
            grid.exact_chunk_indices(&subset).unwrap(),
            Some(indices.clone())
        );
    }
    for subset in [
        ArraySubset::new_with_ranges(&[0..1, 0..1]), // partial subchunk
        ArraySubset::new_with_ranges(&[0..4, 0..3]), // multiple subchunks
        ArraySubset::new_empty(2),
        ArraySubset::new_with_ranges(&[100..102, 0..3]),
        ArraySubset::new_with_ranges(&[0..2, 100..103]),
    ] {
        assert_eq!(
            grid.exact_chunk_indices(&subset).unwrap(),
            None,
            "{subset:?}"
        );
    }
    let wrong_dim = ArraySubset::new_with_shape(vec![1]);
    assert!(grid.exact_chunk_indices(&wrong_dim).is_err());
}

#[test]
fn exact_regular_and_rectilinear_subsets() {
    check(&ChunkGrid::new(
        RegularChunkGrid::new(vec![6, 6], shape(&[2, 3])).unwrap(),
    ));
    let clipped = ChunkGrid::new(
        RectilinearChunkGrid::new(
            vec![5, 7],
            &[
                ChunkEdgeLengths::encode(&shape(&[2, 2, 1])),
                ChunkEdgeLengths::encode(&shape(&[3, 3, 1])),
            ],
        )
        .unwrap(),
    );
    check(&clipped);
    assert_eq!(
        clipped
            .exact_chunk_indices(&ArraySubset::new_with_ranges(&[4..5, 6..7]))
            .unwrap(),
        Some(ArrayIndices::from([2, 2].as_slice()))
    );
    let varying = ChunkGrid::new(
        RectilinearChunkGrid::new(
            vec![6, 6],
            &[
                ChunkEdgeLengths::encode(&shape(&[2, 4])),
                ChunkEdgeLengths::encode(&shape(&[3, 2, 1])),
            ],
        )
        .unwrap(),
    );
    check(&varying);
}

#[test]
fn overhanging_regular_chunks_do_not_make_out_of_bounds_subsets_valid() {
    let grid = ChunkGrid::new(RegularChunkGrid::new(vec![5, 7], shape(&[2, 3])).unwrap());
    // `subset` includes the whole nominal edge chunk. Neither an out-of-bounds chunk nor
    // an arbitrary clipping of it satisfies the exact, in-bounds subset contract.
    for subset in [
        ArraySubset::new_with_ranges(&[4..6, 6..9]),
        ArraySubset::new_with_ranges(&[4..5, 6..7]),
    ] {
        assert_eq!(grid.exact_chunk_indices(&subset).unwrap(), None);
    }
}

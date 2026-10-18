//! Partial encodes through array-to-array codecs that do not map partial encodes.
#![allow(missing_docs)]
#![cfg(feature = "transpose")]

mod subchunk_grid_cases;

use std::sync::Arc;

use subchunk_grid_cases::IdentityCodec;
use zarrs::array::codec::{TransposeCodec, TransposeOrder};
use zarrs::array::{
    ArrayBuilder, ArraySubset, Resources, UnboundArrayToArrayCodecTraits, data_type,
};
use zarrs::storage::store::MemoryStore;

/// A partial write through a codec that does not map partial encodes, alone, after, or before a codec that does.
#[test]
fn partial_encode_through_unmapped_codec() {
    let transpose = || -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap()))
    };
    let identity = || -> Arc<dyn UnboundArrayToArrayCodecTraits> { Arc::new(IdentityCodec) };
    for codecs in [
        vec![identity()],
        vec![transpose(), identity()],
        vec![identity(), transpose()],
    ] {
        let array = ArrayBuilder::new(vec![8, 6], vec![8, 6], data_type::uint16(), 0u16)
            .array_to_array_codecs(codecs)
            .build(Arc::new(MemoryStore::new()), "/")
            .unwrap();
        let resources = Resources::default();
        let mut expected: Vec<u16> = (0..48).collect();
        array
            .store_chunk(&[0, 0], expected.as_slice(), &resources)
            .unwrap();

        let subset = ArraySubset::new_with_ranges(&[2..5, 1..5]);
        let update: Vec<u16> = (100..112).collect();
        array
            .store_array_subset(&subset, update.as_slice(), &resources)
            .unwrap();
        for (i, j) in indices_2d(2..5, 1..5) {
            expected[i * 6 + j] = update[(i - 2) * 4 + (j - 1)];
        }
        let retrieved: Vec<u16> = array
            .retrieve_array_subset(&array.subset_all(), &resources)
            .unwrap();
        assert_eq!(retrieved, expected);
    }
}

fn indices_2d(
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) -> impl Iterator<Item = (usize, usize)> {
    rows.flat_map(move |i| columns.clone().map(move |j| (i, j)))
}

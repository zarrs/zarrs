//! Integration tests for the unstructured cartesian chunk grid.

use std::sync::Arc;

use zarrs::array::{ArrayBuilder, data_type};
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::store::MemoryStore;

#[test]
fn unstructured_cartesian_array_round_trip_by_origin() -> Result<(), Box<dyn std::error::Error>> {
    let chunk_grid = MetadataV3::try_from(
        r#"
        {
            "name": "zarrs.unstructured_cartesian",
            "configuration": {
                "kind": "inline",
                "chunks": [
                    { "origin": [0, 0], "shape": [2, 3] },
                    { "origin": [0, 3], "shape": [2, 1] },
                    { "origin": [2, 0], "shape": [1, 1] },
                    { "origin": [2, 1], "shape": [1, 3] }
                ]
            }
        }"#,
    )?;

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![3, 4], chunk_grid, data_type::uint8(), 0u8)
        .build(store, "/array")?;

    array.store_chunk(&[0, 0], &[1u8, 2, 3, 5, 6, 7])?;
    array.store_chunk(&[0, 3], &[4u8, 8])?;
    array.store_chunk(&[2, 0], &[9u8])?;
    array.store_chunk(&[2, 1], &[10u8, 11, 12])?;

    assert_eq!(
        array.retrieve_array_subset::<Vec<u8>>(&[1..3, 2..4])?,
        vec![7, 8, 11, 12]
    );

    array.store_array_subset(&[1..3, 2..4], &[17u8, 18, 21, 22])?;
    assert_eq!(
        array.retrieve_array_subset::<Vec<u8>>(&[0..3, 0..4])?,
        vec![1, 2, 3, 4, 5, 6, 17, 18, 9, 10, 21, 22]
    );

    Ok(())
}

#[test]
fn unstructured_cartesian_sharded_array_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let chunk_grid = MetadataV3::try_from(
        r#"
        {
            "name": "zarrs.unstructured_cartesian",
            "configuration": {
                "kind": "inline",
                "chunks": [
                    { "origin": [0, 0], "shape": [2, 3] },
                    { "origin": [0, 3], "shape": [2, 1] },
                    { "origin": [2, 0], "shape": [1, 1] },
                    { "origin": [2, 1], "shape": [1, 3] }
                ]
            }
        }"#,
    )?;

    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![3, 4], chunk_grid, data_type::uint8(), 0u8);
    builder.subchunk_shape(vec![1, 1]);
    let array = builder.build(store, "/array")?;

    array.store_array_subset(&[0..3, 0..4], &[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])?;

    assert_eq!(
        array.retrieve_array_subset::<Vec<u8>>(&[1..3, 2..4])?,
        vec![7, 8, 11, 12]
    );

    Ok(())
}

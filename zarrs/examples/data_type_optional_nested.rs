//! This example is similar to `data_type_optional.rs` but demonstrates nested optional types,
//! i.e., `Option<Option<T>>`.
//!
//! The fill value is set to `Some(None)`.

use std::sync::Arc;

use ndarray::ArrayD;
use zarrs::array::{ArrayBuilder, FillValue, data_type};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an in-memory store
    // let store = Arc::new(zarrs::filesystem::FilesystemStore::new(
    //     "zarrs/tests/data/v3/array_optional_nested.zarr",
    // )?);
    let store = Arc::new(zarrs::storage::store::MemoryStore::new());

    // Build the codec chains for the optional codec
    let array = ArrayBuilder::new(
        vec![4, 4],                                     // 4x4 array
        vec![2, 2],                                     // 2x2 chunks
        data_type::uint8().to_optional().to_optional(), // Optional optional uint8 => Option<Option<u8>>
        FillValue::new_optional_null().into_optional(), // Fill value => Some(None)
    )
    .dimension_names(["y", "x"].into())
    .attributes(
        serde_json::json!({
            "description": r"A 4x4 array of optional optional uint8 values with some missing data.
The fill value is null on the inner optional layer, i.e. Some(None).
N marks missing (`None`=`null`) values. SN marks `Some(None)`=`[null]` values:
  N  SN   2   3 
  N   5   N   7 
 SN  SN   N   N 
 SN  SN   N   N",
        })
        .as_object()
        .unwrap()
        .clone(),
    )
    .build(store.clone(), "/array")?;
    array.store_metadata_opt(
        &zarrs::array::ArrayMetadataOptions::default().with_include_zarrs_metadata(false),
    )?;

    println!("Array metadata:\n{}", array.metadata().to_string_pretty());

    // Create some data with missing values
    let data = ndarray::array![
        [None, Some(None), Some(Some(2u8)), Some(Some(3u8))],
        [None, Some(Some(5u8)), None, Some(Some(7u8))],
        [Some(None), Some(None), None, None],
        [Some(None), Some(None), None, None],
    ]
    .into_dyn();

    // Write the data
    array.store_array_subset(&array.subset_all(), data.clone())?;
    println!("Data written to array.");

    // Read back the data
    let data_read: ArrayD<Option<Option<u8>>> = array.retrieve_array_subset(&array.subset_all())?;

    // Verify data integrity
    assert_eq!(data, data_read);

    // Display the data in a grid format
    println!(
        "Data grid. N marks missing (`None`=`null`) values. SN marks `Some(None)`=`[null]` values"
    );
    println!("    0   1   2   3");
    for y in 0..4 {
        print!("{y} ");
        for x in 0..4 {
            match data_read[[y, x]] {
                Some(Some(value)) => print!("{value:3} "),
                Some(None) => print!(" SN "),
                None => print!("  N "),
            }
        }
        println!();
    }
    Ok(())
}

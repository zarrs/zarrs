#![allow(missing_docs)]

use std::{error::Error, sync::Arc};

use zarrs_filesystem::FilesystemStore;

#[test]
fn cities_zarr_python_v3_compat() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/v3_bytes.zarr",
    )?);
    let array = zarrs::array::Array::open(store.clone(), "/")?;
    let subset_all = array.subset_all();
    let cities_out = array.retrieve_array_subset_elements::<Vec<u8>>(&subset_all)?;

    assert_eq!(cities_out[0], b"New York");
    assert_eq!(cities_out[1], b"Los Angeles");
    assert_eq!(cities_out[2], b"Chicago");

    Ok(())
}

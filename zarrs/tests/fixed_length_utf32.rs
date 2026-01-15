#![allow(missing_docs)]

use std::error::Error;
use std::sync::Arc;

use zarrs::array::Array;
use zarrs_filesystem::FilesystemStore;
use zarrs_plugin::ZarrVersion;

#[test]
fn fixed_length_utf32_v3_read() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/fixed_length_utf32.zarr",
    )?);
    let array = Array::open(store, "/")?;

    // Verify metadata
    assert_eq!(array.shape(), vec![12, 12]);

    // The data type should be fixed_length_utf32 with 80 bytes (20 chars)
    let data_type = array.data_type();
    assert_eq!(
        data_type.name(ZarrVersion::V3).as_deref(),
        Some("fixed_length_utf32")
    );
    assert_eq!(data_type.fixed_size(), Some(80));

    Ok(())
}

#[test]
fn fixed_length_utf32_v2_read() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/U20_V2.zarr",
    )?);
    let array = Array::open(store, "/")?;

    // Verify metadata
    assert_eq!(array.shape(), vec![12, 12]);

    // The data type should be fixed_length_utf32 with 80 bytes (20 chars)
    let data_type = array.data_type();
    assert_eq!(
        data_type.name(ZarrVersion::V3).as_deref(),
        Some("fixed_length_utf32")
    );
    assert_eq!(data_type.fixed_size(), Some(80));

    Ok(())
}

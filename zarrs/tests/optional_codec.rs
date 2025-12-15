#![allow(missing_docs)]

use std::sync::Arc;

use zarrs::array::{Array, ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::store::MemoryStore;

/// Test helper to create an array with optional codec
fn create_optional_array(data_type: DataType, fill_value: FillValue) -> Array<MemoryStore> {
    let store = Arc::new(MemoryStore::default());
    let array_path = "/optional_array";

    // The ArrayBuilder automatically creates the correct codec configuration
    // for nested optional types (e.g., Option<Option<T>>)
    ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 4], // chunk shape
        data_type,
        fill_value,
    )
    .build(store, array_path)
    .unwrap()
}

#[test]
fn optional_array_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{array, Array2};

    let array = create_optional_array(DataType::UInt8.into_optional(), None::<u8>.into());

    // Store different patterns in different chunks
    // Chunk [0,0]: mostly Some values
    let data0 = array![
        [Some(1), Some(2), Some(3), Some(4)],
        [Some(5), Some(6), Some(7), Some(8)],
        [Some(9), Some(10), Some(11), Some(12)],
        [Some(13), Some(14), None, Some(16)],
    ];
    array.store_chunk_ndarray(&[0, 0], &data0)?;

    // Chunk [0,1]: half None, half Some
    let data1 = array![
        [None, None, None, None],
        [None, None, None, None],
        [Some(1), Some(2), Some(3), Some(4)],
        [Some(5), Some(6), Some(7), Some(8)],
    ];
    array.store_chunk_ndarray(&[0, 1], &data1)?;

    // Chunk [1,0]: all None
    let data2 = array![
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
    ];
    array.store_chunk_ndarray(&[1, 0], &data2)?;

    // Chunk [1,1]: alternating Some/None
    let data3 = array![
        [Some(0), None, Some(2), None],
        [Some(4), None, Some(6), None],
        [Some(8), None, Some(10), None],
        [Some(12), None, Some(14), None],
    ];
    array.store_chunk_ndarray(&[1, 1], &data3)?;

    // Verify all chunks
    let retrieved0 = array.retrieve_chunk_ndarray::<Option<u8>>(&[0, 0])?;
    let retrieved0: Array2<Option<u8>> = retrieved0.into_dimensionality()?;
    assert_eq!(retrieved0, data0);

    let retrieved1 = array.retrieve_chunk_ndarray::<Option<u8>>(&[0, 1])?;
    let retrieved1: Array2<Option<u8>> = retrieved1.into_dimensionality()?;
    assert_eq!(retrieved1, data1);

    let retrieved2 = array.retrieve_chunk_ndarray::<Option<u8>>(&[1, 0])?;
    let retrieved2: Array2<Option<u8>> = retrieved2.into_dimensionality()?;
    assert_eq!(retrieved2, data2);

    let retrieved3 = array.retrieve_chunk_ndarray::<Option<u8>>(&[1, 1])?;
    let retrieved3: Array2<Option<u8>> = retrieved3.into_dimensionality()?;
    assert_eq!(retrieved3, data3);

    // Verify entire array
    let retrieved_full = array.retrieve_array_subset_ndarray::<Option<u8>>(&array.subset_all())?;
    let retrieved_full: Array2<Option<u8>> = retrieved_full.into_dimensionality()?;
    #[rustfmt::skip]
    let expected_full = array![
        [Some(1), Some(2), Some(3), Some(4), None, None, None, None],
        [Some(5), Some(6), Some(7), Some(8), None, None, None, None],
        [Some(9), Some(10), Some(11), Some(12), Some(1), Some(2), Some(3), Some(4)],
        [Some(13), Some(14), None, Some(16), Some(5), Some(6), Some(7), Some(8)],
        [None, None, None, None, Some(0), None, Some(2), None],
        [None, None, None, None, Some(4), None, Some(6), None],
        [None, None, None, None, Some(8), None, Some(10), None],
        [None, None, None, None, Some(12), None, Some(14), None],
    ];
    assert_eq!(retrieved_full, expected_full);

    // Partially update chunks
    let update_data = array![
        [Some(99), None, Some(98), None],
        [None, Some(97), None, Some(96)],
        [Some(95), None, Some(94), None],
        [None, Some(93), None, Some(92)],
    ];
    array.store_array_subset_ndarray(&[0, 2], &update_data)?;

    // Verify partial update
    let retrieved_update =
        array.retrieve_array_subset_ndarray::<Option<u8>>(&ArraySubset::new_with_ranges(&[
            0..4,
            2..6,
        ]))?;
    let retrieved_update: Array2<Option<u8>> = retrieved_update.into_dimensionality()?;
    assert_eq!(retrieved_update, update_data);
    let chunk0_updated_data = array![
        [Some(1), Some(2), Some(99), None],
        [Some(5), Some(6), None, Some(97)],
        [Some(9), Some(10), Some(95), None],
        [Some(13), Some(14), None, Some(93)],
    ];
    let retrieved0_updated = array.retrieve_chunk_ndarray::<Option<u8>>(&[0, 0])?;
    let retrieved0_updated: Array2<Option<u8>> = retrieved0_updated.into_dimensionality()?;
    assert_eq!(retrieved0_updated, chunk0_updated_data);

    Ok(())
}

#[test]
fn optional_array_nested_2_level() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{array, Array2};

    // Test Option<Option<u8>>
    let array = create_optional_array(
        DataType::UInt8.into_optional().into_optional(),
        None::<Option<u8>>.into(),
    );

    // Store nested optional data
    // Some(Some(v)) - outer valid, inner valid
    // Some(None) - outer valid, inner invalid
    // None - outer invalid
    let data = array![
        [Some(Some(1)), Some(None), None, Some(Some(4))],
        [Some(Some(5)), None, Some(None), Some(Some(8))],
        [Some(Some(9)), None, Some(Some(11)), Some(None)],
        [Some(Some(13)), Some(Some(14)), None, Some(Some(16))],
    ];
    array.store_chunk_ndarray(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved = array.retrieve_chunk_ndarray::<Option<Option<u8>>>(&[0, 0])?;
    let retrieved: Array2<Option<Option<u8>>> = retrieved.into_dimensionality()?;
    assert_eq!(retrieved, data);

    // Specifically verify nested None values
    assert_eq!(retrieved[[0, 1]], Some(None)); // Outer Some, inner None
    assert_eq!(retrieved[[0, 2]], None); // Outer None
    assert_eq!(retrieved[[1, 2]], Some(None)); // Outer Some, inner None
    assert_eq!(retrieved[[2, 3]], Some(None)); // Outer Some, inner None

    Ok(())
}

#[test]
fn optional_array_nested_3_level() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{array, Array2};

    // Test Option<Option<Option<u16>>>
    let array = create_optional_array(
        DataType::UInt16
            .into_optional()
            .into_optional()
            .into_optional(),
        None::<Option<Option<u16>>>.into(),
    );

    // Store 3-level nested optional data
    let data = array![
        [Some(Some(Some(100))), Some(Some(None)), Some(None), None],
        [Some(Some(Some(400))), Some(None), Some(Some(None)), None],
        [
            Some(Some(Some(800))),
            Some(Some(Some(900))),
            None,
            Some(None)
        ],
        [
            Some(Some(Some(1200))),
            Some(Some(None)),
            Some(Some(Some(1400))),
            None
        ],
    ];
    array.store_chunk_ndarray(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved = array.retrieve_chunk_ndarray::<Option<Option<Option<u16>>>>(&[0, 0])?;
    let retrieved: Array2<Option<Option<Option<u16>>>> = retrieved.into_dimensionality()?;
    assert_eq!(retrieved, data);

    // Verify different nesting levels of None
    assert_eq!(retrieved[[0, 0]], Some(Some(Some(100)))); // Fully valid
    assert_eq!(retrieved[[0, 1]], Some(Some(None))); // Inner None
    assert_eq!(retrieved[[0, 2]], Some(None)); // Middle None
    assert_eq!(retrieved[[0, 3]], None); // Outer None
    assert_eq!(retrieved[[1, 2]], Some(Some(None))); // Inner None

    Ok(())
}

#[test]
fn optional_array_with_non_null_fill_value() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{array, Array2};

    // Create an optional array with a non-null fill value
    let array = create_optional_array(
        DataType::UInt8.into_optional(),
        FillValue::from(Some(255u8)),
    );

    // Store data in one chunk
    let data = array![
        [Some(1), None, Some(3), Some(4)],
        [Some(5), Some(6), None, Some(8)],
        [None, Some(10), Some(11), Some(12)],
        [Some(13), Some(14), None, Some(16)],
    ];
    array.store_chunk_ndarray(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved = array.retrieve_chunk_ndarray::<Option<u8>>(&[0, 0])?;
    let retrieved: Array2<Option<u8>> = retrieved.into_dimensionality()?;
    assert_eq!(retrieved, data);

    // The mask should properly distinguish None values from actual data
    assert_eq!(retrieved[[0, 1]], None);
    assert_eq!(retrieved[[1, 2]], None);
    assert_eq!(retrieved[[2, 0]], None);
    assert_eq!(retrieved[[3, 2]], None);

    Ok(())
}

#[test]
fn optional_array_string() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{array, Array2};

    // The ArrayBuilder automatically handles optional strings with the correct vlen codec
    let array = create_optional_array(DataType::String.into_optional(), None::<String>.into());

    // Store chunk with Option<String>
    let data = array![
        [
            Some("hello".to_string()),
            Some("world".to_string()),
            None,
            Some("zarr".to_string())
        ],
        [
            Some("optional".to_string()),
            None,
            Some("string".to_string()),
            Some("test".to_string())
        ],
        [
            Some("data".to_string()),
            Some("codec".to_string()),
            Some("rust".to_string()),
            None
        ],
        [
            None,
            Some("variable".to_string()),
            Some("length".to_string()),
            Some("encoding".to_string())
        ],
    ];
    array.store_chunk_ndarray(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved = array.retrieve_chunk_ndarray::<Option<String>>(&[0, 0])?;
    let retrieved: Array2<Option<String>> = retrieved.into_dimensionality()?;
    assert_eq!(retrieved, data);

    // Verify that None values are properly encoded/decoded
    assert_eq!(retrieved[[0, 2]], None);
    assert_eq!(retrieved[[1, 1]], None);
    assert_eq!(retrieved[[2, 3]], None);
    assert_eq!(retrieved[[3, 0]], None);

    // Verify Some values
    assert_eq!(retrieved[[0, 0]], Some("hello".to_string()));
    assert_eq!(retrieved[[0, 1]], Some("world".to_string()));
    assert_eq!(retrieved[[0, 3]], Some("zarr".to_string()));
    assert_eq!(retrieved[[1, 2]], Some("string".to_string()));

    // Test with different length strings including empty
    let data2 = array![
        [
            Some("".to_string()),
            None,
            Some("x".to_string()),
            Some("longer string with spaces".to_string())
        ],
        [
            None,
            Some("unicode: 你好🌍".to_string()),
            None,
            Some("numbers123".to_string())
        ],
        [
            Some("symbols!@#$%".to_string()),
            None,
            Some("newline\nchar".to_string()),
            Some("tab\tchar".to_string())
        ],
        [None, None, Some("final".to_string()), None],
    ];
    array.store_chunk_ndarray(&[0, 1], &data2)?;

    let retrieved2 = array.retrieve_chunk_ndarray::<Option<String>>(&[0, 1])?;
    let retrieved2: Array2<Option<String>> = retrieved2.into_dimensionality()?;
    assert_eq!(retrieved2, data2);

    // Verify specific complex strings
    assert_eq!(retrieved2[[0, 0]], Some("".to_string()));
    assert_eq!(retrieved2[[1, 1]], Some("unicode: 你好🌍".to_string()));
    assert_eq!(retrieved2[[2, 2]], Some("newline\nchar".to_string()));

    Ok(())
}

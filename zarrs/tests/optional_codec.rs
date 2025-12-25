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
    use ndarray::{Array2, array};

    let array = create_optional_array(DataType::UInt8.into_optional(), None::<u8>.into());

    // Store different patterns in different chunks
    // Chunk [0,0]: mostly Some values
    let data0 = array![
        [Some(1), Some(2), Some(3), Some(4)],
        [Some(5), Some(6), Some(7), Some(8)],
        [Some(9), Some(10), Some(11), Some(12)],
        [Some(13), Some(14), None, Some(16)],
    ];
    array.store_chunk(&[0, 0], data0.clone())?;

    // Chunk [0,1]: half None, half Some
    let data1 = array![
        [None, None, None, None],
        [None, None, None, None],
        [Some(1), Some(2), Some(3), Some(4)],
        [Some(5), Some(6), Some(7), Some(8)],
    ];
    array.store_chunk(&[0, 1], data1.clone())?;

    // Chunk [1,0]: all None
    let data2 = array![
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
    ];
    array.store_chunk(&[1, 0], data2.clone())?;

    // Chunk [1,1]: alternating Some/None
    let data3 = array![
        [Some(0), None, Some(2), None],
        [Some(4), None, Some(6), None],
        [Some(8), None, Some(10), None],
        [Some(12), None, Some(14), None],
    ];
    array.store_chunk(&[1, 1], data3.clone())?;

    // Verify all chunks
    let retrieved0 = array.retrieve_chunk::<ndarray::ArrayD<Option<u8>>>(&[0, 0])?;
    let retrieved0: Array2<Option<u8>> = retrieved0.into_dimensionality()?;
    assert_eq!(retrieved0, data0);

    let retrieved1 = array.retrieve_chunk::<ndarray::ArrayD<Option<u8>>>(&[0, 1])?;
    let retrieved1: Array2<Option<u8>> = retrieved1.into_dimensionality()?;
    assert_eq!(retrieved1, data1);

    let retrieved2 = array.retrieve_chunk::<ndarray::ArrayD<Option<u8>>>(&[1, 0])?;
    let retrieved2: Array2<Option<u8>> = retrieved2.into_dimensionality()?;
    assert_eq!(retrieved2, data2);

    let retrieved3 = array.retrieve_chunk::<ndarray::ArrayD<Option<u8>>>(&[1, 1])?;
    let retrieved3: Array2<Option<u8>> = retrieved3.into_dimensionality()?;
    assert_eq!(retrieved3, data3);

    // Verify entire array
    let retrieved_full =
        array.retrieve_array_subset::<ndarray::ArrayD<Option<u8>>>(&array.subset_all())?;
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
    array.store_array_subset(
        &ArraySubset::new_with_start_shape(
            vec![0, 2],
            update_data.shape().iter().map(|&x| x as u64).collect(),
        )?,
        update_data.clone(),
    )?;

    // Verify partial update
    let retrieved_update = array.retrieve_array_subset::<ndarray::ArrayD<Option<u8>>>(
        &ArraySubset::new_with_ranges(&[0..4, 2..6]),
    )?;
    let retrieved_update: Array2<Option<u8>> = retrieved_update.into_dimensionality()?;
    assert_eq!(retrieved_update, update_data);
    let chunk0_updated_data = array![
        [Some(1), Some(2), Some(99), None],
        [Some(5), Some(6), None, Some(97)],
        [Some(9), Some(10), Some(95), None],
        [Some(13), Some(14), None, Some(93)],
    ];
    let retrieved0_updated = array.retrieve_chunk::<ndarray::ArrayD<Option<u8>>>(&[0, 0])?;
    let retrieved0_updated: Array2<Option<u8>> = retrieved0_updated.into_dimensionality()?;
    assert_eq!(retrieved0_updated, chunk0_updated_data);

    Ok(())
}

#[test]
fn optional_array_nested_2_level() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{Array2, array};

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
    array.store_chunk(&[0, 0], data.clone())?;

    // Retrieve and verify
    let retrieved = array.retrieve_chunk::<ndarray::ArrayD<Option<Option<u8>>>>(&[0, 0])?;
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
    use ndarray::{Array2, array};

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
    array.store_chunk(&[0, 0], data.clone())?;

    // Retrieve and verify
    let retrieved =
        array.retrieve_chunk::<ndarray::ArrayD<Option<Option<Option<u16>>>>>(&[0, 0])?;
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
    use ndarray::{Array2, array};

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
    array.store_chunk(&[0, 0], data.clone())?;

    // Retrieve and verify
    let retrieved = array.retrieve_chunk::<ndarray::ArrayD<Option<u8>>>(&[0, 0])?;
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
    use ndarray::{Array2, array};

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
    array.store_chunk(&[0, 0], data.clone())?;

    // Retrieve and verify
    let retrieved = array.retrieve_chunk::<ndarray::ArrayD<Option<String>>>(&[0, 0])?;
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
    array.store_chunk(&[0, 1], data2.clone())?;

    let retrieved2 = array.retrieve_chunk::<ndarray::ArrayD<Option<String>>>(&[0, 1])?;
    let retrieved2: Array2<Option<String>> = retrieved2.into_dimensionality()?;
    assert_eq!(retrieved2, data2);

    // Verify specific complex strings
    assert_eq!(retrieved2[[0, 0]], Some("".to_string()));
    assert_eq!(retrieved2[[1, 1]], Some("unicode: 你好🌍".to_string()));
    assert_eq!(retrieved2[[2, 2]], Some("newline\nchar".to_string()));

    Ok(())
}

#[test]
fn optional_array_string_multi_chunk() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{Array2, array};

    // 8x8 array with 4x4 chunks = 4 chunks total
    let array = create_optional_array(DataType::String.into_optional(), None::<String>.into());

    // Store data in all 4 chunks with varying patterns
    // Chunk [0,0]: mixed Some/None with various string lengths
    let data0 = array![
        [
            Some("hello".to_string()),
            None,
            Some("world".to_string()),
            Some("test".to_string())
        ],
        [None, Some("a".to_string()), Some("bc".to_string()), None],
        [
            Some("".to_string()),
            Some("data".to_string()),
            None,
            Some("xyz".to_string())
        ],
        [
            Some("chunk0".to_string()),
            None,
            None,
            Some("end".to_string())
        ],
    ];
    array.store_chunk(&[0, 0], data0.clone())?;

    // Chunk [0,1]: more None values
    let data1 = array![
        [None, None, None, None],
        [
            Some("only".to_string()),
            Some("some".to_string()),
            None,
            None
        ],
        [
            None,
            Some("strings".to_string()),
            Some("here".to_string()),
            None
        ],
        [
            Some("more".to_string()),
            None,
            Some("text".to_string()),
            Some("data".to_string())
        ],
    ];
    array.store_chunk(&[0, 1], data1.clone())?;

    // Chunk [1,0]: including unicode and special chars
    let data2 = array![
        [
            Some("lower".to_string()),
            Some("left".to_string()),
            Some("chunk".to_string()),
            None
        ],
        [
            None,
            None,
            Some("mixed".to_string()),
            Some("values".to_string())
        ],
        [
            Some("unicode: 你好".to_string()),
            None,
            None,
            Some("emoji: 🎉".to_string())
        ],
        [
            None,
            Some("final".to_string()),
            None,
            Some("row".to_string())
        ],
    ];
    array.store_chunk(&[1, 0], data2.clone())?;

    // Chunk [1,1]: alternating pattern
    let data3 = array![
        [
            None,
            Some("corner".to_string()),
            None,
            Some("last".to_string())
        ],
        [
            Some("bottom".to_string()),
            None,
            Some("right".to_string()),
            None
        ],
        [
            None,
            Some("almost".to_string()),
            None,
            Some("done".to_string())
        ],
        [
            Some("the".to_string()),
            Some("very".to_string()),
            Some("end".to_string()),
            None
        ],
    ];
    array.store_chunk(&[1, 1], data3.clone())?;

    // Retrieve entire array spanning all chunks
    let retrieved =
        array.retrieve_array_subset::<ndarray::ArrayD<Option<String>>>(&array.subset_all())?;
    let retrieved: Array2<Option<String>> = retrieved.into_dimensionality()?;

    // Verify dimensions
    assert_eq!(retrieved.shape(), &[8, 8]);

    // Verify values from chunk [0,0] region (rows 0-3, cols 0-3)
    assert_eq!(retrieved[[0, 0]], Some("hello".to_string()));
    assert_eq!(retrieved[[0, 1]], None);
    assert_eq!(retrieved[[1, 0]], None);
    assert_eq!(retrieved[[2, 0]], Some("".to_string())); // Empty string Some("")

    // Verify values from chunk [0,1] region (rows 0-3, cols 4-7)
    assert_eq!(retrieved[[0, 4]], None);
    assert_eq!(retrieved[[1, 4]], Some("only".to_string()));
    assert_eq!(retrieved[[2, 5]], Some("strings".to_string()));

    // Verify values from chunk [1,0] region (rows 4-7, cols 0-3)
    assert_eq!(retrieved[[4, 0]], Some("lower".to_string()));
    assert_eq!(retrieved[[6, 0]], Some("unicode: 你好".to_string()));
    assert_eq!(retrieved[[6, 3]], Some("emoji: 🎉".to_string()));

    // Verify values from chunk [1,1] region (rows 4-7, cols 4-7)
    assert_eq!(retrieved[[4, 5]], Some("corner".to_string()));
    assert_eq!(retrieved[[7, 6]], Some("end".to_string()));
    assert_eq!(retrieved[[7, 7]], None);

    Ok(())
}

#[test]
fn optional_array_string_partial_subset() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{Array2, array};

    let array = create_optional_array(DataType::String.into_optional(), None::<String>.into());

    // Store chunks
    let data0 = array![
        [
            Some("a".to_string()),
            Some("b".to_string()),
            Some("c".to_string()),
            Some("d".to_string())
        ],
        [Some("e".to_string()), None, None, Some("h".to_string())],
        [None, Some("j".to_string()), Some("k".to_string()), None],
        [
            Some("m".to_string()),
            Some("n".to_string()),
            Some("o".to_string()),
            Some("p".to_string())
        ],
    ];
    array.store_chunk(&[0, 0], data0)?;

    let data1 = array![
        [
            Some("1".to_string()),
            Some("2".to_string()),
            None,
            Some("4".to_string())
        ],
        [None, Some("6".to_string()), Some("7".to_string()), None],
        [Some("9".to_string()), None, None, Some("12".to_string())],
        [
            Some("13".to_string()),
            Some("14".to_string()),
            Some("15".to_string()),
            Some("16".to_string())
        ],
    ];
    array.store_chunk(&[0, 1], data1)?;

    // Retrieve subset spanning both chunks: rows 1..3, cols 2..6
    let subset = ArraySubset::new_with_ranges(&[1..3, 2..6]);
    let retrieved = array.retrieve_array_subset::<ndarray::ArrayD<Option<String>>>(&subset)?;
    let retrieved: Array2<Option<String>> = retrieved.into_dimensionality()?;

    assert_eq!(retrieved.shape(), &[2, 4]);

    // Row 0 of result: from array row 1, cols 2..6
    // cols 2,3 from chunk[0,0], cols 4,5 from chunk[0,1]
    assert_eq!(retrieved[[0, 0]], None); // array[1,2] from chunk[0,0]
    assert_eq!(retrieved[[0, 1]], Some("h".to_string())); // array[1,3] from chunk[0,0]
    assert_eq!(retrieved[[0, 2]], None); // array[1,4] from chunk[0,1]
    assert_eq!(retrieved[[0, 3]], Some("6".to_string())); // array[1,5] from chunk[0,1]

    // Row 1 of result: from array row 2, cols 2..6
    assert_eq!(retrieved[[1, 0]], Some("k".to_string())); // array[2,2] from chunk[0,0]
    assert_eq!(retrieved[[1, 1]], None); // array[2,3] from chunk[0,0]
    assert_eq!(retrieved[[1, 2]], Some("9".to_string())); // array[2,4] from chunk[0,1]
    assert_eq!(retrieved[[1, 3]], None); // array[2,5] from chunk[0,1]

    Ok(())
}

#[test]
fn optional_array_bytes_multi_chunk() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{Array2, array};

    let array = create_optional_array(DataType::Bytes.into_optional(), None::<Vec<u8>>.into());

    // Store with varying byte lengths
    let data0 = array![
        [Some(vec![1u8]), Some(vec![2, 3]), None, Some(vec![4, 5, 6])],
        [None, Some(vec![]), Some(vec![7, 8, 9, 10]), None],
        [Some(vec![11, 12]), None, None, Some(vec![13])],
        [
            Some(vec![14, 15, 16]),
            Some(vec![17]),
            Some(vec![18, 19]),
            None
        ],
    ];
    array.store_chunk(&[0, 0], data0.clone())?;

    let data1 = array![
        [None, Some(vec![100, 101]), Some(vec![]), None],
        [
            Some(vec![102, 103, 104, 105]),
            None,
            Some(vec![106]),
            Some(vec![107, 108])
        ],
        [
            Some(vec![109]),
            Some(vec![110, 111, 112]),
            None,
            Some(vec![])
        ],
        [None, None, Some(vec![113, 114]), Some(vec![115])],
    ];
    array.store_chunk(&[0, 1], data1.clone())?;

    // Retrieve full array spanning both chunks
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..8]);
    let retrieved = array.retrieve_array_subset::<ndarray::ArrayD<Option<Vec<u8>>>>(&subset)?;
    let retrieved: Array2<Option<Vec<u8>>> = retrieved.into_dimensionality()?;

    // Verify from chunk [0,0]
    assert_eq!(retrieved[[0, 0]], Some(vec![1u8]));
    assert_eq!(retrieved[[0, 1]], Some(vec![2, 3]));
    assert_eq!(retrieved[[0, 2]], None);
    assert_eq!(retrieved[[1, 1]], Some(vec![])); // Empty vec Some(vec![])

    // Verify from chunk [0,1]
    assert_eq!(retrieved[[1, 4]], Some(vec![102, 103, 104, 105]));
    assert_eq!(retrieved[[1, 5]], None);
    assert_eq!(retrieved[[2, 5]], Some(vec![110, 111, 112]));
    assert_eq!(retrieved[[2, 7]], Some(vec![])); // Empty vec

    Ok(())
}

#[test]
fn optional_nested_string_multi_chunk() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::{Array2, array};

    // Option<Option<String>> - testing nested optionals with variable-length inner type
    let array = create_optional_array(
        DataType::String.into_optional().into_optional(),
        None::<Option<String>>.into(),
    );

    // Chunk with all three cases: None, Some(None), Some(Some(str))
    let data0 = array![
        [
            Some(Some("hello".to_string())),
            Some(None),
            None,
            Some(Some("world".to_string()))
        ],
        [None, Some(Some("test".to_string())), Some(None), None],
        [
            Some(Some("".to_string())),
            None,
            Some(None),
            Some(Some("data".to_string()))
        ],
        [Some(None), Some(Some("end".to_string())), None, Some(None)],
    ];
    array.store_chunk(&[0, 0], data0.clone())?;

    let data1 = array![
        [None, Some(None), Some(Some("chunk2".to_string())), None],
        [Some(Some("mixed".to_string())), None, None, Some(None)],
        [
            Some(None),
            Some(Some("values".to_string())),
            Some(None),
            None
        ],
        [
            None,
            None,
            Some(Some("last".to_string())),
            Some(Some("entry".to_string()))
        ],
    ];
    array.store_chunk(&[0, 1], data1.clone())?;

    // Retrieve spanning both chunks
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..8]);
    let retrieved =
        array.retrieve_array_subset::<ndarray::ArrayD<Option<Option<String>>>>(&subset)?;
    let retrieved: Array2<Option<Option<String>>> = retrieved.into_dimensionality()?;

    // Verify nested None values are preserved correctly from chunk [0,0]
    assert_eq!(retrieved[[0, 0]], Some(Some("hello".to_string())));
    assert_eq!(retrieved[[0, 1]], Some(None)); // Some(None) is distinct from None
    assert_eq!(retrieved[[0, 2]], None);
    assert_eq!(retrieved[[2, 0]], Some(Some("".to_string()))); // Empty string Some(Some(""))
    assert_eq!(retrieved[[3, 0]], Some(None));

    // Verify from chunk [0,1]
    assert_eq!(retrieved[[0, 4]], None);
    assert_eq!(retrieved[[0, 5]], Some(None));
    assert_eq!(retrieved[[0, 6]], Some(Some("chunk2".to_string())));
    assert_eq!(retrieved[[1, 4]], Some(Some("mixed".to_string())));
    assert_eq!(retrieved[[3, 6]], Some(Some("last".to_string())));
    assert_eq!(retrieved[[3, 7]], Some(Some("entry".to_string())));

    Ok(())
}

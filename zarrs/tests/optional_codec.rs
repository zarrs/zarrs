#![allow(missing_docs)]

use std::sync::Arc;

use zarrs::array::{Array, ArrayBuilder, DataType, FillValue};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::store::MemoryStore;

/// Test helper to create an array with optional codec
fn create_optional_array(data_type: DataType, fill_value: FillValue) -> Array<MemoryStore> {
    let store = Arc::new(MemoryStore::default());
    let array_path = "/optional_array";

    // Determine the inner data type to configure the bytes codec appropriately
    let inner_type = if let DataType::Optional(inner) = &data_type {
        inner.as_ref()
    } else {
        &data_type
    };

    // Recursively find the innermost non-optional type
    let mut innermost_type = inner_type;
    while let DataType::Optional(inner) = innermost_type {
        innermost_type = inner.as_ref();
    }

    // Configure data codecs based on whether we have nested optional types
    let data_codecs_config = if inner_type.is_optional() {
        // Nested optional type needs its own optional codec
        create_nested_codec_config(inner_type)
    } else if let Some(size) = innermost_type.fixed_size() {
        if size > 1 {
            serde_json::json!([{"name": "bytes", "configuration": {"endian": "little"}}])
        } else {
            serde_json::json!([{"name": "bytes", "configuration": {}}])
        }
    } else {
        serde_json::json!([{"name": "bytes", "configuration": {}}])
    };

    let codec_config: serde_json::Value = serde_json::json!({
        "name": "optional",
        "configuration": {
            "mask_codecs": [{"name": "packbits", "configuration": {}}],
            "data_codecs": data_codecs_config
        }
    });

    let codec =
        zarrs::array::codec::array_to_bytes::optional::OptionalCodec::new_with_configuration(
            &serde_json::from_value(codec_config.get("configuration").unwrap().clone()).unwrap(),
        )
        .unwrap();

    ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 4], // chunk shape
        data_type,
        fill_value,
    )
    .array_to_bytes_codec(Arc::new(codec))
    .build(store, array_path)
    .unwrap()
}

/// Helper to recursively create codec config for nested optional types
fn create_nested_codec_config(data_type: &DataType) -> serde_json::Value {
    if let DataType::Optional(inner) = data_type {
        let inner_codecs = if inner.is_optional() {
            create_nested_codec_config(inner.as_ref())
        } else if let Some(size) = inner.fixed_size() {
            if size > 1 {
                serde_json::json!([{"name": "bytes", "configuration": {"endian": "little"}}])
            } else {
                serde_json::json!([{"name": "bytes", "configuration": {}}])
            }
        } else {
            serde_json::json!([{"name": "bytes", "configuration": {}}])
        };

        serde_json::json!([{
            "name": "optional",
            "configuration": {
                "mask_codecs": [{"name": "packbits", "configuration": {}}],
                "data_codecs": inner_codecs
            }
        }])
    } else if let Some(size) = data_type.fixed_size() {
        if size > 1 {
            serde_json::json!([{"name": "bytes", "configuration": {"endian": "little"}}])
        } else {
            serde_json::json!([{"name": "bytes", "configuration": {}}])
        }
    } else {
        serde_json::json!([{"name": "bytes", "configuration": {}}])
    }
}

#[test]
fn optional_array_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::UInt8)),
        FillValue::new_null(),
    );

    // Store chunk elements with Option<u8>
    let data: Vec<Option<u8>> = vec![
        Some(1),
        Some(2),
        None,
        Some(4),
        Some(5),
        None,
        Some(7),
        Some(8),
        Some(9),
        Some(10),
        Some(11),
        None,
        None,
        Some(14),
        Some(15),
        Some(16),
    ];
    array.store_chunk_elements(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved: Vec<Option<u8>> = array.retrieve_chunk_elements(&[0, 0])?;
    assert_eq!(retrieved, data);

    // Verify that None values are properly encoded/decoded
    assert_eq!(retrieved[2], None);
    assert_eq!(retrieved[5], None);
    assert_eq!(retrieved[11], None);
    assert_eq!(retrieved[12], None);

    // Verify Some values
    assert_eq!(retrieved[0], Some(1));
    assert_eq!(retrieved[1], Some(2));
    assert_eq!(retrieved[3], Some(4));

    Ok(())
}

#[test]
fn optional_array_subset_operations() -> Result<(), Box<dyn std::error::Error>> {
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::Int32)),
        FillValue::new_null(),
    );

    // Store data in first chunk
    let data1: Vec<Option<i32>> = vec![
        Some(100),
        None,
        Some(300),
        Some(400),
        Some(500),
        Some(600),
        None,
        Some(800),
        Some(900),
        Some(1000),
        Some(1100),
        Some(1200),
        None,
        Some(1400),
        Some(1500),
        None,
    ];
    array.store_chunk_elements(&[0, 0], &data1)?;

    // Store data in adjacent chunk
    let data2: Vec<Option<i32>> = vec![
        Some(10),
        Some(20),
        Some(30),
        Some(40),
        None,
        None,
        None,
        None,
        Some(90),
        Some(100),
        Some(110),
        Some(120),
        Some(130),
        Some(140),
        Some(150),
        Some(160),
    ];
    array.store_chunk_elements(&[0, 1], &data2)?;

    // Retrieve array subset spanning both chunks
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..8]);
    let retrieved: Vec<Option<i32>> = array.retrieve_array_subset_elements(&subset)?;

    // Verify the data spans correctly
    assert_eq!(retrieved.len(), 32);
    assert_eq!(retrieved[0], Some(100)); // First element of chunk [0,0]
    assert_eq!(retrieved[4], Some(10)); // First element of chunk [0,1] at row 0
    assert_eq!(retrieved[1], None); // None in chunk [0,0]
    assert_eq!(retrieved[12], None); // None in chunk [0,1] at row 1, col 4 (element 4 of chunk [0,1])

    Ok(())
}

#[test]
fn optional_array_nested_2_level() -> Result<(), Box<dyn std::error::Error>> {
    // Test Option<Option<u8>>
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::Optional(Box::new(DataType::UInt8)))),
        FillValue::new_null(),
    );

    // Store nested optional data
    // Some(Some(v)) - outer valid, inner valid
    // Some(None) - outer valid, inner invalid
    // None - outer invalid
    let data: Vec<Option<Option<u8>>> = vec![
        Some(Some(1)),  // 0: valid value
        Some(None),     // 1: outer Some, inner None
        None,           // 2: outer None
        Some(Some(4)),  // 3: valid value
        Some(Some(5)),  // 4: valid value
        None,           // 5: outer None
        Some(None),     // 6: outer Some, inner None
        Some(Some(8)),  // 7: valid value
        Some(Some(9)),  // 8: valid value
        None,           // 9: outer None
        Some(Some(11)), // 10: valid value
        Some(None),     // 11: outer Some, inner None
        Some(Some(13)), // 12: valid value
        Some(Some(14)), // 13: valid value
        None,           // 14: outer None
        Some(Some(16)), // 15: valid value
    ];
    array.store_chunk_elements(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved: Vec<Option<Option<u8>>> = array.retrieve_chunk_elements(&[0, 0])?;
    assert_eq!(retrieved, data);

    // Specifically verify nested None values
    assert_eq!(retrieved[1], Some(None)); // Outer Some, inner None
    assert_eq!(retrieved[2], None); // Outer None
    assert_eq!(retrieved[6], Some(None)); // Outer Some, inner None
    assert_eq!(retrieved[11], Some(None)); // Outer Some, inner None

    Ok(())
}

#[test]
fn optional_array_nested_3_level() -> Result<(), Box<dyn std::error::Error>> {
    // Test Option<Option<Option<u16>>>
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::Optional(Box::new(DataType::Optional(
            Box::new(DataType::UInt16),
        ))))),
        FillValue::new_null(),
    );

    // Store 3-level nested optional data
    let data: Vec<Option<Option<Option<u16>>>> = vec![
        Some(Some(Some(100))),  // 0: fully valid
        Some(Some(None)),       // 1: outer valid, middle valid, inner None
        Some(None),             // 2: outer valid, middle None
        None,                   // 3: outer None
        Some(Some(Some(400))),  // 4: fully valid
        Some(None),             // 5: outer valid, middle None
        Some(Some(None)),       // 6: outer valid, middle valid, inner None
        None,                   // 7: outer None
        Some(Some(Some(800))),  // 8: fully valid
        Some(Some(Some(900))),  // 9: fully valid
        None,                   // 10: outer None
        Some(None),             // 11: outer valid, middle None
        Some(Some(Some(1200))), // 12: fully valid
        Some(Some(None)),       // 13: outer valid, middle valid, inner None
        Some(Some(Some(1400))), // 14: fully valid
        None,                   // 15: outer None
    ];
    array.store_chunk_elements(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved: Vec<Option<Option<Option<u16>>>> = array.retrieve_chunk_elements(&[0, 0])?;
    assert_eq!(retrieved, data);

    // Verify different nesting levels of None
    assert_eq!(retrieved[0], Some(Some(Some(100)))); // Fully valid
    assert_eq!(retrieved[1], Some(Some(None))); // Inner None
    assert_eq!(retrieved[2], Some(None)); // Middle None
    assert_eq!(retrieved[3], None); // Outer None
    assert_eq!(retrieved[6], Some(Some(None))); // Inner None

    Ok(())
}

#[test]
#[cfg(feature = "ndarray")]
fn optional_array_ndarray_operations() -> Result<(), Box<dyn std::error::Error>> {
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::Float32)),
        FillValue::new_null(),
    );

    // Create test data with Some and None values
    let data: Vec<Option<f32>> = vec![
        Some(1.5),
        None,
        Some(3.5),
        Some(4.5),
        None,
        Some(6.5),
        Some(7.5),
        None,
        Some(9.5),
        Some(10.5),
        None,
        Some(12.5),
        Some(13.5),
        None,
        Some(15.5),
        Some(16.5),
    ];
    array.store_chunk_elements(&[0, 0], &data)?;

    // Retrieve as ndarray
    let retrieved: Vec<Option<f32>> = array.retrieve_chunk_elements(&[0, 0])?;
    assert_eq!(retrieved, data);

    // Verify specific values
    assert_eq!(retrieved[0], Some(1.5));
    assert_eq!(retrieved[1], None);
    assert_eq!(retrieved[4], None);
    assert_eq!(retrieved[5], Some(6.5));

    Ok(())
}

#[test]
fn optional_array_with_non_null_fill_value() -> Result<(), Box<dyn std::error::Error>> {
    // Create an optional array with a non-null fill value
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::UInt8)),
        FillValue::new(vec![255u8]),
    );

    // Store data in one chunk
    let data: Vec<Option<u8>> = vec![
        Some(1),
        None,
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        None,
        Some(8),
        None,
        Some(10),
        Some(11),
        Some(12),
        Some(13),
        Some(14),
        None,
        Some(16),
    ];
    array.store_chunk_elements(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved: Vec<Option<u8>> = array.retrieve_chunk_elements(&[0, 0])?;
    assert_eq!(retrieved, data);

    // The mask should properly distinguish None values from actual data
    assert_eq!(retrieved[1], None);
    assert_eq!(retrieved[6], None);
    assert_eq!(retrieved[8], None);
    assert_eq!(retrieved[14], None);

    Ok(())
}

#[test]
fn optional_array_all_none() -> Result<(), Box<dyn std::error::Error>> {
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::Int16)),
        FillValue::new_null(),
    );

    // Store chunk with all None values
    let data: Vec<Option<i16>> = vec![None; 16];
    array.store_chunk_elements(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved: Vec<Option<i16>> = array.retrieve_chunk_elements(&[0, 0])?;
    assert_eq!(retrieved, data);
    assert!(retrieved.iter().all(|&v| v.is_none()));

    Ok(())
}

#[test]
fn optional_array_all_some() -> Result<(), Box<dyn std::error::Error>> {
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::UInt32)),
        FillValue::new_null(),
    );

    // Store chunk with all Some values
    let data: Vec<Option<u32>> = (0..16).map(|i| Some(i * 10)).collect();
    array.store_chunk_elements(&[0, 0], &data)?;

    // Retrieve and verify
    let retrieved: Vec<Option<u32>> = array.retrieve_chunk_elements(&[0, 0])?;
    assert_eq!(retrieved, data);
    assert!(retrieved.iter().all(|v| v.is_some()));

    Ok(())
}

#[test]
fn optional_array_mixed_chunks() -> Result<(), Box<dyn std::error::Error>> {
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::UInt8)),
        FillValue::new_null(),
    );

    // Store different patterns in different chunks
    // Chunk [0,0]: mostly Some values
    let data1: Vec<Option<u8>> = vec![
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(8),
        Some(9),
        Some(10),
        Some(11),
        Some(12),
        Some(13),
        Some(14),
        None,
        Some(16),
    ];
    array.store_chunk_elements(&[0, 0], &data1)?;

    // Chunk [0,1]: half None, half Some
    let data2: Vec<Option<u8>> = vec![
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(8),
    ];
    array.store_chunk_elements(&[0, 1], &data2)?;

    // Chunk [1,0]: all None
    let data3: Vec<Option<u8>> = vec![None; 16];
    array.store_chunk_elements(&[1, 0], &data3)?;

    // Chunk [1,1]: alternating Some/None
    let data4: Vec<Option<u8>> = (0..16)
        .map(|i| if i % 2 == 0 { Some(i as u8) } else { None })
        .collect();
    array.store_chunk_elements(&[1, 1], &data4)?;

    // Verify all chunks
    assert_eq!(array.retrieve_chunk_elements::<Option<u8>>(&[0, 0])?, data1);
    assert_eq!(array.retrieve_chunk_elements::<Option<u8>>(&[0, 1])?, data2);
    assert_eq!(array.retrieve_chunk_elements::<Option<u8>>(&[1, 0])?, data3);
    assert_eq!(array.retrieve_chunk_elements::<Option<u8>>(&[1, 1])?, data4);

    Ok(())
}

#[test]
fn optional_array_partial_subset_read() -> Result<(), Box<dyn std::error::Error>> {
    let array = create_optional_array(
        DataType::Optional(Box::new(DataType::UInt8)),
        FillValue::new_null(),
    );

    // Store a chunk with known data
    let data: Vec<Option<u8>> = vec![
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        None,
        Some(6),
        Some(7),
        None,
        Some(9),
        None,
        Some(11),
        Some(12),
        Some(13),
        Some(14),
        None,
        Some(16),
    ];
    array.store_chunk_elements(&[0, 0], &data)?;

    // Read partial subsets and verify
    let subset1 = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let retrieved1: Vec<Option<u8>> = array.retrieve_array_subset_elements(&subset1)?;
    assert_eq!(retrieved1, vec![Some(1), Some(2), None, Some(6)]);

    let subset2 = ArraySubset::new_with_ranges(&[2..4, 2..4]);
    let retrieved2: Vec<Option<u8>> = array.retrieve_array_subset_elements(&subset2)?;
    assert_eq!(retrieved2, vec![Some(11), Some(12), None, Some(16)]);

    Ok(())
}

//! A simple example demonstrating the optional data type with the optional codec.
//!
//! This example shows how to work with optional (nullable) data using the zarrs library.
//! The optional data type is useful for representing missing or null values in arrays.
//!
//! This example demonstrates the "optional" codec - a specialized codec for optional data types
//! that separates the validity mask from the actual data:
//! - The mask is compressed with the packbits codec
//! - The data uses the bytes codec for actual values
//! - More efficient storage for arrays with many null/missing values

use std::sync::Arc;

use zarrs::{
    array::{codec::array_to_bytes::optional::OptionalCodec, ArrayBuilder, DataType, FillValue},
    storage::store::MemoryStore,
    storage::ReadableStorageTraits,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an in-memory store
    let store = Arc::new(MemoryStore::new());

    // Build the codec chains for the optional codec
    let mask_codecs = Arc::new(zarrs::array::codec::CodecChain::from_metadata(&vec![
        serde_json::from_str(r#"{"name": "packbits", "configuration": {}}"#)?,
    ])?);
    let data_codecs = Arc::new(zarrs::array::codec::CodecChain::from_metadata(&vec![
        serde_json::from_str(r#"{"name": "bytes", "configuration": {"endian": "little"}}"#)?,
    ])?);

    let array = ArrayBuilder::new(
        vec![4, 4],                                    // 4x4 array
        vec![2, 2],                                    // 2x2 chunks
        DataType::Optional(Box::new(DataType::UInt8)), // Optional uint8
        FillValue::new_null(),                         // Null fill value
    )
    .dimension_names(["y", "x"].into())
    .array_to_bytes_codec(Arc::new(OptionalCodec::new(mask_codecs, data_codecs)))
    .build(store.clone(), "/array")?;

    println!("Array metadata:\n{}", array.metadata().to_string_pretty());

    // Create some data with missing values
    let data = vec![
        Some(0u8),
        None,
        Some(2u8),
        Some(3u8), // Row 0
        None,
        Some(5u8),
        None,
        Some(7u8), // Row 1
        Some(8u8),
        Some(9u8),
        None,
        None, // Row 2
        Some(12u8),
        None,
        Some(14u8),
        Some(15u8), // Row 3
    ];

    // Write the data
    array.store_array_subset_elements(&array.subset_all(), &data)?;

    // Read back the data
    let data_read = array.retrieve_array_subset_elements::<Option<u8>>(&array.subset_all())?;

    // Verify data integrity
    assert_eq!(data, data_read);

    // Display the data in a grid format
    println!("Data grid (X marks missing values):");
    println!("   0  1  2  3");
    for y in 0..4 {
        print!("{} ", y);
        for x in 0..4 {
            let index = y * 4 + x;
            match data_read[index] {
                Some(value) => print!("{:2} ", value),
                None => print!(" X "),
            }
        }
        println!();
    }

    // Print the raw bytes in all chunks
    println!("Raw bytes in all chunks:");
    let chunk_grid_shape = array.chunk_grid_shape();
    for chunk_y in 0..chunk_grid_shape[0] {
        for chunk_x in 0..chunk_grid_shape[1] {
            let chunk_indices = vec![chunk_y, chunk_x];
            let chunk_key = array.chunk_key(&chunk_indices);
            println!("  Chunk [{}, {}] (key: {}):", chunk_y, chunk_x, chunk_key);

            if let Some(chunk_bytes) = store.get(&chunk_key)? {
                println!("    Size: {} bytes", chunk_bytes.len());

                if chunk_bytes.len() >= 16 {
                    // Parse first 8 bytes as mask size (little-endian u64)
                    let mask_size = u64::from_le_bytes([
                        chunk_bytes[0],
                        chunk_bytes[1],
                        chunk_bytes[2],
                        chunk_bytes[3],
                        chunk_bytes[4],
                        chunk_bytes[5],
                        chunk_bytes[6],
                        chunk_bytes[7],
                    ]) as usize;

                    // Parse second 8 bytes as data size (little-endian u64)
                    let data_size = u64::from_le_bytes([
                        chunk_bytes[8],
                        chunk_bytes[9],
                        chunk_bytes[10],
                        chunk_bytes[11],
                        chunk_bytes[12],
                        chunk_bytes[13],
                        chunk_bytes[14],
                        chunk_bytes[15],
                    ]) as usize;

                    // Display mask size header with raw bytes
                    print!("    Mask size: 0b");
                    for byte in &chunk_bytes[0..8] {
                        print!("{:08b}", byte);
                    }
                    println!(" -> {} bytes", mask_size);

                    // Display data size header with raw bytes
                    print!("    Data size: 0b");
                    for byte in &chunk_bytes[8..16] {
                        print!("{:08b}", byte);
                    }
                    println!(" -> {} bytes", data_size);

                    // Show mask and data sections separately
                    if chunk_bytes.len() >= 16 + mask_size + data_size {
                        let mask_start = 16;
                        let data_start = 16 + mask_size;

                        // Show mask as binary
                        if mask_size > 0 {
                            println!("    Mask (binary):");
                            print!("      ");
                            for byte in &chunk_bytes[mask_start..mask_start + mask_size] {
                                print!("0b{:08b} ", byte);
                            }
                            println!();
                        }

                        // Show data as binary
                        if data_size > 0 {
                            println!("    Data (binary):");
                            print!("      ");
                            for byte in &chunk_bytes[data_start..data_start + data_size] {
                                print!("0b{:08b} ", byte);
                            }
                            println!();
                        }
                    }
                } else {
                    println!("    Chunk too small to parse headers");
                }
            } else {
                println!("    Chunk not found in store");
            }
        }
    }
    Ok(())
}

#![allow(missing_docs)]
#![expect(deprecated)]
#![cfg(feature = "bitround")]

use std::{path::PathBuf, sync::Arc};

use zarrs::metadata_ext::codec::bitround::BitroundCodecConfiguration;
use zarrs::{
    array::{ArrayBuilder, ArrayMetadataOptions, DataType, codec::BitroundCodec},
    array_subset::ArraySubset,
};
use zarrs_filesystem::FilesystemStore;

/// Helper function to print binary representation table
fn print_bitround_table_f32(original: &[f32], rounded: &[f32], keepbits: u8) {
    println!("\n## Bitround float32 Encoding (keepbits={})", keepbits);
    println!(
        "| Original           | Rounded           | Original (0b)                      | Rounded (0b)                       |"
    );
    println!(
        "|--------------------|-------------------|------------------------------------|------------------------------------|"
    );

    for (orig, round) in original.iter().zip(rounded.iter()) {
        let orig_bits = orig.to_bits();
        let round_bits = round.to_bits();

        // Format IEEE 754 as sign_exponent_mantissa
        let orig_sign = (orig_bits >> 31) & 1;
        let orig_exp = (orig_bits >> 23) & 0xFF;
        let orig_mant = orig_bits & 0x7FFFFF;

        let round_sign = (round_bits >> 31) & 1;
        let round_exp = (round_bits >> 23) & 0xFF;
        let round_mant = round_bits & 0x7FFFFF;

        println!(
            "| {:>18.6} | {:>17.6} | {:01b}_{:08b}_{:023b} | {:01b}_{:08b}_{:023b} |",
            orig, round, orig_sign, orig_exp, orig_mant, round_sign, round_exp, round_mant
        );
    }
    println!();
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_codec_bitround_float32() -> Result<(), Box<dyn std::error::Error>> {
    let test_data: Vec<f32> = vec![
        0.0,
        0.1,
        1.2,
        12.3,
        123.4,
        1234.5,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];

    let keepbits = 3;

    // Get the test data directory
    let mut test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_dir.push("tests");
    test_dir.push("data");
    test_dir.push("codec");
    test_dir.push("bitround");

    // Create the directory if it doesn't exist
    std::fs::create_dir_all(&test_dir)?;

    // Create array with bitround codec
    let store = Arc::new(FilesystemStore::new(&test_dir)?);
    let array_path = "/bitround_float32.zarr";

    let mut builder = ArrayBuilder::new(
        vec![test_data.len() as u64],
        vec![test_data.len() as u64],
        DataType::Float32,
        0.0f32,
    );

    // Configure bitround codec with keepbits=3
    let codec_config: BitroundCodecConfiguration =
        serde_json::from_str(&format!(r#"{{"keepbits": {}}}"#, keepbits))?;
    let codec = Arc::new(BitroundCodec::new_with_configuration(&codec_config)?);
    builder.array_to_array_codecs(vec![codec]);

    let array = builder.build(store, array_path)?;

    // Write metadata to store
    array
        .store_metadata_opt(&ArrayMetadataOptions::default().with_include_zarrs_metadata(false))?;

    // Store the test data
    let subset = ArraySubset::new_with_ranges(&[0..test_data.len() as u64]);
    array.store_array_subset_elements(&subset, &test_data)?;

    // Retrieve the data
    let retrieved: Vec<f32> = array.retrieve_array_subset_elements(&subset)?;

    // Print the comparison table
    print_bitround_table_f32(&test_data, &retrieved, keepbits);

    // Verify that rounding occurred (data should be different but close)
    for (orig, rounded) in test_data.iter().zip(retrieved.iter()) {
        if orig.is_nan() {
            assert!(rounded.is_nan(), "Expected NaN, got {}", rounded);
            continue;
        }
        if orig.is_infinite() {
            assert!(
                rounded.is_infinite() && orig.signum() == rounded.signum(),
                "Expected {}, got {}",
                orig,
                rounded
            );
            continue;
        }
        // For non-zero values, verify the relative error is reasonable
        if orig.abs() > f32::EPSILON {
            let rel_error = ((orig - rounded) / orig).abs();
            // With keepbits=3, we expect significant precision loss
            // but values should still be in the same ballpark
            assert!(
                rel_error < 0.5 || (orig - rounded).abs() < 1.0,
                "Value {orig} rounded to {rounded} has too much error"
            );
        }
    }

    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_codec_bitround_uint8() -> Result<(), Box<dyn std::error::Error>> {
    let keepbits = 3;

    let test_data_u8: Vec<u8> = vec![0, 1, 10, 11, 100, 123, 200, 208, 209, 255];

    // Get the test data directory
    let mut test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_dir.push("tests");
    test_dir.push("data");
    test_dir.push("codec");
    test_dir.push("bitround");

    // Create the directory if it doesn't exist
    std::fs::create_dir_all(&test_dir)?;

    let store = Arc::new(FilesystemStore::new(&test_dir)?);
    let array_path = "/bitround_uint8.zarr";

    let mut builder = ArrayBuilder::new(
        vec![test_data_u8.len() as u64],
        vec![test_data_u8.len() as u64],
        DataType::UInt8,
        0u8,
    );

    let codec_config: BitroundCodecConfiguration =
        serde_json::from_str(&format!(r#"{{"keepbits": {}}}"#, keepbits))?;
    let codec = Arc::new(BitroundCodec::new_with_configuration(&codec_config)?);
    builder.array_to_array_codecs(vec![codec]);

    let array = builder.build(store, array_path)?;

    // Write metadata to store
    array
        .store_metadata_opt(&ArrayMetadataOptions::default().with_include_zarrs_metadata(false))?;

    // Store and retrieve
    let subset = ArraySubset::new_with_ranges(&[0..test_data_u8.len() as u64]);
    array.store_array_subset_elements(&subset, &test_data_u8)?;
    let retrieved: Vec<u8> = array.retrieve_array_subset_elements(&subset)?;

    println!("\n## Bitround uint8 Encoding (keepbits={})\n", keepbits);
    println!("| Original | Rounded | Original (0b) | Rounded (0b) |");
    println!("|----------|---------|---------------|--------------|");

    for (orig, rounded) in test_data_u8.iter().zip(retrieved.iter()) {
        println!(
            "| {:>8} | {:>7} |      {:08b} |     {:08b} |",
            orig, rounded, orig, rounded
        );
    }
    println!();

    // Verify rounding behavior
    for (orig, rounded) in test_data_u8.iter().zip(retrieved.iter()) {
        // Small values might be exact, large values should be rounded
        if *orig > 0 {
            let leading_zeros = orig.leading_zeros();
            let bits_used = 8 - leading_zeros;

            // If more bits than keepbits are used, expect rounding
            if bits_used > keepbits {
                // Rounded value should be close but not necessarily exact
                let diff = orig.abs_diff(*rounded);
                assert!(
                    diff < orig / 2,
                    "Rounded value {rounded} too far from original {orig}"
                );
            }
        }
    }

    Ok(())
}

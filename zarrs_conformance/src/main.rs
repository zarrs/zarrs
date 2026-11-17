//! Zarr conformance test binary.

use std::sync::Arc;
use zarrs::array::Resources;

use clap::Parser;
use zarrs::array::{Array, ArrayBytes, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;

/// Command-line arguments for the conformance test binary.
#[derive(Parser, Debug)]
#[command(name = "zarr-conformance-cli")]
#[command(about = "Read and print Zarr array elements in C order")]
struct Args {
    /// Path to the Zarr array directory
    #[arg(long = "array_path")]
    array_path: String,
}

type Result<T> = std::result::Result<T, anyhow::Error>;

fn print_elements_as_fill_value_metadata(
    data_type: &DataType,
    element_bytes: ArrayBytes,
) -> Result<()> {
    match (element_bytes, data_type) {
        (ArrayBytes::Fixed(bytes), DataType::Fixed(fixed)) => {
            anyhow::ensure!(fixed.size() != 0, "zero-sized data types are not supported");
            for byte_slice in bytes.chunks_exact(fixed.size()) {
                let fill_value = FillValue::from(byte_slice);
                let fill_value_metadata = data_type.metadata_fill_value(&fill_value).unwrap();
                println!("{fill_value_metadata}");
            }
            Ok(())
        }
        (ArrayBytes::Variable(bytes_and_offsets), DataType::Variable(_)) => {
            let bytes = bytes_and_offsets.bytes();
            let offsets = bytes_and_offsets.offsets();
            for range in offsets.element_ranges() {
                let byte_slice = &bytes[range];
                let fill_value = FillValue::from(byte_slice);
                let fill_value_metadata = data_type.metadata_fill_value(&fill_value).unwrap();
                println!("{fill_value_metadata}");
            }
            Ok(())
        }
        (ArrayBytes::Optional(..), _) | (_, DataType::Optional(_)) => {
            anyhow::bail!("Optional data types are not yet supported in conformance testing");
        }
        _ => anyhow::bail!("array bytes do not match the data type"),
    }
}

/// Main entry point for the conformance test binary.
fn main() -> Result<()> {
    let args = Args::parse();

    // Open the array
    let store = Arc::new(FilesystemStore::new(&args.array_path)?);
    let array = Array::open(store, "/")?;

    // Retrieve the entire array
    let element_bytes = array.retrieve_array_subset(&array.subset_all(), &Resources::default())?;

    // Print the array elements in C order (as fill value metadata)
    print_elements_as_fill_value_metadata(array.data_type(), element_bytes)?;

    Ok(())
}

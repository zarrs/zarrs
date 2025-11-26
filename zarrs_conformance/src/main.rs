//! Zarr conformance test binary.

use std::sync::Arc;

use clap::Parser;
use zarrs::array::{Array, ArrayBytes, FillValue};
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

/// Main entry point for the conformance test binary.
fn main() -> Result<()> {
    let args = Args::parse();

    // Open the array
    let store = Arc::new(FilesystemStore::new(&args.array_path)?);
    let array = Array::open(store, "/")?;

    // Retrieve the entire array
    let element_bytes = array.retrieve_array_subset(&array.subset_all())?;

    // Print the array elements in C order (as fill value metadata)
    match element_bytes {
        ArrayBytes::Fixed(bytes) => {
            for chunk in bytes.chunks_exact(array.data_type().fixed_size().unwrap()) {
                let fill_value = FillValue::from(chunk);
                let fill_value_metadata =
                    array.data_type().metadata_fill_value(&fill_value).unwrap();
                println!("{fill_value_metadata}");
            }
        }
        ArrayBytes::Variable(bytes, offsets) => {
            for (start, end) in offsets.windows(2).map(|w| (w[0], w[1])) {
                let byte_slice = &bytes[start..end];
                let fill_value = FillValue::from(byte_slice);
                let fill_value_metadata =
                    array.data_type().metadata_fill_value(&fill_value).unwrap();
                println!("{fill_value_metadata}");
            }
        }
    }

    Ok(())
}

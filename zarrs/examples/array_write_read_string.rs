#![allow(missing_docs)]

use itertools::Itertools;
use ndarray::{array, Array2, ArrayD};
use zarrs::{
    storage::storage_adapter::usage_log::UsageLogStorageAdapter,
    storage::ReadableWritableListableStorage,
};

fn array_write_read() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use zarrs::{array::DataType, array_subset::ArraySubset, storage::store};

    // Create a store
    // let path = tempfile::TempDir::new()?;
    // let mut store: ReadableWritableListableStorage =
    //     Arc::new(zarrs::filesystem::FilesystemStore::new(path.path())?);
    // let mut store: ReadableWritableListableStorage = Arc::new(
    //     zarrs::filesystem::FilesystemStore::new("zarrs/tests/data/array_write_read.zarr")?,
    // );
    let mut store: ReadableWritableListableStorage = Arc::new(store::MemoryStore::new());
    if let Some(arg1) = std::env::args().collect::<Vec<_>>().get(1) {
        if arg1 == "--usage-log" {
            let log_writer = Arc::new(std::sync::Mutex::new(
                // std::io::BufWriter::new(
                std::io::stdout(),
                //    )
            ));
            store = Arc::new(UsageLogStorageAdapter::new(store, log_writer, || {
                chrono::Utc::now().format("[%T%.3f] ").to_string()
            }));
        }
    }

    // Create the root group
    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")?
        .store_metadata()?;

    // Create a group with attributes
    let group_path = "/group";
    let mut group = zarrs::group::GroupBuilder::new().build(store.clone(), group_path)?;
    group
        .attributes_mut()
        .insert("foo".into(), serde_json::Value::String("bar".into()));
    group.store_metadata()?;

    println!(
        "The group metadata is:\n{}\n",
        group.metadata().to_string_pretty()
    );

    // Create an array
    let array_path = "/group/array";
    let array = zarrs::array::ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        DataType::String,
        "_",
    )
    // .bytes_to_bytes_codecs(vec![]) // uncompressed
    .dimension_names(["y", "x"].into())
    // .storage_transformers(vec![].into())
    .build(store.clone(), array_path)?;

    // Write array metadata to store
    array.store_metadata()?;

    println!(
        "The array metadata is:\n{}\n",
        array.metadata().to_string_pretty()
    );

    // Write some chunks
    array.store_chunk_ndarray(
        &[0, 0],
        &ArrayD::<&str>::from_shape_vec(vec![2, 2], vec!["a", "bb", "ccc", "dddd"]).unwrap(),
    )?;
    array.store_chunk_ndarray(
        &[0, 1],
        &ArrayD::<&str>::from_shape_vec(vec![2, 2], vec!["4444", "333", "22", "1"]).unwrap(),
    )?;
    let subset_all = array.subset_all();
    let data_all = array.retrieve_array_subset_ndarray::<String>(&subset_all)?;
    println!("store_chunk [0, 0] and [0, 1]:\n{data_all}\n");

    // Write a subset spanning multiple chunks, including updating chunks already written
    let ndarray_subset: Array2<&str> = array![["!", "@@"], ["###", "$$$$"]];
    array.store_array_subset_ndarray(
        ArraySubset::new_with_ranges(&[1..3, 1..3]).start(),
        &ndarray_subset,
    )?;
    let data_all = array.retrieve_array_subset_ndarray::<String>(&subset_all)?;
    println!("store_array_subset [1..3, 1..3]:\nndarray::ArrayD<String>\n{data_all}");

    // Retrieve bytes directly, convert into a single string allocation, create a &str ndarray
    // TODO: Add a convenience function for this?
    let data_all = array.retrieve_array_subset(&subset_all)?;
    let (bytes, offsets) = data_all.into_variable()?.into_parts();
    let string = String::from_utf8(bytes.into_owned())?;
    let elements = offsets
        .iter()
        .tuple_windows()
        .map(|(&curr, &next)| &string[curr..next])
        .collect::<Vec<&str>>();
    let ndarray = ArrayD::<&str>::from_shape_vec(subset_all.shape_usize(), elements)?;
    println!("ndarray::ArrayD<&str>:\n{ndarray}");

    Ok(())
}

fn main() {
    if let Err(err) = array_write_read() {
        println!("{:?}", err);
    }
}

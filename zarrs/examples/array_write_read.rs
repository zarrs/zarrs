#![allow(missing_docs)]

use ndarray::ArrayD;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::storage_adapter::usage_log::UsageLogStorageAdapter;

fn array_write_read() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use zarrs::array::{ArraySubset, ZARR_NAN_F32, data_type};
    use zarrs::node::Node;
    use zarrs::storage::store;

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
        vec![8, 8], // array shape
        vec![4, 4], // regular chunk shape
        data_type::float32(),
        ZARR_NAN_F32,
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
    (0..2).into_par_iter().try_for_each(|i| {
        let chunk_indices: Vec<u64> = vec![0, i];
        let chunk_subset = array.chunk_grid().subset(&chunk_indices)?.ok_or_else(|| {
            zarrs::array::ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec())
        })?;
        array.store_chunk(
            &chunk_indices,
            vec![i as f32 * 0.1; chunk_subset.num_elements() as usize],
        )
    })?;

    let subset_all = array.subset_all();
    let data_all: ArrayD<f32> = array.retrieve_array_subset(&subset_all)?;
    println!("store_chunk [0, 0] and [0, 1]:\n{data_all:+4.1}\n");

    // Store multiple chunks
    array.store_chunks(
        &ArraySubset::new_with_ranges(&[1..2, 0..2]),
        &[
            //
            1.0f32, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1,
            //
            1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1,
        ],
    )?;
    let data_all: ArrayD<f32> = array.retrieve_array_subset(&subset_all)?;
    println!("store_chunks [1..2, 0..2]:\n{data_all:+4.1}\n");

    // Write a subset spanning multiple chunks, including updating chunks already written
    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[3..6, 3..6]),
        &[-3.3f32, -3.4, -3.5, -4.3, -4.4, -4.5, -5.3, -5.4, -5.5],
    )?;
    let data_all: ArrayD<f32> = array.retrieve_array_subset(&subset_all)?;
    println!("store_array_subset [3..6, 3..6]:\n{data_all:+4.1}\n");

    // Store array subset
    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[0..8, 6..7]),
        &[-0.6f32, -1.6, -2.6, -3.6, -4.6, -5.6, -6.6, -7.6],
    )?;
    let data_all: ArrayD<f32> = array.retrieve_array_subset(&subset_all)?;
    println!("store_array_subset [0..8, 6..7]:\n{data_all:+4.1}\n");

    // Store chunk subset
    array.store_chunk_subset(
        // chunk indices
        &[1, 1],
        // subset within chunk
        &ArraySubset::new_with_ranges(&[3..4, 0..4]),
        &[-7.4f32, -7.5, -7.6, -7.7],
    )?;
    let data_all: ArrayD<f32> = array.retrieve_array_subset(&subset_all)?;
    println!("store_chunk_subset [3..4, 0..4] of chunk [1, 1]:\n{data_all:+4.1}\n");

    // Erase a chunk
    array.erase_chunk(&[0, 0])?;
    let data_all: ArrayD<f32> = array.retrieve_array_subset(&subset_all)?;
    println!("erase_chunk [0, 0]:\n{data_all:+4.1}\n");

    // Read a chunk
    let chunk_indices = vec![0, 1];
    let data_chunk: ArrayD<f32> = array.retrieve_chunk(&chunk_indices)?;
    println!("retrieve_chunk [0, 1]:\n{data_chunk:+4.1}\n");

    // Read chunks
    let chunks = ArraySubset::new_with_ranges(&[0..2, 1..2]);
    let data_chunks: ArrayD<f32> = array.retrieve_chunks(&chunks)?;
    println!("retrieve_chunks [0..2, 1..2]:\n{data_chunks:+4.1}\n");

    // Retrieve an array subset
    let subset = ArraySubset::new_with_ranges(&[2..6, 3..5]); // the center 4x2 region
    let data_subset: ArrayD<f32> = array.retrieve_array_subset(&subset)?;
    println!("retrieve_array_subset [2..6, 3..5]:\n{data_subset:+4.1}\n");

    // Show the hierarchy
    let node = Node::open(&store, "/").unwrap();
    let tree = node.hierarchy_tree();
    println!("hierarchy_tree:\n{}", tree);

    Ok(())
}

fn main() {
    if let Err(err) = array_write_read() {
        println!("{:?}", err);
    }
}

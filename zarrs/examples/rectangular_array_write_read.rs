#![allow(missing_docs)]

use std::sync::Arc;

use zarrs::metadata::v3::MetadataV3;
use zarrs::{
    array::chunk_grid::RectangularChunkGridConfiguration,
    storage::{
        storage_adapter::usage_log::UsageLogStorageAdapter, ReadableWritableListableStorage,
    },
};

fn rectangular_array_write_read() -> Result<(), Box<dyn std::error::Error>> {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    use zarrs::{
        array::{codec, DataType, ZARR_NAN_F32},
        array_subset::ArraySubset,
        node::Node,
        storage::store,
    };

    // Create a store
    // let path = tempfile::TempDir::new()?;
    // let mut store: ReadableWritableListableStorage =
    //     Arc::new(zarrs::filesystem::FilesystemStore::new(path.path())?);
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
        MetadataV3::new_with_configuration(
            "rectangular",
            RectangularChunkGridConfiguration {
                chunk_shape: vec![[1, 2, 3, 2].try_into()?, 4.try_into()?], // chunk sizes
            },
        ),
        DataType::Float32,
        ZARR_NAN_F32,
    )
    .bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(codec::GzipCodec::new(5)?),
    ])
    .dimension_names(["y", "x"].into())
    // .storage_transformers(vec![].into())
    .build(store.clone(), array_path)?;

    // Write array metadata to store
    array.store_metadata()?;

    // Write some chunks (in parallel)
    (0..4).into_par_iter().try_for_each(|i| {
        let chunk_grid = array.chunk_grid();
        let chunk_indices = vec![i, 0];
        if let Some(chunk_shape) = chunk_grid.chunk_shape(&chunk_indices)? {
            let chunk_array = ndarray::ArrayD::<f32>::from_elem(
                chunk_shape
                    .iter()
                    .map(|u| u.get() as usize)
                    .collect::<Vec<_>>(),
                i as f32,
            );
            array.store_chunk_ndarray(&chunk_indices, &chunk_array)
        } else {
            Err(zarrs::array::ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ))
        }
    })?;

    println!(
        "The array metadata is:\n{}\n",
        array.metadata().to_string_pretty()
    );

    // Write a subset spanning multiple chunks, including updating chunks already written
    array.store_array_subset_ndarray(
        &[3, 3], // start
        &ndarray::ArrayD::<f32>::from_shape_vec(
            vec![3, 3],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )?,
    )?;

    // Store elements directly, in this case set the 7th column to 123.0
    array.store_array_subset_elements::<f32>(
        &ArraySubset::new_with_ranges(&[0..8, 6..7]),
        &[123.0; 8],
    )?;

    // Store elements directly in a chunk, in this case set the last row of the bottom right chunk
    array.store_chunk_subset_elements::<f32>(
        // chunk indices
        &[3, 1],
        // subset within chunk
        &ArraySubset::new_with_ranges(&[1..2, 0..4]),
        &[-4.0; 4],
    )?;

    // Read the whole array
    let data_all = array.retrieve_array_subset_ndarray::<f32>(&array.subset_all())?;
    println!("The whole array is:\n{data_all}\n");

    // Read a chunk back from the store
    let chunk_indices = vec![1, 0];
    let data_chunk = array.retrieve_chunk_ndarray::<f32>(&chunk_indices)?;
    println!("Chunk [1,0] is:\n{data_chunk}\n");

    // Read the central 4x2 subset of the array
    let subset_4x2 = ArraySubset::new_with_ranges(&[2..6, 3..5]); // the center 4x2 region
    let data_4x2 = array.retrieve_array_subset_ndarray::<f32>(&subset_4x2)?;
    println!("The middle 4x2 subset is:\n{data_4x2}\n");

    // Show the hierarchy
    let node = Node::open(&store, "/").unwrap();
    let tree = node.hierarchy_tree();
    println!("The Zarr hierarchy tree is:\n{tree}");

    Ok(())
}

fn main() {
    if let Err(err) = rectangular_array_write_read() {
        println!("{:?}", err);
    }
}

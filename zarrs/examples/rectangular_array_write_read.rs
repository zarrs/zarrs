#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;

use ndarray::ArrayD;

use zarrs::array::chunk_grid::RectangularChunkGridConfiguration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::storage_adapter::usage_log::UsageLogStorageAdapter;

#[allow(clippy::too_many_lines)]
fn rectangular_array_write_read() -> Result<(), Box<dyn std::error::Error>> {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    use zarrs::array::{ArraySubset, ZARR_NAN_F32, codec, data_type};
    use zarrs::node::Node;
    use zarrs::storage::store;

    // Create a store
    // let path = tempfile::TempDir::new()?;
    // let mut store: ReadableWritableListableStorage =
    //     Arc::new(zarrs::filesystem::FilesystemStore::new(path.path())?);
    let mut store: ReadableWritableListableStorage = Arc::new(store::MemoryStore::new());
    if let Some(arg1) = std::env::args().collect::<Vec<_>>().get(1)
        && arg1 == "--usage-log"
    {
        let log_writer = Arc::new(std::sync::Mutex::new(
            // std::io::BufWriter::new(
            std::io::stdout(),
            //    )
        ));
        store = Arc::new(UsageLogStorageAdapter::new(store, log_writer, || {
            chrono::Utc::now().format("[%T%.3f] ").to_string()
        }));
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
                chunk_shape: vec![
                    vec![
                        NonZeroU64::new(1).unwrap(),
                        NonZeroU64::new(2).unwrap(),
                        NonZeroU64::new(3).unwrap(),
                        NonZeroU64::new(2).unwrap(),
                    ]
                    .into(),
                    NonZeroU64::new(4).unwrap().into(),
                ], // chunk sizes
            },
        ),
        data_type::float32(),
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
    (0..4).into_par_iter().try_for_each(|i: u8| {
        let chunk_grid = array.chunk_grid();
        let chunk_indices = vec![u64::from(i), 0];
        if let Some(chunk_shape) = chunk_grid.chunk_shape(&chunk_indices)? {
            let chunk_array = ndarray::ArrayD::<f32>::from_elem(
                chunk_shape
                    .iter()
                    .map(|u| u.get().try_into().unwrap())
                    .collect::<Vec<_>>(),
                f32::from(i),
            );
            array.store_chunk(&chunk_indices, chunk_array)
        } else {
            Err(zarrs::array::ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.clone(),
            ))
        }
    })?;

    println!(
        "The array metadata is:\n{}\n",
        array.metadata().to_string_pretty()
    );

    // Write a subset spanning multiple chunks, including updating chunks already written
    array.store_array_subset(
        &[3..6, 3..6], // start
        ndarray::ArrayD::<f32>::from_shape_vec(
            vec![3, 3],
            vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )?,
    )?;

    // Store elements directly, in this case set the 7th column to 123.0
    array.store_array_subset(&[0..8, 6..7], &[123.0f32; 8])?;

    // Store elements directly in a chunk, in this case set the last row of the bottom right chunk
    array.store_chunk_subset(
        // chunk indices
        &[3, 1],
        // subset within chunk
        &[1..2, 0..4],
        &[-4.0f32; 4],
    )?;

    // Read the whole array
    let data_all: ArrayD<f32> = array.retrieve_array_subset(&array.subset_all())?;
    println!("The whole array is:\n{data_all}\n");

    // Read a chunk back from the store
    let chunk_indices = vec![1, 0];
    let data_chunk: ArrayD<f32> = array.retrieve_chunk(&chunk_indices)?;
    println!("Chunk [1,0] is:\n{data_chunk}\n");

    // Read the central 4x2 subset of the array
    let subset_4x2 = ArraySubset::new_with_ranges(&[2..6, 3..5]); // the center 4x2 region
    let data_4x2: ArrayD<f32> = array.retrieve_array_subset(&subset_4x2)?;
    println!("The middle 4x2 subset is:\n{data_4x2}\n");

    // Show the hierarchy
    let node = Node::open(&store, "/").unwrap();
    let tree = node.hierarchy_tree();
    println!("The Zarr hierarchy tree is:\n{tree}");

    Ok(())
}

fn main() {
    if let Err(err) = rectangular_array_write_read() {
        println!("{err:?}");
    }
}

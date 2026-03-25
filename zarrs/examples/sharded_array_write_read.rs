#![allow(missing_docs)]

use itertools::Itertools;
use ndarray::ArrayD;
use zarrs::array::bytes_to_ndarray;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::storage_adapter::usage_log::UsageLogStorageAdapter;
use zarrs_codec::CodecOptions;

fn sharded_array_write_read() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    use zarrs::array::{ArraySubset, codec, data_type};
    use zarrs::node::Node;
    use zarrs::storage::store;

    // Create a store
    // let path = tempfile::TempDir::new()?;
    // let mut store: ReadableWritableListableStorage =
    //     Arc::new(zarrs::filesystem::FilesystemStore::new(path.path())?);
    // let mut store: ReadableWritableListableStorage = Arc::new(
    //     zarrs::filesystem::FilesystemStore::new("zarrs/tests/data/sharded_array_write_read.zarr")?,
    // );
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

    // Create an array
    let array_path = "/group/array";
    let subchunk_shape = vec![4, 4];
    let array = zarrs::array::ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 8], // chunk (shard) shape
        data_type::uint16(),
        0u16,
    )
    .subchunk_shape(subchunk_shape.clone())
    .bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(codec::GzipCodec::new(5)?),
    ])
    .dimension_names(["y", "x"].into())
    // .storage_transformers(vec![].into())
    .build(store.clone(), array_path)?;

    // Write array metadata to store
    array.store_metadata()?;

    // The array metadata is
    println!(
        "The array metadata is:\n{}\n",
        array.metadata().to_string_pretty()
    );

    // Use default codec options (concurrency etc)
    let options = CodecOptions::default();

    // Write some shards (in parallel)
    (0..2).into_par_iter().try_for_each(|s| {
        let chunk_grid = array.chunk_grid();
        let chunk_indices = vec![s, 0];
        if let Some(chunk_shape) = chunk_grid.chunk_shape(&chunk_indices)? {
            let chunk_array = ndarray::ArrayD::<u16>::from_shape_fn(
                chunk_shape
                    .iter()
                    .map(|u| u.get() as usize)
                    .collect::<Vec<_>>(),
                |ij| {
                    (s * chunk_shape[0].get() * chunk_shape[1].get()
                        + ij[0] as u64 * chunk_shape[1].get()
                        + ij[1] as u64) as u16
                },
            );
            array.store_chunk(&chunk_indices, chunk_array)
        } else {
            Err(zarrs::array::ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ))
        }
    })?;

    // Read the whole array
    let data_all: ArrayD<u16> = array.retrieve_array_subset(&array.subset_all())?;
    println!("The whole array is:\n{data_all}\n");

    // Read a shard back from the store
    let shard_indices = vec![1, 0];
    let data_shard: ArrayD<u16> = array.retrieve_chunk(&shard_indices)?;
    println!("Shard [1,0] is:\n{data_shard}\n");

    // Read a subchunk from the store
    let subset_chunk_1_0 = ArraySubset::new_with_ranges(&[4..8, 0..4]);
    let data_chunk: ArrayD<u16> = array.retrieve_array_subset(&subset_chunk_1_0)?;
    println!("Chunk [1,0] is:\n{data_chunk}\n");

    // Read the central 4x2 subset of the array
    let subset_4x2 = ArraySubset::new_with_ranges(&[2..6, 3..5]); // the center 4x2 region
    let data_4x2: ArrayD<u16> = array.retrieve_array_subset(&subset_4x2)?;
    println!("The middle 4x2 subset is:\n{data_4x2}\n");

    // Decode subchunks
    // In some cases, it might be preferable to decode subchunks in a shard directly.
    // If using the partial decoder, then the shard index will only be read once from the store.
    let partial_decoder = array.partial_decoder(&[0, 0])?;
    println!("Decoded subchunks:");
    for subchunk_subset in [
        ArraySubset::new_with_start_shape(vec![0, 0], subchunk_shape.clone())?,
        ArraySubset::new_with_start_shape(vec![0, 4], subchunk_shape.clone())?,
    ] {
        println!("{subchunk_subset}");
        let decoded_subchunk_bytes = partial_decoder.partial_decode(&subchunk_subset, &options)?;
        let ndarray = bytes_to_ndarray::<u16>(
            &subchunk_shape,
            decoded_subchunk_bytes.into_fixed()?.into_owned(),
        )?;
        println!("{ndarray}\n");
    }

    // Show the hierarchy
    let node = Node::open(&store, "/").unwrap();
    let tree = node.hierarchy_tree();
    println!("The Zarr hierarchy tree is:\n{}", tree);

    println!(
        "The keys in the store are:\n[{}]",
        store.list().unwrap_or_default().iter().format(", ")
    );

    Ok(())
}

fn main() {
    if let Err(err) = sharded_array_write_read() {
        println!("{:?}", err);
    }
}

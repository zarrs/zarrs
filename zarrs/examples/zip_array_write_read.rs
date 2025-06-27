#![allow(missing_docs)]

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
};
use walkdir::WalkDir;

use zarrs::{
    array::{codec, Array, DataType, ZARR_NAN_F32},
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    node::Node,
    storage::storage_adapter::usage_log::UsageLogStorageAdapter,
    storage::{
        ReadableStorageTraits, ReadableWritableListableStorage, ReadableWritableStorageTraits,
        StoreKey,
    },
};
use zarrs_zip::ZipStorageAdapter;

// const ARRAY_PATH: &'static str = "/array";
const ARRAY_PATH: &str = "/";

fn write_array_to_storage<TStorage: ReadableWritableStorageTraits + ?Sized + 'static>(
    storage: Arc<TStorage>,
) -> Result<Array<TStorage>, Box<dyn std::error::Error>> {
    // Create an array
    let array = zarrs::array::ArrayBuilder::new(
        vec![8, 8], // array shape
        DataType::Float32,
        vec![4, 4], // regular chunk shape
        ZARR_NAN_F32,
    )
    .bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(codec::GzipCodec::new(5)?),
    ])
    .dimension_names(["y", "x"].into())
    // .storage_transformers(vec![].into())
    .build(storage, ARRAY_PATH)?;

    // Write array metadata to store
    array.store_metadata()?;

    // Write some chunks (in parallel)
    let _ = (0..2)
        // .into_par_iter()
        .try_for_each(|i| {
            let chunk_grid = array.chunk_grid();
            let chunk_indices: Vec<u64> = vec![i, 0];
            if let Some(chunk_subset) = chunk_grid.subset(&chunk_indices, array.shape())? {
                array.store_chunk_elements(
                    &chunk_indices,
                    &vec![i as f32; chunk_subset.num_elements() as usize],
                )
                // let chunk_shape = chunk_grid.chunk_shape(&chunk_indices, &array.shape())?;
                // let chunk_array = ndarray::ArrayD::<f32>::from_elem(chunk_shape.clone(), i as f32);
                // array.store_chunk_ndarray(&chunk_indices, &chunk_array.view())
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
    array.store_array_subset_elements::<f32>(
        &ArraySubset::new_with_ranges(&[3..6, 3..6]),
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )?;

    // Store elements directly, in this case set the 7th column to 123.0
    array.store_array_subset_elements::<f32>(
        &ArraySubset::new_with_ranges(&[0..8, 6..7]),
        &[123.0; 8],
    )?;

    // Store elements directly in a chunk, in this case set the last row of the bottom right chunk
    array.store_chunk_subset_elements::<f32>(
        // chunk indices
        &[1, 1],
        // subset within chunk
        &ArraySubset::new_with_ranges(&[3..4, 0..4]),
        &[-4.0; 4],
    )?;

    Ok(array)
}

fn read_array_from_store<TStorage: ReadableStorageTraits + 'static>(
    array: Array<TStorage>,
) -> Result<(), Box<dyn std::error::Error>> {
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

    Ok(())
}

// https://github.com/zip-rs/zip/blob/master/examples/write_dir.rs
fn zip_dir(
    it: &mut dyn Iterator<Item = walkdir::DirEntry>,
    prefix: &str,
    writer: File,
    method: zip::CompressionMethod,
) -> zip::result::ZipResult<()> {
    let mut zip = zip::ZipWriter::new(writer);
    let options = zip::write::SimpleFileOptions::default().compression_method(method);
    let mut buffer = Vec::new();
    for entry in it {
        let path = entry.path();
        let name = path.strip_prefix(Path::new(prefix)).unwrap();
        if path.is_file() {
            println!("Storing file {name:?} <- {path:?}");
            #[allow(deprecated)]
            zip.start_file_from_path(name, options)?;
            let mut f = File::open(path)?;
            f.read_to_end(&mut buffer)?;
            zip.write_all(&buffer)?;
            buffer.clear();
        } else if !name.as_os_str().is_empty() {
            println!("Storing dir {name:?} <- {path:?}");
            #[allow(deprecated)]
            zip.add_directory_from_path(name, options)?;
        }
    }
    zip.finish()?;
    Result::Ok(())
}

fn zip_array_write_read() -> Result<(), Box<dyn std::error::Error>> {
    // Create a store
    let path = tempfile::TempDir::new()?;
    let mut zarr_dir = path.path().to_path_buf();
    zarr_dir.push("hierarchy.zarr");
    let mut store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&zarr_dir)?);
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

    // Write the array to the store
    write_array_to_storage(store.clone())?;

    // Write the store to zip
    let mut path_zip = path.path().to_path_buf();
    path_zip.push("zarr_array.zip");
    let file = File::create(&path_zip).unwrap();
    zip_dir(
        &mut WalkDir::new(&zarr_dir).into_iter().filter_map(|e| e.ok()),
        zarr_dir.to_str().unwrap(),
        file,
        zip::CompressionMethod::Stored,
    )?;
    println!("Created zip {path_zip:?} containing {:?}\n", zarr_dir);

    let zip_key = StoreKey::new("zarr_array.zip")?;
    println!(
        "Create a ZipStorageAdapter for store at {:?} with {}",
        path.path(),
        zip_key
    );
    let store = Arc::new(FilesystemStore::new(&path.path())?);
    let store = Arc::new(ZipStorageAdapter::new(store, zip_key)?);
    let array = Array::open(store.clone(), ARRAY_PATH)?;
    read_array_from_store(array)?;

    // Show the hierarchy
    let node = Node::open(&store, "/").unwrap();
    let tree = node.hierarchy_tree();
    println!("The Zarr hierarchy tree is:\n{}", tree);

    Ok(())
}

fn main() {
    if let Err(err) = zip_array_write_read() {
        println!("{:?}", err);
    }
}

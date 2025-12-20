#![allow(missing_docs)]

use futures::TryStreamExt;
use zarrs::storage::{
    AsyncReadableWritableListableStorage, storage_adapter::usage_log::UsageLogStorageAdapter,
};

async fn async_array_write_read() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use futures::StreamExt;
    use zarrs::{
        array::{DataType, ZARR_NAN_F32},
        array_subset::ArraySubset,
        node::Node,
    };

    // Create a store
    let mut store: AsyncReadableWritableListableStorage = Arc::new(
        zarrs_object_store::AsyncObjectStore::new(object_store::memory::InMemory::new()),
    );
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
        .async_store_metadata()
        .await?;

    // Create a group with attributes
    let group_path = "/group";
    let mut group = zarrs::group::GroupBuilder::new().build(store.clone(), group_path)?;
    group
        .attributes_mut()
        .insert("foo".into(), serde_json::Value::String("bar".into()));
    group.async_store_metadata().await?;

    println!(
        "The group metadata is:\n{}\n",
        group.metadata().to_string_pretty()
    );

    // Create an array
    let array_path = "/group/array";
    let array = zarrs::array::ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 4], // regular chunk shape
        DataType::Float32,
        ZARR_NAN_F32,
    )
    // .bytes_to_bytes_codecs(vec![]) // uncompressed
    .dimension_names(["y", "x"].into())
    // .storage_transformers(vec![].into())
    .build_arc(store.clone(), array_path)?;

    // Write array metadata to store
    array.async_store_metadata().await?;

    println!(
        "The array metadata is:\n{}\n",
        array.metadata().to_string_pretty()
    );

    // Write some chunks
    let store_chunk = |i: u64| {
        let array = array.clone();
        async move {
            let chunk_indices: Vec<u64> = vec![0, i];
            let chunk_subset = array.chunk_grid().subset(&chunk_indices)?.ok_or_else(|| {
                zarrs::array::ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec())
            })?;
            array
                .async_store_chunk_elements(
                    &chunk_indices,
                    &vec![i as f32 * 0.1; chunk_subset.num_elements() as usize],
                )
                .await
        }
    };
    futures::stream::iter(0..2)
        .map(Ok)
        .try_for_each_concurrent(None, store_chunk)
        .await?;

    let subset_all = array.subset_all();
    let data_all = array
        .async_retrieve_array_subset_ndarray::<f32>(&subset_all)
        .await?;
    println!("async_store_chunk [0, 0] and [0, 1]:\n{data_all:+4.1}\n");

    // Store multiple chunks
    array
        .async_store_chunks_elements::<f32>(
            &ArraySubset::new_with_ranges(&[1..2, 0..2]),
            &[
                //
                1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1,
                //
                1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1,
            ],
        )
        .await?;
    let data_all = array
        .async_retrieve_array_subset_ndarray::<f32>(&subset_all)
        .await?;
    println!("async_store_chunks [1..2, 0..2]:\n{data_all:+4.1}\n");

    // Write a subset spanning multiple chunks, including updating chunks already written
    array
        .async_store_array_subset_elements::<f32>(
            &ArraySubset::new_with_ranges(&[3..6, 3..6]),
            &[-3.3, -3.4, -3.5, -4.3, -4.4, -4.5, -5.3, -5.4, -5.5],
        )
        .await?;
    let data_all = array
        .async_retrieve_array_subset_ndarray::<f32>(&subset_all)
        .await?;
    println!("async_store_array_subset [3..6, 3..6]:\n{data_all:+4.1}\n");

    // Store array subset
    array
        .async_store_array_subset_elements::<f32>(
            &ArraySubset::new_with_ranges(&[0..8, 6..7]),
            &[-0.6, -1.6, -2.6, -3.6, -4.6, -5.6, -6.6, -7.6],
        )
        .await?;
    let data_all = array
        .async_retrieve_array_subset_ndarray::<f32>(&subset_all)
        .await?;
    println!("async_store_array_subset [0..8, 6..7]:\n{data_all:+4.1}\n");

    // Store chunk subset
    array
        .async_store_chunk_subset_elements::<f32>(
            // chunk indices
            &[1, 1],
            // subset within chunk
            &ArraySubset::new_with_ranges(&[3..4, 0..4]),
            &[-7.4, -7.5, -7.6, -7.7],
        )
        .await?;
    let data_all = array
        .async_retrieve_array_subset_ndarray::<f32>(&subset_all)
        .await?;
    println!("async_store_chunk_subset [3..4, 0..4] of chunk [1, 1]:\n{data_all:+4.1}\n");

    // Erase a chunk
    array.async_erase_chunk(&[0, 0]).await?;
    let data_all = array
        .async_retrieve_array_subset_ndarray::<f32>(&subset_all)
        .await?;
    println!("async_erase_chunk [0, 0]:\n{data_all:+4.1}\n");

    // Read a chunk
    let chunk_indices = vec![0, 1];
    let data_chunk = array
        .async_retrieve_chunk_ndarray::<f32>(&chunk_indices)
        .await?;
    println!("async_retrieve_chunk [0, 1]:\n{data_chunk:+4.1}\n");

    // Read chunks
    let chunks = ArraySubset::new_with_ranges(&[0..2, 1..2]);
    let data_chunks = array.async_retrieve_chunks_ndarray::<f32>(&chunks).await?;
    println!("async_retrieve_chunks [0..2, 1..2]:\n{data_chunks:+4.1}\n");

    // Retrieve an array subset
    let subset = ArraySubset::new_with_ranges(&[2..6, 3..5]); // the center 4x2 region
    let data_subset = array
        .async_retrieve_array_subset_ndarray::<f32>(&subset)
        .await?;
    println!("async_retrieve_array_subset [2..6, 3..5]:\n{data_subset:+4.1}\n");

    // Show the hierarchy
    let node = Node::async_open(store, "/").await.unwrap();
    let tree = node.hierarchy_tree();
    println!("hierarchy_tree:\n{}", tree);

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(err) = async_array_write_read().await {
        println!("{:?}", err);
    }
}

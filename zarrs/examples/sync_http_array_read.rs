#![allow(missing_docs)]

use std::sync::Arc;

use zarrs::{
    array::Array,
    array_subset::ArraySubset,
    storage::{
        storage_adapter::{
            async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter},
            usage_log::UsageLogStorageAdapter,
        },
        ReadableStorage,
    },
};

struct TokioBlockOn(tokio::runtime::Runtime);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

enum Backend {
    // OpenDAL,
    ObjectStore,
}

fn http_array_read(backend: Backend) -> Result<(), Box<dyn std::error::Error>> {
    const HTTP_URL: &str =
        "https://raw.githubusercontent.com/zarrs/zarrs/main/zarrs/tests/data/array_write_read.zarr";
    const ARRAY_PATH: &str = "/group/array";

    // Create a HTTP store
    // let mut store: ReadableStorage = Arc::new(store::HTTPStore::new(HTTP_URL)?);
    let block_on = TokioBlockOn(tokio::runtime::Runtime::new()?);
    let mut store: ReadableStorage = match backend {
        // Backend::OpenDAL => {
        //     let builder = opendal::services::Http::default().endpoint(HTTP_URL);
        //     let operator = opendal::Operator::new(builder)?.finish();
        //     let store = Arc::new(zarrs_opendal::AsyncOpendalStore::new(operator));
        //     Arc::new(AsyncToSyncStorageAdapter::new(store, block_on))
        // }
        Backend::ObjectStore => {
            let options = object_store::ClientOptions::new().with_allow_http(true);
            let store = object_store::http::HttpBuilder::new()
                .with_url(HTTP_URL)
                .with_client_options(options)
                .build()?;
            let store = Arc::new(zarrs_object_store::AsyncObjectStore::new(store));
            Arc::new(AsyncToSyncStorageAdapter::new(store, block_on))
        }
    };
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

    // Init the existing array, reading metadata
    let array = Array::open(store, ARRAY_PATH)?;

    println!(
        "The array metadata is:\n{}\n",
        array.metadata().to_string_pretty()
    );

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("------------ object_store backend ------------");
    http_array_read(Backend::ObjectStore)?;
    // println!("------------   opendal backend    ------------");
    // http_array_read(Backend::OpenDAL)?;
    Ok(())
}

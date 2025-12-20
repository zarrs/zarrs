#![allow(missing_docs)]
#![cfg(all(feature = "async"))]

use std::sync::Arc;

use serde_json::json;
use zarrs::storage::{
    ReadableWritableStorage,
    storage_adapter::async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter},
};
use zarrs::{
    array::{DataType, ZARR_NAN_F32},
    array_subset::ArraySubset,
};

pub struct TokioBlockOn(pub tokio::runtime::Runtime);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

fn readable_writable_store() -> ReadableWritableStorage {
    let block_on = TokioBlockOn(tokio::runtime::Runtime::new().unwrap());
    let store = object_store::memory::InMemory::new();
    let async_store = Arc::new(zarrs_object_store::AsyncObjectStore::new(store));
    Arc::new(AsyncToSyncStorageAdapter::new(async_store, block_on))
}

#[test]
#[cfg_attr(miri, ignore)]
fn array_read_and_write_async_storage_adapter() {
    const GROUP_PATH: &str = "/group";
    const ARRAY_PATH: &str = "/group/array";
    let store = readable_writable_store();

    // Create a group
    let mut group = zarrs::group::GroupBuilder::new()
        .build(store.clone(), GROUP_PATH)
        .unwrap();
    // Update group metadata
    group
        .attributes_mut()
        .insert("foo".into(), serde_json::Value::String("bar".into()));
    // Write group metadata to store
    group.store_metadata().unwrap();
    assert_eq!(group.attributes().get("foo"), Some(&json!("bar")));

    // Create an array
    let array =
        zarrs::array::ArrayBuilder::new(vec![8, 8], vec![4, 4], DataType::Float32, ZARR_NAN_F32)
            .dimension_names(["y", "x"].into())
            .build(store.clone(), ARRAY_PATH)
            .unwrap();
    array.store_metadata().unwrap();
    assert_eq!(array.shape(), &[8, 8]);

    array
        .store_chunk_elements::<f32>(
            &[0, 0],
            &[
                0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3,
            ],
        )
        .unwrap();

    let subset = ArraySubset::new_with_ranges(&[2..4, 2..4]);
    let data = array.retrieve_array_subset_ndarray::<f32>(&subset).unwrap();
    assert_eq!(data, ndarray::array![[2.2, 2.3], [3.2, 3.3]].into_dyn());
}

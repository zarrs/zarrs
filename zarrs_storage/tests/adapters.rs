#![allow(missing_docs)]

use std::{error::Error, sync::Arc};

use zarrs_storage::{
    storage_adapter::{
        async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter},
        sync_to_async::{SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter},
    },
    store::MemoryStore,
};

struct TokioSpawnBlocking;

impl SyncToAsyncSpawnBlocking for TokioSpawnBlocking {
    fn spawn_blocking<F, R>(&self, f: F) -> impl std::future::Future<Output = R> + Send
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        async move { tokio::task::spawn_blocking(f).await.unwrap() }
    }
}

struct TokioBlockOn(tokio::runtime::Runtime);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

#[tokio::test]
async fn sync_to_async_memory_store() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(MemoryStore::new());
    let store = Arc::new(SyncToAsyncStorageAdapter::new(store, TokioSpawnBlocking));
    zarrs_storage::store_test::async_store_write(&store).await?;
    zarrs_storage::store_test::async_store_read(&store).await?;
    zarrs_storage::store_test::async_store_list(&store).await?;
    Ok(())
}

#[test]
fn async_to_sync_memory_store() {
    let store = Arc::new(MemoryStore::new());
    let store = Arc::new(SyncToAsyncStorageAdapter::new(store, TokioSpawnBlocking));
    let store = Arc::new(AsyncToSyncStorageAdapter::new(
        store,
        TokioBlockOn(tokio::runtime::Runtime::new().unwrap()),
    ));
    zarrs_storage::store_test::store_write(&store).unwrap();
    zarrs_storage::store_test::store_read(&store).unwrap();
    zarrs_storage::store_test::store_list(&store).unwrap();
}

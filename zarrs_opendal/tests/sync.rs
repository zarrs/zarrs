#![allow(missing_docs)]

use opendal::Operator;
use zarrs_opendal::AsyncOpendalStore;
use zarrs_storage::storage_adapter::async_to_sync::{
    AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
};

#[derive(Clone)]
struct TokioBlockOn(tokio::runtime::Handle);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

use std::{error::Error, sync::Arc};

#[test]
fn memory() -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Runtime::new()?;
    let block_on = TokioBlockOn(runtime.handle().clone());
    let builder = opendal::services::Memory::default();
    let op = Operator::new(builder)?.finish();
    let store_async = Arc::new(AsyncOpendalStore::new(op));
    let store_sync = AsyncToSyncStorageAdapter::new(store_async, block_on);
    zarrs_storage::store_test::store_write(&store_sync)?;
    zarrs_storage::store_test::store_read(&store_sync)?;
    zarrs_storage::store_test::store_list(&store_sync)?;
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem() -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Runtime::new()?;
    let block_on = TokioBlockOn(runtime.handle().clone());
    let path = tempfile::TempDir::new()?;
    let builder = opendal::services::Fs::default().root(&path.path().to_string_lossy());
    let op = Operator::new(builder)?.finish();
    let store_async = Arc::new(AsyncOpendalStore::new(op));
    let store_sync = AsyncToSyncStorageAdapter::new(store_async, block_on);
    zarrs_storage::store_test::store_write(&store_sync)?;
    zarrs_storage::store_test::store_read(&store_sync)?;
    zarrs_storage::store_test::store_list(&store_sync)?;
    Ok(())
}

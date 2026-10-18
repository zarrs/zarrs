#![allow(missing_docs)]

use std::error::Error;
use std::sync::Arc;

use zarrs_storage::storage_adapter::async_to_sync::{
    AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
};
use zarrs_storage::storage_adapter::sync_to_async::{
    SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter,
};
use zarrs_storage::store::MemoryStore;

struct TokioSpawnBlocking;

impl SyncToAsyncSpawnBlocking for TokioSpawnBlocking {
    async fn spawn_blocking<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        tokio::task::spawn_blocking(f).await.unwrap()
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
    zarrs_storage::store_test::async_store_list_size(&store).await?;
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
    zarrs_storage::store_test::store_list_size(&store).unwrap();
}

use std::sync::atomic::{AtomicUsize, Ordering};
use zarrs_storage::byte_range::ByteRangeIterator;
use zarrs_storage::storage_adapter::usage_log::UsageLogStorageAdapter;
use zarrs_storage::{
    AsyncMaybeBytesIterator, AsyncReadableStorageTraits, Bytes, MaybeBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey,
};

struct ErrorStore(AtomicUsize);

impl ErrorStore {
    fn values(&self, count: usize) -> impl Iterator<Item = Result<Bytes, StorageError>> + '_ {
        (0..count).map(|i| {
            self.0.fetch_add(1, Ordering::SeqCst);
            if i == 1 {
                Err(StorageError::Other("second item".into()))
            } else {
                Ok(Bytes::from(i.to_string()))
            }
        })
    }
}

impl ReadableStorageTraits for ErrorStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        Ok((key.as_str() != "missing").then(|| Box::new(self.values(ranges.count())) as _))
    }

    fn size_key(&self, _: &StoreKey) -> Result<Option<u64>, StorageError> {
        Ok(Some(3))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

#[async_trait::async_trait]
impl AsyncReadableStorageTraits for ErrorStore {
    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        use futures::StreamExt;
        Ok((key.as_str() != "missing").then(|| {
            futures::stream::iter(self.values(ranges.count()))
                .then(|value| async {
                    tokio::task::yield_now().await;
                    value
                })
                .boxed()
        }))
    }

    async fn size_key(&self, _: &StoreKey) -> Result<Option<u64>, StorageError> {
        Ok(Some(3))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

#[test]
fn adapter_collection_errors_empty_and_missing() {
    let source = Arc::new(ErrorStore(AtomicUsize::new(0)));
    let adapter = AsyncToSyncStorageAdapter::new(
        source.clone(),
        TokioBlockOn(tokio::runtime::Runtime::new().unwrap()),
    );
    let key = StoreKey::new("data").unwrap();
    let ranges = || Box::new((0u64..3).map(|i| (i..i + 1).into())) as ByteRangeIterator<'static>;
    let mut results = adapter.get_partial_many(&key, ranges()).unwrap().unwrap();
    assert_eq!(results.next().unwrap().unwrap().as_ref(), b"0");
    assert_eq!(
        results.next().unwrap().unwrap_err().to_string(),
        "second item"
    );
    assert_eq!(results.next().unwrap().unwrap().as_ref(), b"2");
    assert!(results.next().is_none());
    assert_eq!(source.0.load(Ordering::SeqCst), 3);
    assert!(adapter
        .get_partial_many(&key, Box::new(std::iter::empty()))
        .unwrap()
        .unwrap()
        .next()
        .is_none());
    assert!(adapter
        .get_partial_many(&StoreKey::new("missing").unwrap(), ranges())
        .unwrap()
        .is_none());

    let log = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let usage = UsageLogStorageAdapter::new(source.clone(), log.clone(), String::new);
    source.0.store(0, Ordering::SeqCst);
    assert!(ReadableStorageTraits::get_partial_many(&usage, &key, ranges()).is_err());
    assert_eq!(source.0.load(Ordering::SeqCst), 2);
    assert_eq!(*log.lock().unwrap(), Vec::<u8>::new());
    source.0.store(0, Ordering::SeqCst);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    assert!(runtime
        .block_on(AsyncReadableStorageTraits::get_partial_many(
            &usage,
            &key,
            ranges()
        ))
        .is_err());
    assert_eq!(source.0.load(Ordering::SeqCst), 2);
    assert_eq!(*log.lock().unwrap(), Vec::<u8>::new());
}

#![allow(missing_docs)]
//! Writes through an `ArrayCached` that fail invalidate the chunks they may have changed.

use std::error::Error;
use std::sync::{Arc, Mutex};

use zarrs::array::cache::{ChunkCacheDecoded, SyncChunkCacheTraits};
use zarrs::array::{ArrayBuilder, ArrayCached, Resources, data_type};
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    CowBytes, OffsetBytesIterator, StorageError, StoreKey, StorePrefix, SyncMaybeBytesIterator,
    SyncReadableStorageTraits, SyncWritableStorageTraits,
};

type TestResult = Result<(), Box<dyn Error>>;

/// A memory store that fails to write or erase one key.
#[derive(Default)]
struct FailingStore {
    store: MemoryStore,
    failing: Mutex<Option<StoreKey>>,
}

impl FailingStore {
    fn fail(&self, key: Option<StoreKey>) {
        *self.failing.lock().unwrap() = key;
    }

    fn check(&self, key: &StoreKey) -> Result<(), StorageError> {
        if self.failing.lock().unwrap().as_ref() == Some(key) {
            Err(StorageError::Other(format!("failed to write {key}")))
        } else {
            Ok(())
        }
    }
}

impl SyncReadableStorageTraits for FailingStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<SyncMaybeBytesIterator<'a>, StorageError> {
        self.store.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.store.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.store.supports_get_partial()
    }
}

impl SyncWritableStorageTraits for FailingStore {
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        self.check(key)?;
        self.store.set(key, value)
    }

    fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        self.check(key)?;
        self.store.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.check(key)?;
        self.store.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.store.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.store.supports_set_partial()
    }
}

#[test]
fn failed_writes_invalidate_the_chunks_they_may_have_changed() -> TestResult {
    let store = Arc::new(FailingStore::default());
    let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
        .build_arc(store.clone(), "/array")?;
    let cached = ArrayCached::new(array.clone(), ChunkCacheDecoded::new_chunk_limit(2));
    let resources = Resources::default();
    // Each write fails to write chunk 1, after writing any other chunk, and invalidates the chunks it writes
    for write in 0..6 {
        cached.store_chunks(&[0..2], &[1u8, 2, 3, 4], &resources)?;
        cached.retrieve_array_subset::<Vec<u8>>(&[0..4], &resources)?;
        assert_eq!(cached.cache().len(), 2);

        store.fail(Some(array.chunk_key(&[1])?));
        let (result, invalidated) = match write {
            0 => (cached.store_chunk(&[1], &[5u8, 6], &resources), 1),
            1 => (cached.store_chunks(&[0..2], &[5u8, 6, 7, 8], &resources), 2),
            2 => (
                cached.store_array_subset(&[1..4], &[5u8, 6, 7], &resources),
                2,
            ),
            3 => (
                cached.store_partial_chunk(&[1], &[0..1], &[5u8], &resources),
                1,
            ),
            4 => (cached.erase_chunk(&[1]), 1),
            _ => (cached.erase_chunks(&[0..2], &resources), 2),
        };
        store.fail(None);
        assert!(result.is_err());
        assert_eq!(cached.cache().len(), 2 - invalidated);
        assert_eq!(
            cached.retrieve_array_subset::<Vec<u8>>(&[0..4], &resources)?,
            array.retrieve_array_subset::<Vec<u8>>(&[0..4], &resources)?
        );
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_failed_writes_invalidate_the_chunks_they_may_have_changed() -> TestResult {
    use zarrs::array::cache::{AsyncChunkCacheDecoded, AsyncChunkCacheTraits};
    use zarrs::storage::storage_adapter::sync_to_async::{
        SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter,
    };

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

    let store = Arc::new(FailingStore::default());
    let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8).build_arc(
        Arc::new(SyncToAsyncStorageAdapter::new(
            store.clone(),
            TokioSpawnBlocking,
        )),
        "/array",
    )?;
    let cached = ArrayCached::new(array.clone(), AsyncChunkCacheDecoded::new_chunk_limit(2));
    let resources = Resources::default();
    for write in 0..6 {
        cached
            .async_store_chunks(&[0..2], &[1u8, 2, 3, 4], &resources)
            .await?;
        cached
            .async_retrieve_array_subset::<Vec<u8>>(&[0..4], &resources)
            .await?;
        assert_eq!(cached.cache().len().await, 2);

        store.fail(Some(array.chunk_key(&[1])?));
        let (result, invalidated) = match write {
            0 => (
                cached.async_store_chunk(&[1], &[5u8, 6], &resources).await,
                1,
            ),
            1 => (
                cached
                    .async_store_chunks(&[0..2], &[5u8, 6, 7, 8], &resources)
                    .await,
                2,
            ),
            2 => (
                cached
                    .async_store_array_subset(&[1..4], &[5u8, 6, 7], &resources)
                    .await,
                2,
            ),
            3 => (
                cached
                    .async_store_partial_chunk(&[1], &[0..1], &[5u8], &resources)
                    .await,
                1,
            ),
            4 => (cached.async_erase_chunk(&[1]).await, 1),
            _ => (cached.async_erase_chunks(&[0..2], &resources).await, 2),
        };
        store.fail(None);
        assert!(result.is_err());
        assert_eq!(cached.cache().len().await, 2 - invalidated);
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u8>>(&[0..4], &resources)
                .await?,
            array
                .async_retrieve_array_subset::<Vec<u8>>(&[0..4], &resources)
                .await?
        );
    }
    Ok(())
}

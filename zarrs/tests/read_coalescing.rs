#![allow(missing_docs)]

use std::sync::{Arc, Mutex};

use zarrs::array::{Array, ArrayBuilder, ArraySubset, Resources, data_type};
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    MaybeBytesIterator, ReadCoalescing, ReadableStorageTraits, StorageError, StoreKey,
};
use zarrs_codec::ReadDispatch;

/// A store with a read coalescing hint that records the byte ranges of each request.
struct RecordingStore {
    inner: Arc<MemoryStore>,
    coalescing: Option<ReadCoalescing>,
    requests: Mutex<Vec<Vec<ByteRange>>>,
}

impl ReadableStorageTraits for RecordingStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let byte_ranges: Vec<ByteRange> = byte_ranges.collect();
        self.requests.lock().unwrap().push(byte_ranges.clone());
        self.inner
            .get_partial_many(key, Box::new(byte_ranges.into_iter()))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        true
    }

    fn read_coalescing(&self) -> Option<ReadCoalescing> {
        self.coalescing
    }
}

/// The number of storage requests and the elements of a read of a row of the subchunks of a shard.
fn read_subchunk_row(
    coalescing: Option<ReadCoalescing>,
    coalescing_override: Option<ReadCoalescing>,
) -> (usize, Vec<u16>) {
    // One shard of 8 x 8 subchunks of 4 x 4 elements, uncompressed
    let inner = Arc::new(MemoryStore::new());
    let array = ArrayBuilder::new(vec![32, 32], vec![32, 32], data_type::uint16(), 0u16)
        .subchunk_shape(vec![4, 4])
        .build(inner.clone(), "/array")
        .unwrap();
    array.store_metadata().unwrap();
    let elements: Vec<u16> = (0..32 * 32).collect();
    array
        .store_array_subset(&array.subset_all(), elements, &Resources::default())
        .unwrap();
    assert!(
        inner
            .get(&"array/c/0/0".try_into().unwrap())
            .unwrap()
            .is_some()
    );

    let store = Arc::new(RecordingStore {
        inner,
        coalescing,
        requests: Mutex::default(),
    });
    let array = Array::open(store.clone(), "/array").unwrap();
    store.requests.lock().unwrap().clear();
    let resources = Resources::default()
        .with_read_dispatch(Some(ReadDispatch::Pooled))
        .with_read_coalescing(coalescing_override);
    let elements: Vec<u16> = array
        .retrieve_array_subset(&ArraySubset::new_with_ranges(&[4..8, 0..32]), &resources)
        .unwrap();
    let requests = store.requests.lock().unwrap().len();
    (requests, elements)
}

#[test]
fn subchunk_reads_are_coalesced() {
    let expected: Vec<u16> = (4 * 32..8 * 32).collect();

    // The shard index, then each subchunk of the row
    let (requests, elements) = read_subchunk_row(None, None);
    assert_eq!(elements, expected);
    assert_eq!(requests, 1 + 8);

    // The shard index, then the row of subchunks (which need not be adjacent in the shard)
    let coalescing = Some(ReadCoalescing::new(4096, 1024 * 1024));
    let (requests, elements) = read_subchunk_row(coalescing, None);
    assert_eq!(elements, expected);
    assert_eq!(requests, 2);

    // Disabled by a resource override
    let (requests, _) = read_subchunk_row(coalescing, Some(ReadCoalescing::new(0, 0)));
    assert_eq!(requests, 1 + 8);

    // Enabled by a resource override
    let (requests, _) = read_subchunk_row(None, coalescing);
    assert_eq!(requests, 2);
}

#[cfg(feature = "async")]
mod asynchronous {
    use super::*;
    use zarrs::storage::store::AsyncMemoryStore;
    use zarrs::storage::{AsyncMaybeBytesIterator, AsyncReadableStorageTraits};

    /// An asynchronous store with a read coalescing hint that records the byte ranges of each request.
    struct AsyncRecordingStore {
        inner: Arc<AsyncMemoryStore>,
        coalescing: Option<ReadCoalescing>,
        requests: Mutex<Vec<Vec<ByteRange>>>,
    }

    #[async_trait::async_trait]
    impl AsyncReadableStorageTraits for AsyncRecordingStore {
        async fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: ByteRangeIterator<'a>,
        ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
            let byte_ranges: Vec<ByteRange> = byte_ranges.collect();
            self.requests.lock().unwrap().push(byte_ranges.clone());
            self.inner
                .get_partial_many(key, Box::new(byte_ranges.into_iter()))
                .await
        }

        async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.inner.size_key(key).await
        }

        fn supports_get_partial(&self) -> bool {
            true
        }

        fn read_coalescing(&self) -> Option<ReadCoalescing> {
            self.coalescing
        }
    }

    async fn read_subchunk_row(coalescing: Option<ReadCoalescing>) -> (usize, Vec<u16>) {
        let inner = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![32, 32], vec![32, 32], data_type::uint16(), 0u16)
            .subchunk_shape(vec![4, 4])
            .build(inner.clone(), "/array")
            .unwrap();
        array.async_store_metadata().await.unwrap();
        let elements: Vec<u16> = (0..32 * 32).collect();
        array
            .async_store_array_subset(&array.subset_all(), elements, &Resources::default())
            .await
            .unwrap();

        let store = Arc::new(AsyncRecordingStore {
            inner,
            coalescing,
            requests: Mutex::default(),
        });
        let array = Array::async_open(store.clone(), "/array").await.unwrap();
        store.requests.lock().unwrap().clear();
        let elements: Vec<u16> = array
            .async_retrieve_array_subset(
                &ArraySubset::new_with_ranges(&[4..8, 0..32]),
                &Resources::default(),
            )
            .await
            .unwrap();
        let requests = store.requests.lock().unwrap().len();
        (requests, elements)
    }

    #[tokio::test]
    async fn async_subchunk_reads_are_coalesced() {
        let expected: Vec<u16> = (4 * 32..8 * 32).collect();
        let (requests, elements) = read_subchunk_row(None).await;
        assert_eq!(elements, expected);
        assert_eq!(requests, 1 + 8);
        let (requests, elements) =
            read_subchunk_row(Some(ReadCoalescing::new(4096, 1024 * 1024))).await;
        assert_eq!(elements, expected);
        assert_eq!(requests, 2);
    }
}

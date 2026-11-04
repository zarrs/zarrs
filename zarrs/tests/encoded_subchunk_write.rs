#![allow(missing_docs)]
#![cfg(all(feature = "gzip", feature = "crc32c"))]
//! Storing and erasing encoded subchunks through nested sharding with bytes-to-bytes codecs, in stores with and
//! without partial writes.

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{Crc32cCodec, GzipCodec};
use zarrs::array::storage_transformer::{
    StorageTransformer, StorageTransformerChain, StorageTransformerPlugin, StorageTransformerTraits,
};
use zarrs::array::{ArrayBuilder, CowBytes, data_type};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::node::NodePath;
use zarrs::plugin::PluginCreateError;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
#[cfg(feature = "async")]
use zarrs::storage::{
    AsyncListableStorage, AsyncReadableStorage, AsyncReadableWritableStorage, AsyncWritableStorage,
};
use zarrs::storage::{
    ListableStorage, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator, ReadableStorage,
    ReadableStorageTraits, ReadableWritableStorage, ReadableWritableStorageTraits, StorageError,
    StoreKey, StorePrefix, WritableStorage, WritableStorageTraits,
};
use zarrs_storage::byte_range::ByteRangeIterator;

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// A store that does not support partial writes, wrapping a memory store by default.
struct NoPartialWriteStore<T: ?Sized = MemoryStore>(Arc<T>);

impl Default for NoPartialWriteStore {
    fn default() -> Self {
        Self(Arc::default())
    }
}

impl<T: ?Sized + ReadableStorageTraits> ReadableStorageTraits for NoPartialWriteStore<T> {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl<T: ?Sized + WritableStorageTraits> WritableStorageTraits for NoPartialWriteStore<T> {
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        self.0.set(key, value)
    }

    fn set_partial_many<'a>(
        &'a self,
        _key: &StoreKey,
        _offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        Err(StorageError::Unsupported(
            "the store does not support partial writes".to_string(),
        ))
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.0.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.0.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}

zarrs::plugin::impl_extension_aliases!(NoPartialWriteStorageTransformer,
    v3: "zarrs.test_no_partial_write", []
);

inventory::submit! {
    StorageTransformerPlugin::new::<NoPartialWriteStorageTransformer>()
}

/// A storage transformer that does not support partial writes, even if the transformed storage does.
#[derive(Debug)]
struct NoPartialWriteStorageTransformer;

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(
    all(feature = "async", target_arch = "wasm32"),
    async_trait::async_trait(?Send)
)]
impl StorageTransformerTraits for NoPartialWriteStorageTransformer {
    fn create(
        _metadata: &MetadataV3,
        _path: &NodePath,
    ) -> Result<StorageTransformer, PluginCreateError> {
        Ok(Arc::new(Self))
    }

    fn configuration(&self) -> Configuration {
        Configuration::default()
    }

    fn create_readable_transformer(
        self: Arc<Self>,
        storage: ReadableStorage,
    ) -> Result<ReadableStorage, StorageError> {
        Ok(storage)
    }

    fn create_writable_transformer(
        self: Arc<Self>,
        storage: WritableStorage,
    ) -> Result<WritableStorage, StorageError> {
        Ok(Arc::new(NoPartialWriteStore(storage)))
    }

    fn create_readable_writable_transformer(
        self: Arc<Self>,
        storage: ReadableWritableStorage,
    ) -> Result<ReadableWritableStorage, StorageError> {
        Ok(Arc::new(NoPartialWriteStore(storage)))
    }

    fn create_listable_transformer(
        self: Arc<Self>,
        storage: ListableStorage,
    ) -> Result<ListableStorage, StorageError> {
        Ok(storage)
    }

    #[cfg(feature = "async")]
    async fn create_async_readable_transformer(
        self: Arc<Self>,
        storage: AsyncReadableStorage,
    ) -> Result<AsyncReadableStorage, StorageError> {
        Ok(storage)
    }

    #[cfg(feature = "async")]
    async fn create_async_writable_transformer(
        self: Arc<Self>,
        _storage: AsyncWritableStorage,
    ) -> Result<AsyncWritableStorage, StorageError> {
        Err(StorageError::Unsupported("async".to_string()))
    }

    #[cfg(feature = "async")]
    async fn create_async_readable_writable_transformer(
        self: Arc<Self>,
        _storage: AsyncReadableWritableStorage,
    ) -> Result<AsyncReadableWritableStorage, StorageError> {
        Err(StorageError::Unsupported("async".to_string()))
    }

    #[cfg(feature = "async")]
    async fn create_async_listable_transformer(
        self: Arc<Self>,
        storage: AsyncListableStorage,
    ) -> Result<AsyncListableStorage, StorageError> {
        Ok(storage)
    }
}

/// A single chunk array with three levels of sharding, with bytes-to-bytes codecs at the outer two levels.
fn three_level_builder(location: ShardingIndexLocation) -> Result<ArrayBuilder, Box<dyn Error>> {
    let data_type = data_type::uint8();
    let mut inner = ShardingCodecBuilder::new(vec![nz(4)], &data_type);
    inner.index_location(location);
    let mut middle = ShardingCodecBuilder::new(vec![nz(8)], &data_type);
    middle
        .index_location(location)
        .array_to_bytes_codec(inner.build_arc());
    middle.bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5)?)]);
    let mut outer = ShardingCodecBuilder::new(vec![nz(16)], &data_type);
    outer
        .index_location(location)
        .array_to_bytes_codec(middle.build_arc());
    let mut builder = ArrayBuilder::new(vec![32], vec![32], data_type, 0u8);
    builder.array_to_bytes_codec(outer.build_arc());
    builder.bytes_to_bytes_codecs(vec![Arc::new(Crc32cCodec::new())]);
    Ok(builder)
}

#[test]
fn array_store_encoded_subchunks_three_levels_with_bytes_codecs() -> TestResult {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for partial_writes in [true, false] {
            let builder = three_level_builder(location)?;

            let source = builder.build(Arc::new(MemoryStore::default()), "/source")?;
            source.store_chunk(&[0], (1..=32u8).collect::<Vec<_>>())?;
            let check = |target: &dyn Fn(usize, &[u64]) -> TestResult| -> TestResult {
                for (level, indices) in [(2, [3]), (2, [6]), (1, [0]), (0, [1]), (2, [0])] {
                    target(level, &indices)?;
                }
                Ok(())
            };
            macro_rules! store_and_check {
                ($target:expr) => {{
                    let target = $target;
                    check(&|level, indices| {
                        let encoded = source
                            .retrieve_encoded_subchunk_at_level(level, indices)?
                            .expect("the subchunk is stored");
                        // SAFETY: Both arrays have the same codecs and shape.
                        unsafe {
                            target.store_encoded_subchunk_at_level(
                                level,
                                indices,
                                encoded.bytes().clone().into(),
                            )?;
                        }
                        assert_eq!(
                            target
                                .retrieve_encoded_subchunk_at_level(level, indices)?
                                .expect("the subchunk is stored"),
                            encoded
                        );
                        assert_eq!(
                            target.retrieve_subchunk_at_level::<Vec<u8>>(level, indices)?,
                            source.retrieve_subchunk_at_level::<Vec<u8>>(level, indices)?
                        );
                        Ok(())
                    })?;
                    // The subchunks stored above are in the chunk, and the rest is the fill value
                    let mut expected: Vec<u8> = (1..=32).collect();
                    expected[8..12].fill(0);
                    assert_eq!(target.retrieve_chunk::<Vec<u8>>(&[0])?, expected);
                    target.erase_subchunk_at_level(2, &[3])?;
                    expected[12..16].fill(0);
                    assert_eq!(target.retrieve_chunk::<Vec<u8>>(&[0])?, expected);
                    // Erasing every stored subchunk erases the chunk
                    for (level, indices) in [(0, [0]), (1, [3]), (1, [2])] {
                        target.erase_subchunk_at_level(level, &indices)?;
                    }
                    assert!(target.retrieve_encoded_chunk(&[0])?.is_none());
                }};
            }
            if partial_writes {
                store_and_check!(builder.build(Arc::new(MemoryStore::default()), "/target")?);
            } else {
                store_and_check!(
                    builder.build(Arc::new(NoPartialWriteStore::default()), "/target")?
                );
            }
        }
    }
    Ok(())
}

#[test]
fn array_store_encoded_subchunks_repeatedly_compacts_shards() -> TestResult {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for partial_writes in [true, false] {
            let builder = three_level_builder(location)?;
            let source = builder.build(Arc::new(MemoryStore::default()), "/source")?;
            source.store_chunk(&[0], (1..=32u8).collect::<Vec<_>>())?;
            let store: Arc<dyn ReadableWritableStorageTraits> = if partial_writes {
                Arc::new(MemoryStore::default())
            } else {
                Arc::new(NoPartialWriteStore::default())
            };
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));
            let target = builder.build(store.clone(), "/target")?;
            target.store_chunk(&[0], (1..=32u8).collect::<Vec<_>>())?;

            // Every shard is rewritten entirely, in memory or through a bytes-to-bytes codec, so none grow
            let encoded = source
                .retrieve_encoded_subchunk_at_level(2, &[3])?
                .expect("the subchunk is stored");
            let mut size = None;
            for _ in 0..4 {
                // SAFETY: Both arrays have the same codecs and shape.
                unsafe {
                    target.store_encoded_subchunk_at_level(
                        2,
                        &[3],
                        encoded.bytes().clone().into(),
                    )?;
                }
                let stored_size = target
                    .retrieve_encoded_chunk(&[0])?
                    .expect("the chunk is stored")
                    .len();
                assert_eq!(*size.get_or_insert(stored_size), stored_size);
            }
            assert_eq!(
                target.retrieve_chunk::<Vec<u8>>(&[0])?,
                (1..=32).collect::<Vec<_>>()
            );

            // Erasing a subchunk that is not stored does not write the chunk
            target.erase_subchunk_at_level(2, &[3])?;
            let (writes, keys_erased) = (store.writes(), store.keys_erased());
            target.erase_subchunk_at_level(2, &[3])?;
            assert_eq!((store.writes(), store.keys_erased()), (writes, keys_erased));
        }
    }
    Ok(())
}

#[test]
fn array_store_encoded_subchunk_without_partial_writes_through_storage_transformer() -> TestResult {
    // The store supports partial writes, but the storage transformer of the array does not
    let mut builder = three_level_builder(ShardingIndexLocation::End)?;
    let source = builder.build(Arc::new(MemoryStore::default()), "/source")?;
    source.store_chunk(&[0], (1..=32u8).collect::<Vec<_>>())?;
    builder.storage_transformers(StorageTransformerChain::new(vec![Arc::new(
        NoPartialWriteStorageTransformer,
    )]));
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let target = builder.build(store.clone(), "/target")?;
    target.store_chunk(&[0], (1..=32u8).collect::<Vec<_>>())?;

    let encoded = source
        .retrieve_encoded_subchunk_at_level(2, &[3])?
        .expect("the subchunk is stored");
    store.reset();
    // SAFETY: Both arrays have the same codecs and shape.
    unsafe {
        target.store_encoded_subchunk_at_level(2, &[3], encoded.bytes().clone().into())?;
    }
    // The chunk is updated in memory, so it is read and written once
    assert_eq!((store.reads(), store.writes()), (1, 1));
    assert_eq!(
        target.retrieve_chunk::<Vec<u8>>(&[0])?,
        (1..=32).collect::<Vec<_>>()
    );
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_array_store_encoded_subchunks_without_partial_writes() -> TestResult {
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

    let builder = three_level_builder(ShardingIndexLocation::End)?;
    let source = builder.build(Arc::new(MemoryStore::default()), "/source")?;
    source.store_chunk(&[0], (1..=32u8).collect::<Vec<_>>())?;
    let target = builder.build(
        Arc::new(SyncToAsyncStorageAdapter::new(
            Arc::new(NoPartialWriteStore::default()),
            TokioSpawnBlocking,
        )),
        "/target",
    )?;
    for (level, indices) in [(2, [3]), (1, [0]), (0, [1])] {
        let encoded = source
            .retrieve_encoded_subchunk_at_level(level, &indices)?
            .expect("the subchunk is stored");
        // SAFETY: Both arrays have the same codecs and shape.
        unsafe {
            target
                .async_store_encoded_subchunk_at_level(
                    level,
                    &indices,
                    encoded.bytes().clone().into(),
                )
                .await?;
        }
        assert_eq!(
            target
                .async_retrieve_encoded_subchunk_at_level(level, &indices)
                .await?
                .expect("the subchunk is stored"),
            encoded
        );
    }
    let mut expected = vec![0u8; 32];
    expected[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    expected[12..16].copy_from_slice(&[13, 14, 15, 16]);
    expected[16..32].copy_from_slice(&(17..=32).collect::<Vec<u8>>());
    assert_eq!(
        target.async_retrieve_chunk::<Vec<u8>>(&[0]).await?,
        expected
    );

    // Erasing every stored subchunk erases the chunk
    for (level, indices) in [(0, [1]), (1, [0]), (2, [3])] {
        target
            .async_erase_subchunk_at_level(level, &indices)
            .await?;
    }
    assert!(target.async_retrieve_encoded_chunk(&[0]).await?.is_none());
    Ok(())
}

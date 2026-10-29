#![allow(missing_docs)]
#![cfg(all(feature = "gzip", feature = "crc32c"))]
//! Storing and erasing encoded subchunks through nested sharding with bytes-to-bytes codecs, in stores with and
//! without partial writes.

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{Crc32cCodec, GzipCodec};
use zarrs::array::{ArrayBuilder, CowBytes, data_type};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    MaybeBytes, MaybeBytesIterator, OffsetBytesIterator, ReadableStorageTraits, StorageError,
    StoreKey, StorePrefix, WritableStorageTraits,
};
use zarrs_storage::byte_range::ByteRangeIterator;

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// A memory store that does not support partial writes.
#[derive(Debug, Default)]
struct NoPartialWriteStore(MemoryStore);

impl ReadableStorageTraits for NoPartialWriteStore {
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

impl WritableStorageTraits for NoPartialWriteStore {
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

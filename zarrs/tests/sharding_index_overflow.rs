//! Compaction validates shard spans before copying or deciding that the shard is compact.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::ShardingCodecBuilder;
use zarrs::array::{ArrayBuilder, Resources, data_type};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorageTraits, WritableStorageTraits};

const INDEX_SIZE: usize = size_of::<[u64; 8]>();

fn builder(location: ShardingIndexLocation) -> ArrayBuilder {
    let codec = ShardingCodecBuilder::new(vec![NonZeroU64::new(1).unwrap()], &data_type::uint8())
        .index_bytes_to_bytes_codecs(vec![])
        .index_location(location)
        .build_arc();
    let mut builder = ArrayBuilder::new(vec![4], vec![4], data_type::uint8(), 0u8);
    builder.array_to_bytes_codec(codec);
    builder
}

fn shard(location: ShardingIndexLocation, data: &[u8], index: [u64; 8]) -> Vec<u8> {
    let index = index.into_iter().flat_map(u64::to_le_bytes);
    match location {
        ShardingIndexLocation::Start => index.chain(data.iter().copied()).collect(),
        ShardingIndexLocation::End => data.iter().copied().chain(index).collect(),
    }
}

fn corrupt_shards(location: ShardingIndexLocation) -> Vec<Vec<u8>> {
    let mut shards = Vec::new();
    for (offset, size) in [
        (u64::MAX - 1, 2),
        (2, u64::MAX - 1),
        (u64::MAX - 1, 1),
        (0, u64::MAX - 1),
        (u64::MAX, 0),
        (0, u64::MAX),
        (INDEX_SIZE as u64, 1),
        // Within the shard, but in its index
        (0, 1),
    ] {
        let mut index = [u64::MAX; 8];
        index[6] = offset;
        index[7] = size;
        shards.push(shard(location, &[], index));
    }

    // The reported sizes equal the data length: validate spans before the no-op shortcut.
    let mut index = [u64::MAX; 8];
    index[6] = u64::MAX - 1;
    index[7] = 2;
    shards.push(shard(location, &[1, 2], index));

    // A half-missing entry must not be skipped by the no-op shortcut either.
    index[6] = u64::MAX;
    index[7] = 0;
    shards.push(shard(location, &[], index));

    // These reported sizes would overflow an unchecked u64 sum.
    index = [
        0,
        u64::MAX - 1,
        0,
        2,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    ];
    shards.push(shard(location, &[], index));
    shards
}

#[test]
fn compact_rejects_corrupt_indexes_without_writing() {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for encoded in corrupt_shards(location) {
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                MemoryStore::new(),
            )));
            let array = builder(location).build(store.clone(), "/").unwrap();
            let key = array.chunk_key(&[0]).unwrap();
            store.set(&key, encoded.clone().into()).unwrap();
            store.reset();
            let error = array
                .compact_chunk(&[0], &Resources::default())
                .unwrap_err();
            assert!(error.to_string().contains("out-of-bounds bytes"), "{error}");
            assert_eq!(store.writes(), 0);
            assert_eq!(store.keys_erased(), 0);
            assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), encoded);
        }
    }
}

#[test]
fn compact_rejects_truncated_indexes_without_writing() {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for size in [0, INDEX_SIZE - 1] {
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                MemoryStore::new(),
            )));
            let array = builder(location).build(store.clone(), "/").unwrap();
            let key = array.chunk_key(&[0]).unwrap();
            let encoded = vec![0; size];
            store.set(&key, encoded.clone().into()).unwrap();
            store.reset();
            assert!(array.compact_chunk(&[0], &Resources::default()).is_err());
            assert_eq!(store.writes(), 0);
            assert_eq!(store.keys_erased(), 0);
            assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), encoded);
        }
    }
}

#[test]
fn compact_preserves_shared_and_missing_subchunks() {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let data_start = match location {
            ShardingIndexLocation::Start => INDEX_SIZE as u64,
            ShardingIndexLocation::End => 0,
        };
        let mut data = vec![99; 11];
        data[2] = 11;
        data[5] = 23;
        let index = [
            data_start + 2,
            1,
            u64::MAX,
            u64::MAX,
            data_start + 5,
            1,
            data_start + 5,
            1,
        ];
        let expected_index = [
            data_start,
            1,
            u64::MAX,
            u64::MAX,
            data_start + 1,
            1,
            data_start + 2,
            1,
        ];
        let expected = shard(location, &[11, 23, 23], expected_index);
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            MemoryStore::new(),
        )));
        let array = builder(location).build(store.clone(), "/").unwrap();
        let key = array.chunk_key(&[0]).unwrap();
        store
            .set(&key, shard(location, &data, index).into())
            .unwrap();
        assert_eq!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap(),
            [11, 0, 23, 23]
        );
        store.reset();
        assert!(array.compact_chunk(&[0], &Resources::default()).unwrap());
        assert_eq!(store.writes(), 1);
        assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), expected);
        assert_eq!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap(),
            [11, 0, 23, 23]
        );
        store.reset();
        assert!(!array.compact_chunk(&[0], &Resources::default()).unwrap());
        assert_eq!(store.writes(), 0);
        assert_eq!(store.keys_erased(), 0);
        assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), expected);
    }
}

#[test]
fn compact_skips_shards_that_would_not_shrink() {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let data_start = match location {
            ShardingIndexLocation::Start => INDEX_SIZE as u64,
            ShardingIndexLocation::End => 0,
        };
        for (data, index, expected) in [
            // Shared subchunks would be duplicated, growing the shard
            (&[7][..], [data_start, 1].repeat(4), [7, 7, 7, 7]),
            // Shared subchunks would be duplicated, replacing the trailing byte
            (
                &[7, 99][..],
                [
                    data_start,
                    1,
                    data_start,
                    1,
                    u64::MAX,
                    u64::MAX,
                    u64::MAX,
                    u64::MAX,
                ]
                .to_vec(),
                [7, 7, 0, 0],
            ),
        ] {
            let encoded = shard(location, data, index.try_into().unwrap());
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                MemoryStore::new(),
            )));
            let array = builder(location).build(store.clone(), "/").unwrap();
            let key = array.chunk_key(&[0]).unwrap();
            store.set(&key, encoded.clone().into()).unwrap();
            store.reset();
            assert!(!array.compact_chunk(&[0], &Resources::default()).unwrap());
            assert_eq!(store.writes(), 0);
            assert_eq!(store.keys_erased(), 0);
            assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), encoded);
            assert_eq!(
                array
                    .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn compact_preserves_empty_and_missing_ranges() {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let data_start = match location {
            ShardingIndexLocation::Start => INDEX_SIZE as u64,
            ShardingIndexLocation::End => 0,
        };
        for present in [false, true] {
            let mut index = [u64::MAX; 8];
            let mut expected_index = index;
            if present {
                index[0] = data_start + 1;
                index[1] = 0;
                expected_index[0] = data_start;
                expected_index[1] = 0;
            }
            let data: &[u8] = if present { &[99] } else { &[] };
            let expected = shard(location, &[], expected_index);
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                MemoryStore::new(),
            )));
            let array = builder(location).build(store.clone(), "/").unwrap();
            let key = array.chunk_key(&[0]).unwrap();
            store
                .set(&key, shard(location, data, index).into())
                .unwrap();
            store.reset();
            assert_eq!(
                array.compact_chunk(&[0], &Resources::default()).unwrap(),
                present
            );
            assert_eq!(store.writes(), usize::from(present));
            assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), expected);
            store.reset();
            assert!(!array.compact_chunk(&[0], &Resources::default()).unwrap());
            assert_eq!(store.writes(), 0);
        }
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_compact_rejects_corrupt_indexes_without_writing() {
    use zarrs::storage::store::AsyncMemoryStore;
    use zarrs::storage::{AsyncReadableStorageTraits, AsyncWritableStorageTraits};

    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let mut shards = corrupt_shards(location);
        shards.extend([vec![], vec![0; INDEX_SIZE - 1]]);
        for encoded in shards {
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                AsyncMemoryStore::new(),
            )));
            let array = builder(location).build(store.clone(), "/").unwrap();
            let key = array.chunk_key(&[0]).unwrap();
            store.set(&key, encoded.clone().into()).await.unwrap();
            store.reset();
            assert!(
                array
                    .async_compact_chunk(&[0], &Resources::default())
                    .await
                    .is_err()
            );
            assert_eq!(store.writes(), 0);
            assert_eq!(store.keys_erased(), 0);
            assert_eq!(store.get(&key).await.unwrap().unwrap().as_ref(), encoded);
        }
    }
}

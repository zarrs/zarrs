//! Corrupt shard indexes must fail before partial encoders change storage or their cached index.
#![allow(missing_docs)]
#![cfg(feature = "sharding")]

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::ShardingCodecBuilder;
use zarrs::array::{ArrayBuilder, ArrayBytes, ArraySubset, data_type};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorageTraits, WritableStorageTraits};
use zarrs_codec::CodecOptions;

fn indexes(location: ShardingIndexLocation) -> Vec<Vec<u8>> {
    let mut spans = vec![(u64::MAX - 1, 5), (u64::MAX - 2, 1)];
    if location == ShardingIndexLocation::End {
        spans.push((u64::MAX - 10, 1)); // Only the appended index overflows.
    }
    spans
        .into_iter()
        .map(|(offset, size)| {
            let mut index = [u64::MAX; 8];
            index[6] = offset;
            index[7] = size;
            index.into_iter().flat_map(u64::to_le_bytes).collect()
        })
        .collect()
}

fn builder(location: ShardingIndexLocation) -> ArrayBuilder {
    let codec =
        ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint8())
            .index_bytes_to_bytes_codecs(vec![])
            .index_location(location)
            .build_arc();
    let mut builder = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint8(), 0u8);
    builder.array_to_bytes_codec(codec);
    builder
}

#[test]
fn partial_encode_rejects_overflowing_shard_index() {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for encoded in indexes(location) {
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                MemoryStore::new(),
            )));
            let array = builder(location).build(store.clone(), "/").unwrap();
            let key = array.chunk_key(&[0, 0]);
            store.set(&key, encoded.clone().into()).unwrap();
            let encoder = match array.partial_encoder(&[0, 0], &CodecOptions::default()) {
                Ok(encoder) => encoder,
                Err(_) => {
                    assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), encoded);
                    continue;
                }
            };
            store.reset();
            // Reusing the encoder after an error must still reject its original corrupt index.
            for _ in 0..2 {
                assert!(
                    encoder
                        .partial_encode(
                            &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                            &ArrayBytes::new_flen(vec![7u8; 4]),
                            &CodecOptions::default()
                        )
                        .is_err()
                );
                assert_eq!(store.writes(), 0);
                assert_eq!(store.keys_erased(), 0);
                assert_eq!(store.get(&key).unwrap().unwrap().as_ref(), encoded);
            }
        }
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_partial_encode_rejects_overflowing_shard_index() {
    use zarrs::storage::{AsyncReadableStorageTraits, AsyncWritableStorageTraits};
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for encoded in indexes(location) {
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                zarrs::storage::store::AsyncMemoryStore::new(),
            )));
            let array = builder(location).build(store.clone(), "/").unwrap();
            let key = array.chunk_key(&[0, 0]);
            store.set(&key, encoded.clone().into()).await.unwrap();
            let encoder = match array
                .async_partial_encoder(&[0, 0], &CodecOptions::default())
                .await
            {
                Ok(encoder) => encoder,
                Err(_) => {
                    assert_eq!(store.get(&key).await.unwrap().unwrap().as_ref(), encoded);
                    assert_eq!(store.writes(), 1); // Only the corrupt fixture was written.
                    assert_eq!(store.keys_erased(), 0);
                    continue;
                }
            };
            store.reset();
            for _ in 0..2 {
                assert!(
                    encoder
                        .partial_encode(
                            &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                            &ArrayBytes::new_flen(vec![7u8; 4]),
                            &CodecOptions::default()
                        )
                        .await
                        .is_err()
                );
                assert_eq!(store.writes(), 0);
                assert_eq!(store.keys_erased(), 0);
                assert_eq!(store.get(&key).await.unwrap().unwrap().as_ref(), encoded);
            }
        }
    }
}

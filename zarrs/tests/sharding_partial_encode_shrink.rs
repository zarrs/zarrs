//! Partial encodes that leave fewer encoded subchunks must not shrink a shard below its existing length.
//!
//! A shard is never truncated by a partial encode, so a shard with a trailing index that ended before the existing shard would leave the old index as the suffix that readers decode.
#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::ShardingCodecBuilder;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset, data_type};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions, UnboundArrayToBytesCodecTraits};

/// Arrays of one shard with three subchunks of two elements, optionally followed by a `crc32c` codec.
fn arrays<TStorage: ?Sized>(
    store: &Arc<TStorage>,
) -> Vec<(ShardingIndexLocation, bool, Array<TStorage>)> {
    let mut arrays = vec![];
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for crc32c in [false, true] {
            let codec: Arc<dyn UnboundArrayToBytesCodecTraits> =
                ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap()], &data_type::uint8())
                    .index_location(location)
                    .build_arc();
            let mut builder = ArrayBuilder::new(vec![6], vec![6], data_type::uint8(), 0u8);
            builder.array_to_bytes_codec(codec);
            if crc32c {
                let crc32c: Arc<dyn BytesToBytesCodecTraits> =
                    Arc::new(zarrs::array::codec::Crc32cCodec::new());
                builder.bytes_to_bytes_codecs(vec![crc32c]);
            }
            let path = format!("/{location:?}_{crc32c}");
            arrays.push((
                location,
                crc32c,
                builder.build(store.clone(), &path).unwrap(),
            ));
        }
    }
    arrays
}

/// Fill all subchunks, then empty the last subchunk and then the middle subchunk.
fn updates() -> [(ArraySubset, Vec<u8>); 3] {
    [
        (
            ArraySubset::new_with_ranges(&[0..6]),
            vec![1, 2, 3, 4, 5, 6],
        ),
        (ArraySubset::new_with_ranges(&[4..6]), vec![0, 0]),
        (ArraySubset::new_with_ranges(&[2..4]), vec![0, 0]),
    ]
}

const EXPECTED: [u8; 6] = [1, 2, 0, 0, 0, 0];

#[test]
fn partial_encode_shrinking_shard() {
    let store = Arc::new(MemoryStore::new());
    for (location, crc32c, array) in arrays(&store) {
        for reuse_encoder in [false, true] {
            array.erase_chunk(&[0]).unwrap();
            let mut encoder = array
                .partial_encoder(&[0], &zarrs_codec::Resources::default())
                .unwrap();
            for (subset, bytes) in updates() {
                if !reuse_encoder {
                    encoder = array
                        .partial_encoder(&[0], &zarrs_codec::Resources::default())
                        .unwrap();
                }
                encoder
                    .partial_encode(
                        &subset,
                        &ArrayBytes::new_flen(bytes),
                        &CodecOptions::default(),
                        &zarrs_codec::Resources::default(),
                    )
                    .unwrap();
            }
            assert_eq!(
                array
                    .retrieve_chunk::<Vec<u8>>(&[0], &zarrs_codec::Resources::default())
                    .unwrap_or_else(|err| panic!("{location:?} {crc32c} {reuse_encoder}: {err}")),
                EXPECTED,
                "{location:?} {crc32c} {reuse_encoder}"
            );
        }
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_partial_encode_shrinking_shard() {
    let store = Arc::new(zarrs::storage::store::AsyncMemoryStore::new());
    for (location, crc32c, array) in arrays(&store) {
        for reuse_encoder in [false, true] {
            array.async_erase_chunk(&[0]).await.unwrap();
            let mut encoder = array
                .async_partial_encoder(&[0], &zarrs_codec::Resources::default())
                .await
                .unwrap();
            for (subset, bytes) in updates() {
                if !reuse_encoder {
                    encoder = array
                        .async_partial_encoder(&[0], &zarrs_codec::Resources::default())
                        .await
                        .unwrap();
                }
                encoder
                    .partial_encode(
                        &subset,
                        &ArrayBytes::new_flen(bytes),
                        &CodecOptions::default(),
                        &zarrs_codec::Resources::default(),
                    )
                    .await
                    .unwrap();
            }
            assert_eq!(
                array
                    .async_retrieve_chunk::<Vec<u8>>(&[0], &zarrs_codec::Resources::default())
                    .await
                    .unwrap_or_else(|err| panic!("{location:?} {crc32c} {reuse_encoder}: {err}")),
                EXPECTED,
                "{location:?} {crc32c} {reuse_encoder}"
            );
        }
    }
}

/// A partial encode behind `crc32c` replaces the shard after reading it once for the index and once for its subchunks, without querying its size.
#[test]
fn partial_encode_crc32c_reads() {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::new(),
    )));
    for (location, crc32c, array) in arrays(&store) {
        if !crc32c {
            continue;
        }
        let [(subset, bytes), ..] = updates();
        array
            .partial_encoder(&[0], &zarrs_codec::Resources::default())
            .unwrap()
            .partial_encode(
                &subset,
                &ArrayBytes::new_flen(bytes),
                &CodecOptions::default(),
                &zarrs_codec::Resources::default(),
            )
            .unwrap();

        // Update the middle subchunk, which straddles the update, and keep the others
        store.reset();
        array
            .partial_encoder(&[0], &zarrs_codec::Resources::default())
            .unwrap()
            .partial_encode(
                &ArraySubset::new_with_ranges(&[3..4]),
                &ArrayBytes::new_flen(vec![9]),
                &CodecOptions::default(),
                &zarrs_codec::Resources::default(),
            )
            .unwrap();
        assert!(store.reads() <= 2, "{location:?}: {} reads", store.reads());
        assert_eq!(store.size_queries(), 0, "{location:?}");
        assert_eq!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0], &zarrs_codec::Resources::default())
                .unwrap(),
            [1, 2, 3, 9, 5, 6],
            "{location:?}"
        );

        // Overwriting every subchunk reads only the index
        store.reset();
        array
            .partial_encoder(&[0], &zarrs_codec::Resources::default())
            .unwrap()
            .partial_encode(
                &ArraySubset::new_with_ranges(&[0..6]),
                &ArrayBytes::new_flen(vec![6, 5, 4, 3, 2, 1]),
                &CodecOptions::default(),
                &zarrs_codec::Resources::default(),
            )
            .unwrap();
        assert_eq!(store.reads(), 1, "{location:?}");
        assert_eq!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0], &zarrs_codec::Resources::default())
                .unwrap(),
            [6, 5, 4, 3, 2, 1],
            "{location:?}"
        );
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_partial_encode_crc32c_reads() {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        zarrs::storage::store::AsyncMemoryStore::new(),
    )));
    for (location, crc32c, array) in arrays(&store) {
        if !crc32c {
            continue;
        }
        let [(subset, bytes), ..] = updates();
        array
            .async_partial_encoder(&[0], &zarrs_codec::Resources::default())
            .await
            .unwrap()
            .partial_encode(
                &subset,
                &ArrayBytes::new_flen(bytes),
                &CodecOptions::default(),
                &zarrs_codec::Resources::default(),
            )
            .await
            .unwrap();

        store.reset();
        array
            .async_partial_encoder(&[0], &zarrs_codec::Resources::default())
            .await
            .unwrap()
            .partial_encode(
                &ArraySubset::new_with_ranges(&[3..4]),
                &ArrayBytes::new_flen(vec![9]),
                &CodecOptions::default(),
                &zarrs_codec::Resources::default(),
            )
            .await
            .unwrap();
        assert!(store.reads() <= 2, "{location:?}: {} reads", store.reads());
        assert_eq!(store.size_queries(), 0, "{location:?}");
        assert_eq!(
            array
                .async_retrieve_chunk::<Vec<u8>>(&[0], &zarrs_codec::Resources::default())
                .await
                .unwrap(),
            [1, 2, 3, 9, 5, 6],
            "{location:?}"
        );

        store.reset();
        array
            .async_partial_encoder(&[0], &zarrs_codec::Resources::default())
            .await
            .unwrap()
            .partial_encode(
                &ArraySubset::new_with_ranges(&[0..6]),
                &ArrayBytes::new_flen(vec![6, 5, 4, 3, 2, 1]),
                &CodecOptions::default(),
                &zarrs_codec::Resources::default(),
            )
            .await
            .unwrap();
        assert_eq!(store.reads(), 1, "{location:?}");
        assert_eq!(
            array
                .async_retrieve_chunk::<Vec<u8>>(&[0], &zarrs_codec::Resources::default())
                .await
                .unwrap(),
            [6, 5, 4, 3, 2, 1],
            "{location:?}"
        );
    }
}

#![allow(missing_docs)]
#![cfg(all(feature = "zstd", feature = "crc32c"))]

//! Partial encoders that rewrite an entire chunk replace it, rather than erasing it first or leaving stale bytes.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::{
    ArrayBytes, ArraySubset, CodecChain, CodecChainBound, DataType, FillValue, data_type,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorageTraits, StoreKey, WritableStorageTraits};
use zarrs_codec::{ArrayToBytesCodecTraits, CodecOptions, CodecSpecificOptions};

const BYTES: &str = r#"{"name": "bytes", "configuration": {"endian": "little"}}"#;
const ZSTD: &str = r#"{"name": "zstd", "configuration": {"level": 1, "checksum": false}}"#;
const CRC32C: &str = r#"{"name": "crc32c"}"#;

const NUM_ELEMENTS: u64 = 4096;

fn bind(codecs: &[&str], data_type: DataType, fill_value: FillValue) -> Arc<CodecChainBound> {
    let metadata: Vec<MetadataV3> = codecs
        .iter()
        .map(|codec| serde_json::from_str(codec).unwrap())
        .collect();
    CodecChain::from_metadata(&metadata)
        .unwrap()
        .with_context(
            data_type,
            fill_value,
            &CodecSpecificOptions::default(),
        )
        .unwrap()
}

fn shape() -> Vec<NonZeroU64> {
    vec![NonZeroU64::new(NUM_ELEMENTS).unwrap()]
}

/// Incompressible chunk elements.
fn incompressible() -> Vec<u8> {
    let mut state = 12345u32;
    (0..NUM_ELEMENTS)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            state.to_le_bytes()[2]
        })
        .collect()
}

/// An update to the first 4000 elements that makes the chunk compress far smaller.
fn compressible_update() -> (ArraySubset, Vec<u8>) {
    (ArraySubset::new_with_ranges(&[0..4000]), vec![7; 4000])
}

fn updated(mut elements: Vec<u8>, update: &[u8]) -> Vec<u8> {
    elements[..update.len()].copy_from_slice(update);
    elements
}

/// Compression codecs re-encode the entire chunk on a partial encode.
/// The re-encoded chunk replaces the existing chunk, leaving no stale trailing bytes, and stacked codecs read the chunk once.
#[test]
fn partial_encode_compressed_chunk_shrinks() {
    let options = CodecOptions::default();
    let mut reads = vec![];
    for codecs in [&[BYTES, ZSTD][..], &[BYTES, ZSTD, CRC32C]] {
        let codec = bind(codecs, data_type::uint8(), FillValue::from(0u8));
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            MemoryStore::new(),
        )));
        let key = StoreKey::new("c").unwrap();
        let elements = incompressible();
        let encoded = codec
            .encode(ArrayBytes::from(elements.clone()), &shape(), &options)
            .unwrap();
        let len_before = encoded.len();
        store.set(&key, encoded).unwrap();

        let handle = Arc::new((store.clone(), key.clone()));
        let (subset, update) = compressible_update();
        let reads_before = store.reads();
        codec
            .clone()
            .partial_encoder(handle.clone(), &shape(), &options)
            .unwrap()
            .partial_encode(&subset, &ArrayBytes::from(update.clone()), &options)
            .unwrap();
        reads.push(store.reads() - reads_before);

        let len_after = store.get(&key).unwrap().unwrap().len();
        assert!(len_after < len_before / 2, "{codecs:?}: {len_after}");
        let decoded = codec
            .clone()
            .partial_decoder(handle, &shape(), &options)
            .unwrap()
            .partial_decode(&ArraySubset::new_with_ranges(&[0..NUM_ELEMENTS]), &options)
            .unwrap()
            .into_owned();
        assert_eq!(
            decoded,
            ArrayBytes::from(updated(elements, &update)),
            "{codecs:?}"
        );
    }
    // The checksum codec adds no read of the chunk
    assert_eq!(reads[0], reads[1]);
}

#[cfg(feature = "async")]
mod r#async {
    use super::*;
    use zarrs::storage::storage_adapter::sync_to_async::{
        SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter,
    };
    use zarrs::storage::{AsyncReadableStorageTraits, AsyncWritableStorageTraits};

    pub(super) struct TokioSpawnBlocking;

    impl SyncToAsyncSpawnBlocking for TokioSpawnBlocking {
        async fn spawn_blocking<F, R>(&self, f: F) -> R
        where
            F: FnOnce() -> R + Send + 'static,
            R: Send + 'static,
        {
            tokio::task::spawn_blocking(f).await.unwrap()
        }
    }

    #[tokio::test]
    async fn async_partial_encode_compressed_chunk_shrinks() {
        let options = CodecOptions::default();
        let mut reads = vec![];
        for codecs in [&[BYTES, ZSTD][..], &[BYTES, ZSTD, CRC32C]] {
            let codec = bind(codecs, data_type::uint8(), FillValue::from(0u8));
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                SyncToAsyncStorageAdapter::new(Arc::new(MemoryStore::new()), TokioSpawnBlocking),
            )));
            let key = StoreKey::new("c").unwrap();
            let elements = incompressible();
            let encoded = codec
                .encode(ArrayBytes::from(elements.clone()), &shape(), &options)
                .unwrap();
            let len_before = encoded.len();
            store.set(&key, encoded).await.unwrap();

            let handle = Arc::new((store.clone(), key.clone()));
            let (subset, update) = compressible_update();
            let reads_before = store.reads();
            codec
                .clone()
                .async_partial_encoder(handle.clone(), &shape(), &options)
                .await
                .unwrap()
                .partial_encode(&subset, &ArrayBytes::from(update.clone()), &options)
                .await
                .unwrap();
            reads.push(store.reads() - reads_before);

            let len_after = store.get(&key).await.unwrap().unwrap().len();
            assert!(len_after < len_before / 2, "{codecs:?}: {len_after}");
            let decoded = codec
                .clone()
                .async_partial_decoder(handle, &shape(), &options)
                .await
                .unwrap()
                .partial_decode(&ArraySubset::new_with_ranges(&[0..NUM_ELEMENTS]), &options)
                .await
                .unwrap()
                .into_owned();
            assert_eq!(
                decoded,
                ArrayBytes::from(updated(elements, &update)),
                "{codecs:?}"
            );
        }
        // The checksum codec adds no read of the chunk
        assert_eq!(reads[0], reads[1]);
    }
}

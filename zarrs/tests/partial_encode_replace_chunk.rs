#![allow(missing_docs)]

//! Partial encoders that rewrite an entire chunk replace it, rather than erasing it first or leaving stale bytes.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use zarrs::array::{ArrayBytes, ArraySubset, CodecChain, DataType, FillValue, data_type};
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::byte_range::ByteRangeIterator;
#[cfg(all(feature = "zstd", feature = "crc32c"))]
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    CowBytes, MaybeBytesIterator, OffsetBytesIterator, ReadableStorageTraits, StorageError,
    StoreKey, StorePrefix, WritableStorageTraits,
};
use zarrs_codec::{
    ArrayToBytesCodecTraits, CodecOptions, CodecSpecificOptions, ComputeResources, Resources,
};

const BYTES: &str = r#"{"name": "bytes", "configuration": {"endian": "little"}}"#;
#[cfg(feature = "zstd")]
const ZSTD: &str = r#"{"name": "zstd", "configuration": {"level": 1, "checksum": false}}"#;
#[cfg(feature = "crc32c")]
const CRC32C: &str = r#"{"name": "crc32c"}"#;

#[cfg(all(feature = "zstd", feature = "crc32c"))]
const NUM_ELEMENTS: u64 = 4096;

fn bind(codecs: &[&str], data_type: DataType, fill_value: FillValue) -> Arc<TestCodec> {
    let metadata: Vec<MetadataV3> = codecs
        .iter()
        .map(|codec| serde_json::from_str(codec).unwrap())
        .collect();
    let chain = CodecChain::from_metadata(&metadata).unwrap();
    let bound = chain
        .with_context(
            data_type,
            Some(fill_value),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    Arc::new(TestCodec { chain, bound })
}

#[path = "common/planned_codec.rs"]
mod planned_codec;
use planned_codec::TestCodec;

#[cfg(all(feature = "zstd", feature = "crc32c"))]
fn shape() -> Vec<NonZeroU64> {
    vec![NonZeroU64::new(NUM_ELEMENTS).unwrap()]
}

#[cfg(all(feature = "zstd", feature = "crc32c"))]
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

#[cfg(all(feature = "zstd", feature = "crc32c"))]
/// An update to the first 4000 elements that makes the chunk compress far smaller.
fn compressible_update() -> (ArraySubset, Vec<u8>) {
    (ArraySubset::new_with_ranges(&[0..4000]), vec![7; 4000])
}

#[cfg(all(feature = "zstd", feature = "crc32c"))]
fn updated(mut elements: Vec<u8>, update: &[u8]) -> Vec<u8> {
    elements[..update.len()].copy_from_slice(update);
    elements
}

#[cfg(all(feature = "zstd", feature = "crc32c"))]
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
        let key = StoreKey::new("c/0").unwrap();
        let elements = incompressible();
        let encoded = codec
            .encode(
                ArrayBytes::from(elements.clone()),
                &shape(),
                &options,
                &ComputeResources::default(),
            )
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
            .partial_encode(
                &subset,
                &ArrayBytes::from(update.clone()),
                &options,
                &Resources::default(),
            )
            .unwrap();
        reads.push(store.reads() - reads_before);

        let len_after = store.get(&key).unwrap().unwrap().len();
        assert!(len_after < len_before / 2, "{codecs:?}: {len_after}");
        let decoded = codec
            .clone()
            .partial_decoder(handle, &shape(), &options)
            .unwrap()
            .partial_decode(
                &ArraySubset::new_with_ranges(&[0..NUM_ELEMENTS]),
                &options,
                &Resources::default(),
            )
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

#[cfg(all(feature = "zstd", feature = "crc32c"))]
/// A partial encode of the entire chunk replaces it without reading it, so an undecodable chunk is repaired.
#[test]
fn partial_encode_whole_chunk_replaces_undecodable_chunk() {
    let options = CodecOptions::default();
    for codecs in [&[BYTES][..], &[BYTES, ZSTD], &[BYTES, ZSTD, CRC32C]] {
        let codec = bind(codecs, data_type::uint8(), FillValue::from(0u8));
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            MemoryStore::new(),
        )));
        let key = StoreKey::new("c/0").unwrap();
        store.set(&key, CowBytes::from(vec![0xff; 16])).unwrap();
        store.reset();

        let handle = Arc::new((store.clone(), key.clone()));
        let elements = incompressible();
        codec
            .clone()
            .partial_encoder(handle.clone(), &shape(), &options)
            .unwrap()
            .partial_encode(
                &ArraySubset::new_with_ranges(&[0..NUM_ELEMENTS]),
                &ArrayBytes::from(elements.clone()),
                &options,
                &Resources::default(),
            )
            .unwrap();
        assert_eq!(store.reads(), 0, "{codecs:?}");
        assert_eq!(store.writes(), 1, "{codecs:?}");

        let decoded = codec
            .decode(
                store.get(&key).unwrap().unwrap().into(),
                &shape(),
                &options,
                &ComputeResources::default(),
            )
            .unwrap()
            .into_owned();
        assert_eq!(decoded, ArrayBytes::from(elements), "{codecs:?}");
    }
}

/// A store whose writes fail while `fail_writes` is set. Erasing succeeds, as it would if only the write after an erase failed.
struct FailingWritesStorage<T> {
    storage: Arc<T>,
    fail_writes: AtomicBool,
}

impl<T> FailingWritesStorage<T> {
    fn new(storage: Arc<T>) -> Self {
        Self {
            storage,
            fail_writes: AtomicBool::new(false),
        }
    }

    fn fail_writes(&self, fail: bool) {
        self.fail_writes.store(fail, Ordering::Relaxed);
    }

    fn check_write(&self) -> Result<(), StorageError> {
        if self.fail_writes.load(Ordering::Relaxed) {
            Err(StorageError::Other("injected write failure".to_string()))
        } else {
            Ok(())
        }
    }
}

impl<T: ReadableStorageTraits> ReadableStorageTraits for FailingWritesStorage<T> {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.storage.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.storage.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

impl<T: WritableStorageTraits> WritableStorageTraits for FailingWritesStorage<T> {
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        self.check_write()?;
        self.storage.set(key, value)
    }

    fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        self.check_write()?;
        self.storage.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.storage.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.storage.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.storage.supports_set_partial()
    }
}

/// A partial encode of an existing chunk whose write fails.
struct FailedWriteCase {
    name: &'static str,
    codec: Arc<TestCodec>,
    shape: Vec<NonZeroU64>,
    existing: ArrayBytes<'static>,
    subset: ArraySubset,
    update: ArrayBytes<'static>,
    /// `existing` updated with `update`.
    updated: ArrayBytes<'static>,
}

impl FailedWriteCase {
    fn whole(&self) -> ArraySubset {
        ArraySubset::new_with_shape(self.shape.iter().map(|s| s.get()).collect::<Vec<_>>())
    }
}

fn string_bytes(elements: &[&str]) -> ArrayBytes<'static> {
    let mut offsets = vec![0u32];
    for element in elements {
        offsets.push(offsets.last().unwrap() + u32::try_from(element.len()).unwrap());
    }
    ArrayBytes::new_vlen(
        elements.concat().into_bytes(),
        zarrs::array::ArrayBytesOffsets::new(offsets).unwrap(),
    )
    .unwrap()
}

fn failed_write_cases() -> Vec<FailedWriteCase> {
    let mut cases = vec![FailedWriteCase {
        // The default array-to-bytes partial encoder
        name: "vlen-utf8",
        codec: bind(
            &[r#"{"name": "vlen-utf8"}"#],
            data_type::string(),
            FillValue::from(""),
        ),
        shape: vec![NonZeroU64::new(4).unwrap()],
        existing: string_bytes(&["a", "bb", "ccc", "dddd"]),
        subset: ArraySubset::new_with_ranges(&[1..3]),
        update: string_bytes(&["x", "y"]),
        updated: string_bytes(&["a", "x", "y", "dddd"]),
    }];
    cases.push(FailedWriteCase {
        // The default array-to-array partial encoder
        name: "fixedscaleoffset",
        codec: bind(
            &[
                r#"{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 100.0, "scale": 2, "dtype": "f4", "astype": "f8"}}"#,
                r#"{"name": "bytes", "configuration": {"endian": "little"}}"#,
            ],
            data_type::float32(),
            FillValue::from(0f32),
        ),
        shape: vec![NonZeroU64::new(4).unwrap()],
        existing: f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        subset: ArraySubset::new_with_ranges(&[1..3]),
        update: f32_bytes(&[5.0, 6.0]),
        updated: f32_bytes(&[1.0, 5.0, 6.0, 4.0]),
    });
    // The sharding partial encoder rewrites a shard if the update replaces every subchunk in it
    for (name, index_location) in [("sharding-start", "start"), ("sharding-end", "end")] {
        let sharding = format!(
            r#"{{"name": "sharding_indexed", "configuration": {{"chunk_shape": [4], "codecs": [{BYTES}], "index_codecs": [{BYTES}], "index_location": "{index_location}"}}}}"#
        );
        cases.push(FailedWriteCase {
            name,
            codec: bind(&[&sharding], data_type::uint8(), FillValue::from(0u8)),
            shape: vec![NonZeroU64::new(8).unwrap()],
            existing: ArrayBytes::from((1u8..=8).collect::<Vec<u8>>()),
            subset: ArraySubset::new_with_ranges(&[2..6]),
            update: ArrayBytes::from(vec![9u8; 4]),
            updated: ArrayBytes::from(vec![1u8, 2, 9, 9, 9, 9, 7, 8]),
        });
    }
    cases
}

fn f32_bytes(elements: &[f32]) -> ArrayBytes<'static> {
    ArrayBytes::from(
        elements
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<u8>>(),
    )
}

/// A partial encode that rewrites an existing chunk leaves it intact if the write fails.
#[test]
fn failed_partial_encode_keeps_existing_chunk() {
    let options = CodecOptions::default();
    for case in failed_write_cases() {
        let name = case.name;
        let store = Arc::new(FailingWritesStorage::new(Arc::new(MemoryStore::new())));
        let key = StoreKey::new("c/0").unwrap();
        let existing_encoded = case
            .codec
            .encode(
                case.existing.clone(),
                &case.shape,
                &options,
                &ComputeResources::default(),
            )
            .unwrap();
        store.set(&key, existing_encoded).unwrap();
        let handle = Arc::new((store.clone(), key));

        let decode = || {
            case.codec
                .clone()
                .partial_decoder(handle.clone(), &case.shape, &options)
                .unwrap()
                .partial_decode(&case.whole(), &options, &Resources::default())
                .unwrap()
                .into_owned()
        };
        let encoder = case
            .codec
            .clone()
            .partial_encoder(handle.clone(), &case.shape, &options)
            .unwrap();

        store.fail_writes(true);
        let result =
            encoder.partial_encode(&case.subset, &case.update, &options, &Resources::default());
        assert!(result.is_err(), "{name}");
        store.fail_writes(false);
        assert_eq!(decode(), case.existing, "{name}");

        // The encoder remains usable, e.g. a cached shard index still matches the stored shard
        encoder
            .partial_encode(&case.subset, &case.update, &options, &Resources::default())
            .unwrap();
        assert_eq!(decode(), case.updated, "{name}");
    }
}

#[cfg(feature = "async")]
mod r#async {
    use super::*;
    #[cfg(all(feature = "zstd", feature = "crc32c"))]
    use zarrs::storage::AsyncReadableStorageTraits;
    use zarrs::storage::AsyncWritableStorageTraits;
    use zarrs::storage::storage_adapter::sync_to_async::{
        SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter,
    };

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

    #[cfg(all(feature = "zstd", feature = "crc32c"))]
    #[tokio::test]
    async fn async_partial_encode_compressed_chunk_shrinks() {
        let options = CodecOptions::default();
        let mut reads = vec![];
        for codecs in [&[BYTES, ZSTD][..], &[BYTES, ZSTD, CRC32C]] {
            let codec = bind(codecs, data_type::uint8(), FillValue::from(0u8));
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                SyncToAsyncStorageAdapter::new(Arc::new(MemoryStore::new()), TokioSpawnBlocking),
            )));
            let key = StoreKey::new("c/0").unwrap();
            let elements = incompressible();
            let encoded = codec
                .encode(
                    ArrayBytes::from(elements.clone()),
                    &shape(),
                    &options,
                    &ComputeResources::default(),
                )
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
                .partial_encode(
                    &subset,
                    &ArrayBytes::from(update.clone()),
                    &options,
                    &Resources::default(),
                )
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
                .partial_decode(
                    &ArraySubset::new_with_ranges(&[0..NUM_ELEMENTS]),
                    &options,
                    &Resources::default(),
                )
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

    #[cfg(all(feature = "zstd", feature = "crc32c"))]
    #[tokio::test]
    async fn async_partial_encode_whole_chunk_replaces_undecodable_chunk() {
        let options = CodecOptions::default();
        for codecs in [&[BYTES][..], &[BYTES, ZSTD], &[BYTES, ZSTD, CRC32C]] {
            let codec = bind(codecs, data_type::uint8(), FillValue::from(0u8));
            let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                SyncToAsyncStorageAdapter::new(Arc::new(MemoryStore::new()), TokioSpawnBlocking),
            )));
            let key = StoreKey::new("c/0").unwrap();
            store
                .set(&key, CowBytes::from(vec![0xff; 16]))
                .await
                .unwrap();
            store.reset();

            let handle = Arc::new((store.clone(), key.clone()));
            let elements = incompressible();
            codec
                .clone()
                .async_partial_encoder(handle.clone(), &shape(), &options)
                .await
                .unwrap()
                .partial_encode(
                    &ArraySubset::new_with_ranges(&[0..NUM_ELEMENTS]),
                    &ArrayBytes::from(elements.clone()),
                    &options,
                    &Resources::default(),
                )
                .await
                .unwrap();
            assert_eq!(store.reads(), 0, "{codecs:?}");
            assert_eq!(store.writes(), 1, "{codecs:?}");

            let encoded = store.get(&key).await.unwrap().unwrap();
            let decoded = codec
                .decode(
                    encoded.into(),
                    &shape(),
                    &options,
                    &ComputeResources::default(),
                )
                .unwrap()
                .into_owned();
            assert_eq!(decoded, ArrayBytes::from(elements), "{codecs:?}");
        }
    }

    #[tokio::test]
    async fn async_failed_partial_encode_keeps_existing_chunk() {
        let options = CodecOptions::default();
        for case in failed_write_cases() {
            let name = case.name;
            let failing = Arc::new(FailingWritesStorage::new(Arc::new(MemoryStore::new())));
            let store = Arc::new(SyncToAsyncStorageAdapter::new(
                failing.clone(),
                TokioSpawnBlocking,
            ));
            let key = StoreKey::new("c/0").unwrap();
            let existing_encoded = case
                .codec
                .encode(
                    case.existing.clone(),
                    &case.shape,
                    &options,
                    &ComputeResources::default(),
                )
                .unwrap();
            store.set(&key, existing_encoded).await.unwrap();
            let handle = Arc::new((store.clone(), key));

            let decode = async || {
                case.codec
                    .clone()
                    .async_partial_decoder(handle.clone(), &case.shape, &options)
                    .await
                    .unwrap()
                    .partial_decode(&case.whole(), &options, &Resources::default())
                    .await
                    .unwrap()
                    .into_owned()
            };
            let encoder = case
                .codec
                .clone()
                .async_partial_encoder(handle.clone(), &case.shape, &options)
                .await
                .unwrap();

            failing.fail_writes(true);
            let result = encoder
                .partial_encode(&case.subset, &case.update, &options, &Resources::default())
                .await;
            assert!(result.is_err(), "{name}");
            failing.fail_writes(false);
            assert_eq!(decode().await, case.existing, "{name}");

            // The encoder remains usable, e.g. a cached shard index still matches the stored shard
            encoder
                .partial_encode(&case.subset, &case.update, &options, &Resources::default())
                .await
                .unwrap();
            assert_eq!(decode().await, case.updated, "{name}");
        }
    }
}

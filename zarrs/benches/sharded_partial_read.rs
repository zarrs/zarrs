//! Benchmark sharded array reads: full vs partial shard reads.
//!
//! Array data is written to `benches/data/sharded_partial_read/` on the first run and reused
//! on subsequent runs. Delete that directory to regenerate the data.
//!
//! Set [`USE_MEMORY_STORE`] to `true` to benchmark with an in-memory store instead of the
//! filesystem store.
#![allow(missing_docs)]

use std::path::PathBuf;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use zarrs::array::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorage, ReadableWritableListableStorage};

/// If `true`, benchmark with an in-memory store; if `false`, use a filesystem store.
const USE_MEMORY_STORE: bool = false;

// Array dimensions matching the Python snippet
const SHAPE: [u64; 4] = [8192, 4, 128, 128];
const SHARD_SHAPE: [u64; 4] = [4096, 4, 128, 128];
const CHUNK_SHAPE: [u64; 4] = [1, 1, 128, 128];

/// Elements per shard (f64 = 8 bytes each)
const SHARD_ELEMENTS: u64 = SHARD_SHAPE[0] * SHARD_SHAPE[1] * SHARD_SHAPE[2] * SHARD_SHAPE[3];
const SHARD_BYTES: u64 = SHARD_ELEMENTS * 8;

fn data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("data")
        .join("sharded_partial_read")
}

/// Generate pseudo-random f64 values via a simple LCG (no external crate needed).
fn random_data(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x123456789abcdef0;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 * (1.0 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn make_store() -> ReadableWritableListableStorage {
    if USE_MEMORY_STORE {
        Arc::new(MemoryStore::new())
    } else {
        let path = data_path();
        std::fs::create_dir_all(&path).unwrap();
        Arc::new(FilesystemStore::new(&path).unwrap())
    }
}

fn populate_array(store: ReadableWritableListableStorage) {
    let array = zarrs::array::ArrayBuilder::new(
        SHAPE.to_vec(),
        SHARD_SHAPE.to_vec(),
        zarrs::array::data_type::float64(),
        0.0f64,
    )
    .subchunk_shape(CHUNK_SHAPE.to_vec())
    .build(store, "/")
    .unwrap();
    array.store_metadata().unwrap();

    let total_elements = SHAPE.iter().product::<u64>() as usize;
    let data = random_data(total_elements);
    array
        .store_array_subset(
            &ArraySubset::new_with_shape(SHAPE.to_vec()),
            data.as_slice(),
        )
        .unwrap();
}

/// Returns a store loaded with array data, writing it first if needed.
fn open_store() -> ReadableStorage {
    if USE_MEMORY_STORE {
        let store: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
        populate_array(Arc::clone(&store));
        store
    } else {
        let path = data_path();
        let zarr_json = path.join("zarr.json");
        if !zarr_json.exists() {
            populate_array(make_store());
        }
        Arc::new(FilesystemStore::new(path).unwrap())
    }
}

/// Full array read: both shards, shard-boundary-aligned.
fn bench_read_full(c: &mut Criterion) {
    let store = open_store();
    let array = zarrs::array::Array::open(store, "/").unwrap();
    let mut group = c.benchmark_group("sharded_partial_read");
    let full_bytes = SHARD_BYTES * 2; // 2 shards
    let subset = ArraySubset::new_with_shape(SHAPE.to_vec());

    group.throughput(Throughput::Bytes(full_bytes));
    group.bench_function("full_array", |b| {
        b.iter(|| {
            let _: zarrs::array::ArrayBytes = array.retrieve_array_subset(&subset).unwrap();
        });
    });

    group.finish();
}

/// Shard-aligned partial read: exactly the first shard.
fn bench_read_partial_aligned(c: &mut Criterion) {
    let store = open_store();
    let array = zarrs::array::Array::open(store, "/").unwrap();
    let mut group = c.benchmark_group("sharded_partial_read");
    let byte_count = SHARD_BYTES;

    group.throughput(Throughput::Bytes(byte_count));
    group.bench_function("partial_shard_aligned", |b| {
        b.iter(|| {
            let _: zarrs::array::ArrayBytes = array
                .retrieve_array_subset(&[0..SHARD_SHAPE[0], 0..SHAPE[1], 0..SHAPE[2], 0..SHAPE[3]])
                .unwrap();
        });
    });

    group.finish();
}

/// Unaligned partial read: one element short of the first shard boundary.
fn bench_read_partial_unaligned(c: &mut Criterion) {
    let store = open_store();
    let array = zarrs::array::Array::open(store, "/").unwrap();
    let mut group = c.benchmark_group("sharded_partial_read");
    let byte_count = (SHARD_SHAPE[0] - 1) * SHAPE[1] * SHAPE[2] * SHAPE[3] * 8;

    group.throughput(Throughput::Bytes(byte_count));
    group.bench_function("partial_shard_unaligned", |b| {
        b.iter(|| {
            let _: zarrs::array::ArrayBytes = array
                .retrieve_array_subset(&[
                    0..(SHARD_SHAPE[0] - 1),
                    0..SHAPE[1],
                    0..SHAPE[2],
                    0..SHAPE[3],
                ])
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_read_full,
    bench_read_partial_aligned,
    bench_read_partial_unaligned,
);
criterion_main!(benches);

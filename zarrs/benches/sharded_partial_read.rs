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
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs::array::ArraySubset;
use zarrs::array::codec::{ShardingCodecOptions, SubchunkWriteOrder, ZstdCodec};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorage, ReadableWritableListableStorage};

/// If `true`, benchmark with an in-memory store; if `false`, use a filesystem store.
const USE_MEMORY_STORE: bool = false;

// Array dimensions matching the Python snippet
const SHAPE: [u64; 4] = [8192 * 4, 4, 128, 128];
const SHARD_SHAPE: [u64; 4] = [8192 * 4, 4, 128, 128];
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
    .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(5, false))])
    .subchunk_shape(CHUNK_SHAPE.to_vec())
    .codec_specific_options(
        zarrs_codec::CodecSpecificOptions::default().with_option(
            ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
        ),
    )
    .build(store, "/")
    .unwrap();
    array.store_metadata().unwrap();

    let total_elements = SHAPE.iter().product::<u64>() as usize;
    let data = (0..total_elements).map(|e| e as f64).collect::<Vec<_>>();
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

/// 64 disparate chunk reads of size 128 in dim 0, fetched in parallel.
///
/// Subsets are at dim-0 positions 0, 512, 1024, … (stride 512, width 128), covering
/// the full extents of dims 1–3.  Each subset spans `[128, 4, 128, 128]` elements.
fn bench_read_disparate_parallel(c: &mut Criterion) {
    const N_CHUNKS: u64 = 64;
    const CHUNK_SIZE: u64 = 128;
    const STRIDE: u64 = 512;

    let store = open_store();
    let array = zarrs::array::Array::open(store, "/").unwrap();
    let mut group = c.benchmark_group("sharded_partial_read");

    let chunk_bytes = CHUNK_SIZE * SHAPE[1] * SHAPE[2] * SHAPE[3] * 8;
    group.throughput(Throughput::Bytes(chunk_bytes * N_CHUNKS));

    let subsets: Vec<ArraySubset> = (0..N_CHUNKS)
        .map(|k| {
            let start = k * STRIDE;
            ArraySubset::new_with_ranges(&[
                start..start + CHUNK_SIZE,
                0..SHAPE[1],
                0..SHAPE[2],
                0..SHAPE[3],
            ])
        })
        .collect();

    let concurrency = std::thread::available_parallelism().map_or(4, |n| n.get());

    group.bench_function("disparate_parallel", |b| {
        b.iter(|| {
                iter_concurrent_limit!(concurrency, subsets.clone(), for_each, |subset: ArraySubset| {
                    let _: zarrs::array::ArrayBytes = array.retrieve_array_subset(&subset).unwrap();
                });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    // bench_read_partial_aligned,
    // bench_read_partial_unaligned,
    bench_read_disparate_parallel,
);
criterion_main!(benches);

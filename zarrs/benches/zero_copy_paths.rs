//! Benchmarks for the copy-sensitive chunk read/write paths.
//!
//! These target the four paths affected by giving `CowBytes` a shared representation:
//! 1. `retrieve_chunk` as `ArrayBytes` — the store->decode copy.
//! 2. Sharded partial read — subchunk slicing out of a shard.
//! 3. `store_chunk` from a borrowed slice on a pass-through chain.
//! 4. `retrieve_chunk` as `Vec<T>` — the bytemuck path that cannot avoid a copy.
#![allow(missing_docs)]

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset, data_type};
use zarrs::storage::store::MemoryStore;

const CHUNK: u64 = 64; // 64^3 u16 = 512 KiB per chunk

fn uncompressed_array(chunk: u64) -> Array<MemoryStore> {
    let store = Arc::new(MemoryStore::new());
    ArrayBuilder::new(vec![chunk; 3], vec![chunk; 3], data_type::uint16(), 0u16)
        .build(store, "/")
        .unwrap()
}

/// 1. Read a chunk as `ArrayBytes`.
///
/// `MemoryStore::get` returns a cloned `Bytes` (refcount >= 2), which is shared with the codec
/// chain rather than copied into an owned buffer at the decode boundary.
fn retrieve_chunk_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieve_chunk_bytes");
    let num_elements = CHUNK * CHUNK * CHUNK;
    group.throughput(Throughput::Bytes(num_elements * 2));
    let array = uncompressed_array(CHUNK);
    let data = vec![1u16; num_elements as usize];
    array.store_chunk(&[0, 0, 0], &data).unwrap();

    group.bench_function(BenchmarkId::from_parameter(CHUNK), |b| {
        b.iter(|| {
            let bytes: ArrayBytes = array.retrieve_chunk(&[0, 0, 0]).unwrap();
            std::hint::black_box(bytes);
        });
    });
    group.finish();
}

/// 4. Read a chunk as `Vec<u16>`.
///
/// Goes through `transmute_from_bytes_vec`, which needs a uniquely-owned aligned `Vec<u8>`.
/// This is the one path a shared representation cannot make free, so it is kept as a guard
/// against regressing it.
fn retrieve_chunk_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieve_chunk_vec");
    let num_elements = CHUNK * CHUNK * CHUNK;
    group.throughput(Throughput::Bytes(num_elements * 2));
    let array = uncompressed_array(CHUNK);
    let data = vec![1u16; num_elements as usize];
    array.store_chunk(&[0, 0, 0], &data).unwrap();

    group.bench_function(BenchmarkId::from_parameter(CHUNK), |b| {
        b.iter(|| {
            let elements: Vec<u16> = array.retrieve_chunk(&[0, 0, 0]).unwrap();
            std::hint::black_box(elements);
        });
    });
    group.finish();
}

/// 3. Write a chunk from a borrowed slice on a pass-through chain.
///
/// Fixed-length data type, native endianness, no bytes-to-bytes codecs, so the codec chain
/// passes the input through untouched and the slice reaches the store without a copy, for
/// stores that do not retain the value.
fn store_chunk_borrowed(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_chunk_borrowed");
    let num_elements = CHUNK * CHUNK * CHUNK;
    group.throughput(Throughput::Bytes(num_elements * 2));
    let data = vec![1u16; num_elements as usize];

    group.bench_function(BenchmarkId::from_parameter(CHUNK), |b| {
        b.iter(|| {
            let array = uncompressed_array(CHUNK);
            array.store_chunk(&[0, 0, 0], data.as_slice()).unwrap();
        });
    });
    group.finish();
}

/// 3b. The same write, from an owned `Bytes`.
///
/// The store retains the shared buffer without copying it, so this is the reference point
/// that `store_chunk_borrowed` is measured against.
fn store_chunk_shared(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_chunk_shared");
    let num_elements = CHUNK * CHUNK * CHUNK;
    group.throughput(Throughput::Bytes(num_elements * 2));
    let bytes = bytes::Bytes::from(vec![1u8; (num_elements * 2) as usize]);

    group.bench_function(BenchmarkId::from_parameter(CHUNK), |b| {
        b.iter(|| {
            let array = uncompressed_array(CHUNK);
            array.store_chunk(&[0, 0, 0], &bytes).unwrap();
        });
    });
    group.finish();
}

/// 2. Sharded partial read — one subchunk out of a shard.
///
/// Each subchunk is extracted from the encoded shard, which is free while the shard bytes
/// are shared.
fn sharded_subchunk_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharded_subchunk_read");
    let size = 256u64;
    let num_elements = size * size * size;
    group.throughput(Throughput::Bytes(num_elements * 2));

    let store = Arc::new(MemoryStore::new());
    let array = ArrayBuilder::new(vec![size; 3], vec![size; 3], data_type::uint16(), 0u16)
        .subchunk_shape(vec![32; 3])
        .build(store, "/")
        .unwrap();
    let data = vec![1u16; num_elements as usize];
    array
        .store_array_subset(&ArraySubset::new_with_shape(vec![size; 3]), &data)
        .unwrap();

    // Read a single inner chunk out of the shard.
    let subset = ArraySubset::new_with_ranges(&[0..32, 0..32, 0..32]);
    group.bench_function(BenchmarkId::from_parameter(size), |b| {
        b.iter(|| {
            let bytes: ArrayBytes = array.retrieve_array_subset(&subset).unwrap();
            std::hint::black_box(bytes);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    retrieve_chunk_bytes,
    retrieve_chunk_vec,
    store_chunk_borrowed,
    store_chunk_shared,
    sharded_subchunk_read
);
criterion_main!(benches);

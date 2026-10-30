//! Benchmark sync and async array operations on a store with a fixed request latency.
//!
//! This measures how well array operations overlap I/O with each other and with decoding.
//! Each benchmark prints the peak number of in-flight requests it achieved.
#![allow(missing_docs, clippy::cast_possible_truncation)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{LatencyStore, compressible_u16};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset, data_type};
use zarrs::storage::store::MemoryStore;

/// Request latencies: zero measures pure overhead on an in-memory store.
const LATENCIES: [Duration; 2] = [Duration::ZERO, Duration::from_millis(2)];

/// 2 x 2 shards of 256 x 256 `u16`, each with 8 x 8 subchunks of 32 x 32.
const SHAPE: [u64; 2] = [512, 512];
const SHARD_SHAPE: [u64; 2] = [256, 256];
const CHUNK_SHAPE: [u64; 2] = [32, 32];

fn builder(sharded: bool) -> ArrayBuilder {
    let mut builder = ArrayBuilder::new(
        SHAPE.to_vec(),
        if sharded {
            SHARD_SHAPE.to_vec()
        } else {
            CHUNK_SHAPE.to_vec()
        },
        data_type::uint16(),
        0u16,
    );
    // With sharding, the bytes-to-bytes codecs apply to each subchunk
    builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
    if sharded {
        builder.subchunk_shape(CHUNK_SHAPE.to_vec());
    }
    builder
}

fn latencies_sharded() -> impl Iterator<Item = (Duration, bool)> {
    LATENCIES
        .into_iter()
        .flat_map(|latency| [(latency, false), (latency, true)])
}

fn num_elements() -> usize {
    SHAPE.iter().product::<u64>() as usize
}

fn label(sharded: bool) -> &'static str {
    if sharded { "sharded" } else { "unsharded" }
}

/// Subsets read by the benchmarks: the whole array, and a region that straddles all four shards.
fn read_subsets() -> [(&'static str, ArraySubset); 2] {
    [
        ("all", ArraySubset::new_with_shape(SHAPE)),
        (
            "straddle",
            ArraySubset::new_with_ranges(&[200..312, 200..312]),
        ),
    ]
}

fn sync_array(
    sharded: bool,
    latency: Duration,
) -> (
    Arc<LatencyStore<MemoryStore>>,
    Array<LatencyStore<MemoryStore>>,
) {
    let inner = Arc::new(MemoryStore::new());
    let array = builder(sharded).build(inner.clone(), "/").unwrap();
    array
        .store_array_subset(&array.subset_all(), compressible_u16(num_elements()))
        .unwrap();
    let store = Arc::new(LatencyStore::new(inner, latency));
    let array = builder(sharded).build(store.clone(), "/").unwrap();
    (store, array)
}

fn bench_sync_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_latency_sync_read");
    group.sample_size(10);
    for (latency, sharded) in latencies_sharded() {
        let (store, array) = sync_array(sharded, latency);
        for (name, subset) in read_subsets() {
            group.throughput(Throughput::Bytes(subset.num_elements() * 2));
            let id = format!("{}ms/{}/{name}", latency.as_millis(), label(sharded));
            store.reset();
            group.bench_function(&id, |b| {
                b.iter(|| {
                    let _: ArrayBytes = array.retrieve_array_subset(&subset).unwrap();
                });
            });
            eprintln!(
                "io_latency_sync_read/{id}: peak in-flight requests {}",
                store.peak_in_flight()
            );
        }
    }
    group.finish();
}

fn bench_sync_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_latency_sync_write");
    group.sample_size(10);
    let data = compressible_u16(num_elements());
    for (latency, sharded) in latencies_sharded() {
        let store = Arc::new(LatencyStore::new(Arc::new(MemoryStore::new()), latency));
        let array = builder(sharded).build(store.clone(), "/").unwrap();
        group.throughput(Throughput::Bytes(data.len() as u64 * 2));
        let id = format!("{}ms/{}/all", latency.as_millis(), label(sharded));
        store.reset();
        group.bench_function(&id, |b| {
            b.iter(|| {
                array
                    .store_array_subset(&array.subset_all(), &data)
                    .unwrap();
            });
        });
        eprintln!(
            "io_latency_sync_write/{id}: peak in-flight requests {}",
            store.peak_in_flight()
        );
    }
    group.finish();
}

#[cfg(feature = "async")]
mod asynchronous {
    use super::{
        Arc, Array, ArrayBytes, Criterion, Duration, LatencyStore, Throughput, builder,
        compressible_u16, label, latencies_sharded, num_elements, read_subsets,
    };
    use zarrs::storage::store::AsyncMemoryStore;

    fn runtimes() -> [(&'static str, tokio::runtime::Runtime); 2] {
        [
            (
                "multi_thread",
                tokio::runtime::Builder::new_multi_thread()
                    .enable_time()
                    .build()
                    .unwrap(),
            ),
            (
                "current_thread",
                tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .unwrap(),
            ),
        ]
    }

    async fn async_array(
        sharded: bool,
        latency: Duration,
    ) -> (
        Arc<LatencyStore<AsyncMemoryStore>>,
        Array<LatencyStore<AsyncMemoryStore>>,
    ) {
        let inner = Arc::new(AsyncMemoryStore::new());
        let array = builder(sharded).build(inner.clone(), "/").unwrap();
        array
            .async_store_array_subset(&array.subset_all(), compressible_u16(num_elements()))
            .await
            .unwrap();
        let store = Arc::new(LatencyStore::new(inner, latency));
        let array = builder(sharded).build(store.clone(), "/").unwrap();
        (store, array)
    }

    pub(super) fn bench_async_read(c: &mut Criterion) {
        let mut group = c.benchmark_group("io_latency_async_read");
        group.sample_size(10);
        for (runtime_name, runtime) in runtimes() {
            for (latency, sharded) in latencies_sharded() {
                let (store, array) = runtime.block_on(async_array(sharded, latency));
                for (name, subset) in read_subsets() {
                    group.throughput(Throughput::Bytes(subset.num_elements() * 2));
                    let id = format!(
                        "{runtime_name}/{}ms/{}/{name}",
                        latency.as_millis(),
                        label(sharded)
                    );
                    store.reset();
                    group.bench_function(&id, |b| {
                        b.to_async(&runtime).iter(|| async {
                            let _: ArrayBytes =
                                array.async_retrieve_array_subset(&subset).await.unwrap();
                        });
                    });
                    eprintln!(
                        "io_latency_async_read/{id}: peak in-flight requests {}",
                        store.peak_in_flight()
                    );
                }
            }
        }
        group.finish();
    }

    pub(super) fn bench_async_write(c: &mut Criterion) {
        let mut group = c.benchmark_group("io_latency_async_write");
        group.sample_size(10);
        let data = compressible_u16(num_elements());
        for (runtime_name, runtime) in runtimes() {
            for (latency, sharded) in latencies_sharded() {
                let store = Arc::new(LatencyStore::new(
                    Arc::new(AsyncMemoryStore::new()),
                    latency,
                ));
                let array = builder(sharded).build(store.clone(), "/").unwrap();
                group.throughput(Throughput::Bytes(data.len() as u64 * 2));
                let id = format!(
                    "{runtime_name}/{}ms/{}/all",
                    latency.as_millis(),
                    label(sharded)
                );
                store.reset();
                group.bench_function(&id, |b| {
                    b.to_async(&runtime).iter(|| async {
                        array
                            .async_store_array_subset(&array.subset_all(), &data)
                            .await
                            .unwrap();
                    });
                });
                eprintln!(
                    "io_latency_async_write/{id}: peak in-flight requests {}",
                    store.peak_in_flight()
                );
                // Owned data, with cloning outside the timed operation.
                store.reset();
                group.bench_function(format!("{id}_owned"), |b| {
                    b.to_async(&runtime).iter_batched(
                        || data.clone(),
                        |data| async {
                            array
                                .async_store_array_subset(&array.subset_all(), data)
                                .await
                                .unwrap();
                        },
                        criterion::BatchSize::LargeInput,
                    );
                });
                eprintln!(
                    "io_latency_async_write/{id}_owned: peak in-flight requests {}",
                    store.peak_in_flight()
                );
            }
        }
        group.finish();
    }
}

#[cfg(feature = "async")]
criterion_group!(
    benches,
    bench_sync_read,
    bench_sync_write,
    asynchronous::bench_async_read,
    asynchronous::bench_async_write
);
#[cfg(not(feature = "async"))]
criterion_group!(benches, bench_sync_read, bench_sync_write);
criterion_main!(benches);

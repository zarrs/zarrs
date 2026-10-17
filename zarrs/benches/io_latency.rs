//! Benchmark sync and async array operations on a store with a fixed request latency.
//!
//! This measures how well array operations overlap I/O with each other and with decoding.
//! Each benchmark prints the peak number of in-flight requests it achieved.
#![allow(missing_docs)]

mod common;

use std::cell::LazyCell;
use std::sync::Arc;
use std::time::Duration;

use common::{LatencyStore, compressible_u16};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset, Resources, data_type};
use zarrs::storage::ReadCoalescing;
use zarrs::storage::store::MemoryStore;

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

/// Each tuple adds a distinct path: overhead, subsets, latency, or coalescing.
/// Avoid crossing every one of these with every runtime and ownership variant.
fn read_cases() -> impl Iterator<Item = (Duration, bool, &'static str, bool)> {
    [
        (0, false, "all", false),
        (0, true, "all", false),
        (0, true, "straddle", false),
        (2, false, "all", false),
        (2, true, "all", false),
        (2, true, "all", true),
        (2, true, "straddle", true),
    ]
    .into_iter()
    .map(|(ms, sharded, subset, coalesced)| (Duration::from_millis(ms), sharded, subset, coalesced))
}

fn num_elements() -> usize {
    SHAPE.iter().product::<u64>() as usize
}

fn label(sharded: bool) -> &'static str {
    if sharded { "sharded" } else { "unsharded" }
}

fn read_subset(name: &str) -> ArraySubset {
    match name {
        "all" => ArraySubset::new_with_shape(SHAPE),
        "straddle" => ArraySubset::new_with_ranges(&[200..312, 200..312]),
        _ => unreachable!(),
    }
}

fn read_resources(coalesced: bool) -> Resources {
    let resources = Resources::default();
    if coalesced {
        resources.with_read_coalescing(Some(ReadCoalescing::new(1024 * 1024, 16 * 1024 * 1024)))
    } else {
        resources
    }
}

fn read_id(latency: Duration, sharded: bool, name: &str, coalesced: bool) -> String {
    format!(
        "{}ms/{}/{name}{}",
        latency.as_millis(),
        label(sharded),
        if coalesced { "/coalesced" } else { "" }
    )
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
        .store_array_subset(
            &array.subset_all(),
            compressible_u16(num_elements()),
            &Resources::default(),
        )
        .unwrap();
    let store = Arc::new(LatencyStore::new(inner, latency));
    let array = builder(sharded).build(store.clone(), "/").unwrap();
    (store, array)
}

fn bench_sync_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_latency_sync_read");
    group.sample_size(10);
    for (latency, sharded, name, coalesced) in read_cases() {
        let subset = read_subset(name);
        let fixture = LazyCell::new(|| {
            let (store, array) = sync_array(sharded, latency);
            (store, array, read_resources(coalesced))
        });
        group.throughput(Throughput::Bytes(subset.num_elements() * 2));
        let id = read_id(latency, sharded, name, coalesced);
        let mut ran = false;
        group.bench_function(&id, |b| {
            let (_, array, resources) = &*fixture;
            ran = true;
            b.iter(|| {
                let _: ArrayBytes = array.retrieve_array_subset(&subset, resources).unwrap();
            });
        });
        if ran {
            let (store, array, resources) = &*fixture;
            let peak = store.peak_in_flight();
            store.reset();
            let _: ArrayBytes = array.retrieve_array_subset(&subset, resources).unwrap();
            eprintln!(
                "io_latency_sync_read/{id}: peak in-flight requests {peak}, requests per read {}",
                store.requests()
            );
        }
    }
    group.finish();
}

fn bench_sync_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_latency_sync_write");
    group.sample_size(10);
    // Zero-latency writes are already covered by the array-operation suite.
    for sharded in [false, true] {
        let latency = Duration::from_millis(2);
        let fixture = LazyCell::new(|| {
            let store = Arc::new(LatencyStore::new(Arc::new(MemoryStore::new()), latency));
            let array = builder(sharded).build(store.clone(), "/").unwrap();
            (store, array, compressible_u16(num_elements()))
        });
        group.throughput(Throughput::Bytes(num_elements() as u64 * 2));
        let id = format!("{}ms/{}/all", latency.as_millis(), label(sharded));
        let mut ran = false;
        group.bench_function(&id, |b| {
            let (_, array, data) = &*fixture;
            ran = true;
            b.iter(|| {
                array
                    .store_array_subset(&array.subset_all(), data, &Resources::default())
                    .unwrap();
            });
        });
        if ran {
            eprintln!(
                "io_latency_sync_write/{id}: peak in-flight requests {}",
                fixture.0.peak_in_flight()
            );
        }
    }
    group.finish();
}

#[cfg(feature = "async")]
mod asynchronous {
    use super::*;
    use zarrs::storage::store::AsyncMemoryStore;

    fn runtime(name: &str) -> tokio::runtime::Runtime {
        let mut builder = match name {
            "multi_thread" => tokio::runtime::Builder::new_multi_thread(),
            "current_thread" => tokio::runtime::Builder::new_current_thread(),
            _ => unreachable!(),
        };
        builder.enable_time().build().unwrap()
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
            .async_store_array_subset(
                &array.subset_all(),
                compressible_u16(num_elements()),
                &Resources::default(),
            )
            .await
            .unwrap();
        let store = Arc::new(LatencyStore::new(inner, latency));
        let array = builder(sharded).build(store.clone(), "/").unwrap();
        (store, array)
    }

    pub(super) fn bench_async_read(c: &mut Criterion) {
        let mut group = c.benchmark_group("io_latency_async_read");
        group.sample_size(10);
        for runtime_name in ["multi_thread", "current_thread"] {
            let runtime = LazyCell::new(|| runtime(runtime_name));
            for (latency, sharded, name, coalesced) in read_cases() {
                // One genuinely waiting operation explores the other executor.
                if runtime_name == "current_thread"
                    && (latency.is_zero() || sharded || name != "all")
                {
                    continue;
                }
                let subset = read_subset(name);
                let fixture = LazyCell::new(|| {
                    let (store, array) = runtime.block_on(async_array(sharded, latency));
                    (store, array, read_resources(coalesced))
                });
                group.throughput(Throughput::Bytes(subset.num_elements() * 2));
                let id = format!(
                    "{runtime_name}/{}",
                    read_id(latency, sharded, name, coalesced)
                );
                let mut ran = false;
                group.bench_function(&id, |b| {
                    let (_, array, resources) = &*fixture;
                    ran = true;
                    b.to_async(&*runtime).iter(|| async {
                        let _: ArrayBytes = array
                            .async_retrieve_array_subset(&subset, resources)
                            .await
                            .unwrap();
                    });
                });
                if ran {
                    let (store, array, resources) = &*fixture;
                    let peak = store.peak_in_flight();
                    store.reset();
                    let _: ArrayBytes = runtime
                        .block_on(array.async_retrieve_array_subset(&subset, resources))
                        .unwrap();
                    eprintln!(
                        "io_latency_async_read/{id}: peak in-flight requests {peak}, requests per read {}",
                        store.requests()
                    );
                }
            }
        }
        group.finish();
    }

    pub(super) fn bench_async_write(c: &mut Criterion) {
        let mut group = c.benchmark_group("io_latency_async_write");
        group.sample_size(10);
        for runtime_name in ["multi_thread", "current_thread"] {
            let runtime = LazyCell::new(|| runtime(runtime_name));
            for sharded in [false, true] {
                if runtime_name == "current_thread" && sharded {
                    continue;
                }
                let latency = Duration::from_millis(2);
                let fixture = LazyCell::new(|| {
                    let store = Arc::new(LatencyStore::new(
                        Arc::new(AsyncMemoryStore::new()),
                        latency,
                    ));
                    let array = builder(sharded).build(store.clone(), "/").unwrap();
                    (store, array, compressible_u16(num_elements()))
                });
                group.throughput(Throughput::Bytes(num_elements() as u64 * 2));
                let id = format!(
                    "{runtime_name}/{}ms/{}/all",
                    latency.as_millis(),
                    label(sharded)
                );
                for owned in [false, true] {
                    if runtime_name == "current_thread" && owned {
                        continue;
                    }
                    let id = format!("{id}{}", if owned { "_owned" } else { "" });
                    let mut ran = false;
                    group.bench_function(&id, |b| {
                        let (store, array, data) = &*fixture;
                        store.reset();
                        ran = true;
                        if owned {
                            // A write can move owned data to other threads. Clone
                            // outside timing, preserving the ownership comparison.
                            b.to_async(&*runtime).iter_batched(
                                || data.clone(),
                                |data| async {
                                    array
                                        .async_store_array_subset(
                                            &array.subset_all(),
                                            data,
                                            &Resources::default(),
                                        )
                                        .await
                                        .unwrap();
                                },
                                criterion::BatchSize::LargeInput,
                            );
                        } else {
                            b.to_async(&*runtime).iter(|| async {
                                array
                                    .async_store_array_subset(
                                        &array.subset_all(),
                                        data,
                                        &Resources::default(),
                                    )
                                    .await
                                    .unwrap();
                            });
                        }
                    });
                    if ran {
                        eprintln!(
                            "io_latency_async_write/{id}: peak in-flight requests {}",
                            fixture.0.peak_in_flight()
                        );
                    }
                }
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

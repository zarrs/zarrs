//! Benchmark writing shards encoded whole (buffered) and written as they are encoded (streamed, a multipart write).
//!
//! Each case writes zstd subchunks of `u16`, with the shard index at the end or the start, to a memory store, a
//! filesystem store (in a temporary directory), or a memory store with a request latency of 1 ms (where each call of a
//! multipart writer is a request). Small subchunks (16 x 16) make many small encoded subchunks per part, and large
//! subchunks (128 x 128) few. Shards are 2 MiB decoded, and streamed in parts of 1 MiB (the default).
//!
//! - `store_chunk`: one 1024 x 1024 shard.
//! - `store_array_subset`: four 1024 x 1024 shards.
#![allow(missing_docs)]

mod common;

use std::cell::LazyCell;
use std::sync::Arc;
use std::time::Duration;

use common::{LatencyStore, compressible_u16};
use criterion::{Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zarrs::array::codec::ZstdCodec;
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecBuilder, ShardingIndexLocation};
use zarrs::array::{Array, ArrayBuilder, CodecOptions, MultipartWrite, Resources, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableWritableStorageTraits, WritableStorageTraits};

const SHAPE: [u64; 2] = [2048, 2048];
const SHARD_SHAPE: [u64; 2] = [1024, 1024];

fn array<TStorage: ?Sized + WritableStorageTraits + 'static>(
    store: Arc<TStorage>,
    index_location: ShardingIndexLocation,
    subchunk: u64,
    multipart_write: MultipartWrite,
) -> Array<TStorage> {
    let subchunk_shape = vec![std::num::NonZeroU64::new(subchunk).unwrap(); 2];
    let mut builder = ArrayBuilder::new(
        SHAPE.to_vec(),
        SHARD_SHAPE.to_vec(),
        data_type::uint16(),
        0u16,
    );
    builder.array_to_bytes_codec(Arc::new(
        ShardingCodecBuilder::new(subchunk_shape, &data_type::uint16())
            .index_location(index_location)
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
            .build(),
    ));
    let array = builder.build_arc(store, "/array").unwrap();
    array.with_codec_options(CodecOptions::default().with_multipart_write(multipart_write))
}

/// The stores of the benchmark, by name.
enum Store {
    Memory(Arc<MemoryStore>),
    Filesystem(Arc<FilesystemStore>, tempfile::TempDir),
    Latency(Arc<LatencyStore<MemoryStore>>),
}

impl Store {
    fn new(name: &str) -> Self {
        match name {
            "memory" => Self::Memory(Arc::new(MemoryStore::new())),
            "filesystem" => {
                let dir = tempfile::TempDir::new().unwrap();
                Self::Filesystem(Arc::new(FilesystemStore::new(dir.path()).unwrap()), dir)
            }
            "latency" => Self::Latency(Arc::new(LatencyStore::new(
                Arc::new(MemoryStore::new()),
                Duration::from_millis(1),
            ))),
            _ => unreachable!(),
        }
    }
}

/// The operations of the benchmark.
#[derive(Clone, Copy)]
enum Op {
    StoreChunk,
    StoreArraySubset,
}

/// Benchmark `op` on an array in `store` with `index_location`, `subchunk` and `multipart_write`.
fn run<TStorage: ?Sized + ReadableWritableStorageTraits + 'static>(
    b: &mut Bencher,
    store: Arc<TStorage>,
    (index_location, subchunk, multipart_write): (ShardingIndexLocation, u64, MultipartWrite),
    op: Op,
    data: &[u16],
) {
    let array = array(store, index_location, subchunk, multipart_write);
    let resources = Resources::default();
    match op {
        Op::StoreChunk => b.iter(|| array.store_chunk(&[0, 0], data, &resources).unwrap()),
        Op::StoreArraySubset => b.iter(|| {
            array
                .store_array_subset(&array.subset_all(), data, &resources)
                .unwrap();
        }),
    }
}

fn bench_shard_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("shard_write");
    group.sample_size(10);
    let data = LazyCell::new(|| compressible_u16(SHAPE.iter().product::<u64>() as usize));
    let shard_data =
        LazyCell::new(|| compressible_u16(SHARD_SHAPE.iter().product::<u64>() as usize));
    for (op_name, op) in [
        ("store_chunk", Op::StoreChunk),
        ("store_array_subset", Op::StoreArraySubset),
    ] {
        let elements = match op {
            Op::StoreChunk => SHARD_SHAPE.iter().product::<u64>(),
            Op::StoreArraySubset => SHAPE.iter().product::<u64>(),
        };
        group.throughput(Throughput::Bytes(elements * 2));
        for store_name in ["memory", "filesystem", "latency"] {
            for (index_name, index_location) in [
                ("end", ShardingIndexLocation::End),
                ("start", ShardingIndexLocation::Start),
            ] {
                for (subchunk_name, subchunk) in [("small", 16), ("large", 128)] {
                    for (mode, multipart_write) in [
                        ("buffered", MultipartWrite::Never),
                        ("streamed", MultipartWrite::Always),
                    ] {
                        let id = BenchmarkId::new(
                            format!("{op_name}/{store_name}/{index_name}/{subchunk_name}"),
                            mode,
                        );
                        let case = (index_location, subchunk, multipart_write);
                        // Fixtures are created only for selected cases
                        group.bench_function(id, |b| {
                            let data: &[u16] = match op {
                                Op::StoreChunk => &shard_data,
                                Op::StoreArraySubset => &data,
                            };
                            match Store::new(store_name) {
                                Store::Memory(store) => run(b, store, case, op, data),
                                Store::Filesystem(store, _dir) => run(b, store, case, op, data),
                                Store::Latency(store) => run(b, store, case, op, data),
                            }
                        });
                    }
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_shard_write);
criterion_main!(benches);

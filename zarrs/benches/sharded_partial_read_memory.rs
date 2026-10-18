//! Everyday sharded-read scenarios with one shared 32 MiB in-memory fixture.
//! The large filesystem experiment is a separate `sharded_partial_read` target.
#![allow(missing_docs)]

use std::cell::LazyCell;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use zarrs::array::{ArrayBuilder, ArrayBytes, ArraySubset, data_type};
use zarrs::storage::store::MemoryStore;

const SHAPE: [u64; 4] = [64, 4, 128, 128];
const SHARD_SHAPE: [u64; 4] = [32, 4, 128, 128];
const CHUNK_SHAPE: [u64; 4] = [1, 1, 128, 128];

fn sharded_partial_read(c: &mut Criterion) {
    // Shared across selected scenarios, but never populated for --list or an
    // unmatched filter. No filesystem feature or fixture directory is needed.
    let array = LazyCell::new(|| {
        let array = ArrayBuilder::new(
            SHAPE.to_vec(),
            SHARD_SHAPE.to_vec(),
            data_type::float64(),
            0.0f64,
        )
        .subchunk_shape(CHUNK_SHAPE.to_vec())
        .build(Arc::new(MemoryStore::new()), "/")
        .unwrap();
        let data: Vec<f64> = (0..SHAPE.iter().product::<u64>())
            .map(|i| (i % 1024 + 1) as f64)
            .collect();
        array
            .store_array_subset(&array.subset_all(), data.as_slice())
            .unwrap();
        array
    });
    let mut group = c.benchmark_group("sharded_partial_read_memory");
    for (name, end) in [
        ("full_array", SHAPE[0]),
        ("partial_shard_aligned", SHARD_SHAPE[0]),
        ("partial_shard_unaligned", SHARD_SHAPE[0] - 1),
    ] {
        let subset = ArraySubset::new_with_ranges(&[0..end, 0..SHAPE[1], 0..SHAPE[2], 0..SHAPE[3]]);
        group.throughput(Throughput::Bytes(subset.num_elements() * 8));
        group.bench_function(name, |b| {
            let array = &*array;
            b.iter(|| {
                let bytes: ArrayBytes = array.retrieve_array_subset(&subset).unwrap();
                std::hint::black_box(bytes);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, sharded_partial_read);
criterion_main!(benches);

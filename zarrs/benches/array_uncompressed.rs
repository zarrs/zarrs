//! Benchmark uncompressed unsharded and sharded arrays.
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn array_write_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_write_all");
    for size in [128u64, 256u64, 512u64].iter() {
        let num_elements: u64 = size * size * size;
        group.throughput(Throughput::Bytes(num_elements));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let store = zarrs::storage::store::MemoryStore::new();
                let array = zarrs::array::ArrayBuilder::new(
                    vec![size; 3],
                    vec![32; 3],
                    zarrs::array::data_type::uint8(),
                    0u8,
                )
                .build(store.into(), "/")
                .unwrap();
                let data = vec![1u8; num_elements.try_into().unwrap()];
                let subset = zarrs::array_subset::ArraySubset::new_with_shape(vec![size; 3]);
                array.store_array_subset(&subset, &data).unwrap();
            });
        });
    }
    group.finish();
}

fn array_write_all_sharded(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_write_all_sharded");
    for size in [128u64, 256u64, 512u64].iter() {
        let num_elements: u64 = size * size * size;
        group.throughput(Throughput::Bytes(num_elements));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let store = zarrs::storage::store::MemoryStore::new();
                let array = zarrs::array::ArrayBuilder::new(
                    vec![size; 3],
                    vec![size; 3],
                    zarrs::array::data_type::uint16(),
                    0u16,
                )
                .subchunk_shape(vec![32; 3])
                .build(store.into(), "/")
                .unwrap();
                let data = vec![1u16; num_elements.try_into().unwrap()];
                let subset = zarrs::array_subset::ArraySubset::new_with_shape(vec![size; 3]);
                array.store_array_subset(&subset, &data).unwrap();
            });
        });
    }
    group.finish();
}

fn array_read_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_read_all");
    for size in [128u64, 256u64, 512u64].iter() {
        let num_elements: u64 = size * size * size;
        group.throughput(Throughput::Bytes(num_elements));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Write the data
            let store = zarrs::storage::store::MemoryStore::new();
            let array = zarrs::array::ArrayBuilder::new(
                vec![size; 3],
                vec![32; 3],
                zarrs::array::data_type::uint16(),
                0u16,
            )
            .build(store.into(), "/")
            .unwrap();
            let data = vec![1u16; num_elements.try_into().unwrap()];
            let subset = zarrs::array_subset::ArraySubset::new_with_shape(vec![size; 3]);
            array.store_array_subset(&subset, &data).unwrap();

            // Benchmark reading the data
            b.iter(|| {
                let _bytes: zarrs::array::ArrayBytes =
                    array.retrieve_array_subset(&subset).unwrap();
            });
        });
    }
    group.finish();
}

fn array_read_all_sharded(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_read_all_sharded");
    for size in [128u64, 256u64, 512u64].iter() {
        let num_elements: u64 = size * size * size;
        group.throughput(Throughput::Bytes(num_elements));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Write the data
            let store = zarrs::storage::store::MemoryStore::new();
            let array = zarrs::array::ArrayBuilder::new(
                vec![size; 3],
                vec![size; 3],
                zarrs::array::data_type::uint8(),
                1u8,
            )
            .subchunk_shape(vec![32; 3])
            .build(store.into(), "/")
            .unwrap();
            let data = vec![0u8; num_elements.try_into().unwrap()];
            let subset = zarrs::array_subset::ArraySubset::new_with_shape(vec![size; 3]);
            array.store_array_subset(&subset, &data).unwrap();

            // Benchmark reading the data
            b.iter(|| {
                let _bytes: zarrs::array::ArrayBytes =
                    array.retrieve_array_subset(&subset).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    array_write_all,
    array_read_all,
    array_write_all_sharded,
    array_read_all_sharded
);
criterion_main!(benches);

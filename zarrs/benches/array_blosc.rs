//! Benchmark the `blosc` codec in various configurations.
#![allow(missing_docs)]

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zarrs::array::codec::BloscCodec;
use zarrs::metadata_ext::codec::blosc::{BloscCompressionLevel, BloscCompressor, BloscShuffleMode};

fn array_blosc_write_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_blosc_write_all");
    for size in [128u64, 256u64, 512u64].iter() {
        let num_elements: u64 = size * size * size;
        group.throughput(Throughput::Bytes(num_elements));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let store = zarrs::storage::store::MemoryStore::new();
                let array = zarrs::array::ArrayBuilder::new(
                    vec![size; 3],
                    vec![32; 3],
                    zarrs::array::DataType::UInt8,
                    0u8,
                )
                .bytes_to_bytes_codecs(vec![Arc::new(
                    BloscCodec::new(
                        BloscCompressor::BloscLZ,
                        BloscCompressionLevel::try_from(9).unwrap(),
                        None,
                        BloscShuffleMode::BitShuffle,
                        Some(2),
                    )
                    .unwrap(),
                )])
                .build(store.into(), "/")
                .unwrap();
                let data = vec![1u8; num_elements.try_into().unwrap()];
                let subset = zarrs::array_subset::ArraySubset::new_with_shape(vec![size; 3]);
                array.store_array_subset_elements(&subset, &data).unwrap();
            });
        });
    }
    group.finish();
}

fn array_blosc_read_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_blosc_read_all");
    for size in [128u64, 256u64, 512u64].iter() {
        let num_elements: u64 = size * size * size;
        group.throughput(Throughput::Bytes(num_elements));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Write the data
            let store = zarrs::storage::store::MemoryStore::new();
            let array = zarrs::array::ArrayBuilder::new(
                vec![size; 3],
                vec![32; 3],
                zarrs::array::DataType::UInt8,
                0u8,
            )
            .bytes_to_bytes_codecs(vec![Arc::new(
                BloscCodec::new(
                    BloscCompressor::BloscLZ,
                    BloscCompressionLevel::try_from(9).unwrap(),
                    None,
                    BloscShuffleMode::BitShuffle,
                    Some(2),
                )
                .unwrap(),
            )])
            .build(store.into(), "/")
            .unwrap();
            let data = vec![1u8; num_elements.try_into().unwrap()];
            let subset = zarrs::array_subset::ArraySubset::new_with_shape(vec![size; 3]);
            array.store_array_subset_elements(&subset, &data).unwrap();

            // Benchmark reading the data
            b.iter(|| {
                let _bytes = array.retrieve_array_subset(&subset).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, array_blosc_write_all, array_blosc_read_all);
criterion_main!(benches);

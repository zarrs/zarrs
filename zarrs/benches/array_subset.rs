//! Benchmark the `ArraySubset` indices iterators.
#![allow(missing_docs)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use zarrs::array::ArraySubset;

fn array_subset_indices_iterator(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("array_subset_indices_iterator".to_string());
    group.plot_config(plot_config);

    let array_subset_size = 16;
    let array_subset = ArraySubset::new_with_shape(vec![array_subset_size; 3]);
    group.throughput(Throughput::Elements(array_subset.num_elements()));
    group.bench_function(BenchmarkId::new("size", array_subset_size), |b| {
        b.iter(|| {
            array_subset.indices().into_iter().for_each(|indices| {
                std::hint::black_box(indices.first().unwrap());
            });
        });
    });
    group.finish();

    let mut group = c.benchmark_group("array_subset_linearised_indices_iterator".to_string());
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    let array_shape = vec![array_subset_size * 2; 3];
    let array_subset = ArraySubset::new_with_start_shape(
        vec![array_subset_size / 2; 3],
        vec![array_subset_size; 3],
    )
    .unwrap();
    group.throughput(Throughput::Elements(array_subset.num_elements()));
    group.bench_function(BenchmarkId::new("size", array_subset_size), |b| {
        b.iter(|| {
            array_subset
                .linearised_indices(&array_shape)
                .unwrap()
                .into_iter()
                .for_each(|index| {
                    std::hint::black_box(index);
                });
        });
    });
}

criterion_group!(benches, array_subset_indices_iterator);
criterion_main!(benches);

//! Benchmark the `ArraySubset` indices iterator.
#![allow(missing_docs)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use zarrs::array_subset::ArraySubset;

fn array_subset_indices_iterator(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group(format!("array_subset_indices_iterator"));
    group.plot_config(plot_config);

    for array_subset_size in [4, 16, 64, 256] {
        let array_subset = ArraySubset::new_with_shape(vec![array_subset_size; 3]);
        group.throughput(Throughput::Elements(array_subset.num_elements()));
        group.bench_function(BenchmarkId::new("size", array_subset_size), |b| {
            b.iter(|| {
                array_subset.indices().into_iter().for_each(|indices| {
                    std::hint::black_box(indices.first().unwrap());
                })
            });
        });
    }
}

criterion_group!(benches, array_subset_indices_iterator);
criterion_main!(benches);

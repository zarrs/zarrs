//! Benchmark `FillValue::equas_all()`.
#![allow(missing_docs)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use zarrs::array::FillValue;

fn fill_value(c: &mut Criterion) {
    for element_size in [1, 2, 4, 8, 16] {
        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
        let mut group = c.benchmark_group(format!("fill_value_{element_size}"));
        group.plot_config(plot_config);

        for size in &[32, 64, 128] {
            let size3 = size * size * size;
            let num_elements = size3 / element_size;
            let fill_value: FillValue = FillValue::new(vec![0; element_size]);

            let data = vec![0u8; num_elements * element_size];
            group.throughput(Throughput::Bytes((num_elements * element_size) as u64));
            group.bench_function(BenchmarkId::new("equals_all", size3), |b| {
                b.iter(|| fill_value.equals_all(&data));
            });
        }
    }
}

criterion_group!(benches, fill_value);
criterion_main!(benches);

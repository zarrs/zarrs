//! Benchmark `FillValue::equals_all()`.
#![allow(missing_docs)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use std::hint::black_box;
use zarrs::array::FillValue;

fn fill_value(c: &mut Criterion) {
    for element_size in [1usize, 2, 4, 8, 16] {
        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
        let mut group = c.benchmark_group(format!("fill_value_{element_size}"));
        group.plot_config(plot_config);

        for size in &[32, 64, 128] {
            let size3 = size * size * size;
            let fill_value = FillValue::new((1..=u8::try_from(element_size).unwrap()).collect());
            for (name, offset, extra_elements) in [
                ("aligned", 0, 0),
                ("tail", 0, 1),
                ("unaligned", 1, 0),
                ("unaligned_tail", 1, 1),
            ] {
                let len = size3 + extra_elements * element_size;
                // Control the alignment rather than relying on a byte allocation's alignment.
                let mut storage = vec![0u128; (len + offset).div_ceil(16)];
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
                let data = &mut bytes[offset..offset + len];
                data.copy_from_slice(&fill_value.as_ne_bytes().repeat(len / element_size));
                group.throughput(Throughput::Bytes(len as u64));
                group.bench_function(BenchmarkId::new(name, size3), |b| {
                    b.iter(|| black_box(&fill_value).equals_all(black_box(data)));
                });
            }
        }
    }
}

criterion_group!(benches, fill_value);
criterion_main!(benches);

//! Benchmark fill-value comparisons and output filling.
#![allow(missing_docs)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use std::hint::black_box;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, FillValue,
    copy_fill_value_into, data_type,
};

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

fn fill_view(c: &mut Criterion) {
    let shape = [64, 1024];
    for element_size in [1usize, 3, 4, 17] {
        let mut group = c.benchmark_group(format!("fill_view_{element_size}"));
        let pattern: Vec<u8> = (1..=u8::try_from(element_size).unwrap()).collect();
        for (name, subset) in [
            ("contiguous", ArraySubset::new_with_shape(shape)),
            ("strided", ArraySubset::new_with_ranges(&[0..64, 1..1023])),
        ] {
            let mut storage = vec![0u8; 64 * 1024 * element_size];
            // SAFETY: this is the only view of storage.
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut storage),
                    element_size,
                    &shape,
                    subset,
                )
                .unwrap()
            };
            group.throughput(Throughput::Bytes(view.num_elements() * element_size as u64));
            group.bench_function(name, |b| {
                b.iter(|| black_box(&mut view).fill(black_box(&pattern)).unwrap());
            });
        }
    }
}

fn optional_fill_value(c: &mut Criterion) {
    for depth in [1, 3] {
        let mut group = c.benchmark_group(format!("optional_fill_{depth}"));
        let mut fill_value = FillValue::from(0x1234_5678u32);
        let mut data_type = data_type::uint32();
        for _ in 0..depth {
            fill_value = fill_value.into_optional();
            data_type = data_type.to_optional();
        }
        for num_elements in [1usize, 1024] {
            for mismatch in [false, true] {
                let mut bytes =
                    ArrayBytes::new_flen(0x1234_5678u32.to_ne_bytes().repeat(num_elements));
                for level in 0..depth {
                    let mut mask = vec![1u8; num_elements];
                    if mismatch && level == depth - 1 {
                        mask[0] = 0;
                    }
                    bytes = bytes.with_optional_mask(mask);
                }
                let name = if mismatch {
                    "mask_mismatch"
                } else {
                    "equals_all"
                };
                group.bench_function(BenchmarkId::new(name, num_elements), |b| {
                    b.iter(|| black_box(&bytes).is_fill_value(black_box(&fill_value)));
                });
            }
            group.bench_function(BenchmarkId::new("construct", num_elements), |b| {
                b.iter(|| {
                    ArrayBytes::new_fill_value(
                        black_box(&data_type),
                        num_elements as u64,
                        black_box(&fill_value),
                    )
                    .unwrap()
                });
            });
        }
    }
}

fn nested_target<'a>(
    data_view: &'a mut ArrayBytesFixedDisjointView<'a>,
    mask_views: &'a mut [ArrayBytesFixedDisjointView<'a>],
) -> ArrayBytesDecodeIntoTarget<'a> {
    if let Some((mask, rest)) = mask_views.split_first_mut() {
        ArrayBytesDecodeIntoTarget::Optional(Box::new(nested_target(data_view, rest)), mask)
    } else {
        ArrayBytesDecodeIntoTarget::Fixed(data_view)
    }
}

fn fill_output(c: &mut Criterion) {
    let shape = [64, 1024];
    for depth in [0usize, 1, 3] {
        let mut group = c.benchmark_group(format!("fill_output_{depth}"));
        for null in [false, true] {
            if depth == 0 && null {
                continue;
            }
            let mut data_type = data_type::uint32();
            let mut fill_value = FillValue::from(0x1234_5678u32);
            for _ in 0..depth {
                data_type = data_type.to_optional();
                fill_value = fill_value.into_optional();
            }
            if null {
                fill_value = FillValue::new_optional_null();
            }
            for (layout, subset) in [
                ("contiguous", ArraySubset::new_with_shape(shape)),
                ("strided", ArraySubset::new_with_ranges(&[0..64, 1..1023])),
            ] {
                // Allocate output buffers outside the measurement. Construct views
                // and decode targets inside each timed iteration.
                let mut storage = vec![0u8; 64 * 1024 * 4];
                let mut masks = vec![vec![0u8; 64 * 1024]; depth];
                group.throughput(Throughput::Bytes(
                    subset.num_elements() * (4 + depth as u64),
                ));
                let name = format!("{layout}_{}", if null { "null" } else { "value" });
                group.bench_function(name, |b| {
                    b.iter(|| {
                        {
                            // SAFETY: each view is the only view of its backing buffer.
                            let mut data_view = unsafe {
                                ArrayBytesFixedDisjointView::new(
                                    UnsafeCellSlice::new(&mut storage),
                                    4,
                                    &shape,
                                    subset.clone(),
                                )
                                .unwrap()
                            };
                            let mut mask_views: Vec<_> = masks
                                .iter_mut()
                                .map(|mask| unsafe {
                                    ArrayBytesFixedDisjointView::new(
                                        UnsafeCellSlice::new(mask),
                                        1,
                                        &shape,
                                        subset.clone(),
                                    )
                                    .unwrap()
                                })
                                .collect();
                            let target = nested_target(&mut data_view, &mut mask_views);
                            copy_fill_value_into(
                                black_box(&data_type),
                                black_box(&fill_value),
                                target,
                            )
                            .unwrap();
                        }
                        black_box(&storage);
                        black_box(&masks);
                    });
                });
            }
        }
    }
}

criterion_group!(
    benches,
    fill_value,
    fill_view,
    optional_fill_value,
    fill_output
);
criterion_main!(benches);

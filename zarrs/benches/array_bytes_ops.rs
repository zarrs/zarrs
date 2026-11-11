//! Benchmark gathering from and scattering into strided array bytes.
#![allow(missing_docs, clippy::cast_possible_truncation)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{ArrayBytes, ArrayBytesFixedDisjointView, ArraySubset, data_type};

const ELEMENTS: u64 = 32_768;

/// Gather and scatter `ELEMENTS` `f32` elements in a contiguous, strided (one column of two), or block (half of each row) layout.
fn array_bytes_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_bytes_ops");
    group.throughput(Throughput::Bytes(ELEMENTS * 4));
    let values: Vec<u8> = (0..ELEMENTS as u32 * 4).map(|i| i as u8).collect();
    let fill_value = 1.5f32.to_ne_bytes();
    let positions: Vec<usize> = (0..ELEMENTS as usize).collect();
    let sparse_positions: Vec<usize> = positions.iter().copied().step_by(7).collect();
    let sparse_values = &values[..sparse_positions.len() * 4];

    for (layout, shape, subset) in [
        (
            "contiguous",
            [ELEMENTS, 1],
            ArraySubset::new_with_ranges(&[0..ELEMENTS, 0..1]),
        ),
        (
            "strided",
            [ELEMENTS, 2],
            ArraySubset::new_with_ranges(&[0..ELEMENTS, 0..1]),
        ),
        (
            "block",
            [256, 256],
            ArraySubset::new_with_ranges(&[0..256, 0..128]),
        ),
    ] {
        let mut output = vec![0u8; shape.iter().product::<u64>() as usize * 4];

        let input = ArrayBytes::new_flen(output.clone());
        group.bench_function(BenchmarkId::new("extract_array_subset", layout), |b| {
            b.iter(|| {
                std::hint::black_box(
                    input
                        .extract_array_subset(&subset, &shape, &data_type::float32())
                        .unwrap(),
                );
            });
        });

        let mut bench_view = |name: &str, op: &dyn Fn(&mut ArrayBytesFixedDisjointView)| {
            group.bench_function(BenchmarkId::new(name, layout), |b| {
                b.iter(|| {
                    // SAFETY: each iteration owns the only view of this output buffer.
                    let mut view = unsafe {
                        ArrayBytesFixedDisjointView::new(
                            UnsafeCellSlice::new(&mut output),
                            4,
                            &shape,
                            subset.clone(),
                        )
                        .unwrap()
                    };
                    op(&mut view);
                });
                std::hint::black_box(&output);
            });
        };
        bench_view("fill", &|view| view.fill(&fill_value).unwrap());
        bench_view("fill_uniform", &|view| view.fill(&[0; 4]).unwrap());
        bench_view("copy_elements_from_slice", &|view| {
            view.copy_elements_from_slice(&positions, &values).unwrap();
        });
        bench_view("fill_elements", &|view| {
            view.fill_elements(&positions, &fill_value).unwrap();
        });
        bench_view("copy_elements_from_slice_sparse", &|view| {
            view.copy_elements_from_slice(&sparse_positions, sparse_values)
                .unwrap();
        });
    }
    group.finish();
}

criterion_group!(benches, array_bytes_ops);
criterion_main!(benches);

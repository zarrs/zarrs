//! Benchmark semantic equality of optional array bytes.
#![allow(missing_docs, clippy::cast_possible_truncation)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use zarrs::array::{ArrayBytes, ArrayBytesOffsets};

const N: usize = 1 << 20;

/// A deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A mask with each element present with probability `1 - null_fraction`.
    fn mask(&mut self, null_fraction: f64) -> Vec<u8> {
        #[allow(clippy::cast_precision_loss)]
        (0..N)
            .map(|_| u8::from((self.next() as f64 / u64::MAX as f64) >= null_fraction))
            .collect()
    }
}

/// Two copies of fixed data of `width` bytes per element, differing beneath `hidden` elements.
fn fixed_pair(width: usize, hidden: &[bool]) -> (Vec<u8>, Vec<u8>) {
    let a: Vec<u8> = (0..N * width).map(|i| (i % 251) as u8).collect();
    let mut b = a.clone();
    for (i, &hidden) in hidden.iter().enumerate() {
        if hidden {
            b[i * width] ^= 0xff;
        }
    }
    (a, b)
}

/// Two copies of variable data, differing beneath `hidden` elements, with differing offset types.
fn vlen_pair(hidden: &[bool]) -> (ArrayBytes<'static>, ArrayBytes<'static>) {
    let mut bytes_a = vec![];
    let mut bytes_b = vec![];
    let mut offsets_a = vec![0u32];
    let mut offsets_b = vec![0u64];
    for (i, &hidden) in hidden.iter().enumerate() {
        let len = i % 16;
        let element: Vec<u8> = (0..len).map(|j| (i + j) as u8).collect();
        bytes_a.extend_from_slice(&element);
        if hidden {
            bytes_b.extend_from_slice(b"xyz");
        } else {
            bytes_b.extend_from_slice(&element);
        }
        offsets_a.push(bytes_a.len() as u32);
        offsets_b.push(bytes_b.len() as u64);
    }
    (
        ArrayBytes::new_vlen(bytes_a, ArrayBytesOffsets::new(offsets_a).unwrap()).unwrap(),
        ArrayBytes::new_vlen(bytes_b, ArrayBytesOffsets::new(offsets_b).unwrap()).unwrap(),
    )
}

fn array_bytes_eq(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_bytes_optional_eq");
    group.throughput(Throughput::Elements(N as u64));
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    for (name, width, null_fraction) in [
        ("fixed1_null50", 1, 0.5),
        ("fixed4_null50", 4, 0.5),
        ("fixed4_null5", 4, 0.05),
        ("fixed8_null50", 8, 0.5),
    ] {
        let mask = rng.mask(null_fraction);
        let hidden: Vec<bool> = mask.iter().map(|&m| m == 0).collect();
        let (a, b) = fixed_pair(width, &hidden);
        let a = ArrayBytes::new_flen(a).with_optional_mask(mask.clone());
        let b = ArrayBytes::new_flen(b).with_optional_mask(mask.clone());
        assert_eq!(a, b);
        group.bench_function(format!("{name}_eq"), |bench| {
            bench.iter(|| black_box(&a) == black_box(&b));
        });
        if name == "fixed4_null50" {
            group.bench_function(format!("{name}_structural"), |bench| {
                bench.iter(|| black_box(&a) == black_box(&a));
            });
            // Differs only at the last element, which is visible
            let mut hidden_d = hidden.clone();
            hidden_d[N - 1] = false;
            let (mut data_c, data_d) = fixed_pair(width, &hidden_d);
            data_c[(N - 1) * width] ^= 0xff;
            let mut mask_cd = mask.clone();
            mask_cd[N - 1] = 1;
            let c_ = ArrayBytes::new_flen(data_c).with_optional_mask(mask_cd.clone());
            let d = ArrayBytes::new_flen(data_d).with_optional_mask(mask_cd);
            assert_ne!(c_, d);
            group.bench_function(format!("{name}_ne_last"), |bench| {
                bench.iter(|| black_box(&c_) == black_box(&d));
            });
        }
    }

    // Depth 2: each layer 25% null
    let inner_mask = rng.mask(0.25);
    let outer_mask = rng.mask(0.25);
    let hidden: Vec<bool> = inner_mask
        .iter()
        .zip(&outer_mask)
        .map(|(&i, &o)| i == 0 || o == 0)
        .collect();
    let (a, b) = fixed_pair(4, &hidden);
    let mut inner_mask_b = inner_mask.clone();
    for (m, &o) in inner_mask_b.iter_mut().zip(&outer_mask) {
        if o == 0 {
            *m ^= 1;
        }
    }
    let a = ArrayBytes::new_flen(a)
        .with_optional_mask(inner_mask)
        .with_optional_mask(outer_mask.clone());
    let b = ArrayBytes::new_flen(b)
        .with_optional_mask(inner_mask_b)
        .with_optional_mask(outer_mask);
    assert_eq!(a, b);
    group.bench_function("fixed4_depth2_eq", |bench| {
        bench.iter(|| black_box(&a) == black_box(&b));
    });

    // Variable length
    let mask = rng.mask(0.5);
    let hidden: Vec<bool> = mask.iter().map(|&m| m == 0).collect();
    let (a, b) = vlen_pair(&hidden);
    let a = a.with_optional_mask(mask.clone());
    let b = b.with_optional_mask(mask);
    assert_eq!(a, b);
    group.bench_function("vlen_null50_eq", |bench| {
        bench.iter(|| black_box(&a) == black_box(&b));
    });

    group.finish();
}

criterion_group!(benches, array_bytes_eq);
criterion_main!(benches);

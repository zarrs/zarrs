//! Compare fresh and thread-local Zstd compression contexts without a size cutoff.
//!
//! Run with `cargo bench -p zarrs --bench zstd_context -- --noplot`.
//! Pass `--memory` instead to print context allocation sizes as CSV.
#![allow(missing_docs)]

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput};
use zarrs::array::CowBytes;
use zarrs::array::codec::{ZstdCodec, ZstdCodecOptions};
use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions};
use zstd::zstd_safe::{self, CCtx, CParameter};

const SIZES: &[usize] = &[
    64,
    256,
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
];
const LEVELS: &[i32] = &[1, 3, 9, 19];
const PATTERNS: &[&str] = &["zeros", "smooth_u16", "random"];

// Use a raw context to measure memory independently of the production thread-local cache.
fn compress(context: &mut CCtx<'static>, data: &[u8], level: i32) -> Vec<u8> {
    context
        .set_parameter(CParameter::CompressionLevel(level))
        .unwrap();
    context
        .set_parameter(CParameter::ChecksumFlag(false))
        .unwrap();
    let mut output = Vec::with_capacity(zstd_safe::compress_bound(data.len()));
    context.compress2(&mut output, data).unwrap();
    output
}

fn data(pattern: &str, size: usize) -> Vec<u8> {
    match pattern {
        "zeros" => vec![0; size],
        "smooth_u16" => common::compressible_u16(size / 2)
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        "random" => common::random_bytes(size),
        _ => unreachable!(),
    }
}

fn sizes(level: i32) -> impl Iterator<Item = usize> {
    // Keep the most expensive level's run time practical.
    SIZES
        .iter()
        .copied()
        .filter(move |&size| level != 19 || size <= 4 * 1024 * 1024)
}

fn memory() {
    println!(
        "pattern,level,input_bytes,encoded_bytes,context_bytes,after_one_2k_encode,after_256_2k_encodes"
    );
    for &pattern in PATTERNS {
        for &level in LEVELS {
            for size in sizes(level) {
                let data = data(pattern, size);
                let mut context = CCtx::create();
                let encoded_len = compress(&mut context, &data, level).len();
                let context_size = context.sizeof();
                let small = &data[..data.len().min(2048)];
                drop(compress(&mut context, small, level));
                let after_one = context.sizeof();
                for _ in 1..256 {
                    drop(compress(&mut context, small, level));
                }
                println!(
                    "{pattern},{level},{size},{encoded_len},{context_size},{after_one},{}",
                    context.sizeof()
                );
            }
        }
    }
}

fn timings(c: &mut Criterion) {
    for &pattern in PATTERNS {
        for &level in LEVELS {
            let mut group = c.benchmark_group(format!("zstd_context/{pattern}/level_{level}"));
            group.sampling_mode(SamplingMode::Flat);
            for size in sizes(level) {
                let data = data(pattern, size);
                let fresh = ZstdCodec::new(level, false).with_options(
                    ZstdCodecOptions::default().with_compression_context_reuse_threshold(0),
                );
                let reused = ZstdCodec::new(level, false).with_options(
                    ZstdCodecOptions::default()
                        .with_compression_context_reuse_threshold(usize::MAX),
                );
                let options = CodecOptions::default();
                let expected = compress(&mut CCtx::create(), &data, level);
                for codec in [&fresh, &reused] {
                    assert_eq!(
                        &*codec.encode(CowBytes::Borrowed(&data), &options).unwrap(),
                        expected
                    );
                }
                group.throughput(Throughput::Bytes(size as u64));
                group.bench_function(BenchmarkId::new("fresh", size), |b| {
                    b.iter(|| {
                        fresh
                            .encode(CowBytes::Borrowed(black_box(&data)), &options)
                            .unwrap()
                    });
                });
                group.bench_function(BenchmarkId::new("reused", size), |b| {
                    b.iter(|| {
                        reused
                            .encode(CowBytes::Borrowed(black_box(&data)), &options)
                            .unwrap()
                    });
                });
            }
            group.finish();
        }
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--memory") {
        memory();
    } else {
        let mut criterion = Criterion::default()
            .sample_size(20)
            .warm_up_time(Duration::from_millis(100))
            .measurement_time(Duration::from_millis(500))
            .configure_from_args();
        timings(&mut criterion);
        criterion.final_summary();
    }
}

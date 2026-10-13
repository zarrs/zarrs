//! Benchmark various codecs.
#![allow(missing_docs)]

use zarrs::array::{CodecSpecificOptions, CowBytes};

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
#[cfg(feature = "zstd")]
use zarrs::array::codec::ZstdCodec;
use zarrs::array::codec::bytes_to_bytes::blosc::{BloscCompressor, BloscShuffleMode};
use zarrs::array::codec::{BloscCodec, BytesCodec, ShardingCodecBuilder};
use zarrs::array::{BytesRepresentation, Element, Endianness, data_type};
use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions, UnboundArrayToBytesCodecTraits};
use zarrs_data_type::FillValue;

fn codec_bytes(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("codec_bytes");
    group.plot_config(plot_config);

    // Set the endianness to be the opposite of the target endianness, so the codec does work
    #[cfg(target_endian = "big")]
    let codec = BytesCodec::new(Some(Endianness::Little));
    #[cfg(target_endian = "little")]
    let codec = BytesCodec::new(Some(Endianness::Big));

    let fill_value = FillValue::from(0u16);
    let codec = codec
        .with_context(
            data_type::uint16(),
            fill_value,
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    for size in &[32, 64, 128, 256, 512] {
        let num_elements = size * size * size;
        let shape = [num_elements.try_into().unwrap(); 1];
        let data = vec![0u16; num_elements.try_into().unwrap()];
        let bytes = Element::into_array_bytes(&data_type::uint16(), data).unwrap();
        group.throughput(Throughput::Bytes(num_elements * 2));
        // encode and decode have the same implementation
        group.bench_function(BenchmarkId::new("encode_decode", num_elements), |b| {
            b.iter(|| {
                codec
                    .encode(bytes.clone(), &shape, &CodecOptions::default())
                    .unwrap()
            });
        });
    }
}

fn codec_blosc(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("codec_blosc");
    group.plot_config(plot_config);

    let codec = BloscCodec::new(
        BloscCompressor::BloscLZ,
        9.try_into().unwrap(),
        None,
        BloscShuffleMode::BitShuffle,
        Some(2),
    )
    .unwrap();

    for size in &[32, 64, 128, 256, 512] {
        let size3 = size * size * size;
        let rep = BytesRepresentation::FixedSize(size3);

        let data_decoded: Vec<u8> = (0..size3).map(|i| i as u8).collect();
        let data_encoded = codec
            .encode(CowBytes::Borrowed(&data_decoded), &CodecOptions::default())
            .unwrap();
        group.throughput(Throughput::Bytes(size3));
        group.bench_function(BenchmarkId::new("encode", size3), |b| {
            b.iter(|| {
                codec
                    .encode(CowBytes::Borrowed(&data_decoded), &CodecOptions::default())
                    .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("decode", size3), |b| {
            b.iter(|| {
                codec
                    .decode(
                        CowBytes::Borrowed(&data_encoded),
                        &rep,
                        &CodecOptions::default(),
                    )
                    .unwrap()
            });
        });
    }
}

/// Encode and decode a 512 x 512 shard in memory.
fn codec_sharding(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_sharding");
    let nz = |v: u64| std::num::NonZeroU64::new(v).unwrap();
    let shape = [nz(512), nz(512)];
    let data: Vec<u16> = (0..512 * 512u32)
        .map(|i| u16::try_from((i / 64) % 4096).unwrap() ^ u16::try_from(i % 3).unwrap())
        .collect();
    let bytes = Element::into_array_bytes(&data_type::uint16(), data).unwrap();
    for (subchunk, zstd) in [(32, false), (32, true), (128, true)] {
        #[cfg_attr(not(feature = "zstd"), allow(unused_mut))]
        let mut builder = ShardingCodecBuilder::new(vec![nz(subchunk); 2], &data_type::uint16());
        #[cfg(not(feature = "zstd"))]
        if zstd {
            continue;
        }
        #[cfg(feature = "zstd")]
        if zstd {
            builder.bytes_to_bytes_codecs(vec![std::sync::Arc::new(ZstdCodec::new(1, false))]);
        }
        let codec = builder
            .build_arc()
            .with_context(
                data_type::uint16(),
                FillValue::from(0u16),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        let encoded = codec
            .encode(bytes.clone(), &shape, &CodecOptions::default())
            .unwrap()
            .into_static();
        let label = format!("{subchunk}{}", if zstd { "_zstd" } else { "" });
        group.throughput(Throughput::Bytes(bytes.size() as u64));
        group.bench_function(BenchmarkId::new("encode", &label), |b| {
            b.iter(|| {
                codec
                    .encode(bytes.clone(), &shape, &CodecOptions::default())
                    .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("decode", &label), |b| {
            b.iter(|| {
                codec
                    .decode(encoded.clone(), &shape, &CodecOptions::default())
                    .unwrap()
            });
        });
    }
}

criterion_group!(benches, codec_bytes, codec_blosc, codec_sharding);
criterion_main!(benches);

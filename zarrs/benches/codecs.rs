//! Benchmark various codecs.
#![allow(missing_docs)]

use zarrs::array::{ArrayBytes, ArraySubset, CodecSpecificOptions, CowBytes};

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
#[cfg(feature = "zstd")]
use zarrs::array::codec::ZstdCodec;
use zarrs::array::codec::bytes_to_bytes::blosc::{BloscCompressor, BloscShuffleMode};
use zarrs::array::codec::{BloscCodec, BytesCodec, PackBitsCodec, ShardingCodecBuilder};
use zarrs::array::{BytesRepresentation, Element, Endianness, data_type};
use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions, UnboundArrayToBytesCodecTraits};
use zarrs_data_type::FillValue;
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

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

/// Encode and decode 1 Mi elements with the `packbits` codec.
///
/// Also partially decode 8 columns of the elements as a 1024x1024 array, which unpacks each short row separately.
fn codec_packbits(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_packbits");
    let num_elements = 1024 * 1024;
    let shape = [std::num::NonZeroU64::new(num_elements).unwrap()];
    let shape_2d = [std::num::NonZeroU64::new(1024).unwrap(); 2];
    let subset = ArraySubset::new_with_ranges(&[0..1024, 0..8]);
    let float64: Vec<f64> = (0..num_elements).map(|i| i as f64).collect();
    let float64 = Element::into_array_bytes(&data_type::float64(), float64).unwrap();
    let uint4: Vec<u8> = (0..num_elements).map(|i| (i % 16) as u8).collect();
    let uint4 = Element::into_array_bytes(&data_type::uint4(), uint4).unwrap();
    let bool: Vec<bool> = (0..num_elements).map(|i| i % 3 == 0).collect();
    let bool = Element::into_array_bytes(&data_type::bool(), bool).unwrap();
    let configurations = [
        ("float64", data_type::float64(), float64.clone(), None),
        (
            "float64_last_bit_62",
            data_type::float64(),
            float64,
            Some(62),
        ),
        ("uint4", data_type::uint4(), uint4, None),
        ("bool", data_type::bool(), bool, None),
    ];
    for (label, data_type, bytes, last_bit) in configurations {
        let fill_value = FillValue::new(vec![0; data_type.fixed_size().unwrap()]);
        let codec = std::sync::Arc::new(
            PackBitsCodec::new(PackBitsPaddingEncoding::FirstByte, None, last_bit).unwrap(),
        )
        .with_context(data_type, fill_value, &CodecSpecificOptions::default())
        .unwrap();
        let bytes = bytes.into_fixed().unwrap();
        let encoded = codec
            .encode(
                ArrayBytes::new_flen(CowBytes::Borrowed(&bytes)),
                &shape,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_static();
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(BenchmarkId::new("encode", label), |b| {
            b.iter(|| {
                codec
                    .encode(
                        ArrayBytes::new_flen(CowBytes::Borrowed(&bytes)),
                        &shape,
                        &CodecOptions::default(),
                    )
                    .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("decode", label), |b| {
            b.iter(|| {
                codec
                    .decode(
                        CowBytes::Borrowed(&encoded),
                        &shape,
                        &CodecOptions::default(),
                    )
                    .unwrap()
            });
        });
        let partial_decoder = codec
            .clone()
            .partial_decoder(
                std::sync::Arc::new(encoded),
                &shape_2d,
                &CodecOptions::default(),
            )
            .unwrap();
        group.throughput(Throughput::Bytes(
            bytes.len() as u64 / 1024 * subset.shape()[1],
        ));
        group.bench_function(BenchmarkId::new("partial_decode", label), |b| {
            b.iter(|| {
                partial_decoder
                    .partial_decode(&subset, &CodecOptions::default())
                    .unwrap()
            });
        });
    }
}

criterion_group!(
    benches,
    codec_bytes,
    codec_blosc,
    codec_sharding,
    codec_packbits
);
criterion_main!(benches);

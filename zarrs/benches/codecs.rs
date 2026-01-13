//! Benchmark various codecs.
#![allow(missing_docs)]

use std::borrow::Cow;

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use zarrs::array::codec::bytes_to_bytes::blosc::{BloscCompressor, BloscShuffleMode};
use zarrs::array::codec::{BloscCodec, BytesCodec};
use zarrs::array::{BytesRepresentation, Element, Endianness, data_type};
use zarrs_codec::{ArrayToBytesCodecTraits, BytesToBytesCodecTraits, CodecOptions};
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
    for size in [32, 64, 128, 256, 512].iter() {
        let size3 = size * size * size;
        let num_elements = size3 / 2;
        let shape = [num_elements.try_into().unwrap(); 1];
        let data = vec![0u8; size3.try_into().unwrap()];
        let bytes = Element::into_array_bytes(&data_type::uint16(), data).unwrap();
        group.throughput(Throughput::Bytes(size3));
        // encode and decode have the same implementation
        group.bench_function(BenchmarkId::new("encode_decode", size3), |b| {
            b.iter(|| {
                codec
                    .encode(
                        bytes.clone(),
                        &shape,
                        &data_type::uint16(),
                        &fill_value,
                        &CodecOptions::default(),
                    )
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

    for size in [32, 64, 128, 256, 512].iter() {
        let size3 = size * size * size;
        let rep = BytesRepresentation::FixedSize(size3);

        let data_decoded: Vec<u8> = (0..size3).map(|i| i as u8).collect();
        let data_encoded = codec
            .encode(Cow::Borrowed(&data_decoded), &CodecOptions::default())
            .unwrap();
        group.throughput(Throughput::Bytes(size3));
        group.bench_function(BenchmarkId::new("encode", size3), |b| {
            b.iter(|| {
                codec
                    .encode(Cow::Borrowed(&data_decoded), &CodecOptions::default())
                    .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("decode", size3), |b| {
            b.iter(|| {
                codec
                    .decode(Cow::Borrowed(&data_encoded), &rep, &CodecOptions::default())
                    .unwrap()
            });
        });
    }
}

criterion_group!(benches, codec_bytes, codec_blosc);
criterion_main!(benches);

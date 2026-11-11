//! Benchmark various codecs.
#![allow(
    missing_docs,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::sync::Arc;
use zarrs_codec::ArrayToBytesCodecTraits;

use zarrs::array::{
    ArrayBytes, ArraySubset, CodecSpecificOptions, ComputeResources, CowBytes, Resources,
};

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
#[cfg(feature = "zstd")]
use zarrs::array::codec::ZstdCodec;
use zarrs::array::codec::bytes_to_bytes::blosc::{BloscCompressor, BloscShuffleMode};
use zarrs::array::codec::bytes_to_bytes::shuffle::ShuffleCodec;
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
            Some(fill_value),
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
                    .encode(
                        bytes.clone(),
                        &shape,
                        &CodecOptions::default(),
                        &ComputeResources::default(),
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

    for size in &[32, 64, 128, 256, 512] {
        let size3 = size * size * size;
        let rep = BytesRepresentation::FixedSize(size3);

        let data_decoded: Vec<u8> = (0..size3).map(|i| i as u8).collect();
        let data_encoded = codec
            .encode(
                CowBytes::Borrowed(&data_decoded),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        group.throughput(Throughput::Bytes(size3));
        group.bench_function(BenchmarkId::new("encode", size3), |b| {
            b.iter(|| {
                codec
                    .encode(
                        CowBytes::Borrowed(&data_decoded),
                        &CodecOptions::default(),
                        &ComputeResources::default(),
                    )
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
                        &ComputeResources::default(),
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
                Some(FillValue::from(0u16)),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap()
            .into_static();
        let label = format!("{subchunk}{}", if zstd { "_zstd" } else { "" });
        group.throughput(Throughput::Bytes(bytes.size() as u64));
        group.bench_function(BenchmarkId::new("encode", &label), |b| {
            b.iter(|| {
                codec
                    .encode(
                        bytes.clone(),
                        &shape,
                        &CodecOptions::default(),
                        &ComputeResources::default(),
                    )
                    .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("decode", &label), |b| {
            b.iter(|| {
                codec
                    .decode(
                        encoded.clone(),
                        &shape,
                        &CodecOptions::default(),
                        &ComputeResources::default(),
                    )
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
        .with_context(
            data_type,
            Some(fill_value),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
        let bytes = bytes.into_fixed().unwrap();
        let encoded = codec
            .encode(
                ArrayBytes::new_flen(CowBytes::Borrowed(&bytes)),
                &shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
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
                        &ComputeResources::default(),
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
                        &ComputeResources::default(),
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
                &Resources::default(),
            )
            .unwrap();
        group.throughput(Throughput::Bytes(
            bytes.len() as u64 / 1024 * subset.shape()[1],
        ));
        group.bench_function(BenchmarkId::new("partial_decode", label), |b| {
            b.iter(|| {
                partial_decoder
                    .partial_decode(&subset, &CodecOptions::default(), &Resources::default())
                    .unwrap()
            });
        });
    }
}

fn codec_shuffle(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("codec_shuffle");
    group.plot_config(plot_config);

    for elementsize in [2usize, 3, 4, 8, 16] {
        let codec = ShuffleCodec::new(elementsize);
        for len in [1usize << 16, 1 << 20, 1 << 24] {
            let len = len / elementsize * elementsize;
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let encoded = codec
                .encode(
                    CowBytes::Borrowed(&data),
                    &CodecOptions::default(),
                    &ComputeResources::default(),
                )
                .unwrap()
                .to_vec();
            let rep = BytesRepresentation::FixedSize(len as u64);
            group.throughput(Throughput::Bytes(len as u64));
            let id = format!("{elementsize}/{len}");
            group.bench_function(BenchmarkId::new("encode", &id), |b| {
                b.iter(|| {
                    codec
                        .encode(
                            CowBytes::Borrowed(&data),
                            &CodecOptions::default(),
                            &ComputeResources::default(),
                        )
                        .unwrap()
                });
            });
            group.bench_function(BenchmarkId::new("decode", &id), |b| {
                b.iter(|| {
                    codec
                        .decode(
                            CowBytes::Borrowed(&encoded),
                            &rep,
                            &CodecOptions::default(),
                            &ComputeResources::default(),
                        )
                        .unwrap()
                });
            });
        }
    }
}

fn decode_into_bench(c: &mut Criterion, name: &str, codec: zarrs::array::codec::CodecChain) {
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs::array::{ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView};

    let shape = [std::num::NonZeroU64::new(32_768).unwrap()];
    let values: Vec<f32> = (0..32_768).map(|i| (i % 31) as f32 * 0.25).collect();
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let codec = codec
        .with_context(
            data_type::float32(),
            Some(0f32.into()),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    let encoded = codec
        .encode(
            ArrayBytes::new_flen(bytemuck::cast_slice(&values)),
            &shape,
            &options,
            &resources,
        )
        .unwrap();
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(values.len() as u64 * 4));
    group.bench_function(BenchmarkId::new("decode", "owned"), |b| {
        b.iter(|| {
            std::hint::black_box(
                codec
                    .decode(encoded.clone(), &shape, &options, &resources)
                    .unwrap(),
            );
        });
    });
    for (layout, stride) in [("contiguous", 1), ("strided", 2)] {
        let output_shape = [32_768, stride];
        let subset = ArraySubset::new_with_ranges(&[0..32_768, 0..1]);
        let mut output = vec![0f32; 32_768 * stride as usize];
        group.bench_function(BenchmarkId::new("decode_into", layout), |b| {
            b.iter(|| {
                // SAFETY: each iteration owns the only view of this output buffer.
                let view = unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(bytemuck::cast_slice_mut::<f32, u8>(&mut output)),
                        4,
                        &output_shape,
                        subset.clone(),
                    )
                    .unwrap()
                };
                codec
                    .decode_into(
                        encoded.clone(),
                        &shape,
                        &mut ArrayBytesDecodeIntoTarget::Fixed(view),
                        &options,
                        &resources,
                    )
                    .unwrap();
                std::hint::black_box(&output);
            });
        });
    }
}

fn bytes_decode_into_bench(c: &mut Criterion, name: &str, codec: Arc<dyn BytesToBytesCodecTraits>) {
    decode_into_bench(
        c,
        name,
        zarrs::array::codec::CodecChain::new(vec![], Arc::new(BytesCodec::default()), vec![codec]),
    );
}

fn codec_decode_into(c: &mut Criterion) {
    use zarrs::array::codec::CodecChain;

    let opposite_endianness = if cfg!(target_endian = "little") {
        Endianness::Big
    } else {
        Endianness::Little
    };
    decode_into_bench(
        c,
        "codec_bytes",
        CodecChain::new(
            vec![],
            Arc::new(BytesCodec::new(Some(opposite_endianness))),
            vec![],
        ),
    );
    decode_into_bench(
        c,
        "codec_packbits",
        CodecChain::new(vec![], Arc::new(PackBitsCodec::default()), vec![]),
    );
    bytes_decode_into_bench(c, "codec_shuffle", Arc::new(ShuffleCodec::new(4)));
    #[cfg(feature = "zstd")]
    bytes_decode_into_bench(c, "codec_zstd", Arc::new(ZstdCodec::new(1, false)));
    #[cfg(feature = "gzip")]
    bytes_decode_into_bench(
        c,
        "codec_gzip",
        Arc::new(zarrs::array::codec::GzipCodec::new(1).unwrap()),
    );
    #[cfg(feature = "blosc")]
    bytes_decode_into_bench(
        c,
        "codec_blosc",
        Arc::new(
            BloscCodec::new(
                BloscCompressor::BloscLZ,
                1.try_into().unwrap(),
                None,
                BloscShuffleMode::BitShuffle,
                Some(4),
            )
            .unwrap(),
        ),
    );
    #[cfg(feature = "zlib")]
    bytes_decode_into_bench(
        c,
        "codec_zlib",
        Arc::new(zarrs::array::codec::ZlibCodec::new(1u8.try_into().unwrap())),
    );
    #[cfg(feature = "bz2")]
    bytes_decode_into_bench(
        c,
        "codec_bz2",
        Arc::new(zarrs::array::codec::Bz2Codec::new(1u8.try_into().unwrap())),
    );
    #[cfg(feature = "gdeflate")]
    bytes_decode_into_bench(
        c,
        "codec_gdeflate",
        Arc::new(zarrs::array::codec::GDeflateCodec::new(1).unwrap()),
    );
    #[cfg(feature = "pcodec")]
    decode_into_bench(
        c,
        "codec_pcodec",
        CodecChain::new(
            vec![],
            Arc::new(
                zarrs::array::codec::PcodecCodec::new_with_configuration(&Default::default())
                    .unwrap(),
            ),
            vec![],
        ),
    );
    #[cfg(feature = "zfp")]
    decode_into_bench(
        c,
        "codec_zfp",
        CodecChain::new(
            vec![],
            Arc::new(zarrs::array::codec::ZfpCodec::new_reversible()),
            vec![],
        ),
    );
}

criterion_group!(
    benches,
    codec_bytes,
    codec_blosc,
    codec_sharding,
    codec_packbits,
    codec_shuffle,
    codec_decode_into
);
criterion_main!(benches);

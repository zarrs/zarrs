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

    for elementsize in [1usize, 2, 3, 4, 8, 16] {
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
                        encoded.clone().into(),
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

/// Decode with the opposite endianness into strided views with regions of `width` elements, from shared and owned bytes.
fn codec_bytes_decode_into_strided(c: &mut Criterion) {
    use criterion::BatchSize;
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs::array::{ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView};

    const NUM_ELEMENTS: u64 = 32_768;
    let endianness = if cfg!(target_endian = "little") {
        Endianness::Big
    } else {
        Endianness::Little
    };
    let shape = [std::num::NonZeroU64::new(NUM_ELEMENTS).unwrap()];
    let values: Vec<f32> = (0..NUM_ELEMENTS).map(|i| (i % 31) as f32 * 0.25).collect();
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let codec = BytesCodec::new(Some(endianness))
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
    let mut group = c.benchmark_group("codec_bytes");
    group.throughput(Throughput::Bytes(NUM_ELEMENTS * 4));
    for width in [1u64, 8, 64, 1024] {
        let output_shape = [NUM_ELEMENTS / width, 2 * width];
        let subset = ArraySubset::new_with_ranges(&[0..NUM_ELEMENTS / width, 0..width]);
        let mut output = vec![0f32; 2 * NUM_ELEMENTS as usize];
        let mut decode_into = |input: CowBytes<'_>| {
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
                    input.into(),
                    &shape,
                    &mut ArrayBytesDecodeIntoTarget::Fixed(view),
                    &options,
                    &resources,
                )
                .unwrap();
            std::hint::black_box(&output);
        };
        group.bench_function(BenchmarkId::new("decode_into_strided/shared", width), |b| {
            b.iter(|| decode_into(encoded.clone()));
        });
        group.bench_function(BenchmarkId::new("decode_into_strided/owned", width), |b| {
            b.iter_batched(
                || CowBytes::from(encoded.to_vec()),
                &mut decode_into,
                BatchSize::SmallInput,
            );
        });
    }
}

/// Decode a chain with an array-to-array codec of `decoded_shape` `f32` elements, owned and into contiguous, block (half of
/// each innermost row) and strided (every second element) targets, with and without zstd.
fn a2a_decode_into_bench(
    c: &mut Criterion,
    name: &str,
    codec: Arc<dyn zarrs_codec::UnboundArrayToArrayCodecTraits>,
    decoded_shape: &[u64],
) {
    let num_elements = decoded_shape.iter().product::<u64>();
    let values: Vec<f32> = (0..num_elements).map(|i| (i % 31) as f32 * 0.25).collect();
    a2a_decode_into_bench_with(
        c,
        name,
        codec,
        (data_type::float32(), FillValue::from(0f32), 4),
        bytemuck::cast_slice(&values),
        decoded_shape,
    );
}

/// As [`a2a_decode_into_bench`], with elements of a (data type, fill value, size) and their `values`.
fn a2a_decode_into_bench_with(
    c: &mut Criterion,
    name: &str,
    codec: Arc<dyn zarrs_codec::UnboundArrayToArrayCodecTraits>,
    (data_type, fill_value, size): (zarrs::array::DataType, FillValue, usize),
    values: &[u8],
    decoded_shape: &[u64],
) {
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs::array::codec::CodecChain;
    use zarrs::array::{ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView};

    let shape: Vec<std::num::NonZeroU64> = decoded_shape
        .iter()
        .map(|&size| std::num::NonZeroU64::new(size).unwrap())
        .collect();
    let num_elements = decoded_shape.iter().product::<u64>();
    let options = CodecOptions::default();
    let resources = ComputeResources::default();

    let mut block_shape = decoded_shape.to_vec();
    *block_shape.last_mut().unwrap() *= 2;
    let strided_shape = [decoded_shape, &[2]].concat();
    let layouts = [
        (
            "contiguous",
            decoded_shape.to_vec(),
            ArraySubset::new_with_shape(decoded_shape),
        ),
        (
            "block",
            block_shape,
            ArraySubset::new_with_shape(decoded_shape),
        ),
        (
            "strided",
            strided_shape,
            ArraySubset::new_with_shape([decoded_shape, &[1]].concat()),
        ),
    ];

    #[allow(unused_mut)]
    let mut chains = vec![(
        name.to_string(),
        CodecChain::new(vec![codec.clone()], Arc::new(BytesCodec::default()), vec![]),
    )];
    #[cfg(feature = "zstd")]
    chains.push((
        format!("{name}_zstd"),
        CodecChain::new(
            vec![codec],
            Arc::new(BytesCodec::default()),
            vec![Arc::new(ZstdCodec::new(1, false))],
        ),
    ));
    for (name, chain) in chains {
        let chain = chain
            .with_context(
                data_type.clone(),
                Some(fill_value.clone()),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        let encoded = chain
            .encode(ArrayBytes::new_flen(values), &shape, &options, &resources)
            .unwrap();
        let mut group = c.benchmark_group(name);
        group.throughput(Throughput::Bytes(num_elements * size as u64));
        group.bench_function(BenchmarkId::new("decode", "owned"), |b| {
            b.iter(|| {
                std::hint::black_box(
                    chain
                        .decode(encoded.clone(), &shape, &options, &resources)
                        .unwrap(),
                );
            });
        });
        for (layout, output_shape, subset) in &layouts {
            let mut output = vec![0u8; output_shape.iter().product::<u64>() as usize * size];
            group.bench_function(BenchmarkId::new("decode_into", layout), |b| {
                b.iter(|| {
                    // SAFETY: each iteration owns the only view of this output buffer.
                    let view = unsafe {
                        ArrayBytesFixedDisjointView::new(
                            UnsafeCellSlice::new(&mut output),
                            size,
                            output_shape,
                            subset.clone(),
                        )
                        .unwrap()
                    };
                    chain
                        .decode_into(
                            encoded.clone().into(),
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
}

/// Decode chains with an array-to-array codec into contiguous and strided targets.
fn codec_a2a_decode_into(c: &mut Criterion) {
    use zarrs::array::codec::{ReshapeCodec, ReshapeDim, ReshapeShape, SqueezeCodec};

    #[cfg(feature = "transpose")]
    a2a_decode_into_bench(
        c,
        "codec_transpose",
        Arc::new(zarrs::array::codec::TransposeCodec::new(
            zarrs::array::codec::TransposeOrder::new(&[1, 0]).unwrap(),
        )),
        &[128, 256],
    );
    a2a_decode_into_bench(
        c,
        "codec_reshape",
        Arc::new(ReshapeCodec::new(
            ReshapeShape::new([ReshapeDim::from([0, 1])]).unwrap(),
        )),
        &[128, 256],
    );
    a2a_decode_into_bench(
        c,
        "codec_squeeze",
        Arc::new(SqueezeCodec::new()),
        &[128, 1, 256],
    );
    #[cfg(feature = "bitround")]
    a2a_decode_into_bench(
        c,
        "codec_bitround",
        Arc::new(zarrs::array::codec::BitroundCodec::new(10)),
        &[128, 256],
    );

    // Codecs that change the bytes of elements on decode
    let codec = |configuration: &str| -> Arc<dyn zarrs_codec::UnboundArrayToArrayCodecTraits> {
        let metadata: zarrs::metadata::v3::MetadataV3 =
            serde_json::from_str(configuration).unwrap();
        match zarrs_codec::Codec::from_metadata(&metadata).unwrap() {
            zarrs_codec::Codec::ArrayToArray(codec) => codec,
            _ => unreachable!(),
        }
    };
    a2a_decode_into_bench(
        c,
        "codec_fixedscaleoffset",
        codec(
            r#"{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 0, "scale": 100, "dtype": "<f4", "astype": "<i2"}}"#,
        ),
        &[128, 256],
    );
    let values: Vec<u8> = (0..128 * 256i32)
        .flat_map(|i| (i % 31).to_ne_bytes())
        .collect();
    a2a_decode_into_bench_with(
        c,
        "codec_cast_value",
        codec(r#"{"name": "cast_value", "configuration": {"data_type": "int16"}}"#),
        (data_type::int32(), FillValue::from(0i32), 4),
        &values,
        &[128, 256],
    );
    let values: Vec<u8> = (0..128 * 256u32).map(|i| u8::from(i % 3 == 0)).collect();
    a2a_decode_into_bench_with(
        c,
        "codec_numcodecs_packbits",
        codec(r#"{"name": "numcodecs.packbits"}"#),
        (data_type::bool(), FillValue::from(false), 1),
        &values,
        &[128, 256],
    );
}

/// Decode a chain of array-to-array codecs and an array-to-bytes codec of `num_elements` elements into views with regions of each of `widths` elements: every second block of `width` elements of an `[num_elements / width, 2 * width]` array.
fn decode_into_strided_bench(
    c: &mut Criterion,
    name: &str,
    (array_to_array, array_to_bytes): &(
        Vec<Arc<dyn zarrs_codec::UnboundArrayToArrayCodecTraits>>,
        Arc<dyn UnboundArrayToBytesCodecTraits>,
    ),
    (data_type, fill_value, size): (&zarrs::array::DataType, &FillValue, usize),
    values: &[u8],
    widths: &[u64],
) {
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs::array::codec::CodecChain;
    use zarrs::array::{ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView};

    let num_elements = (values.len() / size) as u64;
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let chain = CodecChain::new(array_to_array.clone(), array_to_bytes.clone(), vec![])
        .with_context(
            data_type.clone(),
            Some(fill_value.clone()),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(num_elements * size as u64));
    for &width in widths {
        let shape = [
            std::num::NonZeroU64::new(num_elements / width).unwrap(),
            std::num::NonZeroU64::new(width).unwrap(),
        ];
        let encoded = chain
            .encode(ArrayBytes::new_flen(values), &shape, &options, &resources)
            .unwrap();
        let output_shape = [num_elements / width, 2 * width];
        let subset = ArraySubset::new_with_ranges(&[0..num_elements / width, 0..width]);
        let mut output = vec![0u8; 2 * values.len()];
        group.bench_function(BenchmarkId::new("decode_into_strided", width), |b| {
            b.iter(|| {
                // SAFETY: each iteration owns the only view of this output buffer.
                let view = unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(&mut output),
                        size,
                        &output_shape,
                        subset.clone(),
                    )
                    .unwrap()
                };
                chain
                    .decode_into(
                        encoded.clone().into(),
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

/// Decode chains with fixedscaleoffset, cast_value and numcodecs.packbits into views with regions of 1 to 32768 elements, of a 32768 element chunk (in L2 cache) and a 4 Mi element chunk (larger than L3 cache).
fn a2a_decode_into_strided(c: &mut Criterion) {
    const SMALL: u64 = 32_768;
    const LARGE: u64 = 4 * 1024 * 1024;
    const SMALL_WIDTHS: &[u64] = &[1, 4, 16, 64, 256, 1024, 4096, SMALL];
    const LARGE_WIDTHS: &[u64] = &[256, 4096];

    let mut bench = |name: &str,
                     configuration: &str,
                     (data_type, fill_value, size): (zarrs::array::DataType, FillValue, usize),
                     values: fn(u64) -> Vec<u8>| {
        let metadata: zarrs::metadata::v3::MetadataV3 =
            serde_json::from_str(configuration).unwrap();
        let zarrs_codec::Codec::ArrayToArray(codec) =
            zarrs_codec::Codec::from_metadata(&metadata).unwrap()
        else {
            unreachable!()
        };
        let codecs = (
            vec![codec],
            Arc::new(BytesCodec::default()) as Arc<dyn UnboundArrayToBytesCodecTraits>,
        );
        for (num_elements, widths, suffix) in
            [(SMALL, SMALL_WIDTHS, ""), (LARGE, LARGE_WIDTHS, "_4Mi")]
        {
            decode_into_strided_bench(
                c,
                &format!("a2a_decode_into_strided/{name}{suffix}"),
                &codecs,
                (&data_type, &fill_value, size),
                &values(num_elements),
                widths,
            );
        }
    };
    bench(
        "fixedscaleoffset",
        r#"{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 0, "scale": 100, "dtype": "<f4", "astype": "<i2"}}"#,
        (data_type::float32(), FillValue::from(0f32), 4),
        |n| {
            (0..n)
                .flat_map(|i| ((i % 31) as f32 * 0.25).to_ne_bytes())
                .collect()
        },
    );
    bench(
        "cast_value",
        r#"{"name": "cast_value", "configuration": {"data_type": "int16"}}"#,
        (data_type::int32(), FillValue::from(0i32), 4),
        |n| {
            (0..n)
                .flat_map(|i| ((i % 31) as i32).to_ne_bytes())
                .collect()
        },
    );
    bench(
        "numcodecs_packbits",
        r#"{"name": "numcodecs.packbits"}"#,
        (data_type::bool(), FillValue::from(false), 1),
        |n| (0..n).map(|i| u8::from(i % 3 == 0)).collect(),
    );
}

/// Decode a packbits chain of `uint4` elements into views with regions of 1 to 32768 elements, as in [`a2a_decode_into_strided`].
fn a2b_decode_into_strided(c: &mut Criterion) {
    const SMALL: u64 = 32_768;
    let values: Vec<u8> = (0..SMALL).map(|i| (i % 13) as u8).collect();
    decode_into_strided_bench(
        c,
        "a2b_decode_into_strided/packbits_uint4",
        &(vec![], Arc::new(PackBitsCodec::default())),
        (&data_type::uint4(), &FillValue::from(0u8), 1),
        &values,
        &[1, 4, 16, 64, 256, 1024, 4096, SMALL],
    );
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
    codec_decode_into,
    codec_bytes_decode_into_strided,
    codec_a2a_decode_into,
    a2a_decode_into_strided,
    a2b_decode_into_strided
);
criterion_main!(benches);

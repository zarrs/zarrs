//! Benchmark per-operation overheads of array operations on an in-memory store.
//!
//! - `many_small_chunks`: reading and writing many tiny chunks, dominated by per-chunk overhead.
//! - `single_chunk`: whole-chunk reads and writes of one chunk.
//! - `partial_encode`: writing small regions of a sharded array with partial encoding.
//! - `partial_write`: partial writes to uncompressed arrays in memory and on the filesystem.
//! - `nested_sharding`: reading a shard of shards.
//! - `sharded_read`: whole reads of 1, 4 and 16 shards of 32 x 32 subchunks, of `uint16` and `optional uint16` data.
//! - `partial_decoder`: creating a partial decoder of a chunk, and decoding regions with it.
//! - `partial_decoder_indexer`: decoding generic (coordinate) indexers of a sharded chunk with a partial decoder, directly into a preallocated output.
//! - `chunk_cache`: reading a region through each type of chunk cache, empty and filled.
//! - `subchunk_cache`: reading regions of a sharded array through a decoded subchunk cache, empty and filled.
//! - `async_chunk_cache`: reading a region through each type of async chunk cache, empty and filled (`async` feature).
//! - `async_subchunk_cache`: reading regions of a sharded array through an async decoded subchunk cache, empty and filled (`async` feature).
//! - `element_order`: reading and writing a multi-chunk array in C and F element order, with and without a `transpose` codec.
#![allow(missing_docs, clippy::cast_possible_truncation)]

mod common;

use std::num::NonZeroU64;
use std::sync::Arc;

use common::compressible_u16;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::{Crc32cCodec, ShardingCodecBuilder, ZstdCodec};
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArraySubset, DataType, FillValue, Resources, data_type,
};
use zarrs::storage::SyncReadableWritableListableStorageTraits;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{ArrayBytesFixedDisjointView, ArrayBytesTarget, CodecOptions};

/// 100 000 chunks of 64 bytes.
fn many_small_chunks(c: &mut Criterion) {
    const NUM_CHUNKS: u64 = 100_000;
    const CHUNK_SIZE: u64 = 64;
    let mut group = c.benchmark_group("many_small_chunks");
    group.sample_size(10);
    group.throughput(Throughput::Elements(NUM_CHUNKS));
    let data = common::random_bytes((NUM_CHUNKS * CHUNK_SIZE) as usize);
    for compressed in [false, true] {
        let mut builder = ArrayBuilder::new(
            vec![NUM_CHUNKS * CHUNK_SIZE],
            vec![CHUNK_SIZE],
            data_type::uint8(),
            0u8,
        );
        if compressed {
            builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
        }
        let label = if compressed { "zstd" } else { "uncompressed" };
        let array = builder.build(Arc::new(MemoryStore::new()), "/").unwrap();
        let subset = array.subset_all();
        // The reads below do not depend on the writes being benchmarked
        array
            .store_array_subset(&subset, data.as_slice(), &Resources::default())
            .unwrap();

        group.bench_function(BenchmarkId::new("write", label), |b| {
            b.iter(|| {
                array
                    .store_array_subset(&subset, data.as_slice(), &Resources::default())
                    .unwrap();
            });
        });
        group.bench_function(BenchmarkId::new("read", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = array
                    .retrieve_array_subset(&subset, &Resources::default())
                    .unwrap();
            });
        });
    }
    group.finish();
}

/// One 256 x 256 `u16` chunk.
fn single_chunk(c: &mut Criterion) {
    const SHAPE: [u64; 2] = [256, 256];
    let mut group = c.benchmark_group("single_chunk");
    let data = compressible_u16(SHAPE.iter().product::<u64>() as usize);
    group.throughput(Throughput::Bytes(data.len() as u64 * 2));
    for (label, compressed, sharded) in [
        ("uncompressed", false, false),
        ("zstd", true, false),
        ("sharded_zstd", true, true),
    ] {
        let mut builder =
            ArrayBuilder::new(SHAPE.to_vec(), SHAPE.to_vec(), data_type::uint16(), 0u16);
        if compressed {
            builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
        }
        if sharded {
            builder.subchunk_shape(vec![32, 32]);
        }
        let array = builder.build(Arc::new(MemoryStore::new()), "/").unwrap();
        // The reads below do not depend on the writes being benchmarked
        array
            .store_chunk(&[0, 0], data.as_slice(), &Resources::default())
            .unwrap();

        group.bench_function(BenchmarkId::new("store_chunk", label), |b| {
            b.iter(|| {
                array
                    .store_chunk(&[0, 0], data.as_slice(), &Resources::default())
                    .unwrap();
            });
        });
        group.bench_function(BenchmarkId::new("retrieve_chunk", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = array
                    .retrieve_chunk(&[0, 0], &Resources::default())
                    .unwrap();
            });
        });
        let mut output = vec![0u8; data.len() * 2];
        group.bench_function(BenchmarkId::new("retrieve_chunk_into", label), |b| {
            b.iter(|| {
                // SAFETY: the view covers the entire output and is its only view.
                let view = unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(&mut output),
                        2,
                        &SHAPE,
                        array.subset_all(),
                    )
                }
                .unwrap();
                array
                    .retrieve_array_subset_into(
                        &array.subset_all(),
                        &mut ArrayBytesTarget::Fixed(view),
                        &Resources::default(),
                    )
                    .unwrap();
            });
        });
        let quarter = ArraySubset::new_with_ranges(&[64..192, 64..192]);
        group.bench_function(BenchmarkId::new("retrieve_chunk_subset", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = array
                    .retrieve_array_subset(&quarter, &Resources::default())
                    .unwrap();
            });
        });
    }
    group.finish();
}

/// Small writes into a populated sharded array.
///
/// Shards with a checksum (`read_modify_write`) cannot be partially encoded, so they are read, decoded, updated,
/// encoded and written in full.
fn partial_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_encode");
    let shape = [512u64, 512];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    for partial_encoding in [false, true] {
        let sharding =
            ShardingCodecBuilder::new(vec![NonZeroU64::new(32).unwrap(); 2], &data_type::uint16())
                .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
                .build_arc();
        let mut builder =
            ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::uint16(), 0u16);
        builder.array_to_bytes_codec(sharding);
        if !partial_encoding {
            builder.bytes_to_bytes_codecs(vec![Arc::new(Crc32cCodec::new())]);
        }
        let array = builder
            .build(Arc::new(MemoryStore::new()), "/")
            .unwrap()
            .with_codec_options(
                CodecOptions::default().with_experimental_partial_encoding(partial_encoding),
            );
        array
            .store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
            .unwrap();
        let label = if partial_encoding {
            "partial_encoding"
        } else {
            "read_modify_write"
        };
        // A region straddling 4 x 4 subchunks
        let subset = ArraySubset::new_with_ranges(&[48..144, 48..144]);
        let update = vec![7u16; subset.num_elements_usize()];
        group.throughput(Throughput::Bytes(update.len() as u64 * 2));
        group.bench_function(BenchmarkId::new("straddle", label), |b| {
            b.iter(|| {
                array
                    .store_array_subset(&subset, update.as_slice(), &Resources::default())
                    .unwrap();
            });
        });
    }
    group.finish();
}

/// Writes to part of the chunks of an uncompressed array in memory and on the filesystem.
fn partial_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_write");
    let shape = [1024u64, 1024];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    #[cfg(feature = "filesystem")]
    let dir = tempfile::TempDir::new().unwrap();
    let stores: Vec<(&str, Arc<dyn SyncReadableWritableListableStorageTraits>)> = vec![
        ("memory", Arc::new(MemoryStore::new())),
        #[cfg(feature = "filesystem")]
        (
            "filesystem",
            Arc::new(zarrs::filesystem::SyncFilesystemStore::new(dir.path()).unwrap()),
        ),
    ];
    for (store_label, store) in stores {
        let array = ArrayBuilder::new(shape.to_vec(), vec![512, 512], data_type::uint16(), 0u16)
            .build(store, "/")
            .unwrap()
            .with_codec_options(CodecOptions::default().with_experimental_partial_encoding(true));
        array
            .store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
            .unwrap();
        for (label, ranges) in [
            // A contiguous range of each of 2 chunks
            ("rows", [0..16, 0..1024]),
            // 512 ranges of 32 bytes of each of 2 chunks
            ("columns", [0..1024, 0..16]),
            // 64 ranges of 128 bytes of one chunk
            ("block", [100..164, 100..164]),
        ] {
            let subset = ArraySubset::new_with_ranges(&ranges);
            let update = vec![7u16; subset.num_elements_usize()];
            group.throughput(Throughput::Bytes(update.len() as u64 * 2));
            group.bench_function(BenchmarkId::new(label, store_label), |b| {
                b.iter(|| {
                    array
                        .store_array_subset(&subset, update.as_slice(), &Resources::default())
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

/// A 512 x 512 shard of 128 x 128 shards of 32 x 32 subchunks.
fn nested_sharding(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_sharding");
    let shape = [512u64, 512];
    let data_type = data_type::uint16();
    let nz = |v: u64| NonZeroU64::new(v).unwrap();
    let inner = ShardingCodecBuilder::new(vec![nz(32), nz(32)], &data_type)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
        .build_arc();
    let outer = ShardingCodecBuilder::new(vec![nz(128), nz(128)], &data_type)
        .array_to_bytes_codec(inner)
        .build_arc();
    let array = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type, 0u16)
        .array_to_bytes_codec(outer)
        .build(Arc::new(MemoryStore::new()), "/")
        .unwrap();
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    array
        .store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
        .unwrap();
    for (label, subset) in [
        ("all", array.subset_all()),
        (
            "straddle",
            ArraySubset::new_with_ranges(&[100..300, 100..300]),
        ),
        (
            "one_subchunk",
            ArraySubset::new_with_ranges(&[160..192, 160..192]),
        ),
    ] {
        group.throughput(Throughput::Bytes(subset.num_elements() * 2));
        group.bench_function(BenchmarkId::new("read", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = array
                    .retrieve_array_subset(&subset, &Resources::default())
                    .unwrap();
            });
        });
    }
    group.finish();
}

/// Whole reads of a 512 x 512 `u16` array of 1, 4 or 16 shards of 32 x 32 zstd subchunks.
fn sharded_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharded_read");
    let shape = [512u64, 512];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    let optional_data = optional_u16(&data, 0);
    group.throughput(Throughput::Bytes(data.len() as u64 * 2));
    for shard in [512u64, 256, 128] {
        let num_shards = (shape[0] / shard) * (shape[1] / shard);
        for (label, optional) in [("all", false), ("optional_all", true)] {
            let (data_type, fill_value) = if optional {
                (
                    data_type::uint16().to_optional(),
                    FillValue::new_optional_null(),
                )
            } else {
                (data_type::uint16(), FillValue::from(0u16))
            };
            let array =
                ArrayBuilder::new(shape.to_vec(), vec![shard, shard], data_type, fill_value)
                    .subchunk_shape(vec![32, 32])
                    .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
                    .build(Arc::new(MemoryStore::new()), "/")
                    .unwrap();
            if optional {
                array
                    .store_array_subset(
                        &array.subset_all(),
                        optional_data.as_slice(),
                        &Resources::default(),
                    )
                    .unwrap();
            } else {
                array
                    .store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
                    .unwrap();
            }
            group.bench_function(
                BenchmarkId::new(label, format!("{num_shards}_shards")),
                |b| {
                    b.iter(|| {
                        let _: ArrayBytes = array
                            .retrieve_array_subset(&array.subset_all(), &Resources::default())
                            .unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

/// Every third element of `data` is null, as are the first `num_null` elements.
fn optional_u16(data: &[u16], num_null: usize) -> Vec<Option<u16>> {
    data.iter()
        .enumerate()
        .map(|(i, &value)| (i >= num_null && i % 3 != 0).then_some(value))
        .collect()
}

/// A partial decoder of one 512 x 512 `u16` chunk, created once and reused for each region.
fn partial_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_decoder");
    let shape = [512u64, 512];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    for (label, sharded) in [("zstd", false), ("sharded_zstd", true)] {
        let mut builder =
            ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::uint16(), 0u16);
        builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
        if sharded {
            builder.subchunk_shape(vec![32, 32]);
        }
        let array = builder.build(Arc::new(MemoryStore::new()), "/").unwrap();
        array
            .store_chunk(&[0, 0], data.as_slice(), &Resources::default())
            .unwrap();

        group.throughput(Throughput::Bytes(data.len() as u64 * 2));
        group.bench_function(BenchmarkId::new("create", label), |b| {
            b.iter(|| {
                array
                    .partial_decoder(&[0, 0], &Resources::default())
                    .unwrap()
            });
        });
        let decoder = array
            .partial_decoder(&[0, 0], &Resources::default())
            .unwrap();
        for (region, subset) in [
            (
                "one_subchunk",
                ArraySubset::new_with_ranges(&[160..192, 160..192]),
            ),
            (
                "straddle",
                ArraySubset::new_with_ranges(&[100..300, 100..300]),
            ),
        ] {
            group.throughput(Throughput::Bytes(subset.num_elements() * 2));
            group.bench_function(BenchmarkId::new(region, label), |b| {
                b.iter(|| {
                    decoder
                        .partial_decode(&subset, &CodecOptions::default(), &Resources::default())
                        .unwrap()
                        .into_owned()
                });
            });
        }
    }
    group.finish();
}

/// Generic (coordinate) indexers into one 512 x 512 sharded chunk of 32 x 32 subchunks, decoded with a reused partial decoder.
///
/// - `points_4096`: 4096 pseudorandom elements, spread over every subchunk.
/// - `one_subchunk`: every element of one subchunk, in column-major order.
/// - `region_200x200`: every element of a region straddling 7 x 7 subchunks, in column-major order.
///
/// In `half_missing` chunks, the top half is null, so its subchunks are not stored.
fn partial_decoder_indexer(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_decoder_indexer");
    let shape = [512u64, 512];
    let num_elements = shape.iter().product::<u64>() as usize;
    let data = compressible_u16(num_elements);

    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % 512
    };
    let points: Vec<Vec<u64>> = (0..4096).map(|_| vec![random(), random()]).collect();
    let column_major = |rows: std::ops::Range<u64>, cols: std::ops::Range<u64>| -> Vec<Vec<u64>> {
        cols.flat_map(|j| rows.clone().map(move |i| vec![i, j]))
            .collect()
    };
    let indexers = [
        ("points_4096", points),
        ("one_subchunk", column_major(160..192, 160..192)),
        ("region_200x200", column_major(100..300, 100..300)),
    ];

    let cases: [(&str, DataType, FillValue, usize, bool); 5] = [
        (
            "uint16",
            data_type::uint16(),
            FillValue::from(0u16),
            0,
            false,
        ),
        (
            "optional_uint16",
            data_type::uint16().to_optional(),
            FillValue::new_optional_null(),
            0,
            false,
        ),
        (
            "optional_uint16_zstd",
            data_type::uint16().to_optional(),
            FillValue::new_optional_null(),
            0,
            true,
        ),
        (
            "optional_uint16_half_missing",
            data_type::uint16().to_optional(),
            FillValue::new_optional_null(),
            num_elements / 2,
            false,
        ),
        (
            "optional_uint16_half_missing_zstd",
            data_type::uint16().to_optional(),
            FillValue::new_optional_null(),
            num_elements / 2,
            true,
        ),
    ];
    for (label, data_type, fill_value, num_null, zstd) in cases {
        let optional = data_type.is_optional();
        let mut builder = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type, fill_value);
        builder.subchunk_shape(vec![32, 32]);
        if zstd {
            builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
        }
        let array = builder.build(Arc::new(MemoryStore::new()), "/").unwrap();
        if optional {
            array
                .store_chunk(
                    &[0, 0],
                    optional_u16(&data, num_null).as_slice(),
                    &Resources::default(),
                )
                .unwrap();
        } else {
            array
                .store_chunk(&[0, 0], data.as_slice(), &Resources::default())
                .unwrap();
        }
        let decoder = array
            .partial_decoder(&[0, 0], &Resources::default())
            .unwrap();

        for (indexer_label, indexer) in &indexers {
            let len = indexer.len();
            let output_shape = [len as u64];
            let output_subset = ArraySubset::new_with_shape(output_shape);
            let mut output = vec![0u8; len * 2];
            let mut mask = vec![0u8; len];
            group.throughput(Throughput::Elements(len as u64));
            group.bench_function(BenchmarkId::new(*indexer_label, label), |b| {
                b.iter(|| {
                    // SAFETY: each view covers its entire output and is its only view.
                    let view = unsafe {
                        ArrayBytesFixedDisjointView::new(
                            UnsafeCellSlice::new(&mut output),
                            2,
                            &output_shape,
                            output_subset.clone(),
                        )
                    }
                    .unwrap();
                    let mut target = ArrayBytesTarget::from(view);
                    if optional {
                        // SAFETY: as above.
                        let mask_view = unsafe {
                            ArrayBytesFixedDisjointView::new(
                                UnsafeCellSlice::new(&mut mask),
                                1,
                                &output_shape,
                                output_subset.clone(),
                            )
                        }
                        .unwrap();
                        target = target.with_optional_mask(mask_view).unwrap();
                    }
                    decoder
                        .partial_decode_into(
                            indexer,
                            &mut target,
                            &CodecOptions::default(),
                            &Resources::default(),
                        )
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

/// A 200 x 200 region straddling 2 x 2 128 x 128 `u16` chunks, read through each type of chunk cache.
fn chunk_cache(c: &mut Criterion) {
    use zarrs::array::ArrayCached;
    use zarrs::array::cache::{
        SyncChunkCacheDecoded, SyncChunkCacheEncoded, SyncChunkCachePartialDecoder,
        SyncChunkCacheTraits,
    };

    fn bench_cache<C: SyncChunkCacheTraits>(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        array: &Arc<zarrs::array::Array<MemoryStore>>,
        label: &str,
        cache: impl Fn() -> C,
    ) {
        let subset = ArraySubset::new_with_ranges(&[100..300, 100..300]);
        group.bench_function(BenchmarkId::new("empty", label), |b| {
            b.iter(|| {
                let cached = ArrayCached::new(array.clone(), cache());
                let _: ArrayBytes = cached
                    .retrieve_array_subset(&subset, &Resources::default())
                    .unwrap();
            });
        });
        let cached = ArrayCached::new(array.clone(), cache());
        let _: ArrayBytes = cached
            .retrieve_array_subset(&subset, &Resources::default())
            .unwrap();
        group.bench_function(BenchmarkId::new("filled", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = cached
                    .retrieve_array_subset(&subset, &Resources::default())
                    .unwrap();
            });
        });
    }

    let mut group = c.benchmark_group("chunk_cache");
    let shape = [512u64, 512];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    group.throughput(Throughput::Bytes(200 * 200 * 2));
    for (codecs, sharded) in [("zstd", false), ("sharded_zstd", true)] {
        let mut builder =
            ArrayBuilder::new(shape.to_vec(), vec![128, 128], data_type::uint16(), 0u16);
        builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
        if sharded {
            builder.subchunk_shape(vec![32, 32]);
        }
        let array = Arc::new(builder.build(Arc::new(MemoryStore::new()), "/").unwrap());
        array
            .store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
            .unwrap();
        bench_cache(&mut group, &array, &format!("encoded/{codecs}"), || {
            SyncChunkCacheEncoded::new_chunk_limit(16)
        });
        bench_cache(&mut group, &array, &format!("decoded/{codecs}"), || {
            SyncChunkCacheDecoded::new_chunk_limit(16)
        });
        bench_cache(
            &mut group,
            &array,
            &format!("partial_decoder/{codecs}"),
            || SyncChunkCachePartialDecoder::new_chunk_limit(16),
        );
    }
    group.finish();
}

/// Regions of a 512 x 512 `u16` array of 256 x 256 shards of 32 x 32 subchunks, read through a decoded subchunk cache
/// over a partial decoder chunk cache.
fn subchunk_cache(c: &mut Criterion) {
    use zarrs::array::ArrayCached;
    use zarrs::array::cache::{SubchunkCache, SyncChunkCacheDecoded, SyncChunkCachePartialDecoder};

    let mut group = c.benchmark_group("subchunk_cache");
    let shape = [512u64, 512];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    let array = Arc::new(
        ArrayBuilder::new(shape.to_vec(), vec![256, 256], data_type::uint16(), 0u16)
            .subchunk_shape(vec![32, 32])
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
            .build(Arc::new(MemoryStore::new()), "/")
            .unwrap(),
    );
    array
        .store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
        .unwrap();
    let cache = || {
        SubchunkCache::new(
            0,
            SyncChunkCacheDecoded::new_chunk_limit(1024),
            SyncChunkCachePartialDecoder::new_chunk_limit(16),
        )
    };
    for (region, subset) in [
        (
            "one_subchunk",
            ArraySubset::new_with_ranges(&[160..192, 160..192]),
        ),
        ("small", ArraySubset::new_with_ranges(&[100..140, 100..140])),
        ("chunk", ArraySubset::new_with_ranges(&[0..256, 0..256])),
        (
            "straddle",
            ArraySubset::new_with_ranges(&[100..300, 100..300]),
        ),
    ] {
        group.throughput(Throughput::Bytes(subset.num_elements() * 2));
        group.bench_function(BenchmarkId::new("empty", region), |b| {
            b.iter(|| {
                let cached = ArrayCached::new(array.clone(), cache());
                let _: ArrayBytes = cached
                    .retrieve_array_subset(&subset, &Resources::default())
                    .unwrap();
            });
        });
        let cached = ArrayCached::new(array.clone(), cache());
        let _: ArrayBytes = cached
            .retrieve_array_subset(&subset, &Resources::default())
            .unwrap();
        group.bench_function(BenchmarkId::new("filled", region), |b| {
            b.iter(|| {
                let _: ArrayBytes = cached
                    .retrieve_array_subset(&subset, &Resources::default())
                    .unwrap();
            });
        });
    }
    group.finish();
}

/// A 200 x 200 region straddling 2 x 2 128 x 128 `u16` chunks, read through each type of async chunk cache.
#[cfg(feature = "async")]
fn async_chunk_cache(c: &mut Criterion) {
    use zarrs::array::ArrayCached;
    use zarrs::array::cache::{
        AsyncChunkCacheDecoded, AsyncChunkCacheEncoded, AsyncChunkCachePartialDecoder,
        AsyncChunkCacheTraits,
    };
    use zarrs::storage::store::MemoryStore;

    let rt = tokio::runtime::Runtime::new().unwrap();

    fn bench_cache<C: AsyncChunkCacheTraits>(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        rt: &tokio::runtime::Runtime,
        array: &Arc<zarrs::array::Array<MemoryStore>>,
        label: &str,
        cache: impl Fn() -> C,
    ) {
        let subset = ArraySubset::new_with_ranges(&[100..300, 100..300]);
        group.bench_function(BenchmarkId::new("empty", label), |b| {
            b.to_async(rt).iter(|| async {
                let cached = ArrayCached::new(array.clone(), cache());
                let _: ArrayBytes = cached
                    .async_retrieve_array_subset(&subset, &Resources::default())
                    .await
                    .unwrap();
            });
        });
        let cached = ArrayCached::new(array.clone(), cache());
        let _: ArrayBytes = rt
            .block_on(cached.async_retrieve_array_subset(&subset, &Resources::default()))
            .unwrap();
        group.bench_function(BenchmarkId::new("filled", label), |b| {
            b.to_async(rt).iter(|| async {
                let _: ArrayBytes = cached
                    .async_retrieve_array_subset(&subset, &Resources::default())
                    .await
                    .unwrap();
            });
        });
    }

    let mut group = c.benchmark_group("async_chunk_cache");
    let shape = [512u64, 512];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    group.throughput(Throughput::Bytes(200 * 200 * 2));
    for (codecs, sharded) in [("zstd", false), ("sharded_zstd", true)] {
        let mut builder =
            ArrayBuilder::new(shape.to_vec(), vec![128, 128], data_type::uint16(), 0u16);
        builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
        if sharded {
            builder.subchunk_shape(vec![32, 32]);
        }
        let array = Arc::new(builder.build(Arc::new(MemoryStore::new()), "/").unwrap());
        rt.block_on(array.async_store_array_subset(
            &array.subset_all(),
            data.as_slice(),
            &Resources::default(),
        ))
        .unwrap();
        bench_cache(
            &mut group,
            &rt,
            &array,
            &format!("encoded/{codecs}"),
            || AsyncChunkCacheEncoded::new_chunk_limit(16),
        );
        bench_cache(
            &mut group,
            &rt,
            &array,
            &format!("decoded/{codecs}"),
            || AsyncChunkCacheDecoded::new_chunk_limit(16),
        );
        bench_cache(
            &mut group,
            &rt,
            &array,
            &format!("partial_decoder/{codecs}"),
            || AsyncChunkCachePartialDecoder::new_chunk_limit(16),
        );
    }
    group.finish();
}

#[cfg(not(feature = "async"))]
fn async_chunk_cache(_c: &mut Criterion) {}

/// Regions of a 512 x 512 `u16` array of 256 x 256 shards of 32 x 32 subchunks, read through an async decoded
/// subchunk cache over an async partial decoder chunk cache.
#[cfg(feature = "async")]
fn async_subchunk_cache(c: &mut Criterion) {
    use zarrs::array::ArrayCached;
    use zarrs::array::cache::{
        AsyncChunkCacheDecoded, AsyncChunkCachePartialDecoder, SubchunkCache,
    };
    use zarrs::storage::store::MemoryStore;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("async_subchunk_cache");
    let shape = [512u64, 512];
    let data = compressible_u16(shape.iter().product::<u64>() as usize);
    let array = Arc::new(
        ArrayBuilder::new(shape.to_vec(), vec![256, 256], data_type::uint16(), 0u16)
            .subchunk_shape(vec![32, 32])
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
            .build(Arc::new(MemoryStore::new()), "/")
            .unwrap(),
    );
    rt.block_on(array.async_store_array_subset(
        &array.subset_all(),
        data.as_slice(),
        &Resources::default(),
    ))
    .unwrap();
    let cache = || {
        SubchunkCache::new(
            0,
            AsyncChunkCacheDecoded::new_chunk_limit(1024),
            AsyncChunkCachePartialDecoder::new_chunk_limit(16),
        )
    };
    for (region, subset) in [
        (
            "one_subchunk",
            ArraySubset::new_with_ranges(&[160..192, 160..192]),
        ),
        ("small", ArraySubset::new_with_ranges(&[100..140, 100..140])),
        ("chunk", ArraySubset::new_with_ranges(&[0..256, 0..256])),
        (
            "straddle",
            ArraySubset::new_with_ranges(&[100..300, 100..300]),
        ),
    ] {
        group.throughput(Throughput::Bytes(subset.num_elements() * 2));
        group.bench_function(BenchmarkId::new("empty", region), |b| {
            b.to_async(&rt).iter(|| async {
                let cached = ArrayCached::new(array.clone(), cache());
                let _: ArrayBytes = cached
                    .async_retrieve_array_subset(&subset, &Resources::default())
                    .await
                    .unwrap();
            });
        });
        let cached = ArrayCached::new(array.clone(), cache());
        let _: ArrayBytes = rt
            .block_on(cached.async_retrieve_array_subset(&subset, &Resources::default()))
            .unwrap();
        group.bench_function(BenchmarkId::new("filled", region), |b| {
            b.to_async(&rt).iter(|| async {
                let _: ArrayBytes = cached
                    .async_retrieve_array_subset(&subset, &Resources::default())
                    .await
                    .unwrap();
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "async"))]
fn async_subchunk_cache(_c: &mut Criterion) {}

/// Reading and writing a 1024 x 1024 `u16` array of 256 x 256 chunks in C and F element order.
fn element_order(c: &mut Criterion) {
    use zarrs::array::ElementOrder;
    const SHAPE: [u64; 2] = [1024, 1024];
    const CHUNK_SHAPE: [u64; 2] = [256, 256];
    let mut group = c.benchmark_group("element_order");
    group.sample_size(20);
    let data = compressible_u16(SHAPE.iter().product::<u64>() as usize);
    let array_throughput = Throughput::Bytes(data.len() as u64 * 2);
    let chunk_throughput = Throughput::Bytes(CHUNK_SHAPE.iter().product::<u64>() * 2);
    for (label, compressed, transpose) in [
        ("uncompressed", false, false),
        ("zstd", true, false),
        ("transpose", false, true),
    ] {
        let mut builder = ArrayBuilder::new(
            SHAPE.to_vec(),
            CHUNK_SHAPE.to_vec(),
            data_type::uint16(),
            0u16,
        );
        if compressed {
            builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
        }
        #[cfg(feature = "transpose")]
        if transpose {
            builder.array_to_array_codecs(vec![Arc::new(
                zarrs::array::codec::TransposeCodec::new(
                    zarrs::array::codec::TransposeOrder::new(&[1, 0]).unwrap(),
                ),
            )]);
        }
        #[cfg(not(feature = "transpose"))]
        if transpose {
            continue;
        }
        let array = builder.build(Arc::new(MemoryStore::new()), "/").unwrap();
        let subset = array.subset_all();
        array
            .store_array_subset(&subset, data.as_slice(), &Resources::default())
            .unwrap();
        for order in [ElementOrder::C, ElementOrder::F] {
            let array = array.with_element_order(order);
            let id = format!("{label}_{order:?}");
            group.throughput(array_throughput.clone());
            group.bench_function(BenchmarkId::new("retrieve_array_subset", &id), |b| {
                b.iter(|| {
                    let _: Vec<u16> = array
                        .retrieve_array_subset(&subset, &Resources::default())
                        .unwrap();
                });
            });
            group.throughput(chunk_throughput.clone());
            group.bench_function(BenchmarkId::new("retrieve_chunk", &id), |b| {
                b.iter(|| {
                    let _: Vec<u16> = array
                        .retrieve_chunk(&[1, 2], &Resources::default())
                        .unwrap();
                });
            });
            group.throughput(array_throughput.clone());
            group.bench_function(BenchmarkId::new("store_array_subset", &id), |b| {
                b.iter(|| {
                    array
                        .store_array_subset(&subset, data.as_slice(), &Resources::default())
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    many_small_chunks,
    single_chunk,
    partial_encode,
    partial_write,
    nested_sharding,
    sharded_read,
    partial_decoder,
    partial_decoder_indexer,
    chunk_cache,
    subchunk_cache,
    async_chunk_cache,
    async_subchunk_cache,
    element_order
);
criterion_main!(benches);

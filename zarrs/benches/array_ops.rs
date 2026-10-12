//! Benchmark per-operation overheads of array operations on an in-memory store.
//!
//! - `many_small_chunks`: reading and writing many tiny chunks, dominated by per-chunk overhead.
//! - `single_chunk`: whole-chunk reads and writes of one chunk.
//! - `partial_encode`: writing small regions of a sharded array with partial encoding.
//! - `partial_write`: partial writes to uncompressed arrays in memory and on the filesystem.
//! - `nested_sharding`: reading a shard of shards.
//! - `sharded_read`: whole reads of 1, 4 and 16 shards of 32 x 32 subchunks.
//! - `partial_decoder`: creating a partial decoder of a chunk, and decoding regions with it.
//! - `chunk_cache`: reading a region through each type of chunk cache, empty and filled.
#![allow(missing_docs)]

mod common;

use std::num::NonZeroU64;
use std::sync::Arc;

use common::compressible_u16;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::{Crc32cCodec, ShardingCodecBuilder, ZstdCodec};
use zarrs::array::{ArrayBuilder, ArrayBytes, ArraySubset, data_type};
use zarrs::storage::ReadableWritableListableStorageTraits;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, CodecOptions};

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
        array.store_array_subset(&subset, data.as_slice()).unwrap();

        group.bench_function(BenchmarkId::new("write", label), |b| {
            b.iter(|| {
                array.store_array_subset(&subset, data.as_slice()).unwrap();
            });
        });
        group.bench_function(BenchmarkId::new("read", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = array.retrieve_array_subset(&subset).unwrap();
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
        array.store_chunk(&[0, 0], data.as_slice()).unwrap();

        group.bench_function(BenchmarkId::new("store_chunk", label), |b| {
            b.iter(|| {
                array.store_chunk(&[0, 0], data.as_slice()).unwrap();
            });
        });
        group.bench_function(BenchmarkId::new("retrieve_chunk", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = array.retrieve_chunk(&[0, 0]).unwrap();
            });
        });
        let mut output = vec![0u8; data.len() * 2];
        group.bench_function(BenchmarkId::new("retrieve_chunk_into", label), |b| {
            b.iter(|| {
                // SAFETY: the view covers the entire output and is its only view.
                let mut view = unsafe {
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
                        ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                    )
                    .unwrap();
            });
        });
        let quarter = ArraySubset::new_with_ranges(&[64..192, 64..192]);
        group.bench_function(BenchmarkId::new("retrieve_chunk_subset", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = array.retrieve_array_subset(&quarter).unwrap();
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
            .store_array_subset(&array.subset_all(), data.as_slice())
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
                    .store_array_subset(&subset, update.as_slice())
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
    let stores: Vec<(&str, Arc<dyn ReadableWritableListableStorageTraits>)> = vec![
        ("memory", Arc::new(MemoryStore::new())),
        #[cfg(feature = "filesystem")]
        (
            "filesystem",
            Arc::new(zarrs::filesystem::FilesystemStore::new(dir.path()).unwrap()),
        ),
    ];
    for (store_label, store) in stores {
        let array = ArrayBuilder::new(shape.to_vec(), vec![512, 512], data_type::uint16(), 0u16)
            .build(store, "/")
            .unwrap()
            .with_codec_options(CodecOptions::default().with_experimental_partial_encoding(true));
        array
            .store_array_subset(&array.subset_all(), data.as_slice())
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
                        .store_array_subset(&subset, update.as_slice())
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
        .store_array_subset(&array.subset_all(), data.as_slice())
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
                let _: ArrayBytes = array.retrieve_array_subset(&subset).unwrap();
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
    group.throughput(Throughput::Bytes(data.len() as u64 * 2));
    for shard in [512u64, 256, 128] {
        let array = ArrayBuilder::new(
            shape.to_vec(),
            vec![shard, shard],
            data_type::uint16(),
            0u16,
        )
        .subchunk_shape(vec![32, 32])
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
        .build(Arc::new(MemoryStore::new()), "/")
        .unwrap();
        array
            .store_array_subset(&array.subset_all(), data.as_slice())
            .unwrap();
        let num_shards = (shape[0] / shard) * (shape[1] / shard);
        group.bench_function(
            BenchmarkId::new("all", format!("{num_shards}_shards")),
            |b| {
                b.iter(|| {
                    let _: ArrayBytes = array.retrieve_array_subset(&array.subset_all()).unwrap();
                });
            },
        );
    }
    group.finish();
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
        array.store_chunk(&[0, 0], data.as_slice()).unwrap();

        group.throughput(Throughput::Bytes(data.len() as u64 * 2));
        group.bench_function(BenchmarkId::new("create", label), |b| {
            b.iter(|| array.partial_decoder(&[0, 0]).unwrap());
        });
        let decoder = array.partial_decoder(&[0, 0]).unwrap();
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
                        .partial_decode(&subset, &CodecOptions::default())
                        .unwrap()
                        .into_owned()
                });
            });
        }
    }
    group.finish();
}

/// A 200 x 200 region straddling 2 x 2 128 x 128 `u16` chunks, read through each type of chunk cache.
fn chunk_cache(c: &mut Criterion) {
    use zarrs::array::ArrayCached;
    use zarrs::array::chunk_cache::{
        ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
        ChunkCachePartialDecoderLruChunkLimit,
    };

    fn bench_cache<C: ChunkCache>(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        array: &Arc<zarrs::array::Array<MemoryStore>>,
        label: &str,
        cache: impl Fn() -> C,
    ) {
        let subset = ArraySubset::new_with_ranges(&[100..300, 100..300]);
        group.bench_function(BenchmarkId::new("empty", label), |b| {
            b.iter(|| {
                let cached = ArrayCached::new(array.clone(), cache());
                let _: ArrayBytes = cached.retrieve_array_subset(&subset).unwrap();
            });
        });
        let cached = ArrayCached::new(array.clone(), cache());
        let _: ArrayBytes = cached.retrieve_array_subset(&subset).unwrap();
        group.bench_function(BenchmarkId::new("filled", label), |b| {
            b.iter(|| {
                let _: ArrayBytes = cached.retrieve_array_subset(&subset).unwrap();
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
            .store_array_subset(&array.subset_all(), data.as_slice())
            .unwrap();
        bench_cache(&mut group, &array, &format!("encoded/{codecs}"), || {
            ChunkCacheEncodedLruChunkLimit::new(16)
        });
        bench_cache(&mut group, &array, &format!("decoded/{codecs}"), || {
            ChunkCacheDecodedLruChunkLimit::new(16)
        });
        bench_cache(
            &mut group,
            &array,
            &format!("partial_decoder/{codecs}"),
            || ChunkCachePartialDecoderLruChunkLimit::new(16),
        );
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
    chunk_cache
);
criterion_main!(benches);

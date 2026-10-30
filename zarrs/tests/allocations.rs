#![allow(missing_docs, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
#![cfg(all(feature = "zstd", not(miri)))]

//! Heap allocation counts of representative array operations.
//!
//! A counting global allocator records the number of allocations, the total bytes allocated and the
//! peak growth in live bytes above the lowest balance seen so far during an operation.
//! Operations run on a single-thread Rayon pool with a concurrency target of 1.
//! Counters are enabled only on that worker during the operation, excluding allocations on other
//! threads (including test-harness and Rayon startup/teardown work).
//! Rayon work uses the installed pool, and async operations use `block_on` on the same worker.
//! These checks do not cover work offloaded to independent threads or pools, or native allocations
//! that bypass Rust's global allocator. Revisit this assumption if a tested store or codec starts
//! offloading work.
//!
//! Fill-value equality checks must allocate nothing. Other operations are checked against
//! [`BASELINE`]. A copy of the 128 KiB data exceeds the byte tolerances, so they detect
//! added copies.
//! Update a baseline only when an operation allocates less.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::num::NonZeroU64;
use std::sync::{Arc, mpsc};

use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset, CowBytes, FillValue, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{ArrayToBytesCodecTraits, CodecOptions};

struct CountingAllocator;

thread_local! {
    // Const initialization and a Copy value keep callbacks allocation-free with native TLS.
    // Targets with allocating, pthread-key-based TLS cannot use this allocator safely.
    static COUNTER: Cell<Option<AllocationCounter>> = const { Cell::new(None) };
}

#[derive(Clone, Copy, Default)]
struct AllocationCounter {
    stats: Stats,
    // Inputs can have been allocated before measurement, so dropping them can make this negative.
    live_bytes: isize,
    min_live_bytes: isize,
}

impl CountingAllocator {
    fn record_alloc(size: usize) {
        let _ = COUNTER.try_with(|counter| {
            if let Some(mut value) = counter.get() {
                value.stats.allocations += 1;
                value.stats.allocated_bytes += size;
                value.live_bytes += size as isize;
                value.stats.peak_bytes = value
                    .stats
                    .peak_bytes
                    .max((value.live_bytes - value.min_live_bytes) as usize);
                counter.set(Some(value));
            }
        });
    }

    fn record_dealloc(size: usize) {
        let _ = COUNTER.try_with(|counter| {
            if let Some(mut value) = counter.get() {
                value.live_bytes -= size as isize;
                value.min_live_bytes = value.min_live_bytes.min(value.live_bytes);
                counter.set(Some(value));
            }
        });
    }
}

// SAFETY: forwards to the system allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record_alloc(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record_alloc(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        Self::record_dealloc(layout.size());
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A reallocation counts as a new allocation of the new size.
        Self::record_dealloc(layout.size());
        Self::record_alloc(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    allocations: usize,
    allocated_bytes: usize,
    /// Maximum growth in live bytes above the lowest balance seen before that allocation.
    /// Freeing pre-existing inputs cannot mask subsequent allocations or inflate an earlier peak.
    peak_bytes: usize,
}

/// Measure the allocations of `f` on a single-thread Rayon pool.
fn measure<R>(name: &str, f: impl FnOnce() -> R + Send) -> (R, Stats)
where
    R: Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        COUNTER.with(|counter| counter.set(Some(AllocationCounter::default())));
        let result = f();
        let stats = COUNTER.with(|counter| counter.take().unwrap().stats);
        println!("{name}: {stats:?}");
        (result, stats)
    })
}

#[test]
fn counter_isolation() {
    std::thread::scope(|scope| {
        // The buffer is allocated before measurement, so sending does not allocate or block.
        let (start, started) = mpsc::sync_channel(1);
        let helper = scope.spawn(move || {
            // If setup panics, dropping the measured closure disconnects this channel.
            if started.recv().is_ok() {
                drop(std::hint::black_box(vec![0u8; 4096]));
            }
        });
        let (_, stats) = measure("counter_isolation", move || {
            start.send(()).unwrap();
            helper.join().unwrap();
            std::hint::black_box(vec![0u8; 64])
        });
        assert_eq!(stats.allocations, 1, "{stats:?}");
        assert_eq!(stats.allocated_bytes, 64, "{stats:?}");
        assert_eq!(stats.peak_bytes, 64, "{stats:?}");
    });
}

#[test]
fn counter_drop_input() {
    let bytes = std::hint::black_box(vec![0u8; DECODED_BYTES]);
    let ((), stats) = measure("counter_drop_input", move || drop(bytes));
    assert_eq!(stats.allocations, 0, "{stats:?}");
    assert_eq!(stats.allocated_bytes, 0, "{stats:?}");
    assert_eq!(stats.peak_bytes, 0, "{stats:?}");
}

#[test]
fn counter_copy_after_drop() {
    let input = std::hint::black_box(vec![0u8; 2 * DECODED_BYTES]);
    let source = std::hint::black_box(vec![1u8; DECODED_BYTES]);
    let (_, stats) = measure("counter_copy_after_drop", move || {
        drop(input);
        // The live balance stays negative even with this full-sized copy.
        std::hint::black_box(source.clone())
    });
    assert_eq!(stats.allocations, 1, "{stats:?}");
    assert_eq!(stats.allocated_bytes, DECODED_BYTES, "{stats:?}");
    assert_eq!(stats.peak_bytes, DECODED_BYTES, "{stats:?}");
}

#[test]
fn counter_drop_after_peak() {
    let input = std::hint::black_box(vec![0u8; DECODED_BYTES]);
    let ((), stats) = measure("counter_drop_after_peak", move || {
        drop(std::hint::black_box(vec![0u8; 64]));
        drop(input);
    });
    assert_eq!(stats.allocations, 1, "{stats:?}");
    assert_eq!(stats.allocated_bytes, 64, "{stats:?}");
    assert_eq!(stats.peak_bytes, 64, "{stats:?}");
}

#[test]
fn counter_rayon_work() {
    let (_, stats) = measure("counter_rayon_work", || {
        rayon::join(
            || std::hint::black_box(vec![0u8; 64]),
            || std::hint::black_box(vec![0u8; 128]),
        )
    });
    assert_eq!(stats.allocations, 2, "{stats:?}");
    assert_eq!(stats.allocated_bytes, 192, "{stats:?}");
    assert_eq!(stats.peak_bytes, 192, "{stats:?}");
}

const SHAPE: [u64; 2] = [256, 256];
const DECODED_BYTES: usize = 256 * 256 * 2;

fn data() -> Vec<u16> {
    (0..SHAPE.iter().product::<u64>())
        .map(|i| u16::try_from(i % 4096).unwrap())
        .collect()
}

fn array(
    chunk_shape: &[u64],
    compressed: bool,
    subchunk_shape: Option<&[u64]>,
) -> Array<MemoryStore> {
    let mut builder = ArrayBuilder::new(
        SHAPE.to_vec(),
        chunk_shape.to_vec(),
        data_type::uint16(),
        0u16,
    );
    if compressed {
        builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))]);
    }
    if let Some(subchunk_shape) = subchunk_shape {
        builder.subchunk_shape(subchunk_shape.to_vec());
    }
    let array = builder
        .build(Arc::new(MemoryStore::new()), "/")
        .unwrap()
        .with_codec_options(CodecOptions::default().with_concurrent_target(1));
    array
        .store_array_subset(&array.subset_all(), data())
        .unwrap();
    array
}

/// Allocation baselines: `(name, allocations, allocated bytes, peak bytes)`.
/// Re-measure with `cargo +nightly test -p zarrs --all-features --test allocations -- --nocapture`.
const BASELINE: &[(&str, usize, usize, usize)] = &[
    ("async_retrieve_chunk_zstd", 11, 131_348, 131_258),
    (
        "async_store_array_subset_many_chunks_zstd",
        566,
        327_506,
        145_603,
    ),
    ("partial_decoder_sharded_zstd", 746, 92578, 16340),
    (
        "retrieve_array_subset_many_chunks_zstd",
        438,
        271_346,
        139_994,
    ),
    ("retrieve_chunk_uncompressed", 8, 172, 162),
    ("retrieve_chunk_zstd", 8, 131_236, 131_226),
    ("retrieve_sharded_all_zstd", 1329, 291_048, 134_940),
    ("retrieve_sharded_partial_zstd", 1096, 155_920, 26692),
    ("sharding_decode_borrowed_shard", 1252, 158_646, 132_676),
    ("store_array_subset_many_chunks_zstd", 502, 304_802, 16948),
    (
        "store_array_subset_partial_chunks_zstd",
        696,
        341_618,
        21012,
    ),
    ("store_array_subset_sharded", 1380, 462_966, 139_675),
    ("store_chunk_uncompressed_borrowed", 5, 131_138, 131_122),
    ("store_chunk_zstd", 6, 131_674, 131_658),
    ("store_chunks_sharded", 1372, 462_790, 139_579),
];

/// The baseline of `name`.
fn baseline(name: &str) -> Stats {
    let &(_, allocations, allocated_bytes, peak_bytes) = BASELINE
        .iter()
        .find(|(baseline_name, ..)| *baseline_name == name)
        .unwrap_or_else(|| panic!("no baseline for {name}"));
    Stats {
        allocations,
        allocated_bytes,
        peak_bytes,
    }
}

/// Assert that `stats` are within a small tolerance of the baseline for `name`.
#[track_caller]
fn check(name: &str, stats: Stats) {
    let Stats {
        allocations,
        allocated_bytes,
        peak_bytes,
    } = baseline(name);
    let within = |measured: usize, baseline: usize, percent: usize, slack: usize| {
        measured <= baseline + baseline * percent / 100 + slack
    };
    assert!(
        within(stats.allocations, allocations, 10, 4),
        "{name}: {} allocations exceeds the baseline of {allocations}",
        stats.allocations
    );
    assert!(
        within(stats.allocated_bytes, allocated_bytes, 2, 1024),
        "{name}: {} allocated bytes exceeds the baseline of {allocated_bytes}",
        stats.allocated_bytes
    );
    assert!(
        within(stats.peak_bytes, peak_bytes, 2, 1024),
        "{name}: {} peak bytes exceeds the baseline of {peak_bytes}",
        stats.peak_bytes
    );
}

/// Measure an operation and check it against its allocation baseline.
#[track_caller]
fn measure_and_check<R: Send>(
    name: &str,
    array: &Array<MemoryStore>,
    f: impl Fn(&Array<MemoryStore>) -> R + Sync,
) {
    let (_, stats) = measure(name, || f(array));
    check(name, stats);
}

#[test]
fn retrieve_chunk_uncompressed() {
    let array = array(&SHAPE, false, None);
    measure_and_check("retrieve_chunk_uncompressed", &array, |array| {
        let bytes = array.retrieve_chunk::<ArrayBytes>(&[0, 0]).unwrap();
        assert_eq!(bytes.size(), DECODED_BYTES);
    });
}

#[test]
fn retrieve_chunk_zstd() {
    let array = array(&SHAPE, true, None);
    measure_and_check("retrieve_chunk_zstd", &array, |array| {
        array.retrieve_chunk::<ArrayBytes>(&[0, 0]).unwrap()
    });
}

#[test]
fn retrieve_array_subset_many_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    measure_and_check("retrieve_array_subset_many_chunks_zstd", &array, |array| {
        array
            .retrieve_array_subset::<ArrayBytes>(&array.subset_all())
            .unwrap()
    });
}

#[test]
fn retrieve_sharded_all_zstd() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    measure_and_check("retrieve_sharded_all_zstd", &array, |array| {
        array
            .retrieve_array_subset::<ArrayBytes>(&array.subset_all())
            .unwrap()
    });
}

#[test]
fn retrieve_sharded_partial_zstd() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    let subset = ArraySubset::new_with_ranges(&[16..112, 16..112]);
    measure_and_check("retrieve_sharded_partial_zstd", &array, |array| {
        array.retrieve_array_subset::<ArrayBytes>(&subset).unwrap()
    });
}

#[test]
fn partial_decoder_sharded_zstd() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    measure_and_check("partial_decoder_sharded_zstd", &array, |array| {
        let decoder = array.partial_decoder(&[0, 0]).unwrap();
        for subset in [
            ArraySubset::new_with_ranges(&[40..60, 40..60]),
            ArraySubset::new_with_ranges(&[16..80, 16..80]),
        ] {
            decoder
                .partial_decode(&subset, &CodecOptions::default().with_concurrent_target(1))
                .unwrap();
        }
    });
}

#[test]
fn sharding_decode_borrowed_shard() {
    let array = array(&SHAPE, false, Some(&[32, 32]));
    let encoded = array.retrieve_encoded_chunk(&[0, 0]).unwrap().unwrap();
    let encoded = encoded.to_vec();
    let codecs = array.codecs_bound();
    let shape = SHAPE.map(|s| NonZeroU64::new(s).unwrap());
    let (decoded, stats) = measure("sharding_decode_borrowed_shard", || {
        codecs
            .decode(
                CowBytes::Borrowed(&encoded),
                &shape,
                &CodecOptions::default().with_concurrent_target(1),
            )
            .unwrap()
            .into_owned()
    });
    assert_eq!(decoded.size(), DECODED_BYTES);
    check("sharding_decode_borrowed_shard", stats);
}

#[test]
fn store_chunk_uncompressed_borrowed() {
    let array = array(&SHAPE, false, None);
    let data = data();
    let ((), stats) = measure("store_chunk_uncompressed_borrowed", || {
        array.store_chunk(&[0, 0], data.as_slice()).unwrap();
    });
    check("store_chunk_uncompressed_borrowed", stats);
}

#[test]
fn store_chunk_zstd() {
    let array = array(&SHAPE, true, None);
    let data = data();
    let ((), stats) = measure("store_chunk_zstd", || {
        array.store_chunk(&[0, 0], data.as_slice()).unwrap();
    });
    check("store_chunk_zstd", stats);
}

#[test]
fn store_array_subset_sharded() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    let data = data();
    let ((), stats) = measure("store_array_subset_sharded", || {
        array
            .store_array_subset(&array.subset_all(), data.as_slice())
            .unwrap();
    });
    check("store_array_subset_sharded", stats);
}

#[test]
fn store_array_subset_many_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    let data = data();
    measure_and_check("store_array_subset_many_chunks_zstd", &array, |array| {
        array
            .store_array_subset(&array.subset_all(), data.as_slice())
            .unwrap();
    });
}

#[test]
fn store_array_subset_partial_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    let subset = ArraySubset::new_with_ranges(&[32..224, 32..224]);
    let data: Vec<u16> = vec![1; 192 * 192];
    measure_and_check("store_array_subset_partial_chunks_zstd", &array, |array| {
        array.store_array_subset(&subset, data.as_slice()).unwrap();
    });
}

#[test]
fn store_chunks_sharded() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    let data = data();
    measure_and_check("store_chunks_sharded", &array, |array| {
        array
            .store_chunks(
                &ArraySubset::new_with_ranges(&[0..1, 0..1]),
                data.as_slice(),
            )
            .unwrap();
    });
}

#[cfg(feature = "async")]
#[test]
fn async_retrieve_chunk_zstd() {
    use zarrs::storage::store::AsyncMemoryStore;
    let store = Arc::new(AsyncMemoryStore::new());
    let array = ArrayBuilder::new(SHAPE.to_vec(), SHAPE.to_vec(), data_type::uint16(), 0u16)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
        .build(store, "/")
        .unwrap()
        .with_codec_options(CodecOptions::default().with_concurrent_target(1));
    futures::executor::block_on(array.async_store_chunk(&[0, 0], data())).unwrap();
    let retrieve =
        || futures::executor::block_on(array.async_retrieve_chunk::<ArrayBytes>(&[0, 0])).unwrap();
    let (_, stats) = measure("async_retrieve_chunk_zstd", retrieve);
    check("async_retrieve_chunk_zstd", stats);
}

#[cfg(feature = "async")]
#[test]
fn async_store_array_subset_many_chunks_zstd() {
    use zarrs::storage::store::AsyncMemoryStore;
    let array = ArrayBuilder::new(SHAPE.to_vec(), vec![64, 64], data_type::uint16(), 0u16)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
        .build(Arc::new(AsyncMemoryStore::new()), "/")
        .unwrap()
        .with_codec_options(CodecOptions::default().with_concurrent_target(1));
    let data = data();
    let ((), stats) = measure("async_store_array_subset_many_chunks_zstd", || {
        futures::executor::block_on(
            array.async_store_array_subset(&array.subset_all(), data.as_slice()),
        )
        .unwrap();
    });
    check("async_store_array_subset_many_chunks_zstd", stats);
}

#[test]
fn optional_fill_value_checks() {
    for depth in 1..=3 {
        let mut fill_value = FillValue::from(0x1234u16);
        let mut bytes = ArrayBytes::new_flen(0x1234u16.to_ne_bytes().repeat(32));
        for _ in 0..depth {
            fill_value = fill_value.into_optional();
            bytes = bytes.with_optional_mask(vec![1; 32]);
        }
        let (inner, _) = bytes.clone().into_optional().unwrap().into_parts();
        let mismatched = (*inner).with_optional_mask(vec![0; 32]);
        for (name, bytes, expected) in [
            ("optional_fill_matches", bytes, true),
            ("optional_fill_mask_mismatch", mismatched, false),
        ] {
            let (matches, stats) = measure(name, || {
                std::hint::black_box(&bytes).is_fill_value(std::hint::black_box(&fill_value))
            });
            assert_eq!(matches, expected);
            assert_eq!(stats.allocations, 0, "{name}, depth {depth}: {stats:?}");
        }
    }
}

// Before reservation, 1,024 one-byte ranges used 1,033 allocations / 50,080 bytes
// for both helpers. The outer vector now allocates once: 1,025 / 25,600 total.
#[test]
fn collection_byte_ranges() {
    use zarrs::storage::byte_range::{
        ByteRange, extract_byte_ranges, extract_byte_ranges_read_seek,
    };

    for count in [0, 1, 1024] {
        let ranges = vec![ByteRange::FromStart(0, Some(1)); count];
        let (output, stats) = measure("collection_byte_ranges", || {
            extract_byte_ranges(&[42], ranges.iter().copied()).unwrap()
        });
        assert_eq!(output, vec![vec![42]; count]);
        assert_eq!(
            stats.allocations,
            count + usize::from(count > 0),
            "{stats:?}"
        );
        assert_eq!(stats.allocated_bytes, count * (size_of::<Vec<u8>>() + 1));
        let (output, stats) = measure("collection_byte_ranges_read_seek", || {
            extract_byte_ranges_read_seek(std::io::Cursor::new([42]), ranges.iter().copied())
                .unwrap()
        });
        assert_eq!(output, vec![vec![42]; count]);
        assert_eq!(
            stats.allocations,
            count + usize::from(count > 0),
            "{stats:?}"
        );
        assert_eq!(stats.allocated_bytes, count * (size_of::<Vec<u8>>() + 1));
    }
}

// The 1,024-index baseline was 10 allocations / 16,384 bytes; now 2 / 8,224.
#[test]
fn collection_indexer() {
    use zarrs_chunk_grid::Indexer;

    for count in [0, 1, 1024] {
        let indices: Vec<Vec<u64>> = (0..count).map(|i| vec![i]).collect();
        let (output, stats) = measure("collection_indexer", || {
            indices.iter_linearised_indices(&[1024]).unwrap()
        });
        assert_eq!(output.collect::<Vec<_>>(), (0..count).collect::<Vec<_>>());
        assert_eq!(stats.allocations, 1 + usize::from(count > 0), "{stats:?}");
    }
}

// The 1,024-range baseline was 15 allocations / 114,768 bytes; now 7 / 82,136.
#[cfg(feature = "async")]
#[test]
fn collection_async_partial_decode() {
    use zarrs::storage::byte_range::ByteRange;
    use zarrs::storage::store::AsyncMemoryStore;
    use zarrs::storage::{AsyncWritableStorageTraits, StoreKey};
    use zarrs_codec::AsyncBytesPartialDecoderTraits;

    let store = AsyncMemoryStore::new();
    let key = StoreKey::new("data").unwrap();
    futures::executor::block_on(store.set(&key, vec![42].into())).unwrap();
    let decoder = (store, key);
    for count in [0, 1, 1024] {
        let ranges = vec![ByteRange::FromStart(0, Some(1)); count];
        let (output, stats) =
            measure("collection_async_partial_decode", || {
                futures::executor::block_on(decoder.partial_decode_many(
                    Box::new(ranges.iter().copied()),
                    &CodecOptions::default(),
                ))
                .unwrap()
                .unwrap()
            });
        assert_eq!(output.len(), count);
        assert!(output.iter().all(|bytes| bytes.as_ref() == [42]));
        assert!(stats.allocations <= 7, "{stats:?}");
    }
}

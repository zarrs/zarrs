#![allow(missing_docs)]
#![cfg(all(feature = "zstd", not(miri)))]

//! Heap allocation counts of representative array operations.
//!
//! A counting global allocator records the number of allocations, the total bytes allocated and the
//! peak bytes live during an operation.
//! Operations run on a single-thread Rayon pool with a concurrency target of 1.
//! All scenarios run in one test so other test-harness threads cannot pollute the counters.
//!
//! Each operation is checked against [`BASELINE`], measured on main at c7f5590f. A copy of the 128 KiB data exceeds the byte tolerances, so they detect added copies.
//! Update a baseline only when an operation allocates less.

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset, CowBytes, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{ArrayToBytesCodecTraits, CodecOptions};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

impl CountingAllocator {
    fn record_alloc(size: usize) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
        let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
        PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
    }

    fn record_dealloc(size: usize) {
        LIVE_BYTES.fetch_sub(size, Ordering::Relaxed);
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

#[derive(Debug, Clone, Copy)]
struct Stats {
    allocations: usize,
    allocated_bytes: usize,
    /// Peak live bytes above the live bytes at the start of the measurement.
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
        let start_live = LIVE_BYTES.load(Ordering::Relaxed);
        ALLOCATIONS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        PEAK_LIVE_BYTES.store(start_live, Ordering::Relaxed);
        let result = f();
        let stats = Stats {
            allocations: ALLOCATIONS.load(Ordering::Relaxed),
            allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            peak_bytes: PEAK_LIVE_BYTES
                .load(Ordering::Relaxed)
                .saturating_sub(start_live),
        };
        println!("{name}: {stats:?}");
        (result, stats)
    })
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

/// Allocation counts measured on main at c7f5590f: `(name, allocations, allocated bytes, peak bytes)`.
const BASELINE: &[(&str, usize, usize, usize)] = &[
    ("async_retrieve_chunk_zstd", 14, 131433, 131258),
    (
        "async_store_array_subset_many_chunks_zstd",
        630,
        329122,
        145603,
    ),
    ("partial_decoder_sharded_zstd", 852, 100159, 16623),
    (
        "retrieve_array_subset_many_chunks_zstd",
        582,
        274754,
        140010,
    ),
    ("retrieve_chunk_uncompressed", 11, 257, 162),
    ("retrieve_chunk_zstd", 11, 131321, 131226),
    ("retrieve_sharded_all_zstd", 1729, 300621, 135116),
    ("retrieve_sharded_partial_zstd", 1228, 165813, 26975),
    ("sharding_decode_borrowed_shard", 1649, 168134, 132852),
    ("store_array_subset_many_chunks_zstd", 582, 306498, 16948),
    ("store_array_subset_partial_chunks_zstd", 884, 345870, 21012),
    ("store_array_subset_sharded", 1449, 493792, 139691),
    ("store_chunk_uncompressed_borrowed", 10, 131244, 131122),
    ("store_chunk_zstd", 11, 131780, 131736),
    ("store_chunks_sharded", 1441, 493616, 139595),
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

fn retrieve_chunk_uncompressed() {
    let array = array(&SHAPE, false, None);
    measure_and_check("retrieve_chunk_uncompressed", &array, |array| {
        let bytes = array.retrieve_chunk::<ArrayBytes>(&[0, 0]).unwrap();
        assert_eq!(bytes.size(), DECODED_BYTES);
    });
}

fn retrieve_chunk_zstd() {
    let array = array(&SHAPE, true, None);
    measure_and_check("retrieve_chunk_zstd", &array, |array| {
        array.retrieve_chunk::<ArrayBytes>(&[0, 0]).unwrap()
    });
}

fn retrieve_array_subset_many_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    measure_and_check("retrieve_array_subset_many_chunks_zstd", &array, |array| {
        array
            .retrieve_array_subset::<ArrayBytes>(&array.subset_all())
            .unwrap()
    });
}

fn retrieve_sharded_zstd() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    measure_and_check("retrieve_sharded_all_zstd", &array, |array| {
        array
            .retrieve_array_subset::<ArrayBytes>(&array.subset_all())
            .unwrap()
    });

    let subset = ArraySubset::new_with_ranges(&[16..112, 16..112]);
    measure_and_check("retrieve_sharded_partial_zstd", &array, |array| {
        array.retrieve_array_subset::<ArrayBytes>(&subset).unwrap()
    });
}

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

fn store_chunk_uncompressed_borrowed() {
    let array = array(&SHAPE, false, None);
    let data = data();
    let (_, stats) = measure("store_chunk_uncompressed_borrowed", || {
        array.store_chunk(&[0, 0], data.as_slice()).unwrap();
    });
    check("store_chunk_uncompressed_borrowed", stats);
}

fn store_chunk_zstd() {
    let array = array(&SHAPE, true, None);
    let data = data();
    let (_, stats) = measure("store_chunk_zstd", || {
        array.store_chunk(&[0, 0], data.as_slice()).unwrap();
    });
    check("store_chunk_zstd", stats);
}

fn store_array_subset_sharded() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    let data = data();
    let (_, stats) = measure("store_array_subset_sharded", || {
        array
            .store_array_subset(&array.subset_all(), data.as_slice())
            .unwrap();
    });
    check("store_array_subset_sharded", stats);
}

fn store_array_subset_many_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    let data = data();
    measure_and_check("store_array_subset_many_chunks_zstd", &array, |array| {
        array
            .store_array_subset(&array.subset_all(), data.as_slice())
            .unwrap();
    });
}

fn store_array_subset_partial_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    let subset = ArraySubset::new_with_ranges(&[32..224, 32..224]);
    let data: Vec<u16> = vec![1; 192 * 192];
    measure_and_check("store_array_subset_partial_chunks_zstd", &array, |array| {
        array.store_array_subset(&subset, data.as_slice()).unwrap();
    });
}

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
fn allocation_regressions() {
    retrieve_chunk_uncompressed();
    retrieve_chunk_zstd();
    retrieve_array_subset_many_chunks_zstd();
    retrieve_sharded_zstd();
    partial_decoder_sharded_zstd();
    sharding_decode_borrowed_shard();
    store_chunk_uncompressed_borrowed();
    store_chunk_zstd();
    store_array_subset_sharded();
    store_array_subset_many_chunks_zstd();
    store_array_subset_partial_chunks_zstd();
    store_chunks_sharded();
    #[cfg(feature = "async")]
    {
        async_retrieve_chunk_zstd();
        async_store_array_subset_many_chunks_zstd();
    }
}

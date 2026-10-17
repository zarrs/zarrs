#![allow(missing_docs)]
#![cfg(all(feature = "zstd", not(miri)))]

//! Heap allocation counts of representative array operations.
//!
//! A counting global allocator records the number of allocations, the total bytes allocated and the
//! peak bytes live during an operation.
//! Operations run on a single-thread Rayon pool with a compute target of 1 and storage requests on the compute thread,
//! so the counts are deterministic.
//!
//! Each operation is checked against [`BASELINE`], the counts last measured for it (see [`check`]).
//! A copy of the 128 KiB data exceeds the byte tolerances, so they detect added copies.
//! Lower a baseline when an operation allocates less; raise one only for a deliberate trade-off.

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serial_test::serial;
use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset, CowBytes, Resources, data_type};
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

/// One compute thread, and storage requests on it, so that a slow request on a busy machine does not move requests to
/// the I/O pool (which allocates to start).
fn resources() -> Resources {
    Resources::default()
        .with_compute_target(1)
        .with_read_dispatch(Some(zarrs_codec::ReadDispatch::Inline))
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
    let array = builder.build(Arc::new(MemoryStore::new()), "/").unwrap();
    array
        .store_array_subset(&array.subset_all(), data(), &Resources::default())
        .unwrap();
    array
}

/// Allocation counts of each operation: `(name, allocations, allocated bytes, peak bytes)`.
///
/// Measured on 2026-10-18. Against the legacy reads and writes (before the I/O and compute scheduling rework), sharded
/// reads allocate 86-98% less often, many-chunk reads and writes 40-74% less and sharded writes 39% less, with about
/// the same peak memory. A read or write of a single chunk allocates about half as often, but plans a task, which
/// allocates a few hundred bytes more.
const BASELINE: &[(&str, usize, usize, usize)] = &[
    // Zero-copy from the store: only planning and metadata-sized allocations
    ("retrieve_chunk_uncompressed", 6, 558, 549),
    // One decompressed output
    ("retrieve_chunk_zstd", 6, 131_606, 131_597),
    // The output plus one decoded intermediate per 64 x 64 chunk
    ("retrieve_array_subset_many_chunks_zstd", 82, 273_481, 142_865),
    ("retrieve_sharded_all_zstd", 105, 271_382, 138_133),
    ("retrieve_sharded_partial_zstd", 123, 74_086, 26_229),
    ("partial_decoder_sharded_zstd", 111, 52_182, 16_517),
    // The output only: the borrowed shard is not copied
    ("sharding_decode_borrowed_shard", 34, 139_240, 136_000),
    // The memory store copies the value once
    ("store_chunk_uncompressed_borrowed", 9, 131_326, 131_133),
    ("store_chunk_zstd", 10, 131_862, 131_837),
    ("store_array_subset_sharded", 623, 461_862, 143_272),
    ("async_retrieve_chunk_zstd", 10, 131_902, 131_629),
    ("store_array_subset_many_chunks_zstd", 191, 289_993, 19_341),
    ("store_array_subset_partial_chunks_zstd", 227, 328_177, 23_533),
    ("store_chunks_sharded", 622, 461_798, 143_272),
    ("async_store_array_subset_many_chunks_zstd", 308, 321_361, 20_645),
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
        within(stats.allocations, allocations, 5, 2),
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

/// Measure the operation `f` on `array`, and check it against its baseline.
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
#[serial]
fn retrieve_chunk_uncompressed() {
    let array = array(&SHAPE, false, None);
    measure_and_check("retrieve_chunk_uncompressed", &array, |array| {
        let bytes = array
            .retrieve_chunk::<ArrayBytes>(&[0, 0], &resources())
            .unwrap();
        assert_eq!(bytes.size(), DECODED_BYTES);
    });
}

#[test]
#[serial]
fn retrieve_chunk_zstd() {
    let array = array(&SHAPE, true, None);
    measure_and_check("retrieve_chunk_zstd", &array, |array| {
        array
            .retrieve_chunk::<ArrayBytes>(&[0, 0], &resources())
            .unwrap()
    });
}

#[test]
#[serial]
fn retrieve_array_subset_many_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    // 16 chunks
    measure_and_check("retrieve_array_subset_many_chunks_zstd", &array, |array| {
        array
            .retrieve_array_subset::<ArrayBytes>(&array.subset_all(), &resources())
            .unwrap()
    });
}

#[test]
#[serial]
fn retrieve_sharded_zstd() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    // One shard of 64 subchunks
    measure_and_check("retrieve_sharded_all_zstd", &array, |array| {
        array
            .retrieve_array_subset::<ArrayBytes>(&array.subset_all(), &resources())
            .unwrap()
    });

    // 25 intersecting subchunks
    let subset = ArraySubset::new_with_ranges(&[16..112, 16..112]);
    measure_and_check("retrieve_sharded_partial_zstd", &array, |array| {
        array
            .retrieve_array_subset::<ArrayBytes>(&subset, &resources())
            .unwrap()
    });
}

#[test]
#[serial]
fn partial_decoder_sharded_zstd() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    // The index, one subchunk, then 9 subchunks
    measure_and_check("partial_decoder_sharded_zstd", &array, |array| {
        let decoder = array.partial_decoder(&[0, 0], &resources()).unwrap();
        for subset in [
            ArraySubset::new_with_ranges(&[40..60, 40..60]),
            ArraySubset::new_with_ranges(&[16..80, 16..80]),
        ] {
            decoder
                .partial_decode(&subset, &CodecOptions::default(), &resources())
                .unwrap();
        }
    });
}

#[test]
#[serial]
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
                &CodecOptions::default(),
                &resources(),
            )
            .unwrap()
            .into_owned()
    });
    assert_eq!(decoded.size(), DECODED_BYTES);
    check("sharding_decode_borrowed_shard", stats);
}

#[test]
#[serial]
fn store_chunk_uncompressed_borrowed() {
    let array = array(&SHAPE, false, None);
    let data = data();
    let (_, stats) = measure("store_chunk_uncompressed_borrowed", || {
        array
            .store_chunk(&[0, 0], data.as_slice(), &resources())
            .unwrap();
    });
    check("store_chunk_uncompressed_borrowed", stats);
}

#[test]
#[serial]
fn store_chunk_zstd() {
    let array = array(&SHAPE, true, None);
    let data = data();
    let (_, stats) = measure("store_chunk_zstd", || {
        array
            .store_chunk(&[0, 0], data.as_slice(), &resources())
            .unwrap();
    });
    check("store_chunk_zstd", stats);
}

#[test]
#[serial]
fn store_array_subset_sharded() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    let data = data();
    let (_, stats) = measure("store_array_subset_sharded", || {
        array
            .store_array_subset(&array.subset_all(), data.as_slice(), &resources())
            .unwrap();
    });
    check("store_array_subset_sharded", stats);
}

#[test]
#[serial]
fn store_array_subset_many_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    let data = data();
    measure_and_check("store_array_subset_many_chunks_zstd", &array, |array| {
        array
            .store_array_subset(&array.subset_all(), data.as_slice(), &resources())
            .unwrap();
    });
}

#[test]
#[serial]
fn store_array_subset_partial_chunks_zstd() {
    let array = array(&[64, 64], true, None);
    let subset = ArraySubset::new_with_ranges(&[32..224, 32..224]);
    let data: Vec<u16> = vec![1; 192 * 192];
    measure_and_check("store_array_subset_partial_chunks_zstd", &array, |array| {
        array
            .store_array_subset(&subset, data.as_slice(), &resources())
            .unwrap();
    });
}

#[test]
#[serial]
fn store_chunks_sharded() {
    let array = array(&SHAPE, true, Some(&[32, 32]));
    let data = data();
    measure_and_check("store_chunks_sharded", &array, |array| {
        array
            .store_chunks(
                &ArraySubset::new_with_ranges(&[0..1, 0..1]),
                data.as_slice(),
                &resources(),
            )
            .unwrap();
    });
}

#[cfg(feature = "async")]
#[test]
#[serial]
fn async_retrieve_chunk_zstd() {
    use zarrs::storage::store::AsyncMemoryStore;
    let store = Arc::new(AsyncMemoryStore::new());
    let array = ArrayBuilder::new(SHAPE.to_vec(), SHAPE.to_vec(), data_type::uint16(), 0u16)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
        .build(store, "/")
        .unwrap();
    futures::executor::block_on(array.async_store_chunk(&[0, 0], data(), &Resources::default()))
        .unwrap();
    let retrieve = || {
        futures::executor::block_on(array.async_retrieve_chunk::<ArrayBytes>(&[0, 0], &resources()))
            .unwrap()
    };
    // The first drive of the executor initialises thread-local state
    retrieve();
    let (_, stats) = measure("async_retrieve_chunk_zstd", retrieve);
    check("async_retrieve_chunk_zstd", stats);
}

#[cfg(feature = "async")]
#[test]
#[serial]
fn async_store_array_subset_many_chunks_zstd() {
    use zarrs::storage::store::AsyncMemoryStore;
    let array = ArrayBuilder::new(SHAPE.to_vec(), vec![64, 64], data_type::uint16(), 0u16)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(1, false))])
        .build(Arc::new(AsyncMemoryStore::new()), "/")
        .unwrap();
    let data = data();
    // The first drive of the executor initialises thread-local state
    futures::executor::block_on(array.async_store_array_subset(
        &array.subset_all(),
        data.as_slice(),
        &resources(),
    ))
    .unwrap();
    let ((), stats) = measure("async_store_array_subset_many_chunks_zstd", || {
        futures::executor::block_on(array.async_store_array_subset(
            &array.subset_all(),
            data.as_slice(),
            &resources(),
        ))
        .unwrap();
    });
    check("async_store_array_subset_many_chunks_zstd", stats);
}

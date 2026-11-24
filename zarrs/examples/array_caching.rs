//! Read a sharded array through each kind of cache, counting the storage reads of repeated reads.
//!
//! An [`ArrayCached`] combines an [`Array`] with a cache. Reads are served through the cache, and writes go to the
//! array and invalidate what the cache holds for the written region. The caches differ in what they hold:
//! - an encoded chunk cache holds encoded chunks (here, whole shards), so a hit skips storage but still decodes,
//! - a decoded chunk cache holds decoded chunks, so a hit skips storage and decoding,
//! - a partial decoder cache holds a partial decoder per chunk, which for a shard holds its index, so a hit reads
//!   only the subchunks it needs,
//! - a [`SubchunkCache`] holds decoded subchunks over one of the above, so a hit skips storage and decoding while
//!   holding only the subchunks that were read.
//!
//! Each chunk cache has `LruChunkLimit` and `LruSizeLimit` (bytes) variants. With the `async` feature, the
//! `AsyncChunkCache*` caches serve the `async_` operations of an [`ArrayCached`], and a [`SubchunkCache`] composes
//! them in the same way.
//!
//! Caches are dyn compatible, so caches of different types can be held as a `Box<dyn SyncArrayCache>` (or
//! `Box<dyn AsyncArrayCache>`) and passed to [`ArrayCached::new_boxed`]. Each [`ArrayCached`] owns its cache.
//!
//! The example prints the shards and subchunks of the array with the regions it reads, then a table of the storage
//! reads and cache contents of two passes over the regions through each cache.

use std::error::Error;
use std::fmt::Write;
use std::sync::Arc;

use zarrs::array::cache::{
    ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruSizeLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit, NoCache, SubchunkCache, SyncArrayCache,
    SyncChunkCacheTraits,
};
use zarrs::array::codec::GzipCodec;
use zarrs::array::{Array, ArrayBuilder, ArrayCached, ArraySubset, Resources, data_type};
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;

/// An in-memory store that counts storage reads.
type Store = PerformanceMetricsStorageAdapter<MemoryStore>;

/// The length of each dimension of the array.
const ARRAY: u64 = 256;
/// The length of each dimension of a shard.
const SHARD: u64 = 128;
/// The length of each dimension of a subchunk.
const SUBCHUNK: u64 = 16;
/// The size of a decoded subchunk of `uint16` elements, in bytes.
const SUBCHUNK_BYTES: u64 = SUBCHUNK * SUBCHUNK * 2;

/// Small reads of the array: one subchunk and four subchunks of shard `[0, 0]`, one subchunk of shard `[1, 1]`, and
/// another subchunk of shard `[0, 0]`.
fn regions() -> [ArraySubset; 4] {
    [
        ArraySubset::new_with_ranges(&[3..5, 3..5]),
        ArraySubset::new_with_ranges(&[20..40, 20..40]),
        ArraySubset::new_with_ranges(&[130..134, 200..204]),
        ArraySubset::new_with_ranges(&[100..104, 100..104]),
    ]
}

/// The number of [`regions`] read by each pass.
///
/// The 2nd pass rereads the regions of the 1st pass, and also reads the last region, an unread subchunk of a shard that
/// the 1st pass read.
const PASS_REGIONS: [usize; 2] = [3, 4];

fn main() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::new(),
    )));
    let array = create_array(&store)?;
    print_grid(&array)?;

    println!(
        "Storage reads of two passes through each cache. The 1st pass reads regions 1 to {}, and the 2nd pass\n\
         rereads them and reads region {}:\n",
        PASS_REGIONS[0], PASS_REGIONS[1]
    );
    print_header();
    for subchunked in [false, true] {
        let read = Read {
            array: &array,
            store: &store,
            subchunked,
        };
        read.through("NoCache", NoCache, |_| Vec::new())?;
        read.through("Encoded", ChunkCacheEncodedLruChunkLimit::new(2), |cache| {
            vec![held(cache, "encoded shards")]
        })?;
        read.through("Decoded", ChunkCacheDecodedLruChunkLimit::new(2), |cache| {
            vec![held(cache, "decoded shards")]
        })?;
        read.through(
            "PartialDecoder",
            ChunkCachePartialDecoderLruChunkLimit::new(2),
            |cache| vec![held(cache, "shard indexes")],
        )?;
    }

    println!(
        "\nChunks are held in a ChunkCache{{Encoded,Decoded,PartialDecoder}}LruChunkLimit, or none with NoCache.\n\
         Subchunks are held in a ChunkCacheDecodedLruSizeLimit by a SubchunkCache over the chunk cache."
    );
    Ok(())
}

/// What a cache holds, as a size in bytes and a description.
type Held = Vec<(usize, String)>;

/// What the chunk cache `cache` holds, describing its chunks as `entries`.
fn held(cache: &impl SyncChunkCacheTraits, entries: &str) -> (usize, String) {
    (cache.size(), format!("{} {entries}", cache.len()))
}

/// Reads of an array, each through a new cache.
struct Read<'a> {
    array: &'a Arc<Array<Store>>,
    store: &'a Store,
    /// Whether to read through a [`SubchunkCache`] over each chunk cache, rather than the chunk cache alone.
    subchunked: bool,
}

impl Read<'_> {
    /// Read through the chunk cache `chunks` named `name`, which holds `chunks_held`.
    ///
    /// The chunk caches hold whole chunks (here, whole shards). Without a chunk cache, every read opens the shard: it
    /// reads the shard index, then the subchunks. Under a [`SubchunkCache`], a subchunk miss:
    /// - over [`NoCache`], opens the shard again, but nothing is held per shard,
    /// - over an encoded chunk cache, decodes the subchunk from the cached shard,
    /// - over a decoded chunk cache, copies the subchunk from the cached decoded shard, so the subchunk cache only
    ///   duplicates what the chunk cache holds. This is permitted, but redundant,
    /// - over a partial decoder cache, reads only that subchunk, as the shard index is cached.
    fn through<C: SyncArrayCache>(
        &self,
        name: &str,
        chunks: C,
        chunks_held: impl Fn(&C) -> Held,
    ) -> Result<(), Box<dyn Error>> {
        if self.subchunked {
            // The subchunks are held in a decoded chunk cache keyed by subchunk indices, here with room for 64
            // subchunks. Level 0 is the outermost subchunk grid, the subchunks of the shards.
            let subchunks = ChunkCacheDecodedLruSizeLimit::new(64 * SUBCHUNK_BYTES);
            let cache = SubchunkCache::new(0, subchunks, chunks);
            self.read_twice([name, "Decoded"], cache, |cache| {
                let mut held_ = chunks_held(cache.chunks());
                held_.push(held(cache.subchunks(), "decoded subchunks"));
                held_
            })
        } else {
            self.read_twice([name, "-"], chunks, chunks_held)
        }
    }

    /// Read the [`regions`] through `cache` in two passes (see [`PASS_REGIONS`]), and print a row of the cache
    /// `names`, the storage reads of each pass, and what the cache holds afterwards.
    fn read_twice<C: SyncArrayCache>(
        &self,
        names: [&str; 2],
        cache: C,
        held: impl Fn(&C) -> Held,
    ) -> Result<(), Box<dyn Error>> {
        let cached = ArrayCached::new(self.array.clone(), cache);
        let resources = Resources::default();
        let mut passes = Vec::new();
        for regions_read in PASS_REGIONS {
            self.store.reset();
            for region in &regions()[..regions_read] {
                let _: Vec<u16> = cached.retrieve_array_subset(region, &resources)?;
            }
            passes.push((self.store.reads(), self.store.bytes_read()));
        }
        let held = held(cached.cache());
        let bytes_held: usize = held.iter().map(|(bytes, _)| bytes).sum();
        let contents = held
            .into_iter()
            .map(|(_, contents)| contents)
            .collect::<Vec<_>>();
        print_row(&[
            names[0].to_string(),
            names[1].to_string(),
            passes[0].0.to_string(),
            passes[0].1.to_string(),
            passes[1].0.to_string(),
            passes[1].1.to_string(),
            bytes_held.to_string(),
            if contents.is_empty() {
                "nothing".to_string()
            } else {
                contents.join(" + ")
            },
        ]);
        Ok(())
    }
}

// The functions below create the array and print the grid and table.

/// Create a 256x256 `uint16` array of 128x128 shards, each holding 8x8 gzip compressed subchunks of 16x16 elements.
fn create_array(store: &Arc<Store>) -> Result<Arc<Array<Store>>, Box<dyn Error>> {
    let array = ArrayBuilder::new(
        vec![ARRAY, ARRAY],
        vec![SHARD, SHARD],
        data_type::uint16(),
        0u16,
    )
    .subchunk_shape(vec![SUBCHUNK, SUBCHUNK])
    .bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5)?)])
    .build_arc(store.clone(), "/array")?;
    array.store_metadata()?;
    array.store_array_subset(
        &array.subset_all(),
        (0..=u16::MAX).collect::<Vec<u16>>(),
        &Resources::default(),
    )?;
    Ok(array)
}

/// Print the shards and subchunks of `array`, marking the subchunks of each of the [`regions`] with its number.
fn print_grid(array: &Array<Store>) -> Result<(), Box<dyn Error>> {
    println!(
        "A {ARRAY}x{ARRAY} array of {SHARD}x{SHARD} shards (boxes) of {SUBCHUNK}x{SUBCHUNK} subchunks (dots).\n\
         The subchunks of each region read are marked with its number:"
    );
    for (number, region) in (1..).zip(regions()) {
        let passes = if number <= PASS_REGIONS[0] {
            ""
        } else {
            " (2nd pass only)"
        };
        println!("  {number}: {region}{passes}");
    }
    println!();

    // Mark the subchunks overlapping each region in a grid of subchunks
    let subchunk_grid = array
        .subchunk_grid_at_level(0)
        .as_chunk_grid()
        .ok_or("the array has no subchunk grid")?;
    let subchunks = usize::try_from(ARRAY / SUBCHUNK)?;
    let mut grid = vec![vec!['.'; subchunks]; subchunks];
    for (number, region) in (1u32..).zip(regions()) {
        let region_subchunks = subchunk_grid
            .chunks_in_array_subset(&region)?
            .ok_or("the region has no subchunks")?;
        for indices in &region_subchunks.indices() {
            grid[usize::try_from(indices[0])?][usize::try_from(indices[1])?] =
                char::from_digit(number, 10).ok_or("too many regions")?;
        }
    }

    // Draw the grid, with a box around each shard and the element index of each shard boundary
    let shards = usize::try_from(ARRAY / SHARD)?;
    let shard_subchunks = usize::try_from(SHARD / SUBCHUNK)?;
    let shard_width = 2 * shard_subchunks + 2;
    let mut labels = String::new();
    for boundary in (0..=ARRAY).step_by(usize::try_from(SHARD)?) {
        write!(labels, "{boundary:<shard_width$}")?;
    }
    println!("{:5}{}", "", labels.trim_end());
    let border = format!("+{}", "-".repeat(shard_width - 1)).repeat(shards) + "+";
    for (row, start) in grid
        .iter()
        .zip((0..ARRAY).step_by(usize::try_from(SUBCHUNK)?))
    {
        if start % SHARD == 0 {
            println!("{start:>4} {border}");
        }
        let mut line = format!("{:5}", "");
        for shard in row.chunks(shard_subchunks) {
            line.push('|');
            for cell in shard {
                write!(line, " {cell}")?;
            }
            line.push(' ');
        }
        println!("{line}|");
    }
    println!("{ARRAY:>4} {border}\n");
    Ok(())
}

/// The widths of the table columns, other than the last.
const WIDTHS: [usize; 7] = [14, 9, 5, 10, 5, 10, 10];

/// Print the header of the table.
fn print_header() {
    let span = |columns: std::ops::Range<usize>| {
        WIDTHS[columns.clone()].iter().sum::<usize>() + 2 * (columns.len() - 1)
    };
    println!(
        "{:-^a$}  {:-^b$}  {:-^b$}",
        " cache ",
        " 1st pass ",
        " 2nd pass ",
        a = span(0..2),
        b = span(2..4),
    );
    print_row(
        &[
            "chunks",
            "subchunks",
            "reads",
            "bytes read",
            "reads",
            "bytes read",
            "bytes held",
            "cache contents",
        ]
        .map(String::from),
    );
}

/// Print a row of the table, with the text columns left aligned and the numeric columns right aligned.
fn print_row(cells: &[String; 8]) {
    let row = cells
        .iter()
        .zip(WIDTHS.into_iter().chain([0]))
        .enumerate()
        .map(|(column, (cell, width))| {
            if column < 2 || column == WIDTHS.len() {
                format!("{cell:<width$}")
            } else {
                format!("{cell:>width$}")
            }
        })
        .collect::<Vec<_>>()
        .join("  ");
    println!("{}", row.trim_end());
}

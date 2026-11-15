//! Subchunk caches layered over chunk caches.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "async")]
use super::AsyncChunkCache;
use super::ChunkCache;
#[cfg(feature = "async")]
use super::array_cache::array_cache_sealed::AsyncArrayCacheOps;
use super::array_cache::array_cache_sealed::SyncArrayCacheOps;
use crate::array::array_ops::subchunk_grid_at_level;
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, CodecOptions, Indexer,
};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{ArrayPartialDecoderTraits, Resources};
use zarrs_storage::ReadableStorageTraits;
#[cfg(feature = "async")]
use zarrs_storage::{AsyncReadableStorageTraits, MaybeSend, MaybeSync};

mod decoded;

/// A subchunk cache layered over a chunk cache.
///
/// `SubchunkCached` caches the decoded subchunks of the `level` subchunk grid of an array in the subchunk cache `S`,
/// and serves the subchunks that `S` does not hold through the chunk cache `C`. Use it with
/// [`ArrayCached`](crate::array::ArrayCached) like any other cache.
///
/// The two caches are chosen independently:
/// - The subchunk cache is a decoded chunk cache keyed by subchunk indices rather than chunk indices, e.g. a
///   [`ChunkCacheDecodedLruSizeLimit`](super::ChunkCacheDecodedLruSizeLimit). It sets how many subchunks are held.
///   A subchunk that is missing or entirely the fill value is held as [`None`], which has a size of zero.
/// - The chunk cache sets what is held per chunk to serve a subchunk miss:
///   - a [`ChunkCachePartialDecoderLruChunkLimit`](super::ChunkCachePartialDecoderLruChunkLimit) caches shard indexes,
///     so a miss reads and decodes only the subchunk. This is the recommended chunk cache.
///   - a [`ChunkCacheEncodedLruChunkLimit`](super::ChunkCacheEncodedLruChunkLimit) caches encoded chunks, so misses
///     decode subchunks without reading storage after the first read of a chunk. Each read with misses decodes the
///     shard index again, and the cached subchunks duplicate data held in the encoded chunks.
///   - [`NoChunkCache`] holds nothing and reads through the array, so each read re-reads a shard index for its misses.
///   - a decoded chunk cache holds whole decoded chunks, so a decoded subchunk cache over it is redundant.
///
/// `SubchunkCached` is itself a cache, so it can be the chunk cache of another `SubchunkCached`.
///
/// For asynchronous operations, use an asynchronous decoded chunk cache as the subchunk cache (e.g.
/// `AsyncChunkCacheDecodedLruChunkLimit`) over an asynchronous chunk cache (e.g.
/// `AsyncChunkCachePartialDecoderLruChunkLimit`, or [`NoChunkCache`]).
///
/// The array must have a globally resolvable subchunk grid at `level` (see
/// [`ArrayOps::subchunk_grid_at_level`](crate::array::ArrayOps::subchunk_grid_at_level)), or cached retrievals return
/// [`ArrayError::MissingSubchunkGrid`]. A retrieval of a region beyond the array shape, such as a whole edge chunk,
/// cannot be addressed by subchunks and is served by the chunk cache.
///
/// Writes through [`ArrayCached`](crate::array::ArrayCached) invalidate the cached subchunks of each affected chunk,
/// and what the chunk cache holds for those chunks. Partial writes may re-encode an entire chunk, and lossy codecs
/// can change values outside the requested region. After writes that bypass it, invalidate through
/// [`ArrayCached::invalidate_chunk`](crate::array::ArrayCached::invalidate_chunk) and related methods.
///
/// A partial decoder opened through the cache before its chunk is invalidated stops reading and refilling the
/// subchunk cache, and decodes through the partial decoder it opened from the chunk cache. As for any cache of an
/// [`ArrayCached`](crate::array::ArrayCached), invalidation does not make concurrent reads and writes of the same
/// chunk safe: a subchunk decoded while a write to its chunk invalidates the cache can be inserted after the
/// invalidation, leaving a pre-write subchunk cached.
///
/// If either cache is thread-local, synchronous reads stay on the calling thread, including nested decoding.
#[derive(Debug)]
pub struct SubchunkCached<S, C> {
    level: usize,
    subchunks: Arc<S>,
    chunks: C,
    generations: Arc<Generations>,
}

impl<S, C> SubchunkCached<S, C> {
    /// Create a cache of the subchunks of the `level` subchunk grid in `subchunks`, over the chunk cache `chunks`.
    #[must_use]
    pub fn new(level: usize, subchunks: S, chunks: C) -> Self {
        Self {
            level,
            subchunks: Arc::new(subchunks),
            chunks,
            generations: Arc::new(Generations::new()),
        }
    }

    /// Return the subchunk grid level of the cached subchunks.
    #[must_use]
    pub const fn level(&self) -> usize {
        self.level
    }

    /// Return the subchunk cache.
    ///
    /// Invalidating it directly does not retire the partial decoders opened through this cache, which can refill it
    /// with chunks as they were when opened. Invalidate through [`ArrayCached`](crate::array::ArrayCached) instead
    /// (e.g. [`ArrayCached::invalidate_chunk`](crate::array::ArrayCached::invalidate_chunk)).
    #[must_use]
    pub fn subchunks(&self) -> &S {
        &self.subchunks
    }

    /// Return the chunk cache that serves subchunk misses.
    ///
    /// See [`SubchunkCached::subchunks`] on invalidation.
    #[must_use]
    pub const fn chunks(&self) -> &C {
        &self.chunks
    }
}

impl<S, C> SubchunkCached<S, C> {
    /// Return the region of the subchunk grid overlapping the `array_subset` of `array`.
    ///
    /// Returns [`None`] if `array_subset` is [`None`] or the overlapping subchunks cannot be determined.
    fn overlapping_subchunks<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        array_subset: Option<&(impl ArraySubsetTraits + ?Sized)>,
    ) -> Option<ArraySubset> {
        let array_subset = array_subset?
            .overlap(&ArraySubset::new_with_shape(array.shape()))
            .ok()?;
        subchunk_grid_at_level(array, self.level)
            .ok()?
            .chunks_in_array_subset(&array_subset)
            .ok()?
    }
}

impl<S: ChunkCache, C> SubchunkCached<S, C> {
    /// Invalidate every cached subchunk.
    ///
    /// Call this after invalidating the chunk cache, so that a subchunk miss cannot refill the subchunk cache from a
    /// chunk the chunk cache still holds. The generations advance before the subchunks are invalidated, so that a
    /// partial decoder opened in an older generation cannot refill them afterwards.
    fn invalidate_all_subchunks(&self) {
        self.generations.advance_all();
        self.subchunks.invalidate();
    }

    /// Invalidate the cached subchunks of the chunk at `chunk_indices` of `array`.
    ///
    /// See [`SubchunkCached::invalidate_all_subchunks`].
    fn invalidate_chunk_subchunks<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
    ) {
        self.generations.advance(chunk_indices);
        self.invalidate_subchunks_in(array, array.chunk_subset(chunk_indices).ok().as_ref());
    }

    /// Invalidate the cached subchunks of the `chunks` of `array`.
    ///
    /// A region of chunks invalidates the subchunks of their region of the subchunk grid at once.
    ///
    /// See [`SubchunkCached::invalidate_all_subchunks`].
    fn invalidate_chunks_subchunks<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        chunks: &dyn Indexer,
    ) {
        self.generations.advance_chunks(chunks);
        if let Some(chunks) = chunks.as_array_subset() {
            self.invalidate_subchunks_in(array, array.chunks_subset(chunks).ok().as_ref());
        } else {
            for chunk_indices in chunks.iter_indices() {
                self.invalidate_subchunks_in(
                    array,
                    array.chunk_subset(&chunk_indices).ok().as_ref(),
                );
            }
        }
    }

    /// Invalidate the cached subchunks overlapping the `array_subset` of `array`, or every cached subchunk if they
    /// cannot be determined.
    fn invalidate_subchunks_in<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        array_subset: Option<&ArraySubset>,
    ) {
        if let Some(subchunks) = self.overlapping_subchunks(array, array_subset) {
            self.subchunks.invalidate_chunks(&subchunks);
        } else {
            self.subchunks.invalidate();
        }
    }
}

#[cfg(feature = "async")]
impl<S: AsyncChunkCache, C> SubchunkCached<S, C> {
    /// Asynchronously invalidate every cached subchunk.
    ///
    /// See [`SubchunkCached::invalidate_all_subchunks`].
    async fn async_invalidate_all_subchunks(&self) {
        self.generations.advance_all();
        self.subchunks.invalidate().await;
    }

    /// Asynchronously invalidate the cached subchunks of the chunk at `chunk_indices` of `array`.
    ///
    /// See [`SubchunkCached::invalidate_all_subchunks`].
    async fn async_invalidate_chunk_subchunks<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
    ) {
        self.generations.advance(chunk_indices);
        self.async_invalidate_subchunks_in(array, array.chunk_subset(chunk_indices).ok().as_ref())
            .await;
    }

    /// Asynchronously invalidate the cached subchunks of the `chunks` of `array`.
    ///
    /// See [`SubchunkCached::invalidate_chunks_subchunks`].
    async fn async_invalidate_chunks_subchunks<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        chunks: &dyn Indexer,
    ) {
        self.generations.advance_chunks(chunks);
        if let Some(chunks) = chunks.as_array_subset() {
            self.async_invalidate_subchunks_in(array, array.chunks_subset(chunks).ok().as_ref())
                .await;
        } else {
            for chunk_indices in chunks.iter_indices() {
                self.async_invalidate_subchunks_in(
                    array,
                    array.chunk_subset(&chunk_indices).ok().as_ref(),
                )
                .await;
            }
        }
    }

    /// Asynchronously invalidate the cached subchunks overlapping the `array_subset` of `array`, or every cached
    /// subchunk if they cannot be determined.
    async fn async_invalidate_subchunks_in<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        array_subset: Option<&ArraySubset>,
    ) {
        if let Some(subchunks) = self.overlapping_subchunks(array, array_subset) {
            self.subchunks.invalidate_chunks(&subchunks).await;
        } else {
            self.subchunks.invalidate().await;
        }
    }
}

/// The number of generations shared by the chunks of a [`SubchunkCached`].
const GENERATIONS: usize = 64;

/// The generations of the chunks of a [`SubchunkCached`], which advance as chunks are invalidated.
///
/// A partial decoder opened in an older generation of its chunk decodes its own snapshot of the chunk, rather than
/// reading or refilling the shared subchunk cache. Chunks share a fixed number of generations by a hash of their
/// indices, so invalidating a chunk also retires the partial decoders of the few chunks that share its generation.
#[derive(Debug)]
struct Generations([AtomicU64; GENERATIONS]);

impl Generations {
    fn new() -> Self {
        Self(std::array::from_fn(|_| AtomicU64::new(0)))
    }

    /// Return the index of the generation of the chunk at `chunk_indices`.
    fn index(chunk_indices: &[u64]) -> usize {
        let hash = chunk_indices.iter().fold(0u64, |hash, &index| {
            (hash.rotate_left(5) ^ index).wrapping_mul(0x517c_c1b7_2722_0a95)
        });
        // The high bits of a multiplicative hash are the best mixed
        #[allow(clippy::cast_possible_truncation)]
        let index = (hash >> (u64::BITS - GENERATIONS.ilog2())) as usize;
        index
    }

    /// Return the current generation of the chunk at `chunk_indices`.
    fn snapshot(self: &Arc<Self>, chunk_indices: &[u64]) -> Generation {
        let index = Self::index(chunk_indices);
        Generation {
            generations: self.clone(),
            index,
            value: self.0[index].load(Ordering::Acquire),
        }
    }

    /// Advance the generation of the chunk at `chunk_indices`.
    fn advance(&self, chunk_indices: &[u64]) {
        self.0[Self::index(chunk_indices)].fetch_add(1, Ordering::AcqRel);
    }

    /// Advance the generations of the `chunks`.
    fn advance_chunks(&self, chunks: &dyn Indexer) {
        if chunks.len() >= GENERATIONS as u64 {
            self.advance_all();
        } else {
            for chunk_indices in chunks.iter_indices() {
                self.advance(&chunk_indices);
            }
        }
    }

    /// Advance the generations of every chunk.
    fn advance_all(&self) {
        for generation in &self.0 {
            generation.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// A generation of a chunk of a [`SubchunkCached`].
struct Generation {
    generations: Arc<Generations>,
    index: usize,
    value: u64,
}

impl Generation {
    /// Return whether the chunk is still in this generation.
    fn is_current(&self) -> bool {
        self.generations.0[self.index].load(Ordering::Acquire) == self.value
    }
}

/// A chunk cache that holds nothing, so a [`SubchunkCached`] serves its subchunk misses from the array.
///
/// This holds no memory, but each retrieval with a subchunk miss opens the chunk again, e.g. re-reading a shard index.
/// It is harmless, though pointless, as the cache of an [`ArrayCached`](crate::array::ArrayCached) on its own.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoChunkCache;

impl SyncArrayCacheOps for NoChunkCache {
    fn cached_partial_decoder<TStorage>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
    {
        array.partial_decoder_with_options(chunk_indices, options, resources)
    }

    fn cached_chunk_bytes_if_exists<TStorage>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
    {
        Ok(array
            .retrieve_chunk_if_exists_with_options(chunk_indices, options, resources)?
            .map(Arc::new))
    }

    fn cached_partial_chunk_bytes<TStorage>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
    {
        Ok(Arc::new(array.retrieve_partial_chunk_with_options(
            chunk_indices,
            indexer,
            options,
            resources,
        )?))
    }

    fn invalidate_cached_all<TStorage: ?Sized>(&self, _array: &Array<TStorage>) {}

    fn invalidate_cached_chunk<TStorage: ?Sized>(
        &self,
        _array: &Array<TStorage>,
        _chunk_indices: &[u64],
    ) {
    }
}

#[cfg(feature = "async")]
impl AsyncArrayCacheOps for NoChunkCache {
    async fn async_cached_partial_decoder<TStorage>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    {
        array
            .async_partial_decoder_with_options(chunk_indices, options, resources)
            .await
    }

    async fn async_cached_chunk_bytes_if_exists<TStorage>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    {
        Ok(array
            .async_retrieve_chunk_if_exists_with_options(chunk_indices, options, resources)
            .await?
            .map(Arc::new))
    }

    async fn async_cached_partial_chunk_bytes<TStorage>(
        &self,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    {
        Ok(Arc::new(
            array
                .async_retrieve_partial_chunk_with_options(
                    chunk_indices,
                    indexer,
                    options,
                    resources,
                )
                .await?,
        ))
    }

    async fn async_invalidate_cached_all<TStorage: ?Sized + MaybeSend + MaybeSync>(
        &self,
        _array: &Array<TStorage>,
    ) {
    }

    async fn async_invalidate_cached_chunk<TStorage: ?Sized + MaybeSend + MaybeSync>(
        &self,
        _array: &Array<TStorage>,
        _chunk_indices: &[u64],
    ) {
    }
}

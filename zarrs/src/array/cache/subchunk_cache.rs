//! Subchunk caches layered over chunk caches.

use std::sync::Arc;

use ambisync::ambisync;

#[cfg(feature = "async")]
use super::array_cache::array_cache_sealed::AsyncArrayCacheOps;
use super::array_cache::array_cache_sealed::SyncArrayCacheOps;
#[cfg(feature = "async")]
use super::{AsyncCachedArray, AsyncChunkCacheTraits};
use super::{CachedArrayMeta, SyncCachedArray, SyncChunkCacheTraits};
use crate::array::{ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, CodecOptions, Indexer};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{Resources, SyncArrayPartialDecoderTraits};

mod decoded;

/// A subchunk cache layered over a chunk cache.
///
/// `SubchunkCache` caches the decoded subchunks of the `level` subchunk grid of an array in the subchunk cache `S`,
/// and serves the subchunks that `S` does not hold through the chunk cache `C`. Use it with
/// [`ArrayCached`](crate::array::ArrayCached) like any other cache.
///
/// The two caches are chosen independently:
/// - The subchunk cache is a decoded chunk cache keyed by subchunk indices rather than chunk indices, e.g. a
///   [`ChunkCacheDecoded`](super::ChunkCacheDecoded). It sets how many subchunks are held.
///   A subchunk that is missing or entirely the fill value is held as [`None`], which has a size of zero.
/// - The chunk cache sets what is held per chunk to serve a subchunk miss:
///   - a [`ChunkCachePartialDecoder`](super::ChunkCachePartialDecoder) caches shard indexes,
///     so a miss reads and decodes only the subchunk. This is the recommended chunk cache.
///   - a [`ChunkCacheEncoded`](super::ChunkCacheEncoded) caches encoded chunks, so misses
///     decode subchunks without reading storage after the first read of a chunk. Each read with misses decodes the
///     shard index again, and the cached subchunks duplicate data held in the encoded chunks.
///   - [`NoCache`] holds nothing and reads through the array, so each read re-reads a shard index for its misses.
///   - a decoded chunk cache holds whole decoded chunks, so a decoded subchunk cache over it is redundant.
///
/// `SubchunkCache` is itself a cache, so it can be the chunk cache of another `SubchunkCache`. It owns its subchunk
/// cache, so each level has its own and no subchunk cache is shared between arrays.
///
/// For asynchronous operations, use an asynchronous decoded chunk cache as the subchunk cache (e.g.
/// `AsyncChunkCacheDecoded`) over an asynchronous chunk cache (e.g.
/// `AsyncChunkCachePartialDecoder`, or [`NoCache`]).
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
/// As for any cache of an [`ArrayCached`](crate::array::ArrayCached), reads of a chunk concurrent with a write to it
/// are not supported, and may leave pre-write subchunks cached. This includes using a partial decoder opened through
/// the cache before a write to its chunk.
#[derive(Debug)]
pub struct SubchunkCache<S: ?Sized, C> {
    level: usize,
    subchunks: Arc<S>,
    chunks: C,
}

impl<S, C> SubchunkCache<S, C> {
    /// Create a cache of the subchunks of the `level` subchunk grid in `subchunks`, over the chunk cache `chunks`.
    #[must_use]
    pub fn new(level: usize, subchunks: S, chunks: C) -> Self {
        Self::from_arc(level, Arc::new(subchunks), chunks)
    }
}

impl<S: ?Sized, C> SubchunkCache<S, C> {
    /// Create a cache of the subchunks of the `level` subchunk grid in the boxed `subchunks`, such as a
    /// `Box<dyn SyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>>`, over the chunk cache `chunks`.
    #[must_use]
    pub fn new_boxed(level: usize, subchunks: Box<S>, chunks: C) -> Self {
        Self::from_arc(level, Arc::from(subchunks), chunks)
    }

    fn from_arc(level: usize, subchunks: Arc<S>, chunks: C) -> Self {
        Self {
            level,
            subchunks,
            chunks,
        }
    }

    /// Return the subchunk grid level of the cached subchunks.
    #[must_use]
    pub const fn level(&self) -> usize {
        self.level
    }

    /// Return the subchunk cache.
    ///
    /// Invalidating it directly leaves the chunk cache holding the chunks it was filled from. Invalidate through
    /// [`ArrayCached`](crate::array::ArrayCached) instead (e.g.
    /// [`ArrayCached::invalidate_chunk`](crate::array::ArrayCached::invalidate_chunk)).
    #[must_use]
    pub fn subchunks(&self) -> &S {
        &self.subchunks
    }

    /// Return the chunk cache that serves subchunk misses.
    ///
    /// See [`SubchunkCache::subchunks`] on invalidation.
    #[must_use]
    pub const fn chunks(&self) -> &C {
        &self.chunks
    }

    /// Return the region of the subchunk grid overlapping the `array_subset` of `array`.
    ///
    /// Returns [`None`] if `array_subset` is [`None`] or the overlapping subchunks cannot be determined.
    fn overlapping_subchunks(
        &self,
        array: &(impl CachedArrayMeta + ?Sized),
        array_subset: Option<&(impl ArraySubsetTraits + ?Sized)>,
    ) -> Option<ArraySubset> {
        let array_subset = array_subset?
            .overlap(&ArraySubset::new_with_shape(array.shape()))
            .ok()?;
        array
            .subchunk_grid_at_level(self.level)
            .ok()?
            .chunks_in_array_subset(&array_subset)
            .ok()?
    }
}

#[ambisync(
    sync(fns("async_{}"), types(AsyncChunkCacheTraits => SyncChunkCacheTraits)),
    async(feature = "async")
)]
impl<S: AsyncChunkCacheTraits + ?Sized, C> SubchunkCache<S, C> {
    /// Invalidate every cached subchunk.
    async fn async_invalidate_all_subchunks(&self) {
        self.subchunks.invalidate().await;
    }

    /// Invalidate the cached subchunks of the chunk at `chunk_indices` of `array`.
    async fn async_invalidate_chunk_subchunks(
        &self,
        array: &dyn CachedArrayMeta,
        chunk_indices: &[u64],
    ) {
        self.async_invalidate_subchunks_in(array, array.chunk_subset(chunk_indices).ok().as_ref())
            .await;
    }

    /// Invalidate the cached subchunks of the `chunks` of `array`.
    ///
    /// A region of chunks invalidates the subchunks of their region of the subchunk grid at once.
    async fn async_invalidate_chunks_subchunks(
        &self,
        array: &dyn CachedArrayMeta,
        chunks: &dyn Indexer,
    ) {
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

    /// Invalidate the cached subchunks overlapping the `array_subset` of `array`, or every cached subchunk if they
    /// cannot be determined.
    async fn async_invalidate_subchunks_in(
        &self,
        array: &dyn CachedArrayMeta,
        array_subset: Option<&ArraySubset>,
    ) {
        if let Some(subchunks) = self.overlapping_subchunks(array, array_subset) {
            self.subchunks.invalidate_chunks(&subchunks).await;
        } else {
            self.subchunks.invalidate().await;
        }
    }
}

/// A cache that holds nothing, so a [`SubchunkCache`] serves its subchunk misses from the array.
///
/// This holds no memory, but each retrieval with a subchunk miss opens the chunk again, e.g. re-reading a shard index.
/// It is harmless, though pointless, as the cache of an [`ArrayCached`](crate::array::ArrayCached) on its own.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCache;

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncArrayCacheOps => SyncArrayCacheOps,
            AsyncCachedArray => SyncCachedArray,
            AsyncArrayPartialDecoderTraits => SyncArrayPartialDecoderTraits,
        )
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32"))
    )
)]
impl AsyncArrayCacheOps for NoCache {
    async fn async_cached_partial_decoder(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        array
            .async_partial_decoder_with_options(chunk_indices, options, resources)
            .await
    }

    async fn async_cached_chunk_bytes_if_exists(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
        Ok(array
            .async_retrieve_chunk_if_exists_with_options(chunk_indices, options, resources)
            .await?
            .map(Arc::new))
    }

    async fn async_cached_partial_chunk_bytes(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
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

    async fn async_invalidate_cached_all(&self, _array: &dyn CachedArrayMeta) {}

    async fn async_invalidate_cached_chunk(
        &self,
        _array: &dyn CachedArrayMeta,
        _chunk_indices: &[u64],
    ) {
    }
}

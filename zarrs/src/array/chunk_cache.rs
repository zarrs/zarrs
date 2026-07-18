//! Chunk caching.
//!
//! `zarrs` supports three types of chunk caches:
//! - [`ChunkCacheTypeDecoded`]: caches decoded chunks.
//!   - Preferred where decoding is expensive and memory is abundant.
//! - [`ChunkCacheTypeEncoded`]: caches encoded chunks.
//!   - Preferred where decoding is cheap and memory is scarce, provided that data is well compressed/sparse.
//! - [`ChunkCacheTypePartialDecoder`]: caches partial decoders.
//!   - Preferred where chunks are repeatedly *partially retrieved*.
//!   - Useful for retrieval of subchunks from sharded arrays, as the partial decoder caches shard indexes (but **not** subchunks).
//!   - Memory usage of this cache is highly dependent on the array codecs and whether the codec chain ([`Array::codecs`]) ends up decoding entire chunks or caching inputs based on their [`PartialDecoderCapability`](zarrs_codec::PartialDecoderCapability).
//!   - With the `async` feature, `ChunkCacheTypeAsyncPartialDecoder` is the asynchronous counterpart.
//!
//! `zarrs` implements the following Least Recently Used (LRU) chunk caches:
//!  - [`ChunkCacheDecodedLruChunkLimit`]: a decoded chunk cache with a fixed chunk capacity..
//!  - [`ChunkCacheDecodedLruSizeLimit`]: a decoded chunk cache with a fixed size in bytes.
//!  - [`ChunkCacheEncodedLruChunkLimit`]: an encoded chunk cache with a fixed chunk capacity.
//!  - [`ChunkCacheEncodedLruSizeLimit`]: an encoded chunk cache with a fixed size in bytes.
//!  - [`ChunkCachePartialDecoderLruChunkLimit`]: a partial decoder chunk cache with a fixed chunk capacity
//!  - [`ChunkCachePartialDecoderLruSizeLimit`]: a partial decoder chunk cache with a fixed size in bytes.
//!
//! There are also `ThreadLocal` suffixed variants of all of these caches that have a per-thread cache,
//! and (with the `async` feature) `ChunkCacheAsyncPartialDecoder` prefixed variants of the partial decoder caches.
//! `zarrs` consumers can create custom cache policies by implementing the [`ChunkCache`] trait.
//! Use a cache with [`ArrayCached`](super::ArrayCached) to perform cached array operations.
//!
//! With the `async` feature, [`ArrayCached`](super::ArrayCached) also supports asynchronous array
//! operations with caches holding [`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], or
//! `ChunkCacheTypeAsyncPartialDecoder` values (see `AsyncChunkCacheType`).
//! [`ChunkCacheTypePartialDecoder`] caches only support synchronous retrieval and
//! `ChunkCacheTypeAsyncPartialDecoder` caches only support asynchronous retrieval (see [`SyncChunkCacheType`]).
//!
//! Chunk caching is likely to be effective for remote stores where redundant retrievals are costly.
//! Chunk caching may not outperform disk caching with a filesystem store.
//! The above caches use internal locking to support multithreading, which has a performance overhead.
//! **Prefer not to use a chunk cache if chunks are not accessed repeatedly**.
//! Aside from [`ChunkCacheTypePartialDecoder`]-based caches, caches do not use partial decoders and any intersected chunk is fully retrieved if not present in the cache.
//!
//! For many access patterns, chunk caching may reduce performance.
//! **Benchmark your algorithm/data.**

use std::sync::Arc;

use super::{ArrayBytes, ArrayBytesRaw, ArrayError};
use crate::array::{Array, ArraySubsetTraits, Indexer};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{ArrayPartialDecoderTraits, CodecOptions};

#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{MaybeSend, MaybeSync, ReadableStorageTraits};

mod chunk_cache_lru;
mod chunk_cache_type;
// pub(crate) mod chunk_cache_lru_macros;
pub use chunk_cache_lru::*;
#[cfg(feature = "async")]
pub(crate) use chunk_cache_type::async_retrieve_chunk_bytes;
pub(crate) use chunk_cache_type::{fill_value_bytes, retrieve_chunk_bytes};

/// The chunk type of an encoded chunk cache.
pub type ChunkCacheTypeEncoded = Option<Arc<ArrayBytesRaw<'static>>>;

/// The chunk type of a decoded chunk cache.
pub type ChunkCacheTypeDecoded = Option<Arc<ArrayBytes<'static>>>;

/// The chunk type of a partial decoder chunk cache.
pub type ChunkCacheTypePartialDecoder = Arc<dyn ArrayPartialDecoderTraits>;

/// The chunk type of an asynchronous partial decoder chunk cache.
#[cfg(feature = "async")]
pub type ChunkCacheTypeAsyncPartialDecoder = Arc<dyn AsyncArrayPartialDecoderTraits>;

/// A chunk cache type ([`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], [`ChunkCacheTypePartialDecoder`], or `ChunkCacheTypeAsyncPartialDecoder`).
///
/// Retrieval is provided by the [`SyncChunkCacheType`] and `AsyncChunkCacheType` subtraits.
pub trait ChunkCacheType:
    chunk_cache_type_sealed::Sealed + MaybeSend + MaybeSync + Clone + 'static
{
    /// The size of the chunk in bytes.
    fn size(&self) -> usize;
}

/// A chunk cache type supporting asynchronous retrieval.
///
/// This is implemented for [`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], and
/// [`ChunkCacheTypeAsyncPartialDecoder`].
/// It is not implemented for [`ChunkCacheTypePartialDecoder`], which caches synchronous
/// partial decoders that cannot be created from asynchronous storage.
#[ambisync::ambisync(
    sync(
        declaration {
            /// A chunk cache type supporting synchronous retrieval.
            ///
            /// This is implemented for [`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], and
            /// [`ChunkCacheTypePartialDecoder`].
            /// It is not implemented for `ChunkCacheTypeAsyncPartialDecoder`, which caches asynchronous
            /// partial decoders that cannot operate over synchronous storage.
            pub trait SyncChunkCacheType: ChunkCacheType {}
        },
        fns("async_{}"),
        types(
            AsyncReadableStorageTraits => ReadableStorageTraits,
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
#[allow(async_fn_in_trait)]
pub trait AsyncChunkCacheType: ChunkCacheType {
    #[doc(hidden)]
    async fn async_partial_decoder<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized;

    #[doc(hidden)]
    async fn async_retrieve_chunk_bytes_if_exists<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized;

    #[doc(hidden)]
    async fn async_retrieve_chunk_subset_bytes<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: ChunkCache<Value = Self> + ?Sized;
}

mod chunk_cache_type_sealed {
    #[cfg(feature = "async")]
    use super::ChunkCacheTypeAsyncPartialDecoder;
    use super::{ChunkCacheTypeDecoded, ChunkCacheTypeEncoded, ChunkCacheTypePartialDecoder};

    pub trait Sealed {}

    impl Sealed for ChunkCacheTypeEncoded {}
    impl Sealed for ChunkCacheTypeDecoded {}
    impl Sealed for ChunkCacheTypePartialDecoder {}
    #[cfg(feature = "async")]
    impl Sealed for ChunkCacheTypeAsyncPartialDecoder {}
}

/// A chunk cache.
///
/// A chunk cache stores values by chunk indices. It is intentionally unaware of
/// arrays; [`ArrayCached`](super::ArrayCached) is the entry point for cached
/// array operations.
pub trait ChunkCache: MaybeSend + MaybeSync {
    /// The value stored for each chunk.
    type Value: ChunkCacheType;

    /// Return the cached value for a chunk without inserting, if it is cached.
    ///
    /// For a thread-local cache, queries only the current thread's cache.
    #[must_use]
    fn get(&self, chunk_indices: &[u64]) -> Option<Self::Value>;

    /// Return a cached value or insert the value returned by `f`.
    #[doc(hidden)]
    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<Self::Value, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<Self::Value, ArrayError>;

    /// Invalidate all cached chunks, returning the number of chunks invalidated.
    ///
    /// For a thread-local cache, clears only the current thread's cache.
    fn invalidate(&self) -> usize;

    /// Invalidate a cached chunk, returning true if the chunk was cached.
    ///
    fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool;

    /// Invalidate cached chunks, returning the number of chunks invalidated.
    ///
    fn invalidate_chunks(&self, chunks: &dyn Indexer) -> usize {
        let mut invalidated = 0;
        for chunk_indices in chunks.iter_indices() {
            invalidated += usize::from(self.invalidate_chunk(&chunk_indices));
        }
        invalidated
    }

    /// Return the number of chunks in the cache. For a thread-local cache, returns the number of chunks cached on the current thread.
    #[must_use]
    fn len(&self) -> usize;

    /// Returns true if the cache is empty. For a thread-local cache, returns if the cache is empty on the current thread.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

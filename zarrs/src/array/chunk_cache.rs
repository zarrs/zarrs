//! Chunk caching.
//!
//! `zarrs` supports the following types of chunk caches:
//! - [`ChunkCacheTypeDecoded`]: caches decoded chunks.
//!   - Preferred where decoding is expensive and memory is abundant.
//! - [`ChunkCacheTypeEncoded`]: caches encoded chunks.
//!   - Preferred where decoding is cheap and memory is scarce, provided that data is well compressed/sparse.
//! - [Async][`ChunkCacheTypePartialDecoder`]: caches partial decoders.
//!   - Preferred where chunks are repeatedly *partially retrieved*.
//!   - Useful for retrieval of subchunks from sharded arrays, as the partial decoder caches shard indexes (but **not** subchunks).
//!   - Memory usage of this cache is highly dependent on the array codecs and whether the codec chain ([`Array::codecs`]) ends up decoding entire chunks or caching inputs based on their [`PartialDecoderCapability`](zarrs_codec::PartialDecoderCapability).
//!
//! `zarrs` implements the following Least Recently Used (LRU) chunk caches:
//!  - [`ChunkCacheDecodedLruChunkLimit`]: a decoded chunk cache with a fixed chunk capacity..
//!  - [`ChunkCacheDecodedLruSizeLimit`]: a decoded chunk cache with a fixed size in bytes.
//!  - [`ChunkCacheEncodedLruChunkLimit`]: an encoded chunk cache with a fixed chunk capacity.
//!  - [`ChunkCacheEncodedLruSizeLimit`]: an encoded chunk cache with a fixed size in bytes.
//!  - [`ChunkCachePartialDecoderLruChunkLimit`]: a partial decoder chunk cache with a fixed chunk capacity
//!  - [`ChunkCachePartialDecoderLruSizeLimit`]: a partial decoder chunk cache with a fixed size in bytes.
//!
//! There are also `ThreadLocal` suffixed variants of all of these caches that have a per-thread cache.
//! `zarrs` consumers can create custom cache policies by implementing the [`ChunkCache`] trait.
//! Use a cache with [`ArrayCached`](super::ArrayCached) to perform cached array operations.
//!
//! With the `async` feature, [`ArrayCached`](super::ArrayCached) also supports asynchronous array
//! operations, which need an `AsyncChunkCache` rather than a [`ChunkCache`].
//! The `AsyncChunkCacheLru{ChunkLimit,SizeLimit}` caches are the asynchronous counterparts of the
//! caches above, with an `AsyncChunkCache{Encoded,Decoded,PartialDecoder}Lru{ChunkLimit,SizeLimit}`
//! alias per chunk type.
//!
//! All of the above caches coalesce concurrent retrievals of an uncached chunk where the backing
//! cache implementation supports it, so that a chunk is usually fetched once no matter how many
//! callers request it. This is best effort: the `wasm32` asynchronous caches do not coalesce, and
//! each caller awaits its own retrieval.
//!
//! Chunk caching is likely to be effective for remote stores where redundant retrievals are costly.
//! Chunk caching may not outperform disk caching with a filesystem store.
//! The above caches use internal locking to support multithreading, which has a performance overhead.
//! **Prefer not to use a chunk cache if chunks are not accessed repeatedly**.
//! Aside from [`ChunkCacheTypePartialDecoder`]-based caches, caches do not use partial decoders and any intersected chunk is fully retrieved if not present in the cache.
//!
//! For many access patterns, chunk caching may reduce performance.
//! **Benchmark your algorithm/data.**

#[cfg(feature = "async")]
use std::future::Future;
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

/// A chunk cache type supporting synchronous retrieval.
///
/// This is implemented for [`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], and
/// [`ChunkCacheTypePartialDecoder`].
/// It is not implemented for `ChunkCacheTypeAsyncPartialDecoder`, which caches asynchronous
/// partial decoders that cannot operate over synchronous storage.
///
/// The retrieval operations themselves live on a sealed supertrait, and are internal to `zarrs`.
pub trait SyncChunkCacheType: ChunkCacheType + chunk_cache_type_sealed::SealedSync {}

impl<T: ChunkCacheType + chunk_cache_type_sealed::SealedSync> SyncChunkCacheType for T {}

/// A chunk cache type supporting asynchronous retrieval.
///
/// This is implemented for [`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], and
/// [`ChunkCacheTypeAsyncPartialDecoder`].
/// It is not implemented for [`ChunkCacheTypePartialDecoder`], which caches synchronous
/// partial decoders that cannot be created from asynchronous storage.
///
/// The retrieval operations themselves live on a sealed supertrait, and are internal to `zarrs`.
#[cfg(feature = "async")]
pub trait AsyncChunkCacheType: ChunkCacheType + chunk_cache_type_sealed::SealedAsync {}

#[cfg(feature = "async")]
impl<T: ChunkCacheType + chunk_cache_type_sealed::SealedAsync> AsyncChunkCacheType for T {}

mod chunk_cache_type_sealed {
    use std::sync::Arc;

    #[cfg(feature = "async")]
    use super::{
        AsyncArrayPartialDecoderTraits, AsyncChunkCache, AsyncReadableStorageTraits,
        ChunkCacheTypeAsyncPartialDecoder,
    };
    use super::{
        Array, ArrayBytes, ArrayError, ArrayPartialDecoderTraits, ArraySubsetTraits, ChunkCache,
        ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded, ChunkCacheTypePartialDecoder,
        CodecOptions, ReadableStorageTraits,
    };

    pub trait Sealed {}

    impl Sealed for ChunkCacheTypeEncoded {}
    impl Sealed for ChunkCacheTypeDecoded {}
    impl Sealed for ChunkCacheTypePartialDecoder {}
    #[cfg(feature = "async")]
    impl Sealed for ChunkCacheTypeAsyncPartialDecoder {}

    /// The synchronous retrieval operations of [`SyncChunkCacheType`](super::SyncChunkCacheType).
    ///
    /// This trait cannot be named outside of `zarrs`, which keeps its operations out of the public
    /// API while [`SyncChunkCacheType`](super::SyncChunkCacheType) remains usable as a public bound.
    pub trait SealedSync: ChunkCacheType {
        fn partial_decoder<TStorage, C>(
            cache: &C,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
        ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
            C: ChunkCache<Value = Self> + ?Sized;

        fn retrieve_chunk_bytes_if_exists<TStorage, C>(
            cache: &C,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
            C: ChunkCache<Value = Self> + ?Sized;

        fn retrieve_chunk_subset_bytes<TStorage, C>(
            cache: &C,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            chunk_subset: &dyn ArraySubsetTraits,
            options: &CodecOptions,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
            C: ChunkCache<Value = Self> + ?Sized;
    }

    /// The asynchronous retrieval operations of [`AsyncChunkCacheType`](super::AsyncChunkCacheType).
    ///
    /// This trait cannot be named outside of `zarrs`, which keeps its operations out of the public
    /// API while [`AsyncChunkCacheType`](super::AsyncChunkCacheType) remains usable as a public
    /// bound.
    #[cfg(feature = "async")]
    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    pub trait SealedAsync: ChunkCacheType {
        async fn async_partial_decoder<TStorage, C>(
            cache: &C,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
        ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
            C: AsyncChunkCache<Value = Self> + ?Sized;

        async fn async_retrieve_chunk_bytes_if_exists<TStorage, C>(
            cache: &C,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
            C: AsyncChunkCache<Value = Self> + ?Sized;

        async fn async_retrieve_chunk_subset_bytes<TStorage, C>(
            cache: &C,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            chunk_subset: &dyn ArraySubsetTraits,
            options: &CodecOptions,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
            C: AsyncChunkCache<Value = Self> + ?Sized;
    }
}

#[cfg(feature = "async")]
pub(crate) use chunk_cache_type_sealed::SealedAsync;
pub(crate) use chunk_cache_type_sealed::SealedSync;

/// A chunk cache.
///
/// A chunk cache stores values by chunk indices. It is intentionally unaware of
/// arrays; [`ArrayCached`](super::ArrayCached) is the entry point for cached
/// array operations.
pub trait ChunkCache: MaybeSend + MaybeSync {
    /// The value stored for each chunk.
    ///
    /// The [`SyncChunkCacheType`] bound is what confines this trait to chunk types that support
    /// synchronous retrieval, so an unusable combination such as a
    /// [`ChunkCacheLruChunkLimit`]`<ChunkCacheTypeAsyncPartialDecoder>` is rejected where it is
    /// written rather than at an unrelated retrieval call site.
    type Value: SyncChunkCacheType;

    /// Return a cached value or insert the value returned by `f`.
    ///
    /// Implementations should evaluate `f` at most once per uncached chunk where possible, so
    /// that concurrent retrievals of the same uncached chunk do not each decode it.
    ///
    /// # Errors
    /// Returns the [`ArrayError`] returned by `f`, which may be shared with the other callers
    /// awaiting the same value.
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

/// An asynchronous chunk cache.
///
/// This is the asynchronous counterpart of [`ChunkCache`], and is what
/// [`ArrayCached`](super::ArrayCached) asynchronous operations require. It is a separate trait
/// rather than a [`ChunkCache`] extension because an asynchronous cache is backed by storage
/// whose own operations are asynchronous, and so cannot implement the synchronous [`ChunkCache`]
/// methods.
///
/// A chunk cache stores values by chunk indices. It is intentionally unaware of
/// arrays; [`ArrayCached`](super::ArrayCached) is the entry point for cached
/// array operations.
#[cfg(feature = "async")]
pub trait AsyncChunkCache: MaybeSend + MaybeSync {
    /// The value stored for each chunk.
    ///
    /// The [`AsyncChunkCacheType`] bound is what confines this trait to chunk types that support
    /// asynchronous retrieval, so an unusable combination such as an
    /// [`AsyncChunkCacheLruChunkLimit`]`<`[`ChunkCacheTypePartialDecoder`]`>` is rejected where it
    /// is written rather than at an unrelated retrieval call site.
    type Value: AsyncChunkCacheType;

    /// Return a cached value or insert the value awaited from `init`.
    ///
    /// `init` is awaited only if the chunk is not cached. Implementations should await it at most
    /// once per uncached chunk where possible, so that concurrent retrievals of the same uncached
    /// chunk do not each fetch it.
    ///
    /// # Errors
    /// Returns the [`ArrayError`] returned by `init`, which may be shared with the other callers
    /// awaiting the same value.
    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        init: F,
    ) -> impl Future<Output = Result<Self::Value, Arc<ArrayError>>> + MaybeSend
    where
        F: Future<Output = Result<Self::Value, ArrayError>> + MaybeSend;

    /// Invalidate all cached chunks, returning the number of chunks invalidated.
    fn invalidate(&self) -> impl Future<Output = usize> + MaybeSend;

    /// Invalidate a cached chunk, returning true if the chunk was cached.
    fn invalidate_chunk(&self, chunk_indices: &[u64]) -> impl Future<Output = bool> + MaybeSend;

    /// Invalidate cached chunks, returning the number of chunks invalidated.
    fn invalidate_chunks(&self, chunks: &dyn Indexer) -> impl Future<Output = usize> + MaybeSend {
        async move {
            let mut invalidated = 0;
            for chunk_indices in chunks.iter_indices() {
                invalidated += usize::from(self.invalidate_chunk(&chunk_indices).await);
            }
            invalidated
        }
    }

    /// Return the number of chunks in the cache.
    fn len(&self) -> impl Future<Output = usize> + MaybeSend;

    /// Returns true if the cache is empty.
    fn is_empty(&self) -> impl Future<Output = bool> + MaybeSend {
        async move { self.len().await == 0 }
    }
}

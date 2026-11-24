#[cfg(feature = "async")]
use std::future::Future;
use std::sync::Arc;

#[cfg(feature = "async")]
use super::{
    AsyncChunkCacheInit, AsyncChunkCacheTraits, AsyncChunkCacheType,
    ChunkCacheTypeAsyncPartialDecoder,
};
use super::{
    ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded, ChunkCacheTypePartialDecoder,
    SyncChunkCacheTraits, SyncChunkCacheType,
};
use crate::array::{ArrayError, ArrayIndices, Indexer};

type ChunkIndices = ArrayIndices;

trait CacheTraits<CT: ChunkCacheType> {
    fn len(&self) -> usize;

    /// Return the sum of the sizes of the cached values.
    fn size(&self) -> usize;

    fn remove(&self, chunk_indices: &[u64]) -> bool;

    /// Remove the entries at the indices of `indexer`, returning the number removed.
    fn remove_many(&self, indexer: &dyn Indexer) -> usize {
        indexer
            .iter_indices()
            .filter(|chunk_indices| self.remove(chunk_indices))
            .count()
    }

    fn clear(&self) -> usize;

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>;
}

trait CacheChunkLimitTraits {
    fn new_with_chunk_capacity(chunk_capacity: u64) -> Self;
}

trait CacheSizeLimitTraits {
    fn new_with_size_capacity(size_capacity: u64) -> Self;
}

#[cfg(feature = "async")]
trait AsyncCacheTraits<CT: ChunkCacheType> {
    fn len(&self) -> impl Future<Output = usize>;

    /// Return the sum of the sizes of the cached values.
    fn size(&self) -> impl Future<Output = usize>;

    fn remove(&self, chunk_indices: &[u64]) -> impl Future<Output = bool>;

    /// Remove the entries at the indices of `indexer`, returning the number removed.
    fn remove_many(&self, indexer: &dyn Indexer) -> impl Future<Output = usize> {
        async move {
            let mut removed = 0;
            for chunk_indices in indexer.iter_indices() {
                removed += usize::from(self.remove(&chunk_indices).await);
            }
            removed
        }
    }

    fn clear(&self) -> impl Future<Output = usize>;

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
        init: F,
    ) -> impl Future<Output = Result<CT, Arc<ArrayError>>>
    where
        F: Future<Output = Result<CT, ArrayError>>;
}

#[cfg(not(target_arch = "wasm32"))]
#[path = "chunk_cache_lru_moka.rs"]
mod platform;

#[cfg(target_arch = "wasm32")]
#[path = "chunk_cache_lru_quick_cache.rs"]
mod platform;

#[cfg(all(feature = "async", not(target_arch = "wasm32")))]
#[path = "chunk_cache_lru_async_moka.rs"]
mod platform_async;

#[cfg(all(feature = "async", target_arch = "wasm32"))]
#[path = "chunk_cache_lru_async_lru.rs"]
mod platform_async;

/// A chunk cache with a fixed chunk capacity.
pub struct ChunkCacheLruChunkLimit<CT: SyncChunkCacheType> {
    cache: platform::CacheChunkLimit<CT>,
}

impl<CT: SyncChunkCacheType> ChunkCacheLruChunkLimit<CT> {
    /// Create a new [`ChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(chunk_capacity: u64) -> Self {
        let cache = platform::CacheChunkLimit::new_with_chunk_capacity(chunk_capacity);
        Self { cache }
    }
}

/// A chunk cache with a fixed size capacity.
pub struct ChunkCacheLruSizeLimit<CT: SyncChunkCacheType> {
    cache: platform::CacheSizeLimit<CT>,
}

impl<CT: SyncChunkCacheType> ChunkCacheLruSizeLimit<CT> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = platform::CacheSizeLimit::new_with_size_capacity(capacity);
        Self { cache }
    }
}

/// The [`SyncChunkCacheTraits`] methods of a chunk cache.
macro_rules! impl_ChunkCacheLruCommon {
    () => {
        type Value = CT;

        fn try_get_or_insert_with(
            &self,
            chunk_indices: ChunkIndices,
            f: &mut dyn FnMut() -> Result<Self::Value, ArrayError>,
        ) -> Result<Self::Value, Arc<ArrayError>> {
            self.cache.try_get_or_insert_with(chunk_indices, f)
        }

        fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
            CacheTraits::remove(&self.cache, chunk_indices)
        }

        fn invalidate_chunks(&self, chunks: &dyn Indexer) -> usize {
            CacheTraits::remove_many(&self.cache, chunks)
        }

        fn len(&self) -> usize {
            self.cache.len()
        }

        fn size(&self) -> usize {
            CacheTraits::size(&self.cache)
        }

        fn invalidate(&self) -> usize {
            CacheTraits::clear(&self.cache)
        }
    };
}

impl<CT: SyncChunkCacheType> SyncChunkCacheTraits for ChunkCacheLruChunkLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: SyncChunkCacheType> SyncChunkCacheTraits for ChunkCacheLruSizeLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

/// An asynchronous chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub struct AsyncChunkCacheLruChunkLimit<CT: AsyncChunkCacheType> {
    cache: platform_async::CacheChunkLimit<CT>,
}

#[cfg(feature = "async")]
impl<CT: AsyncChunkCacheType> AsyncChunkCacheLruChunkLimit<CT> {
    /// Create a new [`AsyncChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(chunk_capacity: u64) -> Self {
        let cache = platform_async::CacheChunkLimit::new_with_chunk_capacity(chunk_capacity);
        Self { cache }
    }
}

/// An asynchronous chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub struct AsyncChunkCacheLruSizeLimit<CT: AsyncChunkCacheType> {
    cache: platform_async::CacheSizeLimit<CT>,
}

#[cfg(feature = "async")]
impl<CT: AsyncChunkCacheType> AsyncChunkCacheLruSizeLimit<CT> {
    /// Create a new [`AsyncChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = platform_async::CacheSizeLimit::new_with_size_capacity(capacity);
        Self { cache }
    }
}

#[cfg(feature = "async")]
/// Implement [`AsyncChunkCacheTraits`] for a chunk cache type.
macro_rules! impl_AsyncChunkCacheLru {
    ($ty:ident) => {
        #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
        #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
        impl<CT: AsyncChunkCacheType> AsyncChunkCacheTraits for $ty<CT> {
            type Value = CT;

            async fn try_get_or_insert_with(
                &self,
                chunk_indices: ChunkIndices,
                init: AsyncChunkCacheInit<'_, Self::Value>,
            ) -> Result<Self::Value, Arc<ArrayError>> {
                self.cache.try_get_or_insert_with(chunk_indices, init).await
            }

            async fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
                AsyncCacheTraits::remove(&self.cache, chunk_indices).await
            }

            async fn invalidate_chunks(&self, chunks: &dyn Indexer) -> usize {
                AsyncCacheTraits::remove_many(&self.cache, chunks).await
            }

            async fn len(&self) -> usize {
                AsyncCacheTraits::len(&self.cache).await
            }

            async fn size(&self) -> usize {
                AsyncCacheTraits::size(&self.cache).await
            }

            async fn invalidate(&self) -> usize {
                AsyncCacheTraits::clear(&self.cache).await
            }
        }
    };
}

#[cfg(feature = "async")]
impl_AsyncChunkCacheLru!(AsyncChunkCacheLruChunkLimit);

#[cfg(feature = "async")]
impl_AsyncChunkCacheLru!(AsyncChunkCacheLruSizeLimit);

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) encoded chunk cache with a fixed size capacity.
pub type ChunkCacheEncodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed size capacity .
pub type ChunkCacheDecodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimit =
    ChunkCacheLruChunkLimit<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed size capacity.
pub type ChunkCachePartialDecoderLruSizeLimit =
    ChunkCacheLruSizeLimit<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) asynchronous encoded chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheEncodedLruChunkLimit = AsyncChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) asynchronous encoded chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheEncodedLruSizeLimit = AsyncChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) asynchronous decoded chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheDecodedLruChunkLimit = AsyncChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) asynchronous decoded chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCacheDecodedLruSizeLimit = AsyncChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) asynchronous partial decoder chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCachePartialDecoderLruChunkLimit =
    AsyncChunkCacheLruChunkLimit<ChunkCacheTypeAsyncPartialDecoder>;

/// An LRU (least recently used) asynchronous partial decoder chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type AsyncChunkCachePartialDecoderLruSizeLimit =
    AsyncChunkCacheLruSizeLimit<ChunkCacheTypeAsyncPartialDecoder>;

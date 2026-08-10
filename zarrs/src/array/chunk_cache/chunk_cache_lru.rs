#[cfg(feature = "async")]
use std::future::Future;
use std::sync::{Arc, atomic};

#[cfg(feature = "async")]
use zarrs_storage::MaybeSend;

#[cfg(feature = "async")]
use super::{AsyncChunkCache, AsyncChunkCacheType, ChunkCacheTypeAsyncPartialDecoder};
use super::{
    ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded,
    ChunkCacheTypePartialDecoder, SyncChunkCacheType,
};
use crate::array::{ArrayError, ArrayIndices};

type ChunkIndices = ArrayIndices;

trait CacheTraits<CT: ChunkCacheType> {
    fn len(&self) -> usize;

    fn remove(&self, chunk_indices: &[u64]) -> bool;

    fn clear(&self) -> usize;

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
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

    fn remove(&self, chunk_indices: &[u64]) -> impl Future<Output = bool>;

    fn clear(&self) -> impl Future<Output = usize>;

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
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

/// A thread local chunk cache with a fixed chunk capacity per thread.
pub struct ChunkCacheLruChunkLimitThreadLocal<CT: SyncChunkCacheType> {
    cache: platform::ThreadLocalCacheChunkLimit<CT>,
}

impl<CT: SyncChunkCacheType> ChunkCacheLruChunkLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruChunkLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = platform::ThreadLocalCacheChunkLimit::new_with_chunk_capacity(capacity);
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

/// A thread local chunk cache with a fixed size capacity per thread.
pub struct ChunkCacheLruSizeLimitThreadLocal<CT: SyncChunkCacheType> {
    cache: platform::ThreadLocalCacheSizeLimit<CT>,
}

impl<CT: SyncChunkCacheType> ChunkCacheLruSizeLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruSizeLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = platform::ThreadLocalCacheSizeLimit::new_with_size_capacity(capacity);
        Self { cache }
    }
}

macro_rules! impl_ChunkCacheLruCommon {
    () => {
        type Value = CT;

        fn try_get_or_insert_with<F>(
            &self,
            chunk_indices: Vec<u64>,
            f: F,
        ) -> Result<Self::Value, Arc<ArrayError>>
        where
            F: FnOnce() -> Result<Self::Value, ArrayError>,
        {
            self.cache.try_get_or_insert_with(chunk_indices, f)
        }

        fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
            CacheTraits::remove(&self.cache, chunk_indices)
        }

        fn len(&self) -> usize {
            self.cache.len()
        }

        fn invalidate(&self) -> usize {
            CacheTraits::clear(&self.cache)
        }
    };
}

impl<CT: SyncChunkCacheType> ChunkCache for ChunkCacheLruChunkLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: SyncChunkCacheType> ChunkCache for ChunkCacheLruChunkLimitThreadLocal<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: SyncChunkCacheType> ChunkCache for ChunkCacheLruSizeLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: SyncChunkCacheType> ChunkCache for ChunkCacheLruSizeLimitThreadLocal<CT> {
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
macro_rules! impl_AsyncChunkCacheLruCommon {
    () => {
        type Value = CT;

        async fn try_get_or_insert_with<F>(
            &self,
            chunk_indices: Vec<u64>,
            init: F,
        ) -> Result<Self::Value, Arc<ArrayError>>
        where
            F: Future<Output = Result<Self::Value, ArrayError>> + MaybeSend,
        {
            self.cache.try_get_or_insert_with(chunk_indices, init).await
        }

        async fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
            AsyncCacheTraits::remove(&self.cache, chunk_indices).await
        }

        async fn len(&self) -> usize {
            AsyncCacheTraits::len(&self.cache).await
        }

        async fn invalidate(&self) -> usize {
            AsyncCacheTraits::clear(&self.cache).await
        }
    };
}

#[cfg(feature = "async")]
impl<CT: AsyncChunkCacheType> AsyncChunkCache for AsyncChunkCacheLruChunkLimit<CT> {
    impl_AsyncChunkCacheLruCommon!();
}

#[cfg(feature = "async")]
impl<CT: AsyncChunkCacheType> AsyncChunkCache for AsyncChunkCacheLruSizeLimit<CT> {
    impl_AsyncChunkCacheLruCommon!();
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) encoded chunk cache with a fixed size capacity.
pub type ChunkCacheEncodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) encoded chunk cache with a fixed size capacity.
pub type ChunkCacheEncodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed size capacity .
pub type ChunkCacheDecodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed size capacity.
pub type ChunkCacheDecodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimit =
    ChunkCacheLruChunkLimit<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed size capacity.
pub type ChunkCachePartialDecoderLruSizeLimit =
    ChunkCacheLruSizeLimit<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypePartialDecoder>;

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

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::{
        AsyncChunkCache, AsyncChunkCacheLruChunkLimit, AsyncChunkCacheLruSizeLimit, ChunkCache,
        ChunkCacheLruChunkLimit, ChunkCacheLruSizeLimit, ChunkCacheTypeDecoded,
    };

    /// The synchronous and asynchronous cache policies implement one trait each, which is what
    /// admits each family to the [`ArrayCached`] operations of the matching flavour.
    ///
    /// The `compile_fail` doctest on [`ArrayCached`] illustrates a synchronous cache being
    /// rejected by an asynchronous operation but cannot guard it, since rustdoc does not check
    /// *why* such a doctest fails to compile.
    ///
    /// The implementations are generic over the chunk type, so pinning one chunk type pins every
    /// public alias of these policies.
    ///
    /// [`ArrayCached`]: crate::array::ArrayCached
    #[test]
    fn lru_cache_policies_implement_their_flavour_of_chunk_cache() {
        fn assert_sync<C: ChunkCache>() {}
        fn assert_async<C: AsyncChunkCache>() {}

        assert_sync::<ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>>();
        assert_sync::<ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>>();
        assert_async::<AsyncChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>>();
        assert_async::<AsyncChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>>();
    }
}

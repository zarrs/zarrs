use std::sync::{Arc, atomic};

#[cfg(feature = "async")]
use super::ChunkCacheTypeAsyncPartialDecoder;
use super::{
    ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded,
    ChunkCacheTypePartialDecoder,
};
use crate::array::{ArrayError, ArrayIndices};

type ChunkIndices = ArrayIndices;

trait CacheTraits<CT: ChunkCacheType> {
    fn len(&self) -> usize;

    fn get_cached(&self, chunk_indices: &[u64]) -> Option<CT>;

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

#[cfg(not(target_arch = "wasm32"))]
#[path = "chunk_cache_lru_moka.rs"]
mod platform;

#[cfg(target_arch = "wasm32")]
#[path = "chunk_cache_lru_quick_cache.rs"]
mod platform;

/// A chunk cache with a fixed chunk capacity.
pub struct ChunkCacheLruChunkLimit<CT: ChunkCacheType> {
    cache: platform::CacheChunkLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimit<CT> {
    /// Create a new [`ChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(chunk_capacity: u64) -> Self {
        let cache = platform::CacheChunkLimit::new_with_chunk_capacity(chunk_capacity);
        Self { cache }
    }
}

/// A thread local chunk cache with a fixed chunk capacity per thread.
pub struct ChunkCacheLruChunkLimitThreadLocal<CT: ChunkCacheType> {
    cache: platform::ThreadLocalCacheChunkLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruChunkLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = platform::ThreadLocalCacheChunkLimit::new_with_chunk_capacity(capacity);
        Self { cache }
    }
}

/// A chunk cache with a fixed size capacity.
pub struct ChunkCacheLruSizeLimit<CT: ChunkCacheType> {
    cache: platform::CacheSizeLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimit<CT> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = platform::CacheSizeLimit::new_with_size_capacity(capacity);
        Self { cache }
    }
}

/// A thread local chunk cache with a fixed size capacity per thread.
pub struct ChunkCacheLruSizeLimitThreadLocal<CT: ChunkCacheType> {
    cache: platform::ThreadLocalCacheSizeLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimitThreadLocal<CT> {
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

        fn get(&self, chunk_indices: &[u64]) -> Option<Self::Value> {
            self.cache.get_cached(chunk_indices)
        }

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

impl<CT: ChunkCacheType> ChunkCache for ChunkCacheLruChunkLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: ChunkCacheType> ChunkCache for ChunkCacheLruChunkLimitThreadLocal<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: ChunkCacheType> ChunkCache for ChunkCacheLruSizeLimit<CT> {
    impl_ChunkCacheLruCommon!();
}

impl<CT: ChunkCacheType> ChunkCache for ChunkCacheLruSizeLimitThreadLocal<CT> {
    impl_ChunkCacheLruCommon!();
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

/// An LRU (least recently used) asynchronous partial decoder chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type ChunkCacheAsyncPartialDecoderLruChunkLimit =
    ChunkCacheLruChunkLimit<ChunkCacheTypeAsyncPartialDecoder>;

/// An LRU (least recently used) asynchronous partial decoder chunk cache with a fixed chunk capacity.
#[cfg(feature = "async")]
pub type ChunkCacheAsyncPartialDecoderLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeAsyncPartialDecoder>;

/// An LRU (least recently used) asynchronous partial decoder chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type ChunkCacheAsyncPartialDecoderLruSizeLimit =
    ChunkCacheLruSizeLimit<ChunkCacheTypeAsyncPartialDecoder>;

/// An LRU (least recently used) asynchronous partial decoder chunk cache with a fixed size capacity.
#[cfg(feature = "async")]
pub type ChunkCacheAsyncPartialDecoderLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeAsyncPartialDecoder>;

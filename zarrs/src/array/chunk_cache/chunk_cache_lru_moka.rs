use std::{
    num::NonZeroUsize,
    sync::{atomic, atomic::AtomicUsize, Arc, Mutex},
};

use lru::LruCache;
use moka::{
    policy::EvictionPolicy,
    sync::{Cache, CacheBuilder},
};
use thread_local::ThreadLocal;
use zarrs_storage::ReadableStorageTraits;

use crate::{
    array::{
        chunk_cache::{ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded},
        codec::ArrayToBytesCodecTraits,
        Array, ArrayBytes, ArrayError, ArrayIndices, ArraySize, ChunkCacheTypePartialDecoder,
    },
    array_subset::ArraySubset,
    impl_ChunkCacheLruDecoded, impl_ChunkCacheLruEncoded, impl_ChunkCacheLruPartialDecoder,
    storage::StorageError,
};

use std::borrow::Cow;

type ChunkIndices = ArrayIndices;

/// A chunk cache with a fixed chunk capacity.
pub struct ChunkCacheLruChunkLimit<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: Cache<ChunkIndices, T>,
}

/// A thread local chunk cache with a fixed chunk capacity per thread.
pub struct ChunkCacheLruChunkLimitThreadLocal<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: ThreadLocal<Mutex<LruCache<ChunkIndices, T>>>,
    capacity: u64,
}

/// A chunk cache with a fixed size capacity.
pub struct ChunkCacheLruSizeLimit<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: Cache<ChunkIndices, T>,
}

/// A thread local chunk cache with a fixed chunk capacity per thread.
pub struct ChunkCacheLruSizeLimitThreadLocal<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: ThreadLocal<Mutex<LruCache<ChunkIndices, T>>>,
    capacity: usize,
    size: ThreadLocal<AtomicUsize>,
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) encoded chunk cache with a fixed size capacity in bytes.
pub type ChunkCacheEncodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed size capacity in bytes.
pub type ChunkCacheDecodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeDecoded>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimit =
    ChunkCacheLruChunkLimit<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed size capacity in bytes.
pub type ChunkCachePartialDecoderLruSizeLimit =
    ChunkCacheLruSizeLimit<ChunkCacheTypePartialDecoder>;

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypePartialDecoder>;

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimit<CT> {
    /// Create a new [`ChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, chunk_capacity: u64) -> Self {
        let cache = CacheBuilder::new(chunk_capacity)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        Self { array, cache }
    }

    // fn get(&self, chunk_indices: &[u64]) -> Option<CT> {
    //     self.cache.get(&chunk_indices.to_vec())
    // }

    // fn insert(&self, chunk_indices: ChunkIndices, chunk: CT) {
    //     self.cache.insert(chunk_indices, chunk);
    // }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        self.cache.try_get_with(chunk_indices, f)
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruChunkLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = ThreadLocal::new();
        Self {
            array,
            cache,
            capacity,
        }
    }

    fn cache(&self) -> &Mutex<LruCache<ChunkIndices, CT>> {
        self.cache.get_or(|| {
            Mutex::new(LruCache::new(
                NonZeroUsize::new(usize::try_from(self.capacity).unwrap_or(usize::MAX).max(1))
                    .unwrap(),
            ))
        })
    }

    // fn get(&self, chunk_indices: &[u64]) -> Option<CT> {
    //     self.cache()
    //         .lock()
    //         .unwrap()
    //         .get(&chunk_indices.to_vec())
    //         .cloned()
    // }

    // fn insert(&self, chunk_indices: ChunkIndices, chunk: CT) {
    //     self.cache().lock().unwrap().push(chunk_indices, chunk);
    // }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        self.cache()
            .lock()
            .unwrap()
            .try_get_or_insert(chunk_indices, f)
            .cloned()
            .map_err(Arc::new)
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimit<CT> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = CacheBuilder::new(capacity)
            .eviction_policy(EvictionPolicy::lru())
            .weigher(|_k, v: &CT| u32::try_from(v.size()).unwrap_or(u32::MAX))
            .build();
        Self { array, cache }
    }

    // fn get(&self, chunk_indices: &[u64]) -> Option<CT> {
    //     self.cache.get(&chunk_indices.to_vec())
    // }

    // fn insert(&self, chunk_indices: ChunkIndices, chunk: CT) {
    //     self.cache.insert(chunk_indices, chunk);
    // }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        self.cache.try_get_with(chunk_indices, f)
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruSizeLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = ThreadLocal::new();
        Self {
            array,
            cache,
            capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
            size: ThreadLocal::new(),
        }
    }

    fn cache(&self) -> &Mutex<LruCache<ChunkIndices, CT>> {
        self.cache.get_or(|| Mutex::new(LruCache::unbounded()))
    }

    fn get(&self, chunk_indices: &[u64]) -> Option<CT> {
        self.cache()
            .lock()
            .unwrap()
            .get(&chunk_indices.to_vec())
            .cloned()
    }

    fn insert(&self, chunk_indices: ChunkIndices, chunk: CT) {
        let size = self.size.get_or_default();
        let size_old = size.fetch_add(chunk.size(), atomic::Ordering::SeqCst);
        if size_old + chunk.size() > self.capacity {
            let old = self.cache().lock().unwrap().pop_lru();
            if let Some(old) = old {
                size.fetch_sub(old.1.size(), atomic::Ordering::SeqCst);
            }
        }

        let old = self.cache().lock().unwrap().push(chunk_indices, chunk);
        if let Some(old) = old {
            size.fetch_sub(old.1.size(), atomic::Ordering::SeqCst);
        }
    }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        if let Some(value) = self.get(&chunk_indices) {
            Ok(value)
        } else {
            let value = f()?;
            self.insert(chunk_indices, value.clone());
            Ok(value)
        }

        // self.cache()
        //     .lock()
        //     .unwrap()
        //     .try_get_or_insert(chunk_indices, f)
        //     .cloned()
        //     .map_err(|e| Arc::new(e))
    }
}

macro_rules! impl_ChunkCacheLruCommon {
    () => {
        fn array(&self) -> Arc<Array<dyn ReadableStorageTraits>> {
            self.array.clone()
        }

        fn len(&self) -> usize {
            self.cache.run_pending_tasks();
            usize::try_from(self.cache.entry_count()).unwrap()
        }
    };
}

impl ChunkCache for ChunkCacheEncodedLruChunkLimit {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruCommon!();
}

impl ChunkCache for ChunkCacheEncodedLruSizeLimit {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruCommon!();
}

impl ChunkCache for ChunkCacheDecodedLruChunkLimit {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruCommon!();
}

impl ChunkCache for ChunkCacheDecodedLruSizeLimit {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruCommon!();
}

impl ChunkCache for ChunkCachePartialDecoderLruChunkLimit {
    impl_ChunkCacheLruPartialDecoder!();
    impl_ChunkCacheLruCommon!();
}

impl ChunkCache for ChunkCachePartialDecoderLruSizeLimit {
    impl_ChunkCacheLruPartialDecoder!();
    impl_ChunkCacheLruCommon!();
}

macro_rules! impl_ChunkCacheLruChunkLimitThreadLocal {
    () => {
        fn array(&self) -> Arc<Array<dyn ReadableStorageTraits>> {
            self.array.clone()
        }

        fn len(&self) -> usize {
            self.cache().lock().unwrap().len()
        }
    };
}

impl ChunkCache for ChunkCacheEncodedLruChunkLimitThreadLocal {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruChunkLimitThreadLocal!();
}

impl ChunkCache for ChunkCacheDecodedLruChunkLimitThreadLocal {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruChunkLimitThreadLocal!();
}

impl ChunkCache for ChunkCachePartialDecoderLruChunkLimitThreadLocal {
    impl_ChunkCacheLruPartialDecoder!();
    impl_ChunkCacheLruChunkLimitThreadLocal!();
}

macro_rules! impl_ChunkCacheLruSizeLimitThreadLocal {
    () => {
        fn array(&self) -> Arc<Array<dyn ReadableStorageTraits>> {
            self.array.clone()
        }

        fn len(&self) -> usize {
            self.cache().lock().unwrap().len()
        }
    };
}

impl ChunkCache for ChunkCacheEncodedLruSizeLimitThreadLocal {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruSizeLimitThreadLocal!();
}

impl ChunkCache for ChunkCacheDecodedLruSizeLimitThreadLocal {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruSizeLimitThreadLocal!();
}

impl ChunkCache for ChunkCachePartialDecoderLruSizeLimitThreadLocal {
    impl_ChunkCacheLruPartialDecoder!();
    impl_ChunkCacheLruSizeLimitThreadLocal!();
}

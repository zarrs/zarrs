use std::{num::NonZeroUsize, sync::Arc};

use lru::LruCache;

use zarrs_storage::ReadableStorageTraits;

//#[cfg(target_arch = "wasm32")]
use quick_cache::sync::Cache;
//#[cfg(target_arch = "wasm32")]
use std::cell::{Cell, RefCell};

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
//#[cfg(target_arch = "wasm32")]
pub struct ChunkCacheLruChunkLimit<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: RefCell<Cache<ChunkIndices, T>>,
}

/// A thread local chunk cache with a fixed chunk capacity per thread.
//#[cfg(target_arch = "wasm32")]
pub struct ChunkCacheLruChunkLimitThreadLocal<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: RefCell<LruCache<ChunkIndices, T>>,
    capacity: u64,
}

/// A chunk cache with a fixed size capacity.
//#[cfg(target_arch = "wasm32")]
pub struct ChunkCacheLruSizeLimit<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: RefCell<Cache<ChunkIndices, T>>,
}

/// A thread local chunk cache with a fixed chunk capacity per thread.
//#[cfg(target_arch = "wasm32")]
pub struct ChunkCacheLruSizeLimitThreadLocal<T: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: RefCell<LruCache<ChunkIndices, T>>,
    capacity: usize,
    size: Cell<usize>,
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
    //#[cfg(target_arch = "wasm32")]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, chunk_capacity: u64) -> Self {
        let cache = Cache::new(usize::try_from(chunk_capacity).unwrap_or(usize::MAX));
        Self {
            array,
            cache: RefCell::new(cache),
        }
    }

    // fn get(&self, chunk_indices: &[u64]) -> Option<CT> {
    //     self.cache.get(&chunk_indices.to_vec())
    // }

    // fn insert(&self, chunk_indices: ChunkIndices, chunk: CT) {
    //     self.cache.insert(chunk_indices, chunk);
    // }

    //#[cfg(target_arch = "wasm32")]
    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        let cache = self.cache.borrow_mut();
        cache
            .get_or_insert_with(&chunk_indices, || f().map_err(Arc::new))
            .map_err(|e| Arc::clone(&e))
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruChunkLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    //#[cfg(target_arch = "wasm32")]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cap =
            NonZeroUsize::new(usize::try_from(capacity).unwrap_or(usize::MAX).max(1)).unwrap();
        let cache = RefCell::new(LruCache::new(cap));
        Self {
            array,
            cache,
            capacity,
        }
    }

    //#[cfg(target_arch = "wasm32")]
    fn cache(&self) -> &RefCell<LruCache<ChunkIndices, CT>> {
        &self.cache
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

    //#[cfg(target_arch = "wasm32")]
    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        let mut cache = self.cache.borrow_mut();
        cache
            .try_get_or_insert(chunk_indices, f)
            .cloned()
            .map_err(Arc::new)
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimit<CT> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    //#[cfg(target_arch = "wasm32")]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = Cache::new(usize::try_from(capacity).unwrap_or(usize::MAX));
        //.weigher(|_k, v: &CT| u32::try_from(v.size()).unwrap_or(u32::MAX))
        //.build();
        Self {
            array,
            cache: RefCell::new(cache),
        }
    }

    // fn get(&self, chunk_indices: &[u64]) -> Option<CT> {
    //     self.cache.get(&chunk_indices.to_vec())
    // }

    // fn insert(&self, chunk_indices: ChunkIndices, chunk: CT) {
    //     self.cache.insert(chunk_indices, chunk);
    // }

    //#[cfg(target_arch = "wasm32")]
    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        let cache = self.cache.borrow_mut();
        cache
            .get_or_insert_with(&chunk_indices, || f().map_err(Arc::new))
            .map_err(|e| Arc::clone(&e))
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruSizeLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    //#[cfg(target_arch = "wasm32")]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = RefCell::new(LruCache::unbounded());
        Self {
            array,
            cache,
            capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
            size: Cell::new(0),
        }
    }

    //#[cfg(target_arch = "wasm32")]
    fn cache(&self) -> &RefCell<LruCache<ChunkIndices, CT>> {
        &self.cache
    }

    //#[cfg(target_arch = "wasm32")]
    fn get(&self, chunk_indices: &[u64]) -> Option<CT> {
        self.cache()
            .borrow_mut()
            .get(&chunk_indices.to_vec())
            .cloned()
    }

    //#[cfg(target_arch = "wasm32")]
    fn insert(&self, chunk_indices: ChunkIndices, chunk: CT) {
        let new_chunk_size = chunk.size();
        let mut cache = self.cache().borrow_mut();
        let mut size = self.size.get();
        if size + new_chunk_size > self.capacity {
            if let Some((_k, old)) = cache.pop_lru() {
                size -= old.size();
            }
        }

        let replaced = cache.push(chunk_indices, chunk);
        if let Some((_k, old)) = replaced {
            size -= old.size();
        }
        size += new_chunk_size;
        self.size.set(size);
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

        //#[cfg(target_arch = "wasm32")]
        fn len(&self) -> usize {
            // self.cache.run_pending_tasks();
            usize::try_from(self.cache.borrow().len()).unwrap()
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

        //#[cfg(target_arch = "wasm32")]
        fn len(&self) -> usize {
            self.cache().borrow().len()
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

        //#[cfg(target_arch = "wasm32")]
        fn len(&self) -> usize {
            self.cache().borrow().len()
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

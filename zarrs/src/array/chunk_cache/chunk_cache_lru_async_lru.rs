//! An asynchronous chunk cache fallback for targets without `moka`.
//!
//! Unlike the `moka::future::Cache` backed caches, these do not coalesce concurrent retrievals of
//! an uncached chunk: each awaits its own `init`, and the last to complete is the one retained.
//! The cache lock is never held across an `await`.

use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard};

use lru::LruCache;

use super::{
    Arc, ArrayError, AsyncCacheTraits, CacheChunkLimitTraits, CacheSizeLimitTraits, ChunkCacheType,
    ChunkIndices, Future, atomic,
};

pub(super) struct CacheChunkLimit<CT: ChunkCacheType> {
    cache: Mutex<LruCache<ChunkIndices, CT>>,
}

impl<CT: ChunkCacheType> CacheChunkLimit<CT> {
    fn lock(&self) -> MutexGuard<'_, LruCache<ChunkIndices, CT>> {
        self.cache.lock().unwrap()
    }
}

impl<CT: ChunkCacheType> AsyncCacheTraits<CT> for CacheChunkLimit<CT> {
    async fn len(&self) -> usize {
        self.lock().len()
    }

    async fn remove(&self, chunk_indices: &[u64]) -> bool {
        self.lock().pop(chunk_indices).is_some()
    }

    async fn clear(&self) -> usize {
        let mut cache = self.lock();
        let count = cache.len();
        cache.clear();
        count
    }

    async fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        init: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: Future<Output = Result<CT, ArrayError>>,
    {
        if let Some(value) = self.lock().get(&chunk_indices) {
            return Ok(value.clone());
        }
        let value = init.await.map_err(Arc::new)?;
        self.lock().push(chunk_indices, value.clone());
        Ok(value)
    }
}

impl<CT: ChunkCacheType> CacheChunkLimitTraits for CacheChunkLimit<CT> {
    fn new_with_chunk_capacity(chunk_capacity: u64) -> Self {
        let cache = LruCache::new(
            NonZeroUsize::new(usize::try_from(chunk_capacity).unwrap_or(usize::MAX).max(1))
                .unwrap(),
        );
        Self {
            cache: Mutex::new(cache),
        }
    }
}

pub(super) struct CacheSizeLimit<CT: ChunkCacheType> {
    cache: Mutex<LruCache<ChunkIndices, CT>>,
    capacity: u64,
    size: std::sync::atomic::AtomicUsize,
}

impl<CT: ChunkCacheType> CacheSizeLimit<CT> {
    fn lock(&self) -> MutexGuard<'_, LruCache<ChunkIndices, CT>> {
        self.cache.lock().unwrap()
    }
}

impl<CT: ChunkCacheType> AsyncCacheTraits<CT> for CacheSizeLimit<CT> {
    async fn len(&self) -> usize {
        self.lock().len()
    }

    async fn remove(&self, chunk_indices: &[u64]) -> bool {
        if let Some(chunk) = self.lock().pop(chunk_indices) {
            self.size.fetch_sub(chunk.size(), atomic::Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    async fn clear(&self) -> usize {
        let mut cache = self.lock();
        let count = cache.len();
        cache.clear();
        self.size.store(0, atomic::Ordering::SeqCst);
        count
    }

    async fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        init: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: Future<Output = Result<CT, ArrayError>>,
    {
        if let Some(value) = self.lock().get(&chunk_indices) {
            return Ok(value.clone());
        }
        let chunk = init.await.map_err(Arc::new)?;

        let mut cache = self.lock();
        let size = &self.size;
        let size_old = size.fetch_add(chunk.size(), atomic::Ordering::SeqCst);
        if size_old + chunk.size() > usize::try_from(self.capacity).unwrap() {
            let old = cache.pop_lru();
            if let Some(old) = old {
                size.fetch_sub(old.1.size(), atomic::Ordering::SeqCst);
            }
        }

        let old = cache.push(chunk_indices, chunk.clone());
        if let Some(old) = old {
            size.fetch_sub(old.1.size(), atomic::Ordering::SeqCst);
        }

        Ok(chunk)
    }
}

impl<CT: ChunkCacheType> CacheSizeLimitTraits for CacheSizeLimit<CT> {
    fn new_with_size_capacity(size_capacity: u64) -> Self {
        let cache = LruCache::unbounded();
        Self {
            cache: Mutex::new(cache),
            capacity: size_capacity,
            size: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

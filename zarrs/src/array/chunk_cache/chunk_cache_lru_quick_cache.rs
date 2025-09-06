use super::{
    atomic, Arc, ArrayError, CacheChunkLimitTraits, CacheSizeLimitTraits, CacheTraits,
    ChunkCacheType, ChunkIndices,
};

use std::{
    num::NonZeroUsize,
    sync::{Mutex, MutexGuard},
};

use lru::LruCache;

pub(super) struct CacheChunkLimit<CT: ChunkCacheType> {
    cache: Mutex<LruCache<ChunkIndices, CT>>,
}

pub(super) type ThreadLocalCacheChunkLimit<CT> = CacheChunkLimit<CT>;

impl<CT: ChunkCacheType> CacheChunkLimit<CT> {
    fn get(&self) -> MutexGuard<'_, LruCache<ChunkIndices, CT>> {
        self.cache.lock().unwrap()
    }
}

impl<CT: ChunkCacheType> CacheTraits<CT> for CacheChunkLimit<CT> {
    fn len(&self) -> usize {
        self.get().len()
    }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        let mut cache = self.get();
        if let Some(value) = cache.get(&chunk_indices) {
            Ok(value.clone())
        } else {
            let value = f()?;
            cache.push(chunk_indices, value.clone());
            Ok(value)
        }
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

pub(super) type ThreadLocalCacheSizeLimit<CT> = CacheSizeLimit<CT>;

impl<CT: ChunkCacheType> CacheSizeLimit<CT> {
    fn get(&self) -> MutexGuard<'_, LruCache<ChunkIndices, CT>> {
        self.cache.lock().unwrap()
    }
}

impl<CT: ChunkCacheType> CacheTraits<CT> for CacheSizeLimit<CT> {
    fn len(&self) -> usize {
        self.get().len()
    }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        let mut cache = self.get();

        if let Some(value) = cache.get(&chunk_indices) {
            Ok(value.clone())
        } else {
            let chunk = f()?;
            // self.insert(chunk_indices, value.clone());

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

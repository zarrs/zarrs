use std::{
    num::NonZeroUsize,
    sync::{Mutex, MutexGuard},
};

use lru::LruCache;
use moka::{policy::EvictionPolicy, sync::CacheBuilder};
use thread_local::ThreadLocal;

use super::{
    atomic, Arc, ArrayError, CacheChunkLimitTraits, CacheSizeLimitTraits, CacheTraits,
    ChunkCacheType, ChunkIndices,
};

type Cache<CT> = moka::sync::Cache<ChunkIndices, CT>;

impl<CT: ChunkCacheType> CacheTraits<CT> for Cache<CT> {
    fn len(&self) -> usize {
        self.run_pending_tasks();
        usize::try_from(self.entry_count()).unwrap()
    }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<CT, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<CT, ArrayError>,
    {
        self.try_get_with(chunk_indices, f)
    }
}

impl<CT: ChunkCacheType> CacheChunkLimitTraits for Cache<CT> {
    fn new_with_chunk_capacity(chunk_capacity: u64) -> Self {
        CacheBuilder::new(chunk_capacity)
            .eviction_policy(EvictionPolicy::lru())
            .build()
    }
}

impl<CT: ChunkCacheType> CacheSizeLimitTraits for Cache<CT> {
    fn new_with_size_capacity(size_capacity: u64) -> Self {
        CacheBuilder::new(size_capacity)
            .eviction_policy(EvictionPolicy::lru())
            .weigher(|_k: &Vec<u64>, v: &CT| u32::try_from(v.size()).unwrap_or(u32::MAX))
            .build()
    }
}

pub(super) type CacheChunkLimit<CT> = Cache<CT>;

pub(super) type CacheSizeLimit<CT> = Cache<CT>;

pub(super) struct ThreadLocalCacheChunkLimit<CT: ChunkCacheType> {
    cache: ThreadLocal<Mutex<LruCache<ChunkIndices, CT>>>,
    capacity: u64,
}

impl<CT: ChunkCacheType> ThreadLocalCacheChunkLimit<CT> {
    fn get(&self) -> MutexGuard<'_, LruCache<ChunkIndices, CT>> {
        self.cache
            .get_or(|| {
                let cache = LruCache::new(
                    NonZeroUsize::new(usize::try_from(self.capacity).unwrap_or(usize::MAX).max(1))
                        .unwrap(),
                );
                Mutex::new(cache)
            })
            .lock()
            .unwrap()
    }
}

impl<CT: ChunkCacheType> CacheTraits<CT> for ThreadLocalCacheChunkLimit<CT> {
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

impl<CT: ChunkCacheType> CacheChunkLimitTraits for ThreadLocalCacheChunkLimit<CT> {
    fn new_with_chunk_capacity(capacity: u64) -> Self {
        Self {
            cache: ThreadLocal::default(),
            capacity,
        }
    }
}

pub(super) struct ThreadLocalCacheSizeLimit<CT: ChunkCacheType> {
    cache: ThreadLocal<Mutex<LruCache<ChunkIndices, CT>>>,
    capacity: u64,
    size: ThreadLocal<std::sync::atomic::AtomicUsize>,
}

impl<CT: ChunkCacheType> ThreadLocalCacheSizeLimit<CT> {
    fn get(&self) -> MutexGuard<'_, LruCache<ChunkIndices, CT>> {
        self.cache
            .get_or(|| {
                let cache = LruCache::unbounded();
                Mutex::new(cache)
            })
            .lock()
            .unwrap()
    }
}

impl<CT: ChunkCacheType> CacheTraits<CT> for ThreadLocalCacheSizeLimit<CT> {
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

            let size = self.size.get_or_default();
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

impl<CT: ChunkCacheType> CacheSizeLimitTraits for ThreadLocalCacheSizeLimit<CT> {
    fn new_with_size_capacity(capacity: u64) -> Self {
        Self {
            cache: ThreadLocal::default(),
            capacity,
            size: ThreadLocal::default(),
        }
    }
}

use moka::policy::EvictionPolicy;
use moka::sync::CacheBuilder;

use super::{
    Arc, ArrayError, CacheChunkLimitTraits, CacheSizeLimitTraits, CacheTraits, ChunkCacheType,
    ChunkIndices, Indexer,
};

type Cache<CT> = moka::sync::Cache<ChunkIndices, CT>;

impl<CT: ChunkCacheType> CacheTraits<CT> for Cache<CT> {
    fn len(&self) -> usize {
        self.run_pending_tasks();
        usize::try_from(self.entry_count()).unwrap()
    }

    fn size(&self) -> usize {
        self.run_pending_tasks();
        self.iter().map(|(_, value)| value.size()).sum()
    }

    fn remove(&self, chunk_indices: &[u64]) -> bool {
        let removed = self.remove(chunk_indices).is_some();
        self.run_pending_tasks();
        removed
    }

    fn remove_many(&self, indexer: &dyn Indexer) -> usize {
        let removed = indexer
            .iter_indices()
            .filter(|chunk_indices| self.remove(chunk_indices.as_slice()).is_some())
            .count();
        self.run_pending_tasks();
        removed
    }

    fn clear(&self) -> usize {
        self.run_pending_tasks();
        let count = usize::try_from(self.entry_count()).unwrap();
        self.invalidate_all();
        count
    }

    fn try_get_or_insert_with<F>(
        &self,
        chunk_indices: ChunkIndices,
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
            .weigher(|_k: &ChunkIndices, v: &CT| u32::try_from(v.size()).unwrap_or(u32::MAX))
            .build()
    }
}

pub(super) type CacheChunkLimit<CT> = Cache<CT>;

pub(super) type CacheSizeLimit<CT> = Cache<CT>;

use moka::future::CacheBuilder;
use moka::policy::EvictionPolicy;

use super::{
    Arc, ArrayError, AsyncCacheTraits, CacheChunkLimitTraits, CacheSizeLimitTraits, ChunkCacheType,
    ChunkIndices, Future,
};

type Cache<CT> = moka::future::Cache<ChunkIndices, CT>;

impl<CT: ChunkCacheType> AsyncCacheTraits<CT> for Cache<CT> {
    async fn len(&self) -> usize {
        self.run_pending_tasks().await;
        usize::try_from(self.entry_count()).unwrap()
    }

    async fn remove(&self, chunk_indices: &[u64]) -> bool {
        let removed = moka::future::Cache::remove(self, chunk_indices)
            .await
            .is_some();
        self.run_pending_tasks().await;
        removed
    }

    async fn clear(&self) -> usize {
        self.run_pending_tasks().await;
        let count = usize::try_from(self.entry_count()).unwrap();
        self.invalidate_all();
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
        self.try_get_with(chunk_indices, init).await
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

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

use crate::{
    array::{codec::ArrayToBytesCodecTraits, ArrayBytes, ArrayError, ArrayIndices, ArraySize},
    storage::StorageError,
};

use super::{ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded};

use std::borrow::Cow;

type ChunkIndices = ArrayIndices;

/// A chunk cache with a fixed chunk capacity.
pub struct ChunkCacheLruChunkLimit<T: ChunkCacheType> {
    cache: Cache<ChunkIndices, Arc<T>>,
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

/// A chunk cache with a fixed size capacity.
pub struct ChunkCacheLruSizeLimit<T: ChunkCacheType> {
    cache: Cache<ChunkIndices, Arc<T>>,
}

/// An LRU (least recently used) encoded chunk cache with a fixed size capacity in bytes.
pub type ChunkCacheEncodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed size capacity in bytes.
pub type ChunkCacheDecodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

/// A thread local chunk cache with a fixed chunk capacity per thread.
pub struct ChunkCacheLruChunkLimitThreadLocal<T: ChunkCacheType> {
    cache: ThreadLocal<Mutex<LruCache<ChunkIndices, Arc<T>>>>,
    capacity: u64,
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeDecoded>;

/// A thread local chunk cache with a fixed chunk capacity per thread.
pub struct ChunkCacheLruSizeLimitThreadLocal<T: ChunkCacheType> {
    cache: ThreadLocal<Mutex<LruCache<ChunkIndices, Arc<T>>>>,
    capacity: usize,
    size: ThreadLocal<AtomicUsize>,
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeDecoded>;

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimit<CT> {
    /// Create a new [`ChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(chunk_capacity: u64) -> Self {
        let cache = CacheBuilder::new(chunk_capacity)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        Self { cache }
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimit<CT> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = CacheBuilder::new(capacity)
            .eviction_policy(EvictionPolicy::lru())
            .weigher(|_k, v: &Arc<CT>| u32::try_from(v.size()).unwrap_or(u32::MAX))
            .build();
        Self { cache }
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruChunkLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = ThreadLocal::new();
        Self { cache, capacity }
    }

    fn cache(&self) -> &Mutex<LruCache<ChunkIndices, Arc<CT>>> {
        self.cache.get_or(|| {
            Mutex::new(LruCache::new(
                NonZeroUsize::new(usize::try_from(self.capacity).unwrap_or(usize::MAX).max(1))
                    .unwrap(),
            ))
        })
    }
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruSizeLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = ThreadLocal::new();
        Self {
            cache,
            capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
            size: ThreadLocal::new(),
        }
    }

    fn cache(&self) -> &Mutex<LruCache<ChunkIndices, Arc<CT>>> {
        self.cache.get_or(|| Mutex::new(LruCache::unbounded()))
    }
}

macro_rules! impl_ChunkCacheLruCommon {
    ($ct:ty) => {
        fn get(&self, chunk_indices: &[u64]) -> Option<Arc<$ct>> {
            self.cache.get(&chunk_indices.to_vec())
        }

        fn insert(&self, chunk_indices: ChunkIndices, chunk: Arc<$ct>) {
            self.cache.insert(chunk_indices, chunk);
        }

        fn try_get_or_insert_with<F, E>(
            &self,
            chunk_indices: Vec<u64>,
            f: F,
        ) -> Result<Arc<$ct>, Arc<ArrayError>>
        where
            F: FnOnce() -> Result<Arc<$ct>, ArrayError>,
        {
            self.cache.try_get_with(chunk_indices, f)
        }

        fn len(&self) -> usize {
            self.cache.run_pending_tasks();
            usize::try_from(self.cache.entry_count()).unwrap()
        }
    };
}

macro_rules! impl_ChunkCacheLruEncoded {
    () => {
        fn retrieve_chunk<TStorage: ?Sized + crate::storage::ReadableStorageTraits + 'static>(
            &self,
            array: &crate::array::Array<TStorage>,
            chunk_indices: &[u64],
            options: &crate::array::codec::CodecOptions,
        ) -> Result<Arc<crate::array::ArrayBytes<'static>>, ArrayError> {
            let chunk_encoded = self
                .try_get_or_insert_with::<_, ArrayError>(chunk_indices.to_vec(), || {
                    Ok(Arc::new(
                        array.retrieve_encoded_chunk(chunk_indices)?.map(Cow::Owned),
                    ))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            // let chunk_encoded = if let Some(chunk) = cache.get(chunk_indices) {
            //     chunk
            // } else {
            //     let chunk = self.retrieve_encoded_chunk(chunk_indices)?.map(Cow::Owned);
            //     let chunk = Arc::new(chunk);
            //     cache.insert(chunk_indices.to_vec(), chunk.clone());
            //     chunk
            // };

            if let Some(chunk_encoded) = chunk_encoded.as_ref() {
                let chunk_representation = array.chunk_array_representation(chunk_indices)?;
                let bytes = array
                    .codecs()
                    .decode(Cow::Borrowed(chunk_encoded), &chunk_representation, options)
                    .map_err(ArrayError::CodecError)?;
                bytes.validate(
                    chunk_representation.num_elements(),
                    chunk_representation.data_type().size(),
                )?;
                Ok(Arc::new(bytes.into_owned()))
            } else {
                let chunk_shape = array.chunk_shape(chunk_indices)?;
                let array_size =
                    ArraySize::new(array.data_type().size(), chunk_shape.num_elements_u64());
                Ok(Arc::new(ArrayBytes::new_fill_value(
                    array_size,
                    array.fill_value(),
                )))
            }
        }
    };
}

macro_rules! impl_ChunkCacheLruDecoded {
    () => {
        fn retrieve_chunk<TStorage: ?Sized + crate::storage::ReadableStorageTraits + 'static>(
            &self,
            array: &crate::array::Array<TStorage>,
            chunk_indices: &[u64],
            options: &crate::array::codec::CodecOptions,
        ) -> Result<Arc<crate::array::ArrayBytes<'static>>, ArrayError> {
            self.try_get_or_insert_with::<_, ArrayError>(chunk_indices.to_vec(), || {
                Ok(Arc::new(
                    array
                        .retrieve_chunk_opt(chunk_indices, options)?
                        .into_owned(),
                ))
            })
            .map_err(|err| {
                // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                Arc::try_unwrap(err).unwrap_or_else(|err| {
                    ArrayError::StorageError(StorageError::from(err.to_string()))
                })
            })
            // if let Some(chunk) = cache.get(chunk_indices) {
            //     Ok(chunk)
            // } else {
            //     let chunk = Arc::new(
            //         array.retrieve_chunk_opt(chunk_indices, options)?
            //             .into_owned(),
            //     );
            //     cache.insert(chunk_indices.to_vec(), chunk.clone());
            //     Ok(chunk)
            // }
        }
    };
}

impl ChunkCache<ChunkCacheTypeEncoded> for ChunkCacheEncodedLruChunkLimit {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruCommon!(ChunkCacheTypeEncoded);
}

impl ChunkCache<ChunkCacheTypeDecoded> for ChunkCacheDecodedLruChunkLimit {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruCommon!(ChunkCacheTypeDecoded);
}

impl ChunkCache<ChunkCacheTypeEncoded> for ChunkCacheEncodedLruSizeLimit {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruCommon!(ChunkCacheTypeEncoded);
}

impl ChunkCache<ChunkCacheTypeDecoded> for ChunkCacheDecodedLruSizeLimit {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruCommon!(ChunkCacheTypeDecoded);
}

macro_rules! impl_ChunkCacheLruChunkLimitThreadLocal {
    ($ct:ty) => {
        fn get(&self, chunk_indices: &[u64]) -> Option<Arc<$ct>> {
            self.cache()
                .lock()
                .unwrap()
                .get(&chunk_indices.to_vec())
                .cloned()
        }

        fn insert(&self, chunk_indices: ChunkIndices, chunk: Arc<$ct>) {
            self.cache().lock().unwrap().push(chunk_indices, chunk);
        }

        fn try_get_or_insert_with<F, E>(
            &self,
            chunk_indices: Vec<u64>,
            f: F,
        ) -> Result<Arc<$ct>, Arc<ArrayError>>
        where
            F: FnOnce() -> Result<Arc<$ct>, ArrayError>,
        {
            self.cache()
                .lock()
                .unwrap()
                .try_get_or_insert(chunk_indices, f)
                .cloned()
                .map_err(|e| Arc::new(e))
        }

        fn len(&self) -> usize {
            self.cache().lock().unwrap().len()
        }
    };
}

macro_rules! impl_ChunkCacheLruSizeLimitThreadLocal {
    ($ct:ty) => {
        fn get(&self, chunk_indices: &[u64]) -> Option<Arc<$ct>> {
            self.cache()
                .lock()
                .unwrap()
                .get(&chunk_indices.to_vec())
                .cloned()
        }

        fn insert(&self, chunk_indices: ChunkIndices, chunk: Arc<$ct>) {
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

        fn len(&self) -> usize {
            self.cache().lock().unwrap().len()
        }
    };
}

impl ChunkCache<ChunkCacheTypeEncoded> for ChunkCacheEncodedLruChunkLimitThreadLocal {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruChunkLimitThreadLocal!(ChunkCacheTypeEncoded);
}

impl ChunkCache<ChunkCacheTypeDecoded> for ChunkCacheDecodedLruChunkLimitThreadLocal {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruChunkLimitThreadLocal!(ChunkCacheTypeDecoded);
}

impl ChunkCache<ChunkCacheTypeEncoded> for ChunkCacheEncodedLruSizeLimitThreadLocal {
    impl_ChunkCacheLruEncoded!();
    impl_ChunkCacheLruSizeLimitThreadLocal!(ChunkCacheTypeEncoded);
}

impl ChunkCache<ChunkCacheTypeDecoded> for ChunkCacheDecodedLruSizeLimitThreadLocal {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruSizeLimitThreadLocal!(ChunkCacheTypeDecoded);
}

#[cfg(feature = "ndarray")]
#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::{
        array::{
            codec::CodecOptions, ArrayBuilder, ArrayChunkCacheExt, ChunkCacheDecodedLruChunkLimit,
            ChunkCacheDecodedLruSizeLimit, ChunkCacheEncodedLruChunkLimit,
            ChunkCacheEncodedLruSizeLimit, ChunkCacheType, DataType,
        },
        array_subset::ArraySubset,
        storage::{
            storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter,
            store::MemoryStore,
        },
    };

    fn array_chunk_cache_impl<TChunkCache: ChunkCache<CT>, CT: ChunkCacheType>(
        cache: TChunkCache,
        thread_local: bool,
    ) {
        let store = Arc::new(MemoryStore::default());
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));
        let builder = ArrayBuilder::new(
            vec![8, 8], // array shape
            vec![4, 4], // regular chunk shape
            DataType::UInt8,
            0u8,
        );
        let array = builder.build(store.clone(), "/").unwrap();

        let data: Vec<u8> = (0..array.shape().into_iter().product())
            .map(|i| i as u8)
            .collect();
        array
            .store_array_subset_elements(&array.subset_all(), &data)
            .unwrap();

        assert_eq!(store.reads(), 0);
        assert!(cache.is_empty());
        assert_eq!(
            array
                .retrieve_array_subset_ndarray_opt_cached::<u8, _>(
                    &cache,
                    &ArraySubset::new_with_ranges(&[3..5, 0..4]),
                    &CodecOptions::default()
                )
                .unwrap(),
            ndarray::array![[24, 25, 26, 27], [32, 33, 34, 35]].into_dyn()
        );
        assert_eq!(store.reads(), 2);
        if !thread_local {
            assert_eq!(cache.len(), 2);
        }

        // Retrieve a chunk in cache
        assert_eq!(
            array
                .retrieve_chunk_ndarray_opt_cached::<u8, _>(
                    &cache,
                    &[0, 0],
                    &CodecOptions::default()
                )
                .unwrap(),
            ndarray::array![
                [0, 1, 2, 3],
                [8, 9, 10, 11],
                [16, 17, 18, 19],
                [24, 25, 26, 27]
            ]
            .into_dyn()
        );
        if !thread_local {
            assert_eq!(store.reads(), 2);
            assert_eq!(cache.len(), 2);
            assert!(cache.get(&[0, 0]).is_some());
            assert!(cache.get(&[1, 0]).is_some());
        }

        assert_eq!(
            array
                .retrieve_chunk_subset_ndarray_opt_cached::<u8, _>(
                    &cache,
                    &[0, 0],
                    &ArraySubset::new_with_ranges(&[1..3, 1..3]),
                    &CodecOptions::default()
                )
                .unwrap(),
            ndarray::array![[9, 10], [17, 18],].into_dyn()
        );
        if !thread_local {
            assert_eq!(store.reads(), 2);
            assert_eq!(cache.len(), 2);
            assert!(cache.get(&[0, 0]).is_some());
            assert!(cache.get(&[1, 0]).is_some());
        }

        // Retrieve chunks in the cache
        assert_eq!(
            array
                .retrieve_chunks_ndarray_opt_cached::<u8, _>(
                    &cache,
                    &ArraySubset::new_with_ranges(&[0..2, 0..1]),
                    &CodecOptions::default()
                )
                .unwrap(),
            ndarray::array![
                [0, 1, 2, 3],
                [8, 9, 10, 11],
                [16, 17, 18, 19],
                [24, 25, 26, 27],
                [32, 33, 34, 35],
                [40, 41, 42, 43],
                [48, 49, 50, 51],
                [56, 57, 58, 59]
            ]
            .into_dyn()
        );
        if !thread_local {
            assert_eq!(store.reads(), 2);
            assert_eq!(cache.len(), 2);
            assert!(cache.get(&[0, 0]).is_some());
            assert!(cache.get(&[1, 0]).is_some());
        }

        // Retrieve a chunk not in cache
        assert_eq!(
            array
                .retrieve_chunk_opt_cached(&cache, &[0, 1], &CodecOptions::default())
                .unwrap(),
            Arc::new(vec![4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31].into())
        );
        if !thread_local {
            assert_eq!(store.reads(), 3);
            assert_eq!(cache.len(), 2);
            assert!(cache.get(&[0, 1]).is_some());
            assert!(cache.get(&[0, 0]).is_none() || cache.get(&[1, 0]).is_none());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_chunks() {
        let cache = ChunkCacheEncodedLruChunkLimit::new(2);
        array_chunk_cache_impl(cache, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_size() {
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>();
        let cache = ChunkCacheEncodedLruSizeLimit::new(2 * chunk_size as u64);
        array_chunk_cache_impl(cache, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_chunks() {
        let cache = ChunkCacheDecodedLruChunkLimit::new(2);
        array_chunk_cache_impl(cache, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_size() {
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>();
        let cache = ChunkCacheDecodedLruSizeLimit::new(2 * chunk_size as u64);
        array_chunk_cache_impl(cache, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_chunks_thread_local() {
        let cache = ChunkCacheEncodedLruChunkLimitThreadLocal::new(2);
        array_chunk_cache_impl(cache, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_size_thread_local() {
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>();
        let cache = ChunkCacheEncodedLruSizeLimitThreadLocal::new(2 * chunk_size as u64);
        array_chunk_cache_impl(cache, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_chunks_thread_local() {
        let cache = ChunkCacheDecodedLruChunkLimitThreadLocal::new(2);
        array_chunk_cache_impl(cache, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_size_thread_local() {
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>();
        let cache = ChunkCacheDecodedLruSizeLimitThreadLocal::new(2 * chunk_size as u64);
        array_chunk_cache_impl(cache, true)
    }
}

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
        codec::ArrayToBytesCodecTraits, Array, ArrayBytes, ArrayError, ArrayIndices, ArraySize,
        ChunkCacheTypePartialDecoder,
    },
    array_subset::ArraySubset,
    storage::StorageError,
};

use super::{ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded};

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

macro_rules! impl_ChunkCacheLruEncoded {
    () => {
        fn retrieve_chunk(
            &self,
            chunk_indices: &[u64],
            options: &crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk_encoded = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(self
                        .array
                        .retrieve_encoded_chunk(chunk_indices)?
                        .map(|chunk| Arc::new(Cow::Owned(chunk))))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;

            if let Some(chunk_encoded) = chunk_encoded.as_ref() {
                let chunk_representation = self.array.chunk_array_representation(chunk_indices)?;
                let bytes = self
                    .array
                    .codecs()
                    .decode(Cow::Borrowed(chunk_encoded), &chunk_representation, options)
                    .map_err(ArrayError::CodecError)?;
                bytes.validate(
                    chunk_representation.num_elements(),
                    chunk_representation.data_type().size(),
                )?;
                Ok(Arc::new(bytes.into_owned()))
            } else {
                let chunk_shape = self.array.chunk_shape(chunk_indices)?;
                let array_size = ArraySize::new(
                    self.array.data_type().size(),
                    chunk_shape.num_elements_u64(),
                );
                Ok(Arc::new(ArrayBytes::new_fill_value(
                    array_size,
                    self.array.fill_value(),
                )))
            }
        }

        fn retrieve_chunk_subset(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk_encoded: ChunkCacheTypeEncoded = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(self
                        .array
                        .retrieve_encoded_chunk(chunk_indices)?
                        .map(|chunk| Arc::new(Cow::Owned(chunk))))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;

            if let Some(chunk_encoded) = chunk_encoded {
                let chunk_representation = self.array.chunk_array_representation(chunk_indices)?;
                Ok(self
                    .array
                    .codecs()
                    .partial_decoder(chunk_encoded, &chunk_representation, options)?
                    .partial_decode(&chunk_subset, options)?
                    .into_owned()
                    .into())
            } else {
                let array_size =
                    ArraySize::new(self.array.data_type().size(), chunk_subset.num_elements());
                Ok(Arc::new(ArrayBytes::new_fill_value(
                    array_size,
                    self.array.fill_value(),
                )))
            }
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

macro_rules! impl_ChunkCacheLruDecoded {
    () => {
        fn retrieve_chunk(
            &self,
            chunk_indices: &[u64],
            options: &crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            self.try_get_or_insert_with(chunk_indices.to_vec(), || {
                Ok(Arc::new(
                    self.array
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
        }

        fn retrieve_chunk_subset(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(Arc::new(
                        self.array
                            .retrieve_chunk_opt(chunk_indices, options)?
                            .into_owned(),
                    ))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            let chunk_representation = self.array.chunk_array_representation(chunk_indices)?;
            Ok(chunk
                .extract_array_subset(
                    chunk_subset,
                    &chunk_representation.shape_u64(),
                    self.array.data_type(),
                )?
                .into_owned()
                .into())
        }
    };
}

impl ChunkCache for ChunkCacheDecodedLruChunkLimit {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruCommon!();
}

impl ChunkCache for ChunkCacheDecodedLruSizeLimit {
    impl_ChunkCacheLruDecoded!();
    impl_ChunkCacheLruCommon!();
}

macro_rules! impl_ChunkCacheLruPartialDecoder {
    () => {
        fn retrieve_chunk(
            &self,
            chunk_indices: &[u64],
            options: &crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let partial_decoder = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    self.array.partial_decoder(chunk_indices)
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            let chunk_shape =
                crate::array::chunk_shape_to_array_shape(&self.array.chunk_shape(chunk_indices)?);
            Ok(partial_decoder
                .partial_decode(&ArraySubset::new_with_shape(chunk_shape), options)?
                .into_owned()
                .into())
        }

        fn retrieve_chunk_subset(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let partial_decoder = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    self.array.partial_decoder(chunk_indices)
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            Ok(partial_decoder
                .partial_decode(&chunk_subset, options)?
                .into_owned()
                .into())
        }
    };
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

#[cfg(feature = "ndarray")]
#[cfg(test)]
mod tests {
    use zarrs_storage::{ReadableWritableStorage, ReadableWritableStorageTraits};

    use super::*;

    use std::sync::Arc;

    use crate::{
        array::{
            codec::{CodecOptions, ShardingCodecBuilder},
            ArrayBuilder, ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruSizeLimit,
            ChunkCacheEncodedLruChunkLimit, ChunkCacheEncodedLruSizeLimit, DataType,
        },
        array_subset::ArraySubset,
        storage::{
            storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter,
            store::MemoryStore,
        },
    };

    fn create_store_array() -> (
        Arc<PerformanceMetricsStorageAdapter<dyn ReadableWritableStorageTraits>>,
        Arc<Array<dyn ReadableStorageTraits>>,
    ) {
        // Write the store
        let store: ReadableWritableStorage = Arc::new(MemoryStore::default());
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));
        let array = ArrayBuilder::new(
            vec![12, 8], // array shape
            vec![4, 4],  // regular chunk shape
            DataType::UInt8,
            0u8,
        )
        .array_to_bytes_codec(Arc::new(
            ShardingCodecBuilder::new(vec![2, 2].try_into().unwrap()).build(),
        ))
        .build_arc(store.clone(), "/")
        .unwrap();

        let data: Vec<u8> = (0..8 * 8).map(|i| i as u8).collect();
        array
            .store_array_subset_elements(&ArraySubset::new_with_shape(vec![8, 8]), &data)
            .unwrap();
        array.store_metadata().unwrap();

        // Return a read only version
        let array = Arc::new(array.readable());
        (store, array)
    }

    fn array_chunk_cache_impl<TChunkCache: ChunkCache>(
        store: Arc<PerformanceMetricsStorageAdapter<dyn ReadableWritableStorageTraits>>,
        cache: TChunkCache,
        // yikes
        thread_local: bool,
        size_limit: bool,
        partial_decoder: bool,
        encoded: bool,
    ) {
        assert_eq!(store.reads(), 0);
        assert!(cache.is_empty());
        assert_eq!(
            cache
                .retrieve_array_subset_ndarray::<u8>(
                    &ArraySubset::new_with_ranges(&[3..5, 0..4]),
                    &CodecOptions::default()
                )
                .unwrap(),
            ndarray::array![[24, 25, 26, 27], [32, 33, 34, 35]].into_dyn()
        );
        if !thread_local {
            assert!(!cache.is_empty());
        }
        if partial_decoder {
            assert_eq!(store.reads(), 2 + 8); // 2 index + 8 inner chunks
        } else {
            assert_eq!(store.reads(), 2);
        }
        if !thread_local {
            assert_eq!(cache.len(), 2);
        }

        // Retrieve a chunk in cache
        assert_eq!(
            cache
                .retrieve_chunk_ndarray::<u8>(&[0, 0], &CodecOptions::default())
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
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 8 + 4); // + 4 inner chunks
            } else {
                assert_eq!(store.reads(), 2);
            }
            assert_eq!(cache.len(), 2);
            // assert!(cache.get(&[0, 0]).is_some());
            // assert!(cache.get(&[1, 0]).is_some());
        }

        assert_eq!(
            cache
                .retrieve_chunk_subset_ndarray::<u8>(
                    &[0, 0],
                    &ArraySubset::new_with_ranges(&[1..3, 1..3]),
                    &CodecOptions::default()
                )
                .unwrap(),
            ndarray::array![[9, 10], [17, 18],].into_dyn()
        );
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 8 + 4 + 4); // 4 inner chunks
            } else {
                assert_eq!(store.reads(), 2);
            }
            assert_eq!(cache.len(), 2);
            // assert!(cache.get(&[0, 0]).is_some());
            // assert!(cache.get(&[1, 0]).is_some());
        }

        // Retrieve chunks in the cache
        assert_eq!(
            cache
                .retrieve_chunks_ndarray::<u8>(
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
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 8 + 4 + 4 + 8); // + 8 inner chunks
            } else {
                assert_eq!(store.reads(), 2);
            }
            assert_eq!(cache.len(), 2);
            // assert!(cache.get(&[0, 0]).is_some());
            // assert!(cache.get(&[1, 0]).is_some());
        }

        // Retrieve a chunk not in cache
        assert_eq!(
            cache
                .retrieve_chunk(&[0, 1], &CodecOptions::default())
                .unwrap(),
            Arc::new(vec![4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31].into())
        );
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 8 + 4 + 4 + 8 + 1 + 4); // 1 index + 4 inner chunks
            } else {
                assert_eq!(store.reads(), 3);
            }
            assert_eq!(cache.len(), 2);
            // assert!(cache.get(&[0, 1]).is_some());
            // assert!(cache.get(&[0, 0]).is_none() || cache.get(&[1, 0]).is_none());
        }

        // Partially retrieve from a cached chunk
        cache
            .retrieve_chunk_subset(
                &[0, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 8 + 4 + 4 + 8 + 1 + 4 + 1); // 1 inner chunks
            } else {
                assert_eq!(store.reads(), 3);
            }
            assert_eq!(cache.len(), 2);
        }

        // Partially retrieve from an uncached chunk
        cache
            .retrieve_chunk_subset(
                &[1, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 8 + 4 + 4 + 8 + 1 + 4 + 1 + 1 + 1);
            // 1 index + 1 inner chunk
            } else {
                assert_eq!(store.reads(), 4);
            }
            assert_eq!(cache.len(), 2);
        }

        // Partially retrieve from an empty chunk
        cache
            .retrieve_chunk_subset(
                &[2, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 8 + 4 + 4 + 8 + 1 + 4 + 1 + 1 + 1 + 1);
            // 1 index (empty)
            } else {
                assert_eq!(store.reads(), 5);
            }

            if size_limit && (encoded || partial_decoder) {
                assert_eq!(cache.len(), 2 + 1); // empty chunk is not included in size limit
            } else {
                assert_eq!(cache.len(), 2);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_chunks() {
        let (store, array) = create_store_array();
        let cache = ChunkCacheEncodedLruChunkLimit::new(array, 2);
        array_chunk_cache_impl(store, cache, false, false, false, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_chunks_thread_local() {
        let (store, array) = create_store_array();
        let cache = ChunkCacheEncodedLruChunkLimitThreadLocal::new(array, 2);
        array_chunk_cache_impl(store, cache, true, false, false, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_size() {
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>() + size_of::<u64>() * 2 * 2 * 2 + size_of::<u32>();
        let (store, array) = create_store_array();
        let cache = ChunkCacheEncodedLruSizeLimit::new(array, 2 * chunk_size as u64);
        array_chunk_cache_impl(store, cache, false, true, false, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_encoded_size_thread_local() {
        let (store, array) = create_store_array();
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>() + size_of::<u64>() * 2 * 2 * 2 + size_of::<u32>();
        let cache = ChunkCacheEncodedLruSizeLimitThreadLocal::new(array, 2 * chunk_size as u64);
        array_chunk_cache_impl(store, cache, true, true, false, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_chunks() {
        let (store, array) = create_store_array();
        let cache = ChunkCacheDecodedLruChunkLimit::new(array, 2);
        array_chunk_cache_impl(store, cache, false, false, false, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_chunks_thread_local() {
        let (store, array) = create_store_array();
        let cache = ChunkCacheDecodedLruChunkLimitThreadLocal::new(array, 2);
        array_chunk_cache_impl(store, cache, true, false, false, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_size() {
        let (store, array) = create_store_array();
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>();
        let cache = ChunkCacheDecodedLruSizeLimit::new(array, 2 * chunk_size as u64);
        array_chunk_cache_impl(store, cache, false, true, false, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_decoded_size_thread_local() {
        let (store, array) = create_store_array();
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>();
        let cache = ChunkCacheDecodedLruSizeLimitThreadLocal::new(array, 2 * chunk_size as u64);
        array_chunk_cache_impl(store, cache, true, true, false, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_partial_decoder_chunks() {
        let (store, array) = create_store_array();
        let cache = ChunkCachePartialDecoderLruChunkLimit::new(array, 2);
        array_chunk_cache_impl(store, cache, false, false, true, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_partial_decoder_chunks_thread_local() {
        let (store, array) = create_store_array();
        let cache = ChunkCachePartialDecoderLruChunkLimitThreadLocal::new(array, 2);
        array_chunk_cache_impl(store, cache, true, false, true, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_partial_decoder_size() {
        let (store, array) = create_store_array();
        // Create a cache with a size limit equivalent to 2 chunks (indexes)
        let chunk_size = size_of::<u64>() * 2 * 2 * 2;
        let cache = ChunkCachePartialDecoderLruSizeLimit::new(array, 2 * chunk_size as u64);
        array_chunk_cache_impl(store, cache, false, true, true, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_partial_decoder_size_thread_local() {
        let (store, array) = create_store_array();
        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = size_of::<u64>() * 2 * 2 * 2;
        let cache =
            ChunkCachePartialDecoderLruSizeLimitThreadLocal::new(array, 2 * chunk_size as u64);
        array_chunk_cache_impl(store, cache, true, true, true, false)
    }
}

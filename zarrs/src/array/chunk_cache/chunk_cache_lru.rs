use std::borrow::Cow;
use std::sync::{Arc, atomic};

use super::ChunkCacheType;
use crate::array::codec::{ArrayToBytesCodecTraits, CodecError};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArrayIndices, ChunkCache, ChunkCacheTypeDecoded,
    ChunkCacheTypeEncoded, ChunkCacheTypePartialDecoder, ChunkShapeTraits,
};
use crate::array_subset::ArraySubset;
use crate::storage::{ReadableStorageTraits, StorageError};

type ChunkIndices = ArrayIndices;

trait CacheTraits<CT: ChunkCacheType> {
    fn len(&self) -> usize;

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
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: platform::CacheChunkLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimit<CT> {
    /// Create a new [`ChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, chunk_capacity: u64) -> Self {
        let cache = platform::CacheChunkLimit::new_with_chunk_capacity(chunk_capacity);
        Self { array, cache }
    }
}

/// A thread local chunk cache with a fixed chunk capacity per thread.
pub struct ChunkCacheLruChunkLimitThreadLocal<CT: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: platform::ThreadLocalCacheChunkLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruChunkLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruChunkLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = platform::ThreadLocalCacheChunkLimit::new_with_chunk_capacity(capacity);
        Self { array, cache }
    }
}

/// A chunk cache with a fixed size capacity.
pub struct ChunkCacheLruSizeLimit<CT: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: platform::CacheSizeLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimit<CT> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = platform::CacheSizeLimit::new_with_size_capacity(capacity);
        Self { array, cache }
    }
}

/// A thread local chunk cache with a fixed size capacity per thread.
pub struct ChunkCacheLruSizeLimitThreadLocal<CT: ChunkCacheType> {
    array: Arc<Array<dyn ReadableStorageTraits>>,
    cache: platform::ThreadLocalCacheSizeLimit<CT>,
}

impl<CT: ChunkCacheType> ChunkCacheLruSizeLimitThreadLocal<CT> {
    /// Create a new [`ChunkCacheLruSizeLimitThreadLocal`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(array: Arc<Array<dyn ReadableStorageTraits>>, capacity: u64) -> Self {
        let cache = platform::ThreadLocalCacheSizeLimit::new_with_size_capacity(capacity);
        Self { array, cache }
    }
}

macro_rules! impl_ChunkCacheLruCommon {
    () => {
        fn array(&self) -> Arc<Array<dyn ReadableStorageTraits>> {
            self.array.clone()
        }

        fn len(&self) -> usize {
            self.cache.len()
        }
    };
}

macro_rules! impl_ChunkCacheLruEncoded {
    () => {
        fn retrieve_chunk_bytes(
            &self,
            chunk_indices: &[u64],
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk_encoded = self
                .cache
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
                let chunk_shape = self.array.chunk_shape(chunk_indices)?;
                let bytes = self
                    .array
                    .codecs()
                    .decode(
                        Cow::Borrowed(chunk_encoded),
                        &chunk_shape,
                        self.array.data_type(),
                        self.array.fill_value(),
                        options,
                    )
                    .map_err(ArrayError::CodecError)?;
                bytes.validate(chunk_shape.num_elements_u64(), self.array.data_type())?;
                Ok(Arc::new(bytes.into_owned()))
            } else {
                let chunk_shape = self.array.chunk_shape(chunk_indices)?;
                Ok(Arc::new(
                    ArrayBytes::new_fill_value(
                        self.array.data_type(),
                        chunk_shape.num_elements_u64(),
                        self.array.fill_value(),
                    )
                    .map_err(CodecError::from)
                    .map_err(ArrayError::from)?,
                ))
            }
        }

        fn retrieve_chunk_subset_bytes(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk_encoded: ChunkCacheTypeEncoded = self
                .cache
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
                let chunk_shape = self.array.chunk_shape(chunk_indices)?;
                Ok(self
                    .array
                    .codecs()
                    .partial_decoder(
                        chunk_encoded,
                        &chunk_shape,
                        self.array.data_type(),
                        self.array.fill_value(),
                        options,
                    )?
                    .partial_decode(chunk_subset, options)?
                    .into_owned()
                    .into())
            } else {
                Ok(Arc::new(
                    ArrayBytes::new_fill_value(
                        self.array.data_type(),
                        chunk_subset.num_elements(),
                        self.array.fill_value(),
                    )
                    .map_err(CodecError::from)
                    .map_err(ArrayError::from)?,
                ))
            }
        }
    };
}

macro_rules! impl_ChunkCacheLruDecoded {
    () => {
        fn retrieve_chunk_bytes(
            &self,
            chunk_indices: &[u64],
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            self.cache
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(Arc::new(
                        self.array
                            .retrieve_chunk_opt::<ArrayBytes<'static>>(chunk_indices, options)?,
                    ))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })
        }

        fn retrieve_chunk_subset_bytes(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk = self
                .cache
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(Arc::new(
                        self.array
                            .retrieve_chunk_opt::<ArrayBytes<'static>>(chunk_indices, options)?,
                    ))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            let chunk_shape = self.array.chunk_shape(chunk_indices)?;
            Ok(chunk
                .extract_array_subset(
                    chunk_subset,
                    bytemuck::must_cast_slice(&chunk_shape),
                    self.array.data_type(),
                )?
                .into_owned()
                .into())
        }
    };
}

macro_rules! impl_ChunkCacheLruPartialDecoder {
    () => {
        fn retrieve_chunk_bytes(
            &self,
            chunk_indices: &[u64],
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let partial_decoder = self
                .cache
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
                $crate::array::chunk_shape_to_array_shape(&self.array.chunk_shape(chunk_indices)?);
            Ok(partial_decoder
                .partial_decode(&ArraySubset::new_with_shape(chunk_shape), options)?
                .into_owned()
                .into())
        }

        fn retrieve_chunk_subset_bytes(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let partial_decoder = self
                .cache
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
                .partial_decode(chunk_subset, options)?
                .into_owned()
                .into())
        }
    };
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;
impl ChunkCache for ChunkCacheEncodedLruChunkLimit {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruEncoded!();
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeEncoded>;
impl ChunkCache for ChunkCacheEncodedLruChunkLimitThreadLocal {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruEncoded!();
}

/// An LRU (least recently used) encoded chunk cache with a fixed size capacity.
pub type ChunkCacheEncodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;
impl ChunkCache for ChunkCacheEncodedLruSizeLimit {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruEncoded!();
}

/// An LRU (least recently used) encoded chunk cache with a fixed size capacity.
pub type ChunkCacheEncodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeEncoded>;
impl ChunkCache for ChunkCacheEncodedLruSizeLimitThreadLocal {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruEncoded!();
}

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;
impl ChunkCache for ChunkCacheDecodedLruChunkLimit {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruDecoded!();
}

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypeDecoded>;
impl ChunkCache for ChunkCacheDecodedLruChunkLimitThreadLocal {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruDecoded!();
}

/// An LRU (least recently used) decoded chunk cache with a fixed size capacity .
pub type ChunkCacheDecodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;
impl ChunkCache for ChunkCacheDecodedLruSizeLimit {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruDecoded!();
}

/// An LRU (least recently used) decoded chunk cache with a fixed size capacity.
pub type ChunkCacheDecodedLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypeDecoded>;
impl ChunkCache for ChunkCacheDecodedLruSizeLimitThreadLocal {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruDecoded!();
}

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimit =
    ChunkCacheLruChunkLimit<ChunkCacheTypePartialDecoder>;
impl ChunkCache for ChunkCachePartialDecoderLruChunkLimit {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruPartialDecoder!();
}

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruChunkLimitThreadLocal =
    ChunkCacheLruChunkLimitThreadLocal<ChunkCacheTypePartialDecoder>;
impl ChunkCache for ChunkCachePartialDecoderLruChunkLimitThreadLocal {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruPartialDecoder!();
}

/// An LRU (least recently used) partial decoder chunk cache with a fixed size capacity.
pub type ChunkCachePartialDecoderLruSizeLimit =
    ChunkCacheLruSizeLimit<ChunkCacheTypePartialDecoder>;
impl ChunkCache for ChunkCachePartialDecoderLruSizeLimit {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruPartialDecoder!();
}

/// An LRU (least recently used) partial decoder chunk cache with a fixed chunk capacity.
pub type ChunkCachePartialDecoderLruSizeLimitThreadLocal =
    ChunkCacheLruSizeLimitThreadLocal<ChunkCacheTypePartialDecoder>;
impl ChunkCache for ChunkCachePartialDecoderLruSizeLimitThreadLocal {
    impl_ChunkCacheLruCommon!();
    impl_ChunkCacheLruPartialDecoder!();
}

#[cfg(feature = "ndarray")]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::{
        ReadableStorageTraits, ReadableWritableStorage, ReadableWritableStorageTraits,
    };
    use crate::{
        array::{
            Array, ArrayBuilder, ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruSizeLimit,
            ChunkCacheEncodedLruChunkLimit, ChunkCacheEncodedLruSizeLimit, DataType,
            chunk_cache::ChunkCache, codec::CodecOptions,
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
        .subchunk_shape(vec![2, 2])
        .build_arc(store.clone(), "/")
        .unwrap();

        let data: Vec<u8> = (0..8 * 8).map(|i| i as u8).collect();
        array
            .store_array_subset(&ArraySubset::new_with_shape(vec![8, 8]), &data)
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
                .retrieve_array_subset::<ndarray::ArrayD<u8>>(
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
            assert_eq!(store.reads(), 2 + 4); // 2 index + 4 inner chunks
        } else {
            assert_eq!(store.reads(), 2);
        }
        if !thread_local {
            assert_eq!(cache.len(), 2);
        }

        // Retrieve a chunk in cache
        assert_eq!(
            cache
                .retrieve_chunk::<ndarray::ArrayD<u8>>(&[0, 0], &CodecOptions::default())
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
                assert_eq!(store.reads(), 2 + 4 + 4); // + 4 inner chunks
            } else {
                assert_eq!(store.reads(), 2);
            }
            assert_eq!(cache.len(), 2);
            // assert!(cache.get(&[0, 0]).is_some());
            // assert!(cache.get(&[1, 0]).is_some());
        }

        assert_eq!(
            cache
                .retrieve_chunk_subset::<ndarray::ArrayD<u8>>(
                    &[0, 0],
                    &ArraySubset::new_with_ranges(&[1..3, 1..3]),
                    &CodecOptions::default()
                )
                .unwrap(),
            ndarray::array![[9, 10], [17, 18],].into_dyn()
        );
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 4 + 4 + 4); // 4 inner chunks
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
                .retrieve_chunks::<ndarray::ArrayD<u8>>(
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
                assert_eq!(store.reads(), 2 + 4 + 4 + 4 + 8); // + 8 inner chunks
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
                .retrieve_chunk_bytes(&[0, 1], &CodecOptions::default())
                .unwrap(),
            Arc::new(vec![4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31].into())
        );
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 4 + 4 + 4 + 8 + 1 + 4); // 1 index + 4 inner chunks
            } else {
                assert_eq!(store.reads(), 3);
            }
            assert_eq!(cache.len(), 2);
            // assert!(cache.get(&[0, 1]).is_some());
            // assert!(cache.get(&[0, 0]).is_none() || cache.get(&[1, 0]).is_none());
        }

        // Partially retrieve from a cached chunk
        cache
            .retrieve_chunk_subset_bytes(
                &[0, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 4 + 4 + 4 + 8 + 1 + 4 + 1); // 1 inner chunks
            } else {
                assert_eq!(store.reads(), 3);
            }
            assert_eq!(cache.len(), 2);
        }

        // Partially retrieve from an uncached chunk
        cache
            .retrieve_chunk_subset_bytes(
                &[1, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 4 + 4 + 4 + 8 + 1 + 4 + 1 + 1 + 1);
            // 1 index + 1 inner chunk
            } else {
                assert_eq!(store.reads(), 4);
            }
            assert_eq!(cache.len(), 2);
        }

        // Partially retrieve from an empty chunk
        cache
            .retrieve_chunk_subset_bytes(
                &[2, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 2 + 4 + 4 + 4 + 8 + 1 + 4 + 1 + 1 + 1 + 1);
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

    fn create_store_array_string() -> (
        Arc<PerformanceMetricsStorageAdapter<dyn ReadableWritableStorageTraits>>,
        Arc<Array<dyn ReadableStorageTraits>>,
    ) {
        // Write the store with String data
        let store: ReadableWritableStorage = Arc::new(MemoryStore::default());
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));
        let array = ArrayBuilder::new(
            vec![12, 8], // array shape
            vec![4, 4],  // regular chunk shape
            DataType::String,
            "",
        )
        // Note: Default codec for String is VlenUtf8Codec, no need to set explicitly
        .build_arc(store.clone(), "/")
        .unwrap();

        // Create test data with variable-length strings
        let data: Vec<String> = (0..8 * 8).map(|i| "x".repeat((i % 8) + 1)).collect();
        array
            .store_array_subset(&ArraySubset::new_with_shape(vec![8, 8]), data.as_slice())
            .unwrap();
        array.store_metadata().unwrap();

        // Return a read only version
        let array = Arc::new(array.readable());
        (store, array)
    }

    fn array_chunk_cache_string_impl<TChunkCache: ChunkCache>(
        store: Arc<PerformanceMetricsStorageAdapter<dyn ReadableWritableStorageTraits>>,
        cache: TChunkCache,
        thread_local: bool,
        size_limit: bool,
        partial_decoder: bool,
        encoded: bool,
    ) {
        assert_eq!(store.reads(), 0);
        assert!(cache.is_empty());

        // Retrieve an array subset (within a single chunk to test basic functionality)
        let result = cache
            .retrieve_array_subset::<Vec<String>>(
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        let expected: Vec<String> = vec![
            "x".repeat((0 % 8) + 1), // i=0: row 0, col 0
            "x".repeat((1 % 8) + 1), // i=1: row 0, col 1
            "x".repeat((8 % 8) + 1), // i=8: row 1, col 0
            "x".repeat((9 % 8) + 1), // i=9: row 1, col 1
        ];
        assert_eq!(result, expected);

        assert_eq!(store.reads(), 1); // Single chunk read
        if !thread_local {
            assert!(!cache.is_empty());
            assert_eq!(cache.len(), 1);
        }

        // Retrieve a chunk in cache
        let result = cache
            .retrieve_chunk::<Vec<String>>(&[0, 0], &CodecOptions::default())
            .unwrap();
        let expected: Vec<String> = vec![
            "x".repeat((0 % 8) + 1),  // i=0
            "x".repeat((1 % 8) + 1),  // i=1
            "x".repeat((2 % 8) + 1),  // i=2
            "x".repeat((3 % 8) + 1),  // i=3
            "x".repeat((8 % 8) + 1),  // i=8
            "x".repeat((9 % 8) + 1),  // i=9
            "x".repeat((10 % 8) + 1), // i=10
            "x".repeat((11 % 8) + 1), // i=11
            "x".repeat((16 % 8) + 1), // i=16
            "x".repeat((17 % 8) + 1), // i=17
            "x".repeat((18 % 8) + 1), // i=18
            "x".repeat((19 % 8) + 1), // i=19
            "x".repeat((24 % 8) + 1), // i=24
            "x".repeat((25 % 8) + 1), // i=25
            "x".repeat((26 % 8) + 1), // i=26
            "x".repeat((27 % 8) + 1), // i=27
        ];
        assert_eq!(result, expected);

        assert_eq!(store.reads(), 1); // Still cached
        if !thread_local {
            assert_eq!(cache.len(), 1);
        }

        // Retrieve a chunk subset
        let result = cache
            .retrieve_chunk_subset::<Vec<String>>(
                &[0, 0],
                &ArraySubset::new_with_ranges(&[1..3, 1..3]),
                &CodecOptions::default(),
            )
            .unwrap();
        let expected: Vec<String> = vec![
            "x".repeat((9 % 8) + 1),  // i=9
            "x".repeat((10 % 8) + 1), // i=10
            "x".repeat((17 % 8) + 1), // i=17
            "x".repeat((18 % 8) + 1), // i=18
        ];
        assert_eq!(result, expected);

        assert_eq!(store.reads(), 1); // Still cached
        if !thread_local {
            assert_eq!(cache.len(), 1);
        }

        // Retrieve chunks in the cache
        let result = cache
            .retrieve_chunks::<Vec<String>>(
                &ArraySubset::new_with_ranges(&[0..2, 0..1]),
                &CodecOptions::default(),
            )
            .unwrap();
        // Chunks [0,0] and [1,0]: rows 0-7, cols 0-3 -> indices: 0,1,2,3,8,9,10,11,...,56,57,58,59
        let expected: Vec<String> = vec![
            0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42,
            43, 48, 49, 50, 51, 56, 57, 58, 59,
        ]
        .iter()
        .map(|&i| "x".repeat((i % 8) + 1))
        .collect();
        assert_eq!(result, expected);

        if !thread_local {
            assert_eq!(store.reads(), 2); // Two chunks
            assert_eq!(cache.len(), 2);
        } else {
            assert_eq!(store.reads(), 3);
        }

        // Retrieve a chunk not in cache
        let result = cache
            .retrieve_chunk::<Vec<String>>(&[0, 1], &CodecOptions::default())
            .unwrap();
        let expected: Vec<String> = vec![
            "x".repeat((4 % 8) + 1),  // i=4
            "x".repeat((5 % 8) + 1),  // i=5
            "x".repeat((6 % 8) + 1),  // i=6
            "x".repeat((7 % 8) + 1),  // i=7
            "x".repeat((12 % 8) + 1), // i=12
            "x".repeat((13 % 8) + 1), // i=13
            "x".repeat((14 % 8) + 1), // i=14
            "x".repeat((15 % 8) + 1), // i=15
            "x".repeat((20 % 8) + 1), // i=20
            "x".repeat((21 % 8) + 1), // i=21
            "x".repeat((22 % 8) + 1), // i=22
            "x".repeat((23 % 8) + 1), // i=23
            "x".repeat((28 % 8) + 1), // i=28
            "x".repeat((29 % 8) + 1), // i=29
            "x".repeat((30 % 8) + 1), // i=30
            "x".repeat((31 % 8) + 1), // i=31
        ];
        assert_eq!(result, expected);

        if !thread_local {
            assert_eq!(store.reads(), 3); // One more chunk
            assert_eq!(cache.len(), 2);
        } else {
            assert_eq!(store.reads(), 4);
        }

        // Partially retrieve from a cached chunk
        cache
            .retrieve_chunk_subset_bytes(
                &[0, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            assert_eq!(store.reads(), 3); // Still cached
            assert_eq!(cache.len(), 2);
        } else {
            assert_eq!(store.reads(), 4);
        }

        // Partially retrieve from an uncached chunk
        cache
            .retrieve_chunk_subset_bytes(
                &[1, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            if partial_decoder {
                assert_eq!(store.reads(), 4); // One more chunk
            } else {
                assert_eq!(store.reads(), 4);
            }
            assert_eq!(cache.len(), 2);
        }

        // Partially retrieve from an empty chunk
        cache
            .retrieve_chunk_subset_bytes(
                &[2, 1],
                &ArraySubset::new_with_ranges(&[0..2, 0..2]),
                &CodecOptions::default(),
            )
            .unwrap();
        if !thread_local {
            assert_eq!(store.reads(), 5); // One more empty chunk
            if size_limit && encoded {
                assert_eq!(cache.len(), 2 + 1); // empty chunk is not included in size limit
            }
        } else {
            assert_eq!(store.reads(), 6);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_string_encoded_chunks() {
        let (store, array) = create_store_array_string();
        let cache = ChunkCacheEncodedLruChunkLimit::new(array, 2);
        array_chunk_cache_string_impl(store, cache, false, false, false, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_string_encoded_chunks_thread_local() {
        let (store, array) = create_store_array_string();
        let cache = ChunkCacheEncodedLruChunkLimitThreadLocal::new(array, 2);
        array_chunk_cache_string_impl(store, cache, true, false, false, true)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_string_decoded_chunks() {
        let (store, array) = create_store_array_string();
        let cache = ChunkCacheDecodedLruChunkLimit::new(array, 2);
        array_chunk_cache_string_impl(store, cache, false, false, false, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_string_decoded_chunks_thread_local() {
        let (store, array) = create_store_array_string();
        let cache = ChunkCacheDecodedLruChunkLimitThreadLocal::new(array, 2);
        array_chunk_cache_string_impl(store, cache, true, false, false, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_string_partial_decoder_chunks() {
        let (store, array) = create_store_array_string();
        let cache = ChunkCachePartialDecoderLruChunkLimit::new(array, 2);
        array_chunk_cache_string_impl(store, cache, false, false, true, false)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_chunk_cache_string_partial_decoder_chunks_thread_local() {
        let (store, array) = create_store_array_string();
        let cache = ChunkCachePartialDecoderLruChunkLimitThreadLocal::new(array, 2);
        array_chunk_cache_string_impl(store, cache, true, false, true, false)
    }
}

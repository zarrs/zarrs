#[cfg(not(target_arch = "wasm32"))]
#[path = "chunk_cache_lru_moka.rs"]
mod cache_impl;

#[cfg(target_arch = "wasm32")]
#[path = "chunk_cache_lru_quick_cache.rs"]
mod cache_impl;

pub use cache_impl::{
    ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruChunkLimitThreadLocal,
    ChunkCacheDecodedLruSizeLimit, ChunkCacheDecodedLruSizeLimitThreadLocal,
    ChunkCacheEncodedLruChunkLimit, ChunkCacheEncodedLruChunkLimitThreadLocal,
    ChunkCacheEncodedLruSizeLimit, ChunkCacheEncodedLruSizeLimitThreadLocal,
    ChunkCachePartialDecoderLruChunkLimit, ChunkCachePartialDecoderLruChunkLimitThreadLocal,
    ChunkCachePartialDecoderLruSizeLimit, ChunkCachePartialDecoderLruSizeLimitThreadLocal,
};

#[cfg(feature = "ndarray")]
#[cfg(test)]
mod tests {
    use zarrs_storage::{
        ReadableStorageTraits, ReadableWritableStorage, ReadableWritableStorageTraits,
    };

    use super::*;

    use std::sync::Arc;

    use crate::{
        array::{
            chunk_cache::ChunkCache,
            codec::{CodecOptions, ShardingCodecBuilder},
            Array, ArrayBuilder, ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruSizeLimit,
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

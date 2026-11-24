//! Caches usable by [`ArrayCached`](crate::array::ArrayCached).
//!
//! [`ArrayCached`](crate::array::ArrayCached) dispatches its retrievals and invalidations through the cache, rather
//! than through the value type of a [`SyncChunkCacheTraits`]. This lets a cache that is not keyed by chunk (e.g. one composed of
//! several stores) define how it serves a chunk and what a write evicts.
//!
//! Every [`SyncChunkCacheTraits`] is a [`SyncArrayCache`], with the retrieval behaviour of its value type and chunk-level
//! invalidation. With the `async` feature, every `AsyncChunkCacheTraits` is an `AsyncArrayCache`.

use std::sync::Arc;

use super::chunk_cache_type::validate_chunk_indices;
#[cfg(feature = "async")]
use super::{AsyncChunkCacheTraits, SealedAsync};
use super::{SealedSync, SyncChunkCacheTraits};
use crate::array::array_ops::{fill_value_bytes, subchunk_grid_at_level};
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkShape,
    ChunkShapeTraits, CodecChainBound, CodecOptions, DataType, FillValue, Indexer,
};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{
    ArrayBytesTarget, ArrayPartialDecoderTraits, Resources, decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{Bytes, MaybeSend, MaybeSync, ReadableStorageTraits};

/// Storage-independent array metadata needed by cache operations.
pub trait CachedArrayMeta: MaybeSend + MaybeSync {
    fn shape(&self) -> &[u64];

    fn dimensionality(&self) -> usize;

    fn chunk_grid_shape(&self) -> &[u64];

    fn chunk_shape(&self, chunk_indices: &[u64]) -> Result<ChunkShape, ArrayError>;

    fn chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError>;

    fn chunks_subset(&self, chunks: &dyn ArraySubsetTraits) -> Result<ArraySubset, ArrayError>;

    fn subchunk_grid_at_level(&self, level: usize) -> Result<&ChunkGrid, ArrayError>;

    fn data_type(&self) -> &DataType;

    fn fill_value(&self) -> &FillValue;

    fn codecs_bound(&self) -> &Arc<CodecChainBound>;
}

impl<TStorage: ?Sized + MaybeSend + MaybeSync> CachedArrayMeta for Array<TStorage> {
    fn shape(&self) -> &[u64] {
        self.shape()
    }

    fn dimensionality(&self) -> usize {
        self.dimensionality()
    }

    fn chunk_grid_shape(&self) -> &[u64] {
        self.chunk_grid_shape()
    }

    fn chunk_shape(&self, chunk_indices: &[u64]) -> Result<ChunkShape, ArrayError> {
        self.chunk_shape(chunk_indices)
    }

    fn chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        self.chunk_subset(chunk_indices)
    }

    fn chunks_subset(&self, chunks: &dyn ArraySubsetTraits) -> Result<ArraySubset, ArrayError> {
        self.chunks_subset(chunks)
    }

    fn subchunk_grid_at_level(&self, level: usize) -> Result<&ChunkGrid, ArrayError> {
        subchunk_grid_at_level(self, level)
    }

    fn data_type(&self) -> &DataType {
        self.data_type()
    }

    fn fill_value(&self) -> &FillValue {
        self.fill_value()
    }

    fn codecs_bound(&self) -> &Arc<CodecChainBound> {
        &self.codecs_bound
    }
}

/// Dyn-compatible array facade for synchronous cache reads.
pub trait SyncCachedArray: CachedArrayMeta {
    fn retrieve_encoded_chunk(&self, chunk_indices: &[u64]) -> Result<Option<Bytes>, ArrayError>;

    fn retrieve_chunk_if_exists_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<ArrayBytes<'static>>, ArrayError>;

    fn partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>;

    fn retrieve_partial_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError>;
}

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> SyncCachedArray for Array<TStorage> {
    fn retrieve_encoded_chunk(&self, chunk_indices: &[u64]) -> Result<Option<Bytes>, ArrayError> {
        self.retrieve_encoded_chunk(chunk_indices)
    }

    fn retrieve_chunk_if_exists_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<ArrayBytes<'static>>, ArrayError> {
        self.retrieve_chunk_if_exists_with_options(chunk_indices, options, resources)
    }

    fn partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        self.partial_decoder_with_options(chunk_indices, options, resources)
    }

    fn retrieve_partial_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        self.retrieve_partial_chunk_with_options(chunk_indices, indexer, options, resources)
    }
}

/// Dyn-compatible array facade for asynchronous cache reads.
#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncCachedArray: CachedArrayMeta {
    async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError>;

    async fn async_retrieve_chunk_if_exists_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<ArrayBytes<'static>>, ArrayError>;

    async fn async_partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>;

    async fn async_retrieve_partial_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError>;
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> AsyncCachedArray for Array<TStorage> {
    async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError> {
        self.async_retrieve_encoded_chunk(chunk_indices).await
    }

    async fn async_retrieve_chunk_if_exists_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<ArrayBytes<'static>>, ArrayError> {
        self.async_retrieve_chunk_if_exists_with_options(chunk_indices, options, resources)
            .await
    }

    async fn async_partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        self.async_partial_decoder_with_options(chunk_indices, options, resources)
            .await
    }

    async fn async_retrieve_partial_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        self.async_retrieve_partial_chunk_with_options(chunk_indices, indexer, options, resources)
            .await
    }
}

/// Return the fill value of the chunk at `chunk_indices`, after validating them.
fn fill_value_chunk_bytes(
    array: &(impl CachedArrayMeta + ?Sized),
    chunk_indices: &[u64],
) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
    let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
    fill_value_bytes(
        array.data_type(),
        array.fill_value(),
        chunk_shape.num_elements_u64(),
    )
    .map(Arc::new)
}

/// A cache usable by [`ArrayCached`](crate::array::ArrayCached) for synchronous operations.
///
/// This is implemented for every [`SyncChunkCacheTraits`], [`SubchunkCache`](super::SubchunkCache), and
/// [`NoCache`](super::NoCache). Implement [`SyncChunkCacheTraits`] for a custom cache policy.
///
/// The operations themselves live on a sealed supertrait, and are internal to `zarrs`.
pub trait SyncArrayCache: array_cache_sealed::SyncArrayCacheOps {}

impl<T: array_cache_sealed::SyncArrayCacheOps + ?Sized> SyncArrayCache for T {}

/// A cache usable by [`ArrayCached`](crate::array::ArrayCached) for asynchronous operations.
///
/// This is implemented for every [`AsyncChunkCacheTraits`], [`SubchunkCache`](super::SubchunkCache), and
/// [`NoCache`](super::NoCache). Implement [`AsyncChunkCacheTraits`] for a custom cache policy.
///
/// The operations themselves live on a sealed supertrait, and are internal to `zarrs`.
#[cfg(feature = "async")]
pub trait AsyncArrayCache: array_cache_sealed::AsyncArrayCacheOps {}

#[cfg(feature = "async")]
impl<T: array_cache_sealed::AsyncArrayCacheOps + ?Sized> AsyncArrayCache for T {}

pub(crate) mod array_cache_sealed {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    /// The operations of a [`SyncArrayCache`](super::SyncArrayCache).
    ///
    /// The invalidation methods are named apart from those of [`SyncChunkCacheTraits`], so that
    /// [`ArrayCached`](crate::array::ArrayCached), which is bounded by this trait rather than [`SyncChunkCacheTraits`], cannot
    /// invalidate a cache by chunk without going through it.
    pub trait SyncArrayCacheOps: MaybeSend + MaybeSync {
        /// Return a partial decoder of the chunk at `chunk_indices`, through the cache.
        fn cached_partial_decoder(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>;

        /// Return the decoded chunk at `chunk_indices` through the cache, or [`None`] if it does not exist.
        fn cached_chunk_bytes_if_exists(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>;

        /// Return the decoded chunk at `chunk_indices` through the cache, which is the fill value if it does not exist.
        ///
        /// The default implementation returns the fill value if [`cached_chunk_bytes_if_exists`] returns [`None`].
        /// A cache that knows the chunk is the fill value can return it without checking whether the chunk exists.
        ///
        /// [`cached_chunk_bytes_if_exists`]: SyncArrayCacheOps::cached_chunk_bytes_if_exists
        fn cached_chunk_bytes(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
            if let Some(bytes) =
                self.cached_chunk_bytes_if_exists(array, chunk_indices, options, resources)?
            {
                Ok(bytes)
            } else {
                fill_value_chunk_bytes(array, chunk_indices)
            }
        }

        /// Decode the chunk at `chunk_indices` into `output_target`, through the cache.
        fn cached_chunk_into(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let bytes = self.cached_chunk_bytes(array, chunk_indices, options, resources)?;
            decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
        }

        /// Return the elements of the chunk at `chunk_indices` selected by `indexer`, through the cache.
        fn cached_partial_chunk_bytes(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>;

        /// Decode the elements of the chunk at `chunk_indices` selected by `indexer` into `output_target`, through the
        /// cache.
        ///
        /// The default implementation copies the elements returned by
        /// [`cached_partial_chunk_bytes`](SyncArrayCacheOps::cached_partial_chunk_bytes) into `output_target`.
        fn cached_partial_chunk_into(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let bytes =
                self.cached_partial_chunk_bytes(array, chunk_indices, indexer, options, resources)?;
            decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
        }

        /// Invalidate everything cached for `array`.
        fn invalidate_cached_all(&self, array: &dyn CachedArrayMeta);

        /// Invalidate everything cached for the chunk at `chunk_indices`.
        fn invalidate_cached_chunk(&self, array: &dyn CachedArrayMeta, chunk_indices: &[u64]);

        /// Invalidate everything cached for the chunks of `chunks`.
        fn invalidate_cached_chunks(&self, array: &dyn CachedArrayMeta, chunks: &dyn Indexer) {
            for chunk_indices in chunks.iter_indices() {
                self.invalidate_cached_chunk(array, &chunk_indices);
            }
        }
    }

    impl<C: SyncChunkCacheTraits + ?Sized> SyncArrayCacheOps for C {
        fn cached_partial_decoder(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
            C::Value::partial_decoder(self, array, chunk_indices, options, resources)
        }

        fn cached_chunk_bytes_if_exists(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
            C::Value::retrieve_chunk_bytes_if_exists(self, array, chunk_indices, options, resources)
        }

        fn cached_chunk_into(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            C::Value::retrieve_chunk_into(
                self,
                array,
                chunk_indices,
                output_target,
                options,
                resources,
            )
        }

        fn cached_partial_chunk_bytes(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
            C::Value::retrieve_partial_chunk_bytes(
                self,
                array,
                chunk_indices,
                indexer,
                options,
                resources,
            )
        }

        fn cached_partial_chunk_into(
            &self,
            array: &dyn SyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            C::Value::retrieve_partial_chunk_into(
                self,
                array,
                chunk_indices,
                indexer,
                output_target,
                options,
                resources,
            )
        }

        fn invalidate_cached_all(&self, _array: &dyn CachedArrayMeta) {
            self.invalidate();
        }

        fn invalidate_cached_chunk(&self, _array: &dyn CachedArrayMeta, chunk_indices: &[u64]) {
            self.invalidate_chunk(chunk_indices);
        }

        fn invalidate_cached_chunks(&self, _array: &dyn CachedArrayMeta, chunks: &dyn Indexer) {
            self.invalidate_chunks(chunks);
        }
    }

    macro_rules! impl_SyncArrayCacheOps_deref {
        ($ty:ty) => {
            impl SyncArrayCacheOps for $ty {
                fn cached_partial_decoder(
                    &self,
                    array: &dyn SyncCachedArray,
                    chunk_indices: &[u64],
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
                    (**self).cached_partial_decoder(array, chunk_indices, options, resources)
                }

                fn cached_chunk_bytes_if_exists(
                    &self,
                    array: &dyn SyncCachedArray,
                    chunk_indices: &[u64],
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
                    (**self).cached_chunk_bytes_if_exists(array, chunk_indices, options, resources)
                }

                fn cached_chunk_bytes(
                    &self,
                    array: &dyn SyncCachedArray,
                    chunk_indices: &[u64],
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
                    (**self).cached_chunk_bytes(array, chunk_indices, options, resources)
                }

                fn cached_chunk_into(
                    &self,
                    array: &dyn SyncCachedArray,
                    chunk_indices: &[u64],
                    output_target: &mut ArrayBytesTarget<'_>,
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<(), ArrayError> {
                    (**self).cached_chunk_into(
                        array,
                        chunk_indices,
                        output_target,
                        options,
                        resources,
                    )
                }

                fn cached_partial_chunk_bytes(
                    &self,
                    array: &dyn SyncCachedArray,
                    chunk_indices: &[u64],
                    indexer: &dyn Indexer,
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
                    (**self).cached_partial_chunk_bytes(
                        array,
                        chunk_indices,
                        indexer,
                        options,
                        resources,
                    )
                }

                fn cached_partial_chunk_into(
                    &self,
                    array: &dyn SyncCachedArray,
                    chunk_indices: &[u64],
                    indexer: &dyn Indexer,
                    output_target: &mut ArrayBytesTarget<'_>,
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<(), ArrayError> {
                    (**self).cached_partial_chunk_into(
                        array,
                        chunk_indices,
                        indexer,
                        output_target,
                        options,
                        resources,
                    )
                }

                fn invalidate_cached_all(&self, array: &dyn CachedArrayMeta) {
                    (**self).invalidate_cached_all(array);
                }

                fn invalidate_cached_chunk(
                    &self,
                    array: &dyn CachedArrayMeta,
                    chunk_indices: &[u64],
                ) {
                    (**self).invalidate_cached_chunk(array, chunk_indices);
                }

                fn invalidate_cached_chunks(
                    &self,
                    array: &dyn CachedArrayMeta,
                    chunks: &dyn Indexer,
                ) {
                    (**self).invalidate_cached_chunks(array, chunks);
                }
            }
        };
    }

    impl_SyncArrayCacheOps_deref!(Box<dyn SyncArrayCache>);

    /// The operations of an [`AsyncArrayCache`](super::AsyncArrayCache).
    ///
    /// See [`SyncArrayCacheOps`].
    #[cfg(feature = "async")]
    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    pub trait AsyncArrayCacheOps: MaybeSend + MaybeSync {
        /// Return an asynchronous partial decoder of the chunk at `chunk_indices`, through the cache.
        async fn async_cached_partial_decoder(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>;

        /// Return the decoded chunk at `chunk_indices` through the cache, or [`None`] if it does not exist.
        async fn async_cached_chunk_bytes_if_exists(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>;

        /// Return the decoded chunk at `chunk_indices` through the cache, which is the fill value if it does not exist.
        ///
        /// See [`SyncArrayCacheOps::cached_chunk_bytes`].
        async fn async_cached_chunk_bytes(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
            if let Some(bytes) = self
                .async_cached_chunk_bytes_if_exists(array, chunk_indices, options, resources)
                .await?
            {
                Ok(bytes)
            } else {
                fill_value_chunk_bytes(array, chunk_indices)
            }
        }

        /// Decode the chunk at `chunk_indices` into `output_target`, through the cache.
        async fn async_cached_chunk_into(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let bytes = self
                .async_cached_chunk_bytes(array, chunk_indices, options, resources)
                .await?;
            decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
        }

        /// Return the elements of the chunk at `chunk_indices` selected by `indexer`, through the cache.
        async fn async_cached_partial_chunk_bytes(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>;

        /// Decode the elements of the chunk at `chunk_indices` selected by `indexer` into `output_target`, through the
        /// cache.
        ///
        /// See [`SyncArrayCacheOps::cached_partial_chunk_into`].
        async fn async_cached_partial_chunk_into(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let bytes = self
                .async_cached_partial_chunk_bytes(array, chunk_indices, indexer, options, resources)
                .await?;
            decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
        }

        /// Invalidate everything cached for `array`.
        async fn async_invalidate_cached_all(&self, array: &dyn CachedArrayMeta);

        /// Invalidate everything cached for the chunk at `chunk_indices`.
        async fn async_invalidate_cached_chunk(
            &self,
            array: &dyn CachedArrayMeta,
            chunk_indices: &[u64],
        );

        /// Invalidate everything cached for the chunks of `chunks`.
        async fn async_invalidate_cached_chunks(
            &self,
            array: &dyn CachedArrayMeta,
            chunks: &dyn Indexer,
        ) {
            for chunk_indices in chunks.iter_indices() {
                self.async_invalidate_cached_chunk(array, &chunk_indices)
                    .await;
            }
        }
    }

    #[cfg(feature = "async")]
    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl<C: AsyncChunkCacheTraits + ?Sized> AsyncArrayCacheOps for C {
        async fn async_cached_partial_decoder(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
            C::Value::async_partial_decoder(self, array, chunk_indices, options, resources).await
        }

        async fn async_cached_chunk_bytes_if_exists(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
            C::Value::async_retrieve_chunk_bytes_if_exists(
                self,
                array,
                chunk_indices,
                options,
                resources,
            )
            .await
        }

        async fn async_cached_chunk_bytes(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
            if let Some(bytes) = C::Value::async_retrieve_chunk_bytes_if_exists(
                self,
                array,
                chunk_indices,
                options,
                resources,
            )
            .await?
            {
                Ok(bytes)
            } else {
                fill_value_chunk_bytes(array, chunk_indices)
            }
        }

        async fn async_cached_chunk_into(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            C::Value::async_retrieve_chunk_into(
                self,
                array,
                chunk_indices,
                output_target,
                options,
                resources,
            )
            .await
        }

        async fn async_cached_partial_chunk_bytes(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
            C::Value::async_retrieve_partial_chunk_bytes(
                self,
                array,
                chunk_indices,
                indexer,
                options,
                resources,
            )
            .await
        }

        async fn async_cached_partial_chunk_into(
            &self,
            array: &dyn AsyncCachedArray,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            C::Value::async_retrieve_partial_chunk_into(
                self,
                array,
                chunk_indices,
                indexer,
                output_target,
                options,
                resources,
            )
            .await
        }

        async fn async_invalidate_cached_all(&self, _array: &dyn CachedArrayMeta) {
            self.invalidate().await;
        }

        async fn async_invalidate_cached_chunk(
            &self,
            _array: &dyn CachedArrayMeta,
            chunk_indices: &[u64],
        ) {
            self.invalidate_chunk(chunk_indices).await;
        }

        async fn async_invalidate_cached_chunks(
            &self,
            _array: &dyn CachedArrayMeta,
            chunks: &dyn Indexer,
        ) {
            self.invalidate_chunks(chunks).await;
        }
    }

    #[cfg(feature = "async")]
    macro_rules! impl_AsyncArrayCacheOps_deref {
        ($ty:ty) => {
            #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
            #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
            impl AsyncArrayCacheOps for $ty {
                async fn async_cached_partial_decoder(
                    &self,
                    array: &dyn AsyncCachedArray,
                    chunk_indices: &[u64],
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
                    (**self)
                        .async_cached_partial_decoder(array, chunk_indices, options, resources)
                        .await
                }

                async fn async_cached_chunk_bytes_if_exists(
                    &self,
                    array: &dyn AsyncCachedArray,
                    chunk_indices: &[u64],
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
                    (**self)
                        .async_cached_chunk_bytes_if_exists(
                            array,
                            chunk_indices,
                            options,
                            resources,
                        )
                        .await
                }

                async fn async_cached_chunk_bytes(
                    &self,
                    array: &dyn AsyncCachedArray,
                    chunk_indices: &[u64],
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
                    (**self)
                        .async_cached_chunk_bytes(array, chunk_indices, options, resources)
                        .await
                }

                async fn async_cached_chunk_into(
                    &self,
                    array: &dyn AsyncCachedArray,
                    chunk_indices: &[u64],
                    output_target: &mut ArrayBytesTarget<'_>,
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<(), ArrayError> {
                    (**self)
                        .async_cached_chunk_into(
                            array,
                            chunk_indices,
                            output_target,
                            options,
                            resources,
                        )
                        .await
                }

                async fn async_cached_partial_chunk_bytes(
                    &self,
                    array: &dyn AsyncCachedArray,
                    chunk_indices: &[u64],
                    indexer: &dyn Indexer,
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
                    (**self)
                        .async_cached_partial_chunk_bytes(
                            array,
                            chunk_indices,
                            indexer,
                            options,
                            resources,
                        )
                        .await
                }

                async fn async_cached_partial_chunk_into(
                    &self,
                    array: &dyn AsyncCachedArray,
                    chunk_indices: &[u64],
                    indexer: &dyn Indexer,
                    output_target: &mut ArrayBytesTarget<'_>,
                    options: &CodecOptions,
                    resources: &Resources,
                ) -> Result<(), ArrayError> {
                    (**self)
                        .async_cached_partial_chunk_into(
                            array,
                            chunk_indices,
                            indexer,
                            output_target,
                            options,
                            resources,
                        )
                        .await
                }

                async fn async_invalidate_cached_all(&self, array: &dyn CachedArrayMeta) {
                    (**self).async_invalidate_cached_all(array).await;
                }

                async fn async_invalidate_cached_chunk(
                    &self,
                    array: &dyn CachedArrayMeta,
                    chunk_indices: &[u64],
                ) {
                    (**self)
                        .async_invalidate_cached_chunk(array, chunk_indices)
                        .await;
                }

                async fn async_invalidate_cached_chunks(
                    &self,
                    array: &dyn CachedArrayMeta,
                    chunks: &dyn Indexer,
                ) {
                    (**self).async_invalidate_cached_chunks(array, chunks).await;
                }
            }
        };
    }

    #[cfg(feature = "async")]
    impl_AsyncArrayCacheOps_deref!(Box<dyn AsyncArrayCache>);
}

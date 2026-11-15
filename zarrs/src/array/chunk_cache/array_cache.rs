//! Caches usable by [`ArrayCached`](crate::array::ArrayCached).
//!
//! [`ArrayCached`](crate::array::ArrayCached) dispatches its retrievals and invalidations through the cache, rather
//! than through the value type of a [`ChunkCache`]. This lets a cache that is not keyed by chunk (e.g. one composed of
//! several stores) define how it serves a chunk and what a write evicts.
//!
//! Every [`ChunkCache`] is a [`SyncArrayCache`], with the retrieval behaviour of its value type and chunk-level
//! invalidation. With the `async` feature, every `AsyncChunkCache` is an `AsyncArrayCache`.

use std::borrow::Cow;
#[cfg(feature = "async")]
use std::future::Future;
use std::sync::Arc;

use super::chunk_cache_type::validate_chunk_indices;
#[cfg(feature = "async")]
use super::{AsyncChunkCache, SealedAsync};
use super::{ChunkCache, SealedSync};
use crate::array::array_ops::fill_value_bytes;
use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubsetTraits, ChunkShapeTraits, CodecOptions, Indexer,
};
#[cfg(feature = "async")]
use zarrs_codec::AsyncArrayPartialDecoderTraits;
use zarrs_codec::{
    ArrayBytesTarget, ArrayPartialDecoderTraits, Resources, decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{MaybeSend, MaybeSync, ReadableStorageTraits};

/// A cache usable by [`ArrayCached`](crate::array::ArrayCached) for synchronous operations.
///
/// This is implemented for every [`ChunkCache`]. Implement [`ChunkCache`] for a custom cache policy.
///
/// The operations themselves live on a sealed supertrait, and are internal to `zarrs`.
pub trait SyncArrayCache: array_cache_sealed::SyncArrayCacheOps {}

impl<T: array_cache_sealed::SyncArrayCacheOps + ?Sized> SyncArrayCache for T {}

/// A cache usable by [`ArrayCached`](crate::array::ArrayCached) for asynchronous operations.
///
/// This is implemented for every [`AsyncChunkCache`]. Implement [`AsyncChunkCache`] for a custom cache policy.
///
/// The operations themselves live on a sealed supertrait, and are internal to `zarrs`.
#[cfg(feature = "async")]
pub trait AsyncArrayCache: array_cache_sealed::AsyncArrayCacheOps {}

#[cfg(feature = "async")]
impl<T: array_cache_sealed::AsyncArrayCacheOps + ?Sized> AsyncArrayCache for T {}

/// Keep thread-local cache access and any nested decoding on the calling thread, if `thread_local`.
pub(crate) fn thread_local_resources(
    thread_local: bool,
    resources: &Resources,
) -> Cow<'_, Resources> {
    if thread_local && resources.compute_target() != 1 {
        Cow::Owned(resources.clone().with_compute_target(1))
    } else {
        Cow::Borrowed(resources)
    }
}

pub(crate) mod array_cache_sealed {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    /// The operations of a [`SyncArrayCache`](super::SyncArrayCache).
    ///
    /// The invalidation methods are named apart from those of [`ChunkCache`], so that
    /// [`ArrayCached`](crate::array::ArrayCached), which is bounded by this trait rather than [`ChunkCache`], cannot
    /// invalidate a cache by chunk without going through it.
    pub trait SyncArrayCacheOps: MaybeSend + MaybeSync {
        /// Return whether any layer uses a thread-local cache.
        fn uses_thread_local_cache(&self) -> bool {
            false
        }

        /// Return the `resources` of an access of the cache, which keep any nested decoding on the calling thread if a
        /// layer uses a thread-local cache.
        fn cache_resources<'a>(&self, resources: &'a Resources) -> Cow<'a, Resources> {
            thread_local_resources(self.uses_thread_local_cache(), resources)
        }

        /// Return the minimum number of chunks to retrieve concurrently through the cache, given the
        /// `chunk_concurrent_minimum` of the array, and the resources of the retrieval.
        ///
        /// Chunks are retrieved one at a time on the calling thread if a layer uses a thread-local cache.
        fn cache_concurrency<'a>(
            &self,
            chunk_concurrent_minimum: usize,
            resources: &'a Resources,
        ) -> (usize, Cow<'a, Resources>) {
            if self.uses_thread_local_cache() {
                (1, thread_local_resources(true, resources))
            } else {
                (chunk_concurrent_minimum, Cow::Borrowed(resources))
            }
        }

        /// Return a partial decoder of the chunk at `chunk_indices`, through the cache.
        fn cached_partial_decoder<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static;

        /// Return the decoded chunk at `chunk_indices` through the cache, or [`None`] if it does not exist.
        fn cached_chunk_bytes_if_exists<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static;

        /// Return the decoded chunk at `chunk_indices` through the cache, which is the fill value if it does not exist.
        ///
        /// The default implementation returns the fill value if [`cached_chunk_bytes_if_exists`] returns [`None`].
        /// A cache that knows the chunk is the fill value can return it without checking whether the chunk exists.
        ///
        /// [`cached_chunk_bytes_if_exists`]: SyncArrayCacheOps::cached_chunk_bytes_if_exists
        fn cached_chunk_bytes<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            if let Some(bytes) =
                self.cached_chunk_bytes_if_exists(array, chunk_indices, options, resources)?
            {
                Ok(bytes)
            } else {
                let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
                fill_value_bytes(array, chunk_shape.num_elements_u64()).map(Arc::new)
            }
        }

        /// Decode the chunk at `chunk_indices` into `output_target`, through the cache.
        fn cached_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            let bytes = self.cached_chunk_bytes(array, chunk_indices, options, resources)?;
            decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
        }

        /// Return the elements of the chunk at `chunk_indices` selected by `indexer`, through the cache.
        fn cached_partial_chunk_bytes<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static;

        /// Decode the elements of the chunk at `chunk_indices` selected by `indexer` into `output_target`, through the
        /// cache.
        ///
        /// The default implementation copies the elements returned by
        /// [`cached_partial_chunk_bytes`](SyncArrayCacheOps::cached_partial_chunk_bytes) into `output_target`.
        fn cached_partial_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            let bytes =
                self.cached_partial_chunk_bytes(array, chunk_indices, indexer, options, resources)?;
            decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
        }

        /// Invalidate everything cached for `array`.
        fn invalidate_cached_all<TStorage: ?Sized>(&self, array: &Array<TStorage>);

        /// Invalidate everything cached for the chunk at `chunk_indices`.
        fn invalidate_cached_chunk<TStorage: ?Sized>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
        );

        /// Invalidate everything cached for the chunks of `chunks`.
        fn invalidate_cached_chunks<TStorage: ?Sized>(
            &self,
            array: &Array<TStorage>,
            chunks: &dyn Indexer,
        ) {
            for chunk_indices in chunks.iter_indices() {
                self.invalidate_cached_chunk(array, &chunk_indices);
            }
        }

        /// Invalidate what is cached for the `array_subset` of `array`.
        ///
        /// The default implementation invalidates the intersecting chunks, or everything if they cannot be determined.
        ///
        /// # Errors
        /// Returns an [`ArrayError`] if `array_subset` has an incompatible dimensionality.
        fn invalidate_cached_array_subset<TStorage: ?Sized>(
            &self,
            array: &Array<TStorage>,
            array_subset: &dyn ArraySubsetTraits,
        ) -> Result<(), ArrayError> {
            if let Some(chunks) = array.chunks_in_array_subset(array_subset)? {
                self.invalidate_cached_chunks(array, &chunks);
            } else {
                self.invalidate_cached_all(array);
            }
            Ok(())
        }

        /// Invalidate what is cached for the elements selected by `indexer` in the chunk at `chunk_indices`.
        ///
        /// The default implementation invalidates the whole chunk.
        fn invalidate_cached_chunk_indexer<TStorage: ?Sized>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
        ) {
            let _ = indexer;
            self.invalidate_cached_chunk(array, chunk_indices);
        }
    }

    impl<C: ChunkCache + ?Sized> SyncArrayCacheOps for C {
        fn uses_thread_local_cache(&self) -> bool {
            self.is_thread_local()
        }

        fn cached_partial_decoder<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            let resources = self.cache_resources(resources);
            C::Value::partial_decoder(self, array, chunk_indices, options, &resources)
        }

        fn cached_chunk_bytes_if_exists<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            let resources = self.cache_resources(resources);
            C::Value::retrieve_chunk_bytes_if_exists(
                self,
                array,
                chunk_indices,
                options,
                &resources,
            )
        }

        fn cached_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            let resources = self.cache_resources(resources);
            C::Value::retrieve_chunk_into(
                self,
                array,
                chunk_indices,
                output_target,
                options,
                &resources,
            )
        }

        fn cached_partial_chunk_bytes<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            let resources = self.cache_resources(resources);
            C::Value::retrieve_partial_chunk_bytes(
                self,
                array,
                chunk_indices,
                indexer,
                options,
                &resources,
            )
        }

        fn cached_partial_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError>
        where
            TStorage: ?Sized + ReadableStorageTraits + 'static,
        {
            let resources = self.cache_resources(resources);
            C::Value::retrieve_partial_chunk_into(
                self,
                array,
                chunk_indices,
                indexer,
                output_target,
                options,
                &resources,
            )
        }

        fn invalidate_cached_all<TStorage: ?Sized>(&self, _array: &Array<TStorage>) {
            self.invalidate();
        }

        fn invalidate_cached_chunk<TStorage: ?Sized>(
            &self,
            _array: &Array<TStorage>,
            chunk_indices: &[u64],
        ) {
            self.invalidate_chunk(chunk_indices);
        }

        fn invalidate_cached_chunks<TStorage: ?Sized>(
            &self,
            _array: &Array<TStorage>,
            chunks: &dyn Indexer,
        ) {
            self.invalidate_chunks(chunks);
        }
    }

    /// The operations of an [`AsyncArrayCache`](super::AsyncArrayCache).
    ///
    /// See [`SyncArrayCacheOps`].
    #[cfg(feature = "async")]
    pub trait AsyncArrayCacheOps: MaybeSend + MaybeSync {
        /// Return an asynchronous partial decoder of the chunk at `chunk_indices`, through the cache.
        fn async_cached_partial_decoder<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> impl Future<Output = Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>> + MaybeSend
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static;

        /// Return the decoded chunk at `chunk_indices` through the cache, or [`None`] if it does not exist.
        fn async_cached_chunk_bytes_if_exists<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> impl Future<Output = Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>> + MaybeSend
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static;

        /// Return the decoded chunk at `chunk_indices` through the cache, which is the fill value if it does not exist.
        ///
        /// See [`SyncArrayCacheOps::cached_chunk_bytes`].
        fn async_cached_chunk_bytes<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> impl Future<Output = Result<Arc<ArrayBytes<'static>>, ArrayError>> + MaybeSend
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
            async move {
                if let Some(bytes) = self
                    .async_cached_chunk_bytes_if_exists(array, chunk_indices, options, resources)
                    .await?
                {
                    Ok(bytes)
                } else {
                    let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
                    fill_value_bytes(array, chunk_shape.num_elements_u64()).map(Arc::new)
                }
            }
        }

        /// Decode the chunk at `chunk_indices` into `output_target`, through the cache.
        fn async_cached_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> impl Future<Output = Result<(), ArrayError>> + MaybeSend
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
            async move {
                let bytes = self
                    .async_cached_chunk_bytes(array, chunk_indices, options, resources)
                    .await?;
                decode_into_array_bytes_target(&bytes, output_target)
                    .map_err(ArrayError::CodecError)
            }
        }

        /// Return the elements of the chunk at `chunk_indices` selected by `indexer`, through the cache.
        fn async_cached_partial_chunk_bytes<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> impl Future<Output = Result<Arc<ArrayBytes<'static>>, ArrayError>> + MaybeSend
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static;

        /// Decode the elements of the chunk at `chunk_indices` selected by `indexer` into `output_target`, through the
        /// cache.
        ///
        /// See [`SyncArrayCacheOps::cached_partial_chunk_into`].
        fn async_cached_partial_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> impl Future<Output = Result<(), ArrayError>> + MaybeSend
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
            async move {
                let bytes = self
                    .async_cached_partial_chunk_bytes(
                        array,
                        chunk_indices,
                        indexer,
                        options,
                        resources,
                    )
                    .await?;
                decode_into_array_bytes_target(&bytes, output_target)
                    .map_err(ArrayError::CodecError)
            }
        }

        /// Invalidate everything cached for `array`.
        fn async_invalidate_cached_all<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            array: &Array<TStorage>,
        ) -> impl Future<Output = ()> + MaybeSend;

        /// Invalidate everything cached for the chunk at `chunk_indices`.
        fn async_invalidate_cached_chunk<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
        ) -> impl Future<Output = ()> + MaybeSend;

        /// Invalidate everything cached for the chunks of `chunks`.
        fn async_invalidate_cached_chunks<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            array: &Array<TStorage>,
            chunks: &dyn Indexer,
        ) -> impl Future<Output = ()> + MaybeSend {
            async move {
                for chunk_indices in chunks.iter_indices() {
                    self.async_invalidate_cached_chunk(array, &chunk_indices)
                        .await;
                }
            }
        }

        /// Invalidate what is cached for the `array_subset` of `array`.
        ///
        /// The default implementation invalidates the intersecting chunks, or everything if they cannot be determined.
        ///
        /// # Errors
        /// Returns an [`ArrayError`] if `array_subset` has an incompatible dimensionality.
        fn async_invalidate_cached_array_subset<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            array: &Array<TStorage>,
            array_subset: &dyn ArraySubsetTraits,
        ) -> impl Future<Output = Result<(), ArrayError>> + MaybeSend {
            async move {
                if let Some(chunks) = array.chunks_in_array_subset(array_subset)? {
                    self.async_invalidate_cached_chunks(array, &chunks).await;
                } else {
                    self.async_invalidate_cached_all(array).await;
                }
                Ok(())
            }
        }

        /// Invalidate what is cached for the elements selected by `indexer` in the chunk at `chunk_indices`.
        ///
        /// The default implementation invalidates the whole chunk.
        fn async_invalidate_cached_chunk_indexer<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
        ) -> impl Future<Output = ()> + MaybeSend {
            let _ = indexer;
            self.async_invalidate_cached_chunk(array, chunk_indices)
        }
    }

    #[cfg(feature = "async")]
    impl<C: AsyncChunkCache + ?Sized> AsyncArrayCacheOps for C {
        async fn async_cached_partial_decoder<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
            C::Value::async_partial_decoder(self, array, chunk_indices, options, resources).await
        }

        async fn async_cached_chunk_bytes_if_exists<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
            C::Value::async_retrieve_chunk_bytes_if_exists(
                self,
                array,
                chunk_indices,
                options,
                resources,
            )
            .await
        }

        async fn async_cached_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
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

        async fn async_cached_partial_chunk_bytes<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
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

        async fn async_cached_partial_chunk_into<TStorage>(
            &self,
            array: &Array<TStorage>,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            output_target: &mut ArrayBytesTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError>
        where
            TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        {
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

        async fn async_invalidate_cached_all<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            _array: &Array<TStorage>,
        ) {
            self.invalidate().await;
        }

        async fn async_invalidate_cached_chunk<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            _array: &Array<TStorage>,
            chunk_indices: &[u64],
        ) {
            self.invalidate_chunk(chunk_indices).await;
        }

        async fn async_invalidate_cached_chunks<TStorage: ?Sized + MaybeSend + MaybeSync>(
            &self,
            _array: &Array<TStorage>,
            chunks: &dyn Indexer,
        ) {
            self.invalidate_chunks(chunks).await;
        }
    }
}

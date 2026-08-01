use std::sync::Arc;

#[cfg(feature = "async")]
use super::AsyncChunkCacheType;
use super::{ChunkCache, SyncChunkCacheType};
use crate::array::{Array, ArrayBytes, ArrayError, ChunkShape, ChunkShapeTraits, CodecOptions};
use zarrs_codec::CodecError;
#[cfg(feature = "async")]
use zarrs_storage::AsyncReadableStorageTraits;
use zarrs_storage::{ReadableStorageTraits, StorageError};

mod decoded;
mod encoded;
mod partial_decoder;
#[cfg(feature = "async")]
mod partial_decoder_async;

pub(super) fn cache_error(error: Arc<ArrayError>) -> ArrayError {
    Arc::try_unwrap(error)
        .unwrap_or_else(|error| ArrayError::StorageError(StorageError::from(error.to_string())))
}

pub(super) fn validate_chunk_indices<TStorage: ?Sized>(
    array: &Array<TStorage>,
    chunk_indices: &[u64],
) -> Result<ChunkShape, ArrayError> {
    if chunk_indices.len() != array.dimensionality()
        || chunk_indices
            .iter()
            .zip(array.chunk_grid_shape())
            .any(|(&index, &size)| index >= size)
    {
        return Err(ArrayError::InvalidChunkGridIndicesError(
            chunk_indices.to_vec(),
        ));
    }
    array.chunk_shape(chunk_indices)
}

pub(crate) fn fill_value_bytes(
    array: &Array<impl ?Sized>,
    num_elements: u64,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
    Ok(
        ArrayBytes::new_fill_value(array.data_type(), num_elements, array.fill_value())
            .map_err(CodecError::from)
            .map_err(ArrayError::from)?
            .into(),
    )
}

pub(crate) fn retrieve_chunk_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
    C::Value: SyncChunkCacheType,
{
    if let Some(bytes) =
        C::Value::retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options)?
    {
        Ok(bytes)
    } else {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        fill_value_bytes(array, chunk_shape.num_elements_u64())
    }
}

#[cfg(feature = "async")]
pub(crate) async fn async_retrieve_chunk_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    chunk_indices: &[u64],
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
    C::Value: AsyncChunkCacheType,
{
    if let Some(bytes) =
        C::Value::async_retrieve_chunk_bytes_if_exists(cache, array, chunk_indices, options).await?
    {
        Ok(bytes)
    } else {
        let chunk_shape = validate_chunk_indices(array, chunk_indices)?;
        fill_value_bytes(array, chunk_shape.num_elements_u64())
    }
}

/// Return a cached value or insert the value returned by `f`.
///
/// A chunk fetched by `f` is dropped from the cache if an invalidation overlapped the fetch, so
/// that a value fetched before a concurrent write is not left cached. Such a fetch still returns
/// its value to the caller.
pub(super) fn sync_try_get_or_insert_with<C, F>(
    cache: &C,
    chunk_indices: &[u64],
    f: F,
) -> Result<C::Value, Arc<ArrayError>>
where
    C: ChunkCache + ?Sized,
    F: FnOnce() -> Result<C::Value, ArrayError>,
{
    // Read the generation inside `f` rather than before the call, so that a cache hit (where `f`
    // is never invoked) leaves the generation `None` and skips the check below. Checking on a hit
    // would drop a perfectly good entry whenever any unrelated write raced the read.
    let generation = std::cell::Cell::new(None);
    let value = cache.try_get_or_insert_with(chunk_indices.to_vec(), || {
        generation.set(Some(cache.invalidation_generation()));
        f()
    })?;
    if let Some(generation) = generation.get() {
        cache.retain_since(chunk_indices, generation);
    }
    Ok(value)
}

/// Return a cached value or insert the value returned by the asynchronous `f`.
///
/// Unlike [`sync_try_get_or_insert_with`], concurrent calls for an uncached chunk may each invoke
/// `f`; at most one of the retrieved values is retained by the cache.
///
/// As in [`sync_try_get_or_insert_with`], a chunk fetched by `f` is dropped from the cache if an
/// invalidation overlapped the fetch, in which case *none* of the retrieved values is retained.
#[cfg(feature = "async")]
pub(super) async fn async_try_get_or_insert_with<C, F>(
    cache: &C,
    chunk_indices: &[u64],
    f: F,
) -> Result<C::Value, Arc<ArrayError>>
where
    C: ChunkCache + ?Sized,
    F: AsyncFnOnce() -> Result<C::Value, ArrayError>,
{
    if let Some(value) = cache.get(chunk_indices) {
        return Ok(value);
    }
    // Unlike the synchronous case the fetch happens outside `try_get_or_insert_with`, so the
    // generation has to be read here, before the await.
    let generation = cache.invalidation_generation();
    let value = f().await.map_err(Arc::new)?;
    let value = cache.try_get_or_insert_with(chunk_indices.to_vec(), move || Ok(value))?;
    cache.retain_since(chunk_indices, generation);
    Ok(value)
}

/// Expose an in-memory synchronous partial decoder as an asynchronous partial decoder.
///
/// The wrapped decoder must not perform storage operations (i.e. it must be backed by
/// in-memory chunk data), since its methods are called directly from an asynchronous context.
#[cfg(feature = "async")]
pub(super) struct SyncPartialDecoderAsAsync(pub Arc<dyn zarrs_codec::ArrayPartialDecoderTraits>);

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl zarrs_codec::AsyncArrayPartialDecoderTraits for SyncPartialDecoderAsAsync {
    fn data_type(&self) -> &zarrs_data_type::DataType {
        self.0.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.0.exists()
    }

    fn size_held(&self) -> usize {
        self.0.size_held()
    }

    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<zarrs_chunk_grid::ChunkGrid>>, CodecError> {
        self.0.local_subchunk_grids(options)
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.0.partial_decode(indexer, options)
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn crate::array::Indexer,
        output_target: zarrs_codec::ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.0.partial_decode_into(indexer, output_target, options)
    }

    fn supports_partial_decode(&self) -> bool {
        self.0.supports_partial_decode()
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use super::ChunkCache;
    use crate::array::chunk_cache::{
        ChunkCacheEncodedLruChunkLimit, ChunkCacheEncodedLruSizeLimit, ChunkCacheTypeEncoded,
    };

    fn value(byte: u8) -> ChunkCacheTypeEncoded {
        Some(Arc::new(Cow::Owned(vec![byte])))
    }

    fn bytes(value: &ChunkCacheTypeEncoded) -> Option<Vec<u8>> {
        value.as_ref().map(|value| value.to_vec())
    }

    /// A retrieval that fetches a chunk must not leave that chunk cached if a write invalidated
    /// it while the fetch was in flight. The retrieval may still *return* the pre-write value.
    #[cfg(feature = "async")]
    async fn async_invalidation_during_fetch_is_not_reinserted<C>(cache: C)
    where
        C: ChunkCache<Value = ChunkCacheTypeEncoded>,
    {
        use futures::{pin_mut, poll};

        let (gate_tx, gate_rx) = futures::channel::oneshot::channel::<()>();

        // Park a retrieval inside its fetch, as though it were waiting on the store.
        let retrieve = super::async_try_get_or_insert_with(&cache, &[0], async || {
            gate_rx.await.unwrap();
            Ok(value(1))
        });
        pin_mut!(retrieve);
        assert!(poll!(&mut retrieve).is_pending());

        // A concurrent write stores new data and invalidates the chunk.
        cache.invalidate_chunk(&[0]);

        // The parked fetch now completes with the pre-write value.
        gate_tx.send(()).unwrap();
        let retrieved = retrieve.await.unwrap();

        // Returning the pre-write value is permitted, but it must not have been cached.
        assert_eq!(bytes(&retrieved), Some(vec![1]));
        assert_eq!(
            cache.len(),
            0,
            "pre-write value was reinserted after invalidation"
        );

        // A subsequent retrieval sees the post-write value, and caches it.
        let retrieved = super::async_try_get_or_insert_with(&cache, &[0], async || Ok(value(2)))
            .await
            .unwrap();
        assert_eq!(bytes(&retrieved), Some(vec![2]));
        assert_eq!(cache.len(), 1);
        assert_eq!(bytes(&cache.get(&[0]).unwrap()), Some(vec![2]));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_invalidation_during_fetch_is_not_reinserted_chunk_limit() {
        async_invalidation_during_fetch_is_not_reinserted(ChunkCacheEncodedLruChunkLimit::new(4))
            .await;
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_invalidation_during_fetch_is_not_reinserted_size_limit() {
        async_invalidation_during_fetch_is_not_reinserted(ChunkCacheEncodedLruSizeLimit::new(1024))
            .await;
    }

    /// The synchronous equivalent. Gated off `wasm32`, where the backing cache holds a mutex
    /// across the fetch and `invalidate_chunk` would deadlock against the parked closure.
    #[cfg(not(target_arch = "wasm32"))]
    fn sync_invalidation_during_fetch_is_not_reinserted<C>(cache: C)
    where
        C: ChunkCache<Value = ChunkCacheTypeEncoded> + Sync,
    {
        use std::sync::mpsc;

        let (started_tx, started_rx) = mpsc::channel();
        let (gate_tx, gate_rx) = mpsc::channel();

        let cache = &cache;
        std::thread::scope(|scope| {
            scope.spawn(move || {
                let retrieved = super::sync_try_get_or_insert_with(cache, &[0], || {
                    started_tx.send(()).unwrap();
                    gate_rx.recv().unwrap();
                    Ok(value(1))
                })
                .unwrap();
                assert_eq!(bytes(&retrieved), Some(vec![1]));
            });

            started_rx.recv().unwrap();
            cache.invalidate_chunk(&[0]);
            gate_tx.send(()).unwrap();
        });

        assert_eq!(
            cache.len(),
            0,
            "pre-write value was reinserted after invalidation"
        );

        let retrieved = super::sync_try_get_or_insert_with(cache, &[0], || Ok(value(2))).unwrap();
        assert_eq!(bytes(&retrieved), Some(vec![2]));
        assert_eq!(cache.len(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sync_invalidation_during_fetch_is_not_reinserted_chunk_limit() {
        sync_invalidation_during_fetch_is_not_reinserted(ChunkCacheEncodedLruChunkLimit::new(4));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sync_invalidation_during_fetch_is_not_reinserted_size_limit() {
        sync_invalidation_during_fetch_is_not_reinserted(ChunkCacheEncodedLruSizeLimit::new(1024));
    }

    /// A cache hit must not be dropped just because an unrelated write raced the read.
    #[test]
    fn unrelated_invalidation_does_not_drop_a_cache_hit() {
        let cache = ChunkCacheEncodedLruChunkLimit::new(4);
        super::sync_try_get_or_insert_with(&cache, &[0], || Ok(value(1))).unwrap();
        assert_eq!(cache.len(), 1);

        for _ in 0..8 {
            cache.invalidate_chunk(&[1]); // a write to a different chunk
            super::sync_try_get_or_insert_with(&cache, &[0], || {
                panic!("chunk 0 should still be cached")
            })
            .unwrap();
        }
        assert_eq!(cache.len(), 1);
    }

    /// Every invalidation must advance the generation, even when nothing was removed, otherwise
    /// an in-flight fetch cannot tell that it raced a write.
    #[test]
    fn invalidation_advances_the_generation_even_when_nothing_is_cached() {
        let cache = ChunkCacheEncodedLruChunkLimit::new(4);
        assert!(cache.is_empty());

        let generation = cache.invalidation_generation();
        assert!(!cache.invalidate_chunk(&[0]));
        assert_ne!(cache.invalidation_generation(), generation);

        let generation = cache.invalidation_generation();
        assert_eq!(cache.invalidate(), 0);
        assert_ne!(cache.invalidation_generation(), generation);
    }

    /// `retain_since` must not itself advance the generation, or conservative removals cascade
    /// into other in-flight fetches.
    #[test]
    fn retain_since_does_not_advance_the_generation() {
        let cache = ChunkCacheEncodedLruChunkLimit::new(4);
        super::sync_try_get_or_insert_with(&cache, &[0], || Ok(value(1))).unwrap();

        let generation = cache.invalidation_generation();
        assert!(cache.retain_since(&[0], generation));
        assert_eq!(cache.invalidation_generation(), generation);
        assert_eq!(cache.len(), 1);

        cache.invalidate_chunk(&[1]);
        let generation_after_write = cache.invalidation_generation();
        assert!(!cache.retain_since(&[0], generation));
        assert_eq!(cache.invalidation_generation(), generation_after_write);
        assert_eq!(cache.len(), 0);
    }
}

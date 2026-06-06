use std::sync::Arc;

use super::Array;

/// A cached array wrapper.
///
/// `ArrayCached` combines an [`Array`] with a chunk cache. Read operations use
/// the cache, while write/update operations delegate to the inner array and
/// invalidate cached chunks that may have changed.
///
/// ## Cache Behavior
///
/// ### Read Operations
///
/// Most read operations (`retrieve_*`) check the cache first and populate it on
/// a miss. The specific cache behavior depends on the cache type (encoded,
/// decoded, or partial decoder).
///
/// #### Encoded Chunk Retrieval
///
/// Methods that retrieve encoded chunk bytes (`retrieve_encoded_chunk`,
/// `retrieve_encoded_chunks`, `retrieve_encoded_subchunk`) bypass the cache
/// and delegate directly to the inner [`Array`].
///
/// ### Write Operations
///
/// Write operations delegate to the inner [`Array`] and invalidate cache entries:
///
/// - `store_metadata` / `erase_metadata` — invalidates the entire cache.
/// - `store_chunk` / `erase_chunk` — invalidates the affected chunk.
/// - `store_chunks` / `erase_chunks` — invalidates the affected chunks.
/// - `store_encoded_chunk` — invalidates the affected chunk.
///
/// ### Update Operations
///
/// Update operations (partial writes) also delegate to the inner [`Array`]:
///
/// - `store_chunk_subset` — invalidates the affected chunk.
/// - `store_array_subset` — invalidates all intersecting chunks, or the entire
///   cache if the affected chunks cannot be determined.
/// - `compact_chunk` — invalidates the chunk only if compaction occurred.
#[derive(Debug)]
pub struct ArrayCached<TStorage: ?Sized, C> {
    array: Arc<Array<TStorage>>,
    cache: C,
}

impl<TStorage: ?Sized, C> ArrayCached<TStorage, C> {
    /// Create a new cached array wrapper.
    #[must_use]
    pub fn new(array: Arc<Array<TStorage>>, cache: C) -> Self {
        Self { array, cache }
    }

    /// Return the inner array.
    #[must_use]
    pub fn array(&self) -> &Arc<Array<TStorage>> {
        &self.array
    }

    /// Return the chunk cache.
    #[must_use]
    pub const fn cache(&self) -> &C {
        &self.cache
    }

    /// Split into the inner array and cache.
    #[must_use]
    pub fn into_inner(self) -> (Arc<Array<TStorage>>, C) {
        (self.array, self.cache)
    }
}

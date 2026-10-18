//! Array operation traits.
//!
//! These traits decouple the array operations from the [`Array`] type, so that they can also be
//! implemented by wrappers such as [`ArrayCached`].
//!
//! # Resources
//!
//! Operations that encode, decode, or access multiple chunks take a [`Resources`] argument.
//! It sets the resources (e.g. concurrency) that the operation may use, and is chosen per call.
//!
//! # Codec and metadata options
//!
//! Operations do not take options as arguments. They use the options stored on the array:
//! [`codec_options`](ArrayOps::codec_options), [`metadata_options`](ArrayOps::metadata_options)
//! and [`metadata_erase_version`](ArrayOps::metadata_erase_version).
//!
//! To use different options, derive an array that carries them with
//! [`with_codec_options`](ArrayOps::with_codec_options),
//! [`with_metadata_options`](ArrayOps::with_metadata_options) or
//! [`with_metadata_erase_version`](ArrayOps::with_metadata_erase_version). These methods consume
//! the receiver. Clone explicitly when retaining the original; [`Array`] shares its storage, data
//! type, chunk grid, codec chain and metadata behind an [`Arc`], and [`ArrayCached`] additionally
//! shares its chunk cache.
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use zarrs::array::{Array, ArrayOps, ArrayReadOps, CodecOptions, Resources};
//! # let store = Arc::new(zarrs::storage::store::MemoryStore::new());
//! # let array = Array::open(store, "/group/array")?;
//! // Default options and resources
//! let chunk: Vec<f32> = array.retrieve_chunk(&[0, 0], &Resources::default())?;
//!
//! // Limited resources
//! let resources = Resources::default().with_compute_target(1);
//! let chunk: Vec<f32> = array.retrieve_chunk(&[0, 0], &resources)?;
//!
//! // Overridden options. Derive once and reuse, rather than per operation.
//! let tuned = array.with_codec_options(CodecOptions::default().with_validate_checksums(false));
//! let chunk: Vec<f32> = tuned.retrieve_chunk(&[0, 0], &resources)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Because these are [`ArrayOps`] methods, code generic over the operation traits can override
//! options too:
//!
//! ```rust
//! # use zarrs::array::{ArrayOps, ArrayWriteOps};
//! # use zarrs::config::MetadataEraseVersion;
//! fn erase_all_metadata<A: ArrayWriteOps>(
//!     array: &A,
//! ) -> Result<(), zarrs::storage::StorageError> {
//!     array
//!         .with_metadata_erase_version(MetadataEraseVersion::All)
//!         .erase_metadata()
//! }
//! ```
//!
//! [`ArrayMutOps::set_codec_options`] can instead update an array through a mutable reference.

use std::sync::Arc;

use super::chunk_cache::ChunkCache;
use super::{
    Array, ArrayBuilder, ArrayCached, ArrayCreateError, ArrayError, ArrayIndices, ArrayMetadata,
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits,
    ArrayToBytesCodecSubchunkingTraits, ArrayToBytesCodecTraits, ChunkGrid, ChunkGridDecoded,
    ChunkGridDecodedRef, ChunkKeyEncoding, ChunkShape, ChunkShapeTraits, CodecChain,
    CodecChainBound, CodecCreateError, CodecOptions, CodecSpecificOptions, DataType, DimensionName,
    EncodedSubchunk, FillValue, FromArrayBytes, IncompatibleDimensionalityError, Indexer,
    IntoArrayBytes, NodePath, Resources, StorageTransformerChain,
};
use crate::config::MetadataEraseVersion;
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits, AsyncWritableStorageTraits,
    MaybeSend,
};
use zarrs_storage::{
    ReadableStorageTraits, ReadableWritableStorageTraits, StorageError, StoreKey,
    WritableStorageTraits,
};

/// The storage of the chunks of an operation.
///
/// The storage of an array without storage transformers is borrowed, rather than wrapped for each operation.
enum OpStorage<'s, TStorage: ?Sized, TTransformed: ?Sized> {
    /// The storage of the array.
    Array(&'s TStorage),
    /// The storage created by the storage transformers of the array.
    Transformed(Arc<TTransformed>),
}

/// Evaluate `$body` with `$storage` bound to a reference to the storage of the [`OpStorage`] `$op_storage`.
macro_rules! with_op_storage {
    ($op_storage:expr, |$storage:ident| $body:expr) => {
        match &$op_storage {
            $crate::array::array_ops::OpStorage::Array($storage) => {
                let $storage = &**$storage;
                $body
            }
            $crate::array::array_ops::OpStorage::Transformed($storage) => {
                let $storage = &**$storage;
                $body
            }
        }
    };
}

mod array_mut_ops;
mod array_mut_ops_array;
#[allow(clippy::module_inception)]
mod array_ops;
mod array_ops_array;
mod array_ops_array_cached;
mod array_read_ops;
mod array_read_ops_array;
mod array_read_ops_array_cached;
mod array_read_ops_common;
mod array_read_ops_planned;
mod array_update_ops;
mod array_update_ops_array;
mod array_update_ops_array_cached;
mod array_write_ops;
mod array_write_ops_array;
mod array_write_ops_array_cached;
mod array_write_ops_planned;
#[cfg(feature = "async")]
mod async_array_read_ops;
#[cfg(feature = "async")]
mod async_array_read_ops_array;
#[cfg(feature = "async")]
mod async_array_read_ops_array_cached;
#[cfg(feature = "async")]
mod async_array_read_ops_common;
#[cfg(feature = "async")]
mod async_array_update_ops;
#[cfg(feature = "async")]
mod async_array_update_ops_array;
#[cfg(feature = "async")]
mod async_array_update_ops_array_cached;
#[cfg(feature = "async")]
mod async_array_write_ops;
#[cfg(feature = "async")]
mod async_array_write_ops_array;
#[cfg(feature = "async")]
mod async_array_write_ops_array_cached;

pub use array_mut_ops::ArrayMutOps;
pub use array_ops::ArrayOps;
pub use array_read_ops::ArrayReadOps;
pub use array_update_ops::ArrayUpdateOps;
pub use array_write_ops::ArrayWriteOps;
#[cfg(feature = "async")]
pub use async_array_read_ops::AsyncArrayReadOps;
#[cfg(feature = "async")]
pub use async_array_update_ops::AsyncArrayUpdateOps;
#[cfg(feature = "async")]
pub use async_array_write_ops::AsyncArrayWriteOps;

/// The subset of the array of the chunk at `chunk_indices` of `chunk_grid`.
fn chunk_grid_subset(
    chunk_grid: &ChunkGrid,
    chunk_indices: &[u64],
) -> Result<ArraySubset, ArrayError> {
    chunk_grid
        .subset(chunk_indices)
        .map_err(|_| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))
}

/// Split `resources` between `num_chunks` chunks retrieved concurrently.
///
/// Returns the number of chunks to retrieve concurrently, and the resources of each.
fn split_chunk_resources(num_chunks: usize, resources: &Resources) -> (usize, Resources) {
    let concurrent_chunks = num_chunks.clamp(1, resources.compute_limit());
    (
        concurrent_chunks,
        resources.with_compute_share(concurrent_chunks),
    )
}

/// Return true if `subset` covers the whole chunk with shape `chunk_shape`.
fn subset_is_whole_chunk(subset: &dyn ArraySubsetTraits, chunk_shape: &[u64]) -> bool {
    subset.start().iter().all(|&o| o == 0) && subset.shape().as_ref() == chunk_shape
}

/// Return `chunk` if it exists, otherwise the fill value of the chunk at `chunk_indices`.
fn chunk_or_fill_value<A: ArrayOps + ?Sized, T: FromArrayBytes>(
    array: &A,
    chunk_indices: &[u64],
    chunk: Option<T>,
) -> Result<T, ArrayError> {
    if let Some(chunk) = chunk {
        return Ok(chunk);
    }
    let chunk_shape = array.chunk_shape(chunk_indices)?;
    let bytes = super::ArrayBytes::new_fill_value(
        array.data_type(),
        chunk_shape.num_elements_u64(),
        array.fill_value(),
    )
    .map_err(zarrs_codec::CodecError::from)
    .map_err(ArrayError::from)?;
    T::from_array_bytes(
        bytes,
        bytemuck::must_cast_slice(&chunk_shape),
        array.data_type(),
    )
}

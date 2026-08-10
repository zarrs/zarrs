//! Array operation traits.
//!
//! These traits decouple the array operations from the [`Array`] type, so that they can also be
//! implemented by wrappers such as [`ArrayCached`].
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
//! # use zarrs::array::{Array, ArrayOps, ArrayReadOps, CodecOptions};
//! # let store = Arc::new(zarrs::storage::store::MemoryStore::new());
//! # let array = Array::open(store, "/group/array")?;
//! // Default options
//! let chunk: Vec<f32> = array.retrieve_chunk(&[0, 0])?;
//!
//! // Overridden options. Derive once and reuse, rather than per operation.
//! let tuned = array.with_codec_options(CodecOptions::default().with_concurrent_target(1));
//! let chunk: Vec<f32> = tuned.retrieve_chunk(&[0, 0])?;
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
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkGridDecoded,
    ChunkGridDecodedRef, ChunkKeyEncoding, ChunkShape, ChunkShapeTraits, CodecChain,
    CodecChainBound, CodecCreateError, CodecOptions, CodecSpecificOptions, DataType, DimensionName,
    FillValue, FromArrayBytes, IncompatibleDimensionalityError, IntoArrayBytes, NodePath,
    StorageTransformerChain,
};
use crate::config::MetadataEraseVersion;
use zarrs_codec::{ArrayCodecTraits, RecommendedConcurrency};
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits, AsyncWritableStorageTraits,
    MaybeSend,
};
use zarrs_storage::{
    ReadableStorageTraits, ReadableWritableStorageTraits, StorageError, StoreKey,
    WritableStorageTraits,
};

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
mod array_update_ops;
mod array_update_ops_array;
mod array_update_ops_array_cached;
mod array_write_ops;
mod array_write_ops_array;
mod array_write_ops_array_cached;
#[cfg(feature = "async")]
mod async_array_read_ops;
#[cfg(feature = "async")]
mod async_array_read_ops_array;
#[cfg(feature = "async")]
mod async_array_read_ops_common;
#[cfg(feature = "async")]
mod async_array_update_ops;
#[cfg(feature = "async")]
mod async_array_update_ops_array;
#[cfg(feature = "async")]
mod async_array_write_ops;
#[cfg(feature = "async")]
mod async_array_write_ops_array;

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

fn recommended_codec_concurrency(
    array: &impl ArrayOps,
    chunk_shape: &[std::num::NonZeroU64],
) -> Result<RecommendedConcurrency, ArrayError> {
    Ok(array.codecs_bound().recommended_concurrency(chunk_shape)?)
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

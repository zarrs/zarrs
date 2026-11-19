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
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkGridDecoded,
    ChunkGridDecodedRef, ChunkKeyEncoding, ChunkShape, ChunkShapeTraits, CodecChain,
    CodecChainBound, CodecCreateError, CodecOptions, CodecSpecificOptions, DataType, DimensionName,
    FillValue, FromArrayBytes, IncompatibleDimensionalityError, Indexer, IntoArrayBytes, NodePath,
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
mod array_storage;
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

fn recommended_codec_concurrency(
    array: &impl ArrayOps,
    chunk_shape: &[std::num::NonZeroU64],
) -> Result<RecommendedConcurrency, ArrayError> {
    Ok(array.codecs_bound().recommended_concurrency(chunk_shape)?)
}

/// Return the fill value of `array` for `num_elements` elements.
fn fill_value_bytes(
    array: &(impl ArrayOps + ?Sized),
    num_elements: u64,
) -> Result<super::ArrayBytes<'static>, ArrayError> {
    super::ArrayBytes::new_fill_value(array.data_type(), num_elements, array.fill_value())
        .map_err(zarrs_codec::CodecError::from)
        .map_err(ArrayError::from)
}

/// Return the chunks of `array` intersecting `array_subset`, or an error if it is not an array subset of `array`.
fn array_subset_chunks(
    array: &(impl ArrayOps + ?Sized),
    array_subset: &dyn ArraySubsetTraits,
) -> Result<ArraySubset, ArrayError> {
    if array_subset.dimensionality() == array.dimensionality()
        && let Some(chunks) = array.chunks_in_array_subset(array_subset)?
    {
        return Ok(chunks);
    }
    Err(ArrayError::InvalidArraySubset(
        Box::new(array_subset.to_array_subset()),
        array.shape().to_vec(),
    ))
}

/// Return an error if `indexer` is an array subset that is not within `chunk_shape`.
pub(in crate::array) fn validate_partial_chunk_subset(
    indexer: &dyn Indexer,
    chunk_shape: &[u64],
) -> Result<(), ArrayError> {
    if let Some(chunk_subset) = indexer.as_array_subset()
        && !chunk_subset.inbounds_shape(chunk_shape)
    {
        return Err(ArrayError::InvalidArraySubset(
            Box::new(chunk_subset.to_array_subset()),
            chunk_shape.to_vec(),
        ));
    }
    Ok(())
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
    T::from_array_bytes(
        fill_value_bytes(array, chunk_shape.num_elements_u64())?,
        bytemuck::must_cast_slice(&chunk_shape),
        array.data_type(),
    )
}

/// Validate `data`'s shape (if it has a layout) and convert it into validated [`ArrayBytes`](super::ArrayBytes).
fn into_validated_array_bytes<'a, T: super::IntoArrayBytes<'a>>(
    data: T,
    data_type: &DataType,
    shape: &[u64],
    num_elements: u64,
) -> Result<super::ArrayBytes<'a>, ArrayError> {
    data.validate_shape(shape)?;
    let bytes = data.into_array_bytes(data_type)?;
    bytes.validate(num_elements, data_type)?;
    Ok(bytes)
}

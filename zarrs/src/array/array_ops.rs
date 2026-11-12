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

use crate::array::ElementOrder;
use std::sync::Arc;

use super::array_bytes_internal::{FixedDecodeBuffers, reordered};
use zarrs_codec::ArrayBytesDecodeIntoTarget;

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

/// Return `T` from F-ordered `buffers` of `shape`.
///
/// # Safety
/// Every element of the target returned by [`FixedDecodeBuffers::target_in_order`] must have been written.
unsafe fn from_f_order_buffers<T: FromArrayBytes>(
    buffers: FixedDecodeBuffers,
    shape: &[u64],
    data_type: &DataType,
) -> Result<T, ArrayError> {
    // SAFETY: the caller guarantees that every element has been written
    let bytes = unsafe { buffers.into_array_bytes() };
    T::from_array_bytes(bytes, shape, data_type, ElementOrder::F)
}

/// Return `T` from C-ordered `bytes` of `shape` on an array with `order`.
///
/// `T` is in `order` if `order` places the elements of `shape` as C order does, and in C order otherwise.
fn from_c_order_bytes<T: FromArrayBytes>(
    bytes: super::ArrayBytes<'static>,
    shape: &[u64],
    data_type: &DataType,
    order: ElementOrder,
) -> Result<T, ArrayError> {
    T::from_array_bytes(bytes, shape, data_type, order.for_c_order_elements(shape))
}

/// Return `T` from shared C-ordered `bytes` of `shape` on an array with `order`.
///
/// This is the shared counterpart of [`from_c_order_bytes`].
fn from_c_order_bytes_arc<T: FromArrayBytes>(
    bytes: std::sync::Arc<super::ArrayBytes<'static>>,
    shape: &[u64],
    data_type: &DataType,
    order: ElementOrder,
) -> Result<T, ArrayError> {
    T::from_array_bytes_arc(bytes, shape, data_type, order.for_c_order_elements(shape))
}

/// Return `T` with the elements of C-ordered `bytes` of `shape` in F order.
fn from_c_order_in_f_order<T: FromArrayBytes>(
    bytes: &super::ArrayBytes<'_>,
    shape: &[u64],
    data_type: &DataType,
) -> Result<T, ArrayError> {
    let bytes = reordered(bytes, data_type, shape, ElementOrder::C, ElementOrder::F)?;
    T::from_array_bytes(bytes, shape, data_type, ElementOrder::F)
}

/// Return the shape of the chunk at `chunk_indices` of `array`.
fn chunk_shape_u64(array: &impl ArrayOps, chunk_indices: &[u64]) -> Result<Vec<u64>, ArrayError> {
    Ok(bytemuck::allocation::cast_vec(
        array.chunk_shape(chunk_indices)?,
    ))
}

/// Return the element order of the elements selected by `indexer` on an array with `order`.
///
/// Only array subsets have a multidimensional output shape with an element order, so the elements of other indexers are in C order.
fn indexer_element_order(indexer: &dyn Indexer, order: ElementOrder) -> ElementOrder {
    if indexer.as_array_subset().is_some() {
        order
    } else {
        ElementOrder::C
    }
}

/// Return an error if `indexer` is an array subset that is not within `chunk_shape`.
fn validate_partial_chunk_subset(
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

/// Retrieve `T` with the elements of `shape` in `order`.
///
/// `shape` is only evaluated for [`ElementOrder::F`].
/// If `order` reorders the elements of `shape`, fixed length data is decoded by `retrieve_into` into buffers with an F-ordered layout, and variable length data is retrieved in C order by `retrieve_c_order` and reordered.
/// Otherwise, the C and F orders of the elements coincide, and `T` is returned by `retrieve`.
fn retrieve_in_order<T, S, B>(
    data_type: &DataType,
    order: ElementOrder,
    shape: impl FnOnce() -> Result<S, ArrayError>,
    retrieve_into: impl FnOnce(&mut ArrayBytesDecodeIntoTarget<'_>) -> Result<(), ArrayError>,
    retrieve_c_order: impl FnOnce() -> Result<B, ArrayError>,
    retrieve: impl FnOnce() -> Result<T, ArrayError>,
) -> Result<T, ArrayError>
where
    T: FromArrayBytes,
    S: AsRef<[u64]>,
    B: std::borrow::Borrow<super::ArrayBytes<'static>>,
{
    if order == ElementOrder::F {
        let shape = shape()?;
        let shape = shape.as_ref();
        if order.is_reordered(shape) {
            return if let Some(mut buffers) = FixedDecodeBuffers::new(data_type, shape) {
                retrieve_into(&mut buffers.target_in_order(ElementOrder::F)?)?;
                // SAFETY: a successful retrieval into a target writes every element
                unsafe { from_f_order_buffers(buffers, shape, data_type) }
            } else {
                from_c_order_in_f_order(retrieve_c_order()?.borrow(), shape, data_type)
            };
        }
    }
    retrieve()
}

/// Retrieve `T` with the elements of `shape` in `order` asynchronously.
///
/// This is the asynchronous counterpart of [`retrieve_in_order`].
#[cfg(feature = "async")]
async fn async_retrieve_in_order<T, S, B>(
    data_type: &DataType,
    order: ElementOrder,
    shape: impl FnOnce() -> Result<S, ArrayError>,
    retrieve_into: impl AsyncFnOnce(&mut ArrayBytesDecodeIntoTarget<'_>) -> Result<(), ArrayError>,
    retrieve_c_order: impl AsyncFnOnce() -> Result<B, ArrayError>,
    retrieve: impl AsyncFnOnce() -> Result<T, ArrayError>,
) -> Result<T, ArrayError>
where
    T: FromArrayBytes,
    S: AsRef<[u64]>,
    B: std::borrow::Borrow<super::ArrayBytes<'static>>,
{
    if order == ElementOrder::F {
        let shape = shape()?;
        let shape = shape.as_ref();
        if order.is_reordered(shape) {
            return if let Some(mut buffers) = FixedDecodeBuffers::new(data_type, shape) {
                retrieve_into(&mut buffers.target_in_order(ElementOrder::F)?).await?;
                // SAFETY: a successful retrieval into a target writes every element
                unsafe { from_f_order_buffers(buffers, shape, data_type) }
            } else {
                from_c_order_in_f_order(retrieve_c_order().await?.borrow(), shape, data_type)
            };
        }
    }
    retrieve().await
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
        array.element_order(),
    )
}

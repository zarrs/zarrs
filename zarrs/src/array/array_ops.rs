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
//! # use zarrs::array::{Array, ArrayOps, SyncArrayReadOps, CodecOptions, Resources};
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

use super::array_bytes_internal::{FixedDecodeBuffers, reordered};
use zarrs_codec::ArrayBytesTarget;

use super::{
    Array, ArrayBuilder, ArrayCached, ArrayCreateError, ArrayError, ArrayIndices, ArrayMetadata,
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkGridDecoded,
    ChunkGridDecodedRef, ChunkKeyEncoding, ChunkShape, ChunkShapeTraits, CodecChain,
    CodecChainBound, CodecCreateError, CodecOptions, CodecSpecificOptions, DataType, DimensionName,
    ElementOrder, FillValue, FromArrayBytes, IncompatibleDimensionalityError, Indexer,
    IntoArrayBytes, NodePath, StorageTransformerChain,
};
use crate::config::MetadataEraseVersion;
use zarrs_codec::{ArrayCodecTraits, RecommendedConcurrency};
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits, AsyncWritableStorageTraits,
    MaybeSend,
};
use zarrs_storage::{
    StorageError, StoreKey, SyncReadableStorageTraits, SyncReadableWritableStorageTraits,
    SyncWritableStorageTraits,
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
pub use array_read_ops::SyncArrayReadOps;
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
/// Every element of the target returned by [`FixedDecodeBuffers::target_in_order`] must have been written, with mask bytes of 0 or 1.
unsafe fn from_f_order_buffers<T: FromArrayBytes>(
    buffers: FixedDecodeBuffers,
    shape: &[u64],
    data_type: &DataType,
) -> Result<T, ArrayError> {
    // SAFETY: the caller guarantees that every element has been written, with mask bytes of 0 or 1
    let bytes = unsafe { buffers.into_array_bytes() };
    T::from_array_bytes(bytes, shape, data_type, ElementOrder::F)
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

/// C-ordered array bytes, owned or shared.
trait COrderBytes: std::borrow::Borrow<super::ArrayBytes<'static>> {
    /// Return `T` from these bytes of `shape`, with elements labelled `order`.
    fn into_elements<T: FromArrayBytes>(
        self,
        shape: &[u64],
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<T, ArrayError>;
}

impl COrderBytes for super::ArrayBytes<'static> {
    fn into_elements<T: FromArrayBytes>(
        self,
        shape: &[u64],
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<T, ArrayError> {
        T::from_array_bytes(self, shape, data_type, order)
    }
}

impl COrderBytes for Arc<super::ArrayBytes<'static>> {
    fn into_elements<T: FromArrayBytes>(
        self,
        shape: &[u64],
        data_type: &DataType,
        order: ElementOrder,
    ) -> Result<T, ArrayError> {
        T::from_array_bytes_arc(self, shape, data_type, order)
    }
}

/// Return the subchunk grid of `array` at `level`.
///
/// # Errors
/// Returns [`ArrayError::MissingSubchunkGrid`] if the subchunk grid at `level` is not globally resolvable.
pub(crate) fn subchunk_grid_at_level(
    array: &(impl ArrayOps + ?Sized),
    level: usize,
) -> Result<&ChunkGrid, ArrayError> {
    array
        .subchunk_grid_at_level(level)
        .as_chunk_grid()
        .ok_or(ArrayError::MissingSubchunkGrid)
}

/// Return `num_elements` elements of `fill_value`.
pub(crate) fn fill_value_bytes(
    data_type: &DataType,
    fill_value: &FillValue,
    num_elements: u64,
) -> Result<super::ArrayBytes<'static>, ArrayError> {
    super::ArrayBytes::new_fill_value(data_type, num_elements, fill_value)
        .map_err(zarrs_codec::CodecError::from)
        .map_err(ArrayError::from)
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

/// Retrieve `T` with the elements of `shape` in `order`.
///
/// `shape` must be validated by the caller, since it sizes the output of F order before any retrieval.
/// If `order` reorders the elements of `shape`, fixed length data is decoded by `retrieve_into` into buffers with an F-ordered layout, and variable length data is retrieved in C order by `retrieve_c_order` and reordered.
/// Otherwise, the C and F orders of the elements coincide, and the elements retrieved by `retrieve_c_order` are labelled `order`.
fn retrieve_in_order<T, B>(
    data_type: &DataType,
    order: ElementOrder,
    shape: &[u64],
    retrieve_into: impl FnOnce(&mut ArrayBytesTarget<'_>) -> Result<(), ArrayError>,
    retrieve_c_order: impl FnOnce() -> Result<B, ArrayError>,
) -> Result<T, ArrayError>
where
    T: FromArrayBytes,
    B: COrderBytes,
{
    if !order.is_reordered(shape) {
        return retrieve_c_order()?.into_elements(shape, data_type, order);
    }
    if let Some(mut buffers) = FixedDecodeBuffers::new(data_type, shape) {
        retrieve_into(&mut buffers.target_in_order(ElementOrder::F)?)?;
        // SAFETY: a successful retrieval into a target writes every element, and only 0 or 1 to masks (the `decode_into` contract)
        unsafe { from_f_order_buffers(buffers, shape, data_type) }
    } else {
        from_c_order_in_f_order(retrieve_c_order()?.borrow(), shape, data_type)
    }
}

/// Retrieve `T` with the elements of `shape` in `order` asynchronously.
///
/// This is the asynchronous counterpart of [`retrieve_in_order`].
#[cfg(feature = "async")]
async fn async_retrieve_in_order<T, B>(
    data_type: &DataType,
    order: ElementOrder,
    shape: &[u64],
    retrieve_into: impl AsyncFnOnce(&mut ArrayBytesTarget<'_>) -> Result<(), ArrayError>,
    retrieve_c_order: impl AsyncFnOnce() -> Result<B, ArrayError>,
) -> Result<T, ArrayError>
where
    T: FromArrayBytes,
    B: COrderBytes,
{
    if !order.is_reordered(shape) {
        return retrieve_c_order()
            .await?
            .into_elements(shape, data_type, order);
    }
    if let Some(mut buffers) = FixedDecodeBuffers::new(data_type, shape) {
        retrieve_into(&mut buffers.target_in_order(ElementOrder::F)?).await?;
        // SAFETY: a successful retrieval into a target writes every element, and only 0 or 1 to masks (the `decode_into` contract)
        unsafe { from_f_order_buffers(buffers, shape, data_type) }
    } else {
        from_c_order_in_f_order(retrieve_c_order().await?.borrow(), shape, data_type)
    }
}

/// Return `chunk`, in the element order of `array`, if it exists, otherwise the fill value of the chunk at `chunk_indices`.
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
        fill_value_bytes(
            array.data_type(),
            array.fill_value(),
            chunk_shape.num_elements_u64(),
        )?,
        bytemuck::must_cast_slice(&chunk_shape),
        array.data_type(),
        array.element_order(),
    )
}

/// Validate `data`'s shape (if it has a layout) and convert it into validated [`ArrayBytes`](super::ArrayBytes) with its resolved element order.
fn into_ordered_array_bytes<'a, T: super::IntoArrayBytes<'a>>(
    data: T,
    data_type: &DataType,
    shape: &[u64],
    num_elements: u64,
    order: ElementOrder,
) -> Result<(super::ArrayBytes<'a>, ElementOrder), ArrayError> {
    data.validate_shape(shape)?;
    let order = data.element_order(order);
    let bytes = data.into_array_bytes(data_type, order)?;
    bytes.validate(num_elements, data_type)?;
    Ok((bytes, order))
}

/// Validate `data`'s shape (if it has a layout) and convert `data` of `shape` into validated C-order [`ArrayBytes`](super::ArrayBytes).
fn into_c_order_array_bytes<'a, T: super::IntoArrayBytes<'a>>(
    data: T,
    data_type: &DataType,
    shape: &[u64],
    num_elements: u64,
    order: ElementOrder,
) -> Result<super::ArrayBytes<'a>, ArrayError> {
    let (bytes, order) = into_ordered_array_bytes(data, data_type, shape, num_elements, order)?;
    super::array_bytes_internal::reorder(bytes, data_type, shape, order, ElementOrder::C)
        .map_err(ArrayError::CodecError)
}

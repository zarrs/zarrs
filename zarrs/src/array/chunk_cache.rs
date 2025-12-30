use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use unsafe_cell_slice::UnsafeCellSlice;

use super::{ArrayBytes, ArrayBytesRaw, ArrayError, codec::CodecOptions};
use crate::iter_concurrent_limit;
use crate::storage::ReadableStorageTraits;
use crate::storage::{MaybeSend, MaybeSync};
use crate::{
    array::{
        Array, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec, DataTypeExt, ElementOwned,
        array_bytes::{merge_chunks_vlen, merge_chunks_vlen_optional, optional_nesting_depth},
        codec::{ArrayPartialDecoderTraits, CodecError},
        concurrency::concurrency_chunks_and_codec,
        from_array_bytes::FromArrayBytes,
    },
    array_subset::{ArraySubset, IncompatibleDimensionalityError},
};

pub(crate) mod chunk_cache_lru;
// pub(crate) mod chunk_cache_lru_macros;

/// The chunk type of an encoded chunk cache.
pub type ChunkCacheTypeEncoded = Option<Arc<ArrayBytesRaw<'static>>>;

/// The chunk type of a decoded chunk cache.
pub type ChunkCacheTypeDecoded = Arc<ArrayBytes<'static>>;

/// The chunk type of a partial decoder chunk cache.
pub type ChunkCacheTypePartialDecoder = Arc<dyn ArrayPartialDecoderTraits>;

/// A chunk cache type ([`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], or [`ChunkCacheTypePartialDecoder`]).
pub trait ChunkCacheType: MaybeSend + MaybeSync + Clone + 'static {
    /// The size of the chunk in bytes.
    fn size(&self) -> usize;
}

impl ChunkCacheType for ChunkCacheTypeEncoded {
    fn size(&self) -> usize {
        self.as_ref().map_or(0, |v| v.len())
    }
}

impl ChunkCacheType for ChunkCacheTypeDecoded {
    fn size(&self) -> usize {
        ArrayBytes::size(self)
    }
}

impl ChunkCacheType for ChunkCacheTypePartialDecoder {
    fn size(&self) -> usize {
        self.as_ref().size_held()
    }
}

/// Traits for a chunk cache.
pub trait ChunkCache: MaybeSend + MaybeSync {
    /// Return the array associated with the chunk cache.
    fn array(&self) -> Arc<Array<dyn ReadableStorageTraits>>;

    /// Cached variant of [`retrieve_chunk_opt`](Array::retrieve_chunk_opt) returning the cached bytes.
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_bytes(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ChunkCacheTypeDecoded, ArrayError>;

    /// Cached variant of [`retrieve_chunk_opt`](Array::retrieve_chunk_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>
    where
        Self: Sized,
    {
        let bytes = self.retrieve_chunk_bytes(chunk_indices, options)?;
        let shape = self
            .array()
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        T::from_array_bytes_arc(bytes, &shape, self.array().data_type())
    }

    #[deprecated(since = "0.23.0", note = "Use retrieve_chunk::<Vec<T>>() instead")]
    /// Cached variant of [`retrieve_chunk_elements_opt`](Array::retrieve_chunk_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_elements<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_chunk(chunk_indices, options)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk::<ndarray::ArrayD<T>>() instead"
    )]
    /// Cached variant of [`retrieve_chunk_ndarray_opt`](Array::retrieve_chunk_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_ndarray<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_chunk(chunk_indices, options)
    }

    /// Cached variant of [`retrieve_chunk_subset_opt`](Array::retrieve_chunk_subset_opt) returning the cached bytes.
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_bytes(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ChunkCacheTypeDecoded, ArrayError>;

    /// Cached variant of [`retrieve_chunk_subset_opt`](Array::retrieve_chunk_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>
    where
        Self: Sized,
    {
        let bytes = self.retrieve_chunk_subset_bytes(chunk_indices, chunk_subset, options)?;
        T::from_array_bytes_arc(bytes, chunk_subset.shape(), self.array().data_type())
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_subset::<Vec<T>>() instead"
    )]
    /// Cached variant of [`retrieve_chunk_subset_elements_opt`](Array::retrieve_chunk_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_elements<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_chunk_subset(chunk_indices, chunk_subset, options)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_subset::<ndarray::ArrayD<T>>() instead"
    )]
    /// Cached variant of [`retrieve_chunk_subset_ndarray_opt`](Array::retrieve_chunk_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_ndarray<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_chunk_subset(chunk_indices, chunk_subset, options)
    }

    /// Cached variant of [`retrieve_array_subset_opt`](Array::retrieve_array_subset_opt) returning the cached bytes.
    #[allow(clippy::missing_errors_doc)]
    #[allow(clippy::too_many_lines)]
    fn retrieve_array_subset_bytes(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
        let array = self.array();
        if array_subset.dimensionality() != array.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.clone(),
                array.shape().to_vec(),
            ));
        }

        // Find the chunks intersecting this array subset
        let chunks = array.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.clone(),
                array.shape().to_vec(),
            ));
        };

        let chunk_shape0 = array.chunk_shape(&vec![0; array.dimensionality()])?;

        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => Ok(ArrayBytes::new_fill_value(
                array.data_type(),
                array_subset.num_elements(),
                array.fill_value(),
            )
            .map_err(CodecError::from)
            .map_err(ArrayError::from)?
            .into()),
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = array.chunk_subset(chunk_indices)?;
                if &chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    self.retrieve_chunk_bytes(chunk_indices, options)
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.retrieve_chunk_subset_bytes(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        options,
                    )
                }
            }
            _ => {
                // Calculate chunk/codec concurrency
                let num_chunks = chunks.num_elements_usize();
                let codec_concurrency =
                    array.recommended_codec_concurrency(&chunk_shape0, array.data_type())?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                // Delegate to appropriate helper based on data type size
                if array.data_type().is_fixed() {
                    retrieve_multi_chunk_fixed_impl(
                        self,
                        &array,
                        array_subset,
                        &chunks,
                        chunk_concurrent_limit,
                        &options,
                    )
                } else {
                    retrieve_multi_chunk_variable_impl(
                        self,
                        &array,
                        array_subset,
                        &chunks,
                        chunk_concurrent_limit,
                        &options,
                    )
                }
            }
        }
    }

    /// Cached variant of [`retrieve_array_subset_opt`](Array::retrieve_array_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>
    where
        Self: Sized,
    {
        let bytes = self.retrieve_array_subset_bytes(array_subset, options)?;
        T::from_array_bytes_arc(bytes, array_subset.shape(), self.array().data_type())
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_array_subset::<Vec<T>>() instead"
    )]
    /// Cached variant of [`retrieve_array_subset_elements_opt`](Array::retrieve_array_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_elements<T: ElementOwned>(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_array_subset(array_subset, options)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_array_subset::<ndarray::ArrayD<T>>() instead"
    )]
    /// Cached variant of [`retrieve_array_subset_ndarray_opt`](Array::retrieve_array_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_ndarray<T: ElementOwned>(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_array_subset(array_subset, options)
    }

    /// Cached variant of [`retrieve_chunks_opt`](Array::retrieve_chunks_opt) returning the cached bytes.
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_bytes(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
        if chunks.dimensionality() != self.array().dimensionality() {
            return Err(IncompatibleDimensionalityError::new(
                chunks.dimensionality(),
                self.array().dimensionality(),
            )
            .into());
        }

        let array_subset = self.array().chunks_subset(chunks)?;
        self.retrieve_array_subset_bytes(&array_subset, options)
    }

    /// Cached variant of [`retrieve_chunks_opt`](Array::retrieve_chunks_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>
    where
        Self: Sized,
    {
        let bytes = self.retrieve_chunks_bytes(chunks, options)?;
        let array_subset = self.array().chunks_subset(chunks)?;
        T::from_array_bytes_arc(bytes, array_subset.shape(), self.array().data_type())
    }

    #[deprecated(since = "0.23.0", note = "Use retrieve_chunks::<Vec<T>>() instead")]
    /// Cached variant of [`retrieve_chunks_elements_opt`](Array::retrieve_chunks_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_elements<T: ElementOwned>(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_chunks(chunks, options)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunks::<ndarray::ArrayD<T>>() instead"
    )]
    /// Cached variant of [`retrieve_chunks_ndarray_opt`](Array::retrieve_chunks_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_ndarray<T: ElementOwned>(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>
    where
        Self: Sized,
    {
        self.retrieve_chunks(chunks, options)
    }

    /// Return the number of chunks in the cache. For a thread-local cache, returns the number of chunks cached on the current thread.
    #[must_use]
    fn len(&self) -> usize;

    /// Returns true if the cache is empty. For a thread-local cache, returns if the cache is empty on the current thread.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Helper function to retrieve multiple chunks with variable-length data.
/// Also handles optional data types with variable-length inner types (including nested optionals).
fn retrieve_multi_chunk_variable_impl<CC: ChunkCache + ?Sized>(
    cache: &CC,
    array: &Array<dyn ReadableStorageTraits>,
    array_subset: &ArraySubset,
    chunks: &ArraySubset,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
) -> Result<ChunkCacheTypeDecoded, ArrayError> {
    let nesting_depth = optional_nesting_depth(array.data_type());

    // Retrieve chunks for variable-length data
    let indices = chunks.indices();
    let chunk_bytes_and_subsets =
        iter_concurrent_limit!(chunk_concurrent_limit, indices, map, |chunk_indices| {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            cache
                .retrieve_chunk_bytes(&chunk_indices, options)
                .map(|bytes| (bytes, chunk_subset))
        })
        .collect::<Result<Vec<_>, ArrayError>>()?;

    // Arc<ArrayBytes> -> ArrayBytes (not copied, but a bit wasteful, change merge_chunks_vlen?)
    let chunk_bytes_and_subsets = chunk_bytes_and_subsets
        .iter()
        .map(|(chunk_bytes, chunk_subset)| (ArrayBytes::clone(chunk_bytes), chunk_subset.clone()))
        .collect();

    if nesting_depth > 0 {
        Ok(merge_chunks_vlen_optional(
            chunk_bytes_and_subsets,
            array_subset.shape(),
            nesting_depth,
        )?
        .into())
    } else {
        Ok(merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?.into())
    }
}

/// Helper method to retrieve multiple chunks with fixed-length data types.
/// Also handles optional data types with fixed-length inner types.
fn retrieve_multi_chunk_fixed_impl<CC: ChunkCache + ?Sized>(
    cache: &CC,
    array: &Array<dyn ReadableStorageTraits>,
    array_subset: &ArraySubset,
    chunks: &ArraySubset,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
) -> Result<ChunkCacheTypeDecoded, ArrayError> {
    // Allocate data buffer and optional mask buffer
    let data_type_size = array
        .data_type()
        .fixed_size()
        .expect("data_type must have fixed size");
    let num_elements = array_subset.num_elements_usize();
    let size_output = num_elements * data_type_size;
    if size_output == 0 {
        return Ok(ArrayBytes::new_flen(vec![]).into());
    }
    let is_optional = array.data_type().is_optional();
    let mut data_output = Vec::with_capacity(size_output);
    let mut mask_output = if is_optional {
        Some(Vec::with_capacity(num_elements))
    } else {
        None
    };

    {
        let data_output_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut data_output);
        let mask_output_slice = mask_output
            .as_mut()
            .map(UnsafeCellSlice::new_from_vec_with_spare_capacity);

        let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
            let chunk_subset_in_array = chunk_subset_overlap.relative_to(array_subset.start())?;

            // Retrieve the chunk subset bytes
            let chunk_subset_bytes = cache.retrieve_chunk_subset_bytes(
                &chunk_indices,
                &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                options,
            )?;

            // Create views for output
            let mut data_view = unsafe {
                // SAFETY: chunks represent disjoint array subsets
                ArrayBytesFixedDisjointView::new(
                    data_output_slice,
                    data_type_size,
                    array_subset.shape(),
                    chunk_subset_in_array.clone(),
                )?
            };

            let mut mask_view = mask_output_slice
                .map(|mask_slice| unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        mask_slice,
                        1, // 1 byte per element for mask
                        array_subset.shape(),
                        chunk_subset_in_array.clone(),
                    )
                })
                .transpose()?;

            // Copy data from chunk_subset_bytes into the views
            match chunk_subset_bytes.as_ref() {
                ArrayBytes::Fixed(bytes) => {
                    data_view.copy_from_slice(bytes).map_err(CodecError::from)?;
                }
                ArrayBytes::Optional(optional_bytes) => {
                    // Extract the data bytes from the boxed ArrayBytes
                    let data_bytes = match optional_bytes.data() {
                        ArrayBytes::Fixed(bytes) => bytes.as_ref(),
                        ArrayBytes::Variable(..) | ArrayBytes::Optional(..) => {
                            unreachable!("Optional data should contain Fixed array bytes")
                        }
                    };
                    data_view
                        .copy_from_slice(data_bytes)
                        .map_err(CodecError::from)?;
                    if let Some(ref mut mask_view) = mask_view {
                        mask_view
                            .copy_from_slice(optional_bytes.mask().as_ref())
                            .map_err(CodecError::from)?;
                    }
                }
                ArrayBytes::Variable(..) => {
                    unreachable!("Variable-length data should not reach this code path");
                }
            }

            Ok::<_, ArrayError>(())
        };

        let indices = chunks.indices();
        iter_concurrent_limit!(
            chunk_concurrent_limit,
            indices,
            try_for_each,
            retrieve_chunk
        )?;
    }

    unsafe { data_output.set_len(size_output) };
    if let Some(ref mut mask) = mask_output {
        unsafe { mask.set_len(num_elements) };
    }

    let array_bytes = ArrayBytes::from(data_output);
    Ok(if let Some(mask) = mask_output {
        array_bytes.with_optional_mask(mask).into()
    } else {
        array_bytes.into()
    })
}

// TODO: AsyncChunkCache

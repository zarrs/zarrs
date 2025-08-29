use std::sync::Arc;

use crate::iter_concurrent_limit;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_metadata::DataTypeSize;
use zarrs_storage::ReadableStorageTraits;
use zarrs_storage::{MaybeSend, MaybeSync};

use crate::{
    array::{
        array_bytes::merge_chunks_vlen,
        codec::{ArrayPartialDecoderTraits, CodecError},
        concurrency::concurrency_chunks_and_codec,
        Array, ArrayBytesFixedDisjointView, ArraySize, ElementOwned,
    },
    array_subset::{ArraySubset, IncompatibleDimensionalityError},
};

use super::{codec::CodecOptions, ArrayBytes, ArrayError, RawBytes};

pub(crate) mod chunk_cache_lru;
pub(crate) mod chunk_cache_lru_macros;

/// The chunk type of an encoded chunk cache.
pub type ChunkCacheTypeEncoded = Option<Arc<RawBytes<'static>>>;

/// The chunk type of a decoded chunk cache.
pub type ChunkCacheTypeDecoded = Arc<ArrayBytes<'static>>;

/// The chunk type of a partial decoder chunk cache.
pub type ChunkCacheTypePartialDecoder = Arc<dyn ArrayPartialDecoderTraits>;

/// A chunk type ([`ChunkCacheTypeEncoded`], [`ChunkCacheTypeDecoded`], or [`ChunkCacheTypePartialDecoder`]).
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
        self.as_ref().size()
    }
}

/// Traits for a chunk cache.
pub trait ChunkCache: MaybeSend + MaybeSync {
    /// Return the array associated with the chunk cache.
    fn array(&self) -> Arc<Array<dyn ReadableStorageTraits>>;

    /// Cached variant of [`retrieve_chunk_opt`](Array::retrieve_chunk_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ChunkCacheTypeDecoded, ArrayError>;

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
        T::from_array_bytes(
            self.array().data_type(),
            Arc::unwrap_or_clone(self.retrieve_chunk(chunk_indices, options)?),
        )
    }

    #[cfg(feature = "ndarray")]
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
        let shape = self
            .array()
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        let elements = self.retrieve_chunk_elements(chunk_indices, options)?;
        crate::array::elements_to_ndarray(&shape, elements)
    }

    /// Cached variant of [`retrieve_chunk_subset_opt`](Array::retrieve_chunk_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ChunkCacheTypeDecoded, ArrayError>;

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
        T::from_array_bytes(
            self.array().data_type(),
            Arc::unwrap_or_clone(self.retrieve_chunk_subset(
                chunk_indices,
                chunk_subset,
                options,
            )?),
        )
    }

    #[cfg(feature = "ndarray")]
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
        let elements = self.retrieve_chunk_subset_elements(chunk_indices, chunk_subset, options)?;
        crate::array::elements_to_ndarray(chunk_subset.shape(), elements)
    }

    /// Cached variant of [`retrieve_array_subset_opt`](Array::retrieve_array_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    #[allow(clippy::too_many_lines)]
    fn retrieve_array_subset(
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

        let chunk_representation0 =
            array.chunk_array_representation(&vec![0; array.dimensionality()])?;

        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let array_size =
                    ArraySize::new(array.data_type().size(), array_subset.num_elements());
                Ok(ArrayBytes::new_fill_value(array_size, array.fill_value()).into())
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = array.chunk_subset(chunk_indices)?;
                if &chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    Ok(self.retrieve_chunk(chunk_indices, options)?)
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.retrieve_chunk_subset(
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
                    array.recommended_codec_concurrency(&chunk_representation0)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                // Retrieve chunks
                let indices = chunks.indices();
                let chunk_bytes_and_subsets =
                    iter_concurrent_limit!(chunk_concurrent_limit, indices, map, |chunk_indices| {
                        let chunk_subset = array.chunk_subset(&chunk_indices)?;
                        self.retrieve_chunk(&chunk_indices, &options)
                            .map(|bytes| (bytes, chunk_subset))
                    })
                    .collect::<Result<Vec<_>, ArrayError>>()?;

                // Merge
                match array.data_type().size() {
                    DataTypeSize::Variable => {
                        // Arc<ArrayBytes> -> ArrayBytes (not copied, but a bit wasteful, change merge_chunks_vlen?)
                        let chunk_bytes_and_subsets = chunk_bytes_and_subsets
                            .iter()
                            .map(|(chunk_bytes, chunk_subset)| {
                                (ArrayBytes::clone(chunk_bytes), chunk_subset.clone())
                            })
                            .collect();
                        Ok(
                            merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?
                                .into(),
                        )
                    }
                    DataTypeSize::Fixed(data_type_size) => {
                        // Allocate the output
                        let size_output = array_subset.num_elements_usize() * data_type_size;
                        if size_output == 0 {
                            return Ok(ArrayBytes::new_flen(vec![]).into());
                        }
                        let mut output = Vec::with_capacity(size_output);

                        {
                            let output =
                                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
                            let update_output = |(chunk_subset_bytes, chunk_subset): (
                                Arc<ArrayBytes>,
                                ArraySubset,
                            )| {
                                // Extract the overlapping bytes
                                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                                let chunk_subset_bytes = if chunk_subset_overlap == chunk_subset {
                                    chunk_subset_bytes
                                } else {
                                    Arc::new(chunk_subset_bytes.extract_array_subset(
                                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                                        chunk_subset.shape(),
                                        array.data_type(),
                                    )?)
                                };

                                let fixed = match chunk_subset_bytes.as_ref() {
                                    ArrayBytes::Fixed(fixed) => fixed,
                                    ArrayBytes::Variable(_, _) => unreachable!(),
                                };

                                let mut output_view = unsafe {
                                    // SAFETY: chunks represent disjoint array subsets
                                    ArrayBytesFixedDisjointView::new(
                                        output,
                                        data_type_size,
                                        array_subset.shape(),
                                        chunk_subset_overlap.relative_to(array_subset.start())?,
                                    )?
                                };
                                output_view
                                    .copy_from_slice(fixed)
                                    .map_err(CodecError::from)?;
                                Ok::<_, ArrayError>(())
                            };
                            iter_concurrent_limit!(
                                chunk_concurrent_limit,
                                chunk_bytes_and_subsets,
                                try_for_each,
                                update_output
                            )?;
                        }
                        unsafe { output.set_len(size_output) };
                        Ok(ArrayBytes::from(output).into())
                    }
                }
            }
        }
    }

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
        T::from_array_bytes(
            self.array().data_type(),
            Arc::unwrap_or_clone(self.retrieve_array_subset(array_subset, options)?),
        )
    }

    #[cfg(feature = "ndarray")]
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
        let elements = self.retrieve_array_subset_elements(array_subset, options)?;
        crate::array::elements_to_ndarray(array_subset.shape(), elements)
    }

    /// Cached variant of [`retrieve_chunks_opt`](Array::retrieve_chunks_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks(
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
        self.retrieve_array_subset(&array_subset, options)
    }

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
        T::from_array_bytes(
            self.array().data_type(),
            Arc::unwrap_or_clone(self.retrieve_chunks(chunks, options)?),
        )
    }

    #[cfg(feature = "ndarray")]
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
        let array_subset = self.array().chunks_subset(chunks)?;
        let elements = self.retrieve_chunks_elements(chunks, options)?;
        crate::array::elements_to_ndarray(array_subset.shape(), elements)
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

// TODO: AsyncChunkCache

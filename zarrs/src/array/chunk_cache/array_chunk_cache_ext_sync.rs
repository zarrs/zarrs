use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;

use crate::{
    array::{
        array_bytes::merge_chunks_vlen,
        codec::{CodecError, CodecOptions},
        concurrency::concurrency_chunks_and_codec,
        Array, ArrayBytes, ArrayBytesFixedDisjointView, ArrayError, ArraySize, DataTypeSize,
        ElementOwned,
    },
    array_subset::ArraySubset,
    storage::ReadableStorageTraits,
};

use super::{ChunkCache, ChunkCacheType};

/// An [`Array`] extension trait to support reading with a chunk cache.
///
/// Note that these methods never perform partial decoding and always fully decode chunks intersected that are not in the cache.
pub trait ArrayChunkCacheExt<TStorage: ?Sized + ReadableStorageTraits + 'static>:
    private::Sealed
{
    /// Cached variant of [`retrieve_chunk_opt`](Array::retrieve_chunk_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>;

    /// Cached variant of [`retrieve_chunk_elements_opt`](Array::retrieve_chunk_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_chunk_ndarray_opt`](Array::retrieve_chunk_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Cached variant of [`retrieve_chunks_opt`](Array::retrieve_chunks_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Cached variant of [`retrieve_chunks_elements_opt`](Array::retrieve_chunks_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_chunks_ndarray_opt`](Array::retrieve_chunks_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Cached variant of [`retrieve_chunk_subset_opt`](Array::retrieve_chunk_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Cached variant of [`retrieve_chunk_subset_elements_opt`](Array::retrieve_chunk_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_chunk_subset_ndarray_opt`](Array::retrieve_chunk_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Cached variant of [`retrieve_array_subset_opt`](Array::retrieve_array_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Cached variant of [`retrieve_array_subset_elements_opt`](Array::retrieve_array_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_array_subset_ndarray_opt`](Array::retrieve_array_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;
}

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> ArrayChunkCacheExt<TStorage>
    for Array<TStorage>
{
    fn retrieve_chunk_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        cache.retrieve_chunk(self, chunk_indices, options)
    }

    fn retrieve_chunk_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            Arc::unwrap_or_clone(self.retrieve_chunk_opt_cached(cache, chunk_indices, options)?),
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_chunk_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        crate::array::elements_to_ndarray(
            &shape,
            self.retrieve_chunk_elements_opt_cached(cache, chunk_indices, options)?,
        )
    }

    fn retrieve_chunks_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if chunks.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                chunks.clone(),
                self.shape().to_vec(),
            ));
        }

        let array_subset = self.chunks_subset(chunks)?;
        self.retrieve_array_subset_opt_cached(cache, &array_subset, options)
    }

    fn retrieve_chunks_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunks_opt_cached(cache, chunks, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_chunks_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        let elements = self.retrieve_chunks_elements_opt_cached(cache, chunks, options)?;
        crate::array::elements_to_ndarray(array_subset.shape(), elements)
    }

    fn retrieve_chunk_subset_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let chunk_representation = self.chunk_array_representation(chunk_indices)?;
        if !chunk_subset.inbounds_shape(&chunk_representation.shape_u64()) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.clone(),
                self.shape().to_vec(),
            ));
        }

        let chunk_bytes = self.retrieve_chunk_opt_cached(cache, chunk_indices, options)?;

        let chunk_subset_bytes = if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape() == chunk_representation.shape_u64()
        {
            // Fast path if `chunk_subset` encompasses the whole chunk
            Arc::unwrap_or_clone(chunk_bytes)
        } else {
            chunk_bytes
                .extract_array_subset(
                    chunk_subset,
                    &chunk_representation.shape_u64(),
                    self.data_type(),
                )?
                .into_owned()
        };
        Ok(chunk_subset_bytes)
    }

    fn retrieve_chunk_subset_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunk_subset_opt_cached(cache, chunk_indices, chunk_subset, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_chunk_subset_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements = self.retrieve_chunk_subset_elements_opt_cached(
            cache,
            chunk_indices,
            chunk_subset,
            options,
        )?;
        crate::array::elements_to_ndarray(chunk_subset.shape(), elements)
    }

    #[allow(clippy::too_many_lines)]
    fn retrieve_array_subset_opt_cached<CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if array_subset.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.clone(),
                self.shape().to_vec(),
            ));
        }

        // Find the chunks intersecting this array subset
        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.clone(),
                self.shape().to_vec(),
            ));
        };

        let chunk_representation0 =
            self.chunk_array_representation(&vec![0; self.dimensionality()])?;

        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let array_size =
                    ArraySize::new(self.data_type().size(), array_subset.num_elements());
                Ok(ArrayBytes::new_fill_value(array_size, self.fill_value()))
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if &chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    Ok(Arc::unwrap_or_clone(self.retrieve_chunk_opt_cached(
                        cache,
                        chunk_indices,
                        options,
                    )?))
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.retrieve_chunk_subset_opt_cached(
                        cache,
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
                    self.recommended_codec_concurrency(&chunk_representation0)?;
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
                        let chunk_subset = self.chunk_subset(&chunk_indices)?;
                        self.retrieve_chunk_opt_cached(cache, &chunk_indices, &options)
                            .map(|bytes| (bytes, chunk_subset))
                    })
                    .collect::<Result<Vec<_>, ArrayError>>()?;

                // Merge
                match self.data_type().size() {
                    DataTypeSize::Variable => {
                        // Arc<ArrayBytes> -> ArrayBytes (not copied, but a bit wasteful, change merge_chunks_vlen?)
                        let chunk_bytes_and_subsets = chunk_bytes_and_subsets
                            .iter()
                            .map(|(chunk_bytes, chunk_subset)| {
                                (ArrayBytes::clone(chunk_bytes), chunk_subset.clone())
                            })
                            .collect();
                        Ok(merge_chunks_vlen(
                            chunk_bytes_and_subsets,
                            array_subset.shape(),
                        )?)
                    }
                    DataTypeSize::Fixed(data_type_size) => {
                        // Allocate the output
                        let size_output = array_subset.num_elements_usize() * data_type_size;
                        if size_output == 0 {
                            return Ok(ArrayBytes::new_flen(vec![]));
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
                                        self.data_type(),
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
                        Ok(ArrayBytes::from(output))
                    }
                }
            }
        }
    }

    fn retrieve_array_subset_elements_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_array_subset_opt_cached(cache, array_subset, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_array_subset_ndarray_opt_cached<T: ElementOwned, CT: ChunkCacheType>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements =
            self.retrieve_array_subset_elements_opt_cached(cache, array_subset, options)?;
        crate::array::elements_to_ndarray(array_subset.shape(), elements)
    }
}

mod private {
    use super::{Array, ReadableStorageTraits};

    pub trait Sealed {}

    impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Sealed for Array<TStorage> {}
}

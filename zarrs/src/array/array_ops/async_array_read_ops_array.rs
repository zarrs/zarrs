use inherent::inherent;
use std::borrow::Cow;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use unsafe_cell_slice::UnsafeCellSlice;

use super::super::array_bytes_internal::{
    build_nested_optional_target, extract_target_views, merge_chunks_vlen,
    merge_chunks_vlen_optional, optional_nesting_depth,
};
use super::super::concurrency::concurrency_chunks_and_codec;
use super::super::{ArrayBytesFixedDisjointView, ArrayIndicesTinyVec};
use super::{AsyncArrayReadOps, *};
use crate::array::{ArrayBytes, ArraySubset, ChunkShapeTraits};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
    CodecError, InvalidNumberOfElementsError, copy_fill_value_into,
};
use zarrs_storage::{Bytes, StorageHandle};

#[cfg(feature = "async")]
#[inherent]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> AsyncArrayReadOps
    for Array<TStorage>
{
    pub async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<T, ArrayError> {
        if let Some(chunk) = self
            .async_retrieve_chunk_if_exists::<T>(chunk_indices)
            .await?
        {
            Ok(chunk)
        } else {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = ArrayBytes::new_fill_value(
                self.data_type(),
                chunk_shape.num_elements_u64(),
                self.fill_value(),
            )
            .map_err(CodecError::from)
            .map_err(ArrayError::from)?;
            T::from_array_bytes(
                bytes,
                bytemuck::must_cast_slice(&chunk_shape),
                self.data_type(),
            )
        }
    }

    pub async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .await
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            self.codecs_bound()
                .decode_into(
                    Cow::Owned(chunk_encoded.into()),
                    &chunk_shape,
                    output_target,
                    options,
                )
                .map_err(ArrayError::CodecError)
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError)
        }
    }

    pub async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    pub async fn async_retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if !chunk_subset.inbounds_shape(chunk_shape_u64) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        let chunk_subset_shape = chunk_subset.shape();
        if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset_shape.as_ref() == chunk_shape_u64
        {
            self.async_retrieve_chunk(chunk_indices).await
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_async_readable_transformer(storage_handle)
                .await?;
            let input_handle = Arc::new((storage_transformer, self.chunk_key(chunk_indices)));
            let bytes = self
                .codecs_bound()
                .async_partial_decoder(input_handle, &chunk_shape, options)
                .await?
                .partial_decode(chunk_subset, options)
                .await?
                .into_owned();
            bytes.validate(chunk_subset.num_elements(), self.data_type())?;
            T::from_array_bytes(bytes, &chunk_subset_shape, self.data_type())
        }
    }

    pub async fn async_retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if !chunk_subset.inbounds_shape(chunk_shape_u64) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape().as_ref() == chunk_shape_u64
        {
            self.async_retrieve_chunk_into(chunk_indices, output_target)
                .await
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_async_readable_transformer(storage_handle)
                .await?;
            let input_handle = Arc::new((storage_transformer, self.chunk_key(chunk_indices)));
            self.codecs_bound()
                .async_partial_decoder(input_handle, &chunk_shape, options)
                .await?
                .partial_decode_into(chunk_subset, output_target, options)
                .await?;
            Ok(())
        }
    }

    pub async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        if array_subset.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        };

        let num_chunks = chunks.num_elements_usize();
        let array_subset_shape = array_subset.shape();
        match num_chunks {
            0 => {
                let bytes = ArrayBytes::new_fill_value(
                    self.data_type(),
                    array_subset.num_elements(),
                    self.fill_value(),
                )
                .map_err(CodecError::from)
                .map_err(ArrayError::from)?;
                T::from_array_bytes(bytes, &array_subset_shape, self.data_type())
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    self.async_retrieve_chunk(chunk_indices).await
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.async_retrieve_chunk_subset(chunk_indices, &array_subset_in_chunk_subset)
                        .await
                }
            }
            _ => {
                let chunk_shape = self.chunk_shape(chunks.start())?;
                let codec_concurrency = self.recommended_codec_concurrency(&chunk_shape)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                let bytes = if self.data_type().is_fixed() {
                    self.async_retrieve_multi_chunk_fixed(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )
                    .await?
                } else {
                    self.async_retrieve_multi_chunk_variable(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )
                    .await?
                };
                T::from_array_bytes(bytes.into_owned(), &array_subset_shape, self.data_type())
            }
        }
    }

    pub async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, StorageError> {
        let options = self.codec_options();
        let _ = options;
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;

        storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .await
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    pub async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError> {
        let options = self.codec_options();
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .await
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = self
                .codecs_bound()
                .decode(Cow::Owned(chunk_encoded.into()), &chunk_shape, options)
                .map_err(ArrayError::CodecError)?;
            bytes.validate(chunk_shape.num_elements_u64(), self.data_type())?;
            Ok(Some(T::from_array_bytes(
                bytes.into_owned(),
                bytemuck::must_cast_slice(&chunk_shape),
                self.data_type(),
            )?))
        } else {
            Ok(None)
        }
    }

    pub async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<Option<Bytes>>, StorageError> {
        let options = self.codec_options();
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;

        let retrieve_encoded_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            let storage_transformer = storage_transformer.clone();
            async move {
                storage_transformer
                    .get(&self.chunk_key(&chunk_indices))
                    .await
            }
        };

        let indices = chunks.indices();
        let futures = indices.into_iter().map(retrieve_encoded_chunk);
        futures::stream::iter(futures)
            .buffered(options.concurrent_target())
            .try_collect()
            .await
    }

    pub async fn async_retrieve_subchunk<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<T, ArrayError>;

    pub async fn async_retrieve_subchunks<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    pub async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        if array_subset.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        if !self.data_type().is_fixed() {
            return Err(ArrayError::CodecError(CodecError::Other(
                "retrieve_array_subset_into does not support variable-length data types"
                    .to_string(),
            )));
        }

        if output_target.num_elements() != array_subset.num_elements() {
            return Err(ArrayError::CodecError(
                InvalidNumberOfElementsError::new(
                    output_target.num_elements(),
                    array_subset.num_elements(),
                )
                .into(),
            ));
        }

        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        };

        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError),
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    self.async_retrieve_chunk_into(chunk_indices, output_target)
                        .await
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.async_retrieve_chunk_subset_into(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        output_target,
                    )
                    .await
                }
            }
            _ => {
                let chunk_shape = self.chunk_shape(chunks.start())?;
                let codec_concurrency = self.recommended_codec_concurrency(&chunk_shape)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                self.async_retrieve_multi_chunk_fixed_into(
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    &output_target,
                    &options,
                )
                .await
            }
        }
    }

    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        let options = self.codec_options();
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;
        let input_handle = Arc::new((storage_transformer, self.chunk_key(chunk_indices)));
        Ok(self
            .codecs_bound()
            .async_partial_decoder(input_handle, &self.chunk_shape(chunk_indices)?, options)
            .await?)
    }

    pub async fn async_local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkGrid>, ArrayError>;

    pub async fn async_local_subchunk_grid_at_level(
        &self,
        level: usize,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkGrid>, ArrayError>;
}

impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
    /// Helper method to retrieve multiple chunks with variable-length data types (async).
    /// Also handles optional data types with variable-length inner types (including nested optionals).
    pub(crate) async fn async_retrieve_multi_chunk_variable(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        // Per-chunk retrieval must use the concurrency-adjusted options, so operate
        // through an array carrying them. Cloned once, outside the loop.
        let tuned_array = self.with_codec_options(*options);

        let nesting_depth = optional_nesting_depth(data_type);
        let array_subset_start = array_subset.start();
        let array_subset_shape = array_subset.shape();

        if nesting_depth > 0 {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let array_subset_start = &array_subset_start;
                let tuned_array = &tuned_array;
                async move {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                    Ok::<_, ArrayError>((
                        tuned_array
                            .async_retrieve_chunk_subset::<ArrayBytes>(
                                &chunk_indices,
                                &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                            )
                            .await?
                            .into_optional()?,
                        chunk_subset_overlap.relative_to(array_subset_start)?,
                    ))
                }
            };

            let chunk_bytes_and_subsets: Vec<_> = futures::stream::iter(chunks.indices().iter())
                .map(|chunk_indices| retrieve_chunk(chunk_indices.clone()))
                .buffered(chunk_concurrent_limit)
                .try_collect()
                .await?;

            Ok(ArrayBytes::Optional(merge_chunks_vlen_optional(
                chunk_bytes_and_subsets,
                &array_subset_shape,
                nesting_depth,
            )?))
        } else {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let array_subset_start = &array_subset_start;
                let tuned_array = &tuned_array;
                async move {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                    Ok::<_, ArrayError>((
                        tuned_array
                            .async_retrieve_chunk_subset::<ArrayBytes>(
                                &chunk_indices,
                                &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                            )
                            .await?
                            .into_variable()?,
                        chunk_subset_overlap.relative_to(array_subset_start)?,
                    ))
                }
            };

            let chunk_bytes_and_subsets: Vec<_> = futures::stream::iter(chunks.indices().iter())
                .map(|chunk_indices| retrieve_chunk(chunk_indices.clone()))
                .buffered(chunk_concurrent_limit)
                .try_collect()
                .await?;

            Ok(ArrayBytes::Variable(merge_chunks_vlen(
                chunk_bytes_and_subsets,
                &array_subset_shape,
            )))
        }
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types (async).
    /// Also handles optional data types with fixed-length inner types.
    pub(crate) async fn async_retrieve_multi_chunk_fixed(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        // Per-chunk retrieval must use the concurrency-adjusted options, so operate
        // through an array carrying them. Cloned once, outside the loop.
        let tuned_array = self.with_codec_options(*options);

        let data_type_size = data_type
            .fixed_size()
            .expect("data_type must have fixed size");
        let num_elements = array_subset.num_elements_usize();
        let size_output = num_elements * data_type_size;
        let nesting_depth = optional_nesting_depth(data_type);
        let mut data_output = Vec::with_capacity(size_output);
        let mut mask_outputs: Vec<Vec<u8>> = (0..nesting_depth)
            .map(|_| Vec::with_capacity(num_elements))
            .collect();

        {
            let data_output_slice =
                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut data_output);
            let mask_output_slices: Vec<_> = mask_outputs
                .iter_mut()
                .map(UnsafeCellSlice::new_from_vec_with_spare_capacity)
                .collect();
            let mask_output_slices = mask_output_slices.as_slice();
            let array_subset_start = array_subset.start();
            let array_subset_shape = array_subset.shape();

            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let array_subset_start = &array_subset_start;
                let array_subset_shape = &array_subset_shape;
                let tuned_array = &tuned_array;
                async move {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                    let chunk_subset_in_array =
                        chunk_subset_overlap.relative_to(array_subset_start)?;

                    let mut data_view = unsafe {
                        // SAFETY: chunks represent disjoint array subsets
                        ArrayBytesFixedDisjointView::new(
                            data_output_slice,
                            data_type_size,
                            array_subset_shape,
                            chunk_subset_in_array.clone(),
                        )?
                    };

                    let mut mask_views: Vec<ArrayBytesFixedDisjointView<'_>> = mask_output_slices
                        .iter()
                        .map(|mask_slice| unsafe {
                            // SAFETY: chunks represent disjoint array subsets
                            ArrayBytesFixedDisjointView::new(
                                *mask_slice,
                                1,
                                array_subset_shape,
                                chunk_subset_in_array.clone(),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let target =
                        build_nested_optional_target(&mut data_view, mask_views.as_mut_slice());

                    tuned_array
                        .async_retrieve_chunk_subset_into(
                            &chunk_indices,
                            &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                            target,
                        )
                        .await?;
                    Ok::<_, ArrayError>(())
                }
            };

            futures::stream::iter(&chunks.indices())
                .map(Ok)
                .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
                .await?;
        }

        unsafe { data_output.set_len(size_output) };
        for mask in &mut mask_outputs {
            unsafe { mask.set_len(num_elements) };
        }

        let mut array_bytes = ArrayBytes::new_flen(data_output);
        for mask in mask_outputs.into_iter().rev() {
            array_bytes = array_bytes.with_optional_mask(mask);
        }
        Ok(array_bytes)
    }

    async fn async_retrieve_multi_chunk_fixed_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        chunk_concurrent_limit: usize,
        output_target: &ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        // Per-chunk retrieval must use the concurrency-adjusted options, so operate
        // through an array carrying them. Cloned once, outside the loop.
        let tuned_array = self.with_codec_options(*options);

        let (data_view_ref, mask_view_refs) = extract_target_views(output_target);
        let parent_start = data_view_ref.subset().start().to_vec();
        let array_subset_start = array_subset.start();

        let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            let array_subset_start = &array_subset_start;
            let parent_start = &parent_start;
            let mask_view_refs = &mask_view_refs;
            let tuned_array = &tuned_array;
            async move {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                let chunk_subset_in_array = chunk_subset_overlap.relative_to(array_subset_start)?;

                let chunk_start_in_view: Vec<u64> = chunk_subset_in_array
                    .start()
                    .iter()
                    .zip(parent_start.iter())
                    .map(|(&c, &p)| c + p)
                    .collect();
                let chunk_subset_in_view = ArraySubset::new_with_start_shape(
                    chunk_start_in_view,
                    chunk_subset_in_array.shape().to_vec(),
                )?;

                let mut data_sub = unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    data_view_ref.subdivide(chunk_subset_in_view.clone())?
                };

                let mut mask_subs: Vec<ArrayBytesFixedDisjointView<'_>> = mask_view_refs
                    .iter()
                    .map(|mask_view| unsafe {
                        // SAFETY: chunks represent disjoint array subsets
                        mask_view.subdivide(chunk_subset_in_view.clone())
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let target = build_nested_optional_target(&mut data_sub, mask_subs.as_mut_slice());

                tuned_array
                    .async_retrieve_chunk_subset_into(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        target,
                    )
                    .await?;
                Ok::<_, ArrayError>(())
            }
        };

        futures::stream::iter(&chunks.indices())
            .map(Ok)
            .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
            .await?;

        Ok(())
    }
}

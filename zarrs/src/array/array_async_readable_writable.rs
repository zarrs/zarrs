use futures::{StreamExt, TryStreamExt};
use zarrs_storage::AsyncReadableStorageTraits;

use crate::{
    array::ArrayBytes, array_subset::ArraySubset, storage::AsyncReadableWritableStorageTraits,
};

use super::{
    array_bytes::update_array_bytes, codec::CodecOptions,
    concurrency::concurrency_chunks_and_codec, Array, ArrayError, Element,
};

impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// Return a read-only instantiation of the array.
    #[must_use]
    pub fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits> {
        self.with_storage(self.storage.clone().readable())
    }

    /// Async variant of [`store_chunk_subset`](Array::store_chunk_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_bytes: impl Into<ArrayBytes<'a>> + Send,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_bytes,
            &CodecOptions::default(),
        )
        .await
    }

    /// Async variant of [`store_chunk_subset_elements`](Array::store_chunk_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_elements<T: Element + Send + Sync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_elements_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_elements,
            &CodecOptions::default(),
        )
        .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`store_chunk_subset_ndarray`](Array::store_chunk_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_ndarray<
        T: Element + Send + Sync,
        D: ndarray::Dimension,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset_start: &[u64],
        chunk_subset_array: impl Into<ndarray::Array<T, D>> + Send,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_ndarray_opt(
            chunk_indices,
            chunk_subset_start,
            chunk_subset_array,
            &CodecOptions::default(),
        )
        .await
    }

    /// Async variant of [`store_array_subset`](Array::store_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_array_subset<'a>(
        &self,
        array_subset: &ArraySubset,
        subset_bytes: impl Into<ArrayBytes<'a>> + Send,
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_opt(array_subset, subset_bytes, &CodecOptions::default())
            .await
    }

    /// Async variant of [`store_array_subset_elements`](Array::store_array_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_array_subset_elements<T: Element + Send + Sync>(
        &self,
        array_subset: &ArraySubset,
        subset_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_elements_opt(
            array_subset,
            subset_elements,
            &CodecOptions::default(),
        )
        .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`store_array_subset_ndarray`](Array::store_array_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_array_subset_ndarray<
        T: Element + Send + Sync,
        D: ndarray::Dimension,
    >(
        &self,
        subset_start: &[u64],
        subset_array: impl Into<ndarray::Array<T, D>> + Send,
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_ndarray_opt(
            subset_start,
            subset_array,
            &CodecOptions::default(),
        )
        .await
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    /// Async variant of [`store_chunk_subset_opt`](Array::store_chunk_subset_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_opt<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_bytes: impl Into<ArrayBytes<'a>> + Send,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        if std::iter::zip(chunk_subset.end_exc(), &chunk_shape)
            .any(|(end_exc, shape)| end_exc > *shape)
        {
            return Err(ArrayError::InvalidChunkSubset(
                chunk_subset.clone(),
                chunk_indices.to_vec(),
                chunk_shape,
            ));
        }

        if chunk_subset.shape() == chunk_shape && chunk_subset.start().iter().all(|&x| x == 0) {
            // The subset spans the whole chunk, so store the bytes directly and skip decoding
            self.async_store_chunk_opt(chunk_indices, chunk_subset_bytes, options)
                .await
        } else {
            let chunk_subset_bytes = chunk_subset_bytes.into();
            chunk_subset_bytes.validate(chunk_subset.num_elements(), self.data_type().size())?;

            // Lock the chunk
            // let key = self.chunk_key(chunk_indices);
            // let mutex = self.storage.mutex(&key).await?;
            // let _lock = mutex.lock();

            // TODO: Add async partial encoding

            // Decode the entire chunk
            let chunk_bytes_old = self
                .async_retrieve_chunk_opt(chunk_indices, options)
                .await?;

            // Update the chunk
            let chunk_bytes_new = update_array_bytes(
                chunk_bytes_old,
                &chunk_shape,
                chunk_subset,
                &chunk_subset_bytes,
                self.data_type().size(),
            )?;

            // Store the updated chunk
            self.async_store_chunk_opt(chunk_indices, chunk_bytes_new, options)
                .await
        }
    }

    /// Async variant of [`store_chunk_subset_elements_opt`](Array::store_chunk_subset_elements_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_elements_opt<T: Element + Send + Sync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_subset_bytes = T::into_array_bytes(self.data_type(), chunk_subset_elements)?;
        self.async_store_chunk_subset_opt(chunk_indices, chunk_subset, chunk_subset_bytes, options)
            .await
    }

    /// Async variant of [`store_chunk_subset_ndarray_opt`](Array::store_chunk_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_ndarray_opt<
        T: Element + Send + Sync,
        D: ndarray::Dimension,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset_start: &[u64],
        chunk_subset_array: impl Into<ndarray::Array<T, D>> + Send,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_subset_array: ndarray::Array<T, D> = chunk_subset_array.into();
        let subset = ArraySubset::new_with_start_shape(
            chunk_subset_start.to_vec(),
            chunk_subset_array
                .shape()
                .iter()
                .map(|u| *u as u64)
                .collect(),
        )?;
        let chunk_subset_array = super::ndarray_into_vec(chunk_subset_array);
        self.async_store_chunk_subset_elements_opt(
            chunk_indices,
            &subset,
            &chunk_subset_array,
            options,
        )
        .await
    }

    /// Async variant of [`store_array_subset_opt`](Array::store_array_subset_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    #[allow(clippy::too_many_lines)]
    pub async fn async_store_array_subset_opt<'a>(
        &self,
        array_subset: &ArraySubset,
        subset_bytes: impl Into<ArrayBytes<'a>> + Send,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        // Validation
        if array_subset.dimensionality() != self.shape().len() {
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
        let num_chunks = chunks.num_elements_usize();
        if num_chunks == 1 {
            let chunk_indices = chunks.start();
            let chunk_subset = self.chunk_subset(chunk_indices)?;
            if array_subset == &chunk_subset {
                // A fast path if the array subset matches the chunk subset
                // This skips the internal decoding occurring in store_chunk_subset
                self.async_store_chunk_opt(chunk_indices, subset_bytes, options)
                    .await?;
            } else {
                // Store the chunk subset
                self.async_store_chunk_subset_opt(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    subset_bytes,
                    options,
                )
                .await?;
            }
        } else {
            let subset_bytes = subset_bytes.into();
            subset_bytes.validate(array_subset.num_elements(), self.data_type().size())?;

            // Calculate chunk/codec concurrency
            let chunk_representation =
                self.chunk_array_representation(&vec![0; self.dimensionality()])?;
            let codec_concurrency = self.recommended_codec_concurrency(&chunk_representation)?;
            let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                options.concurrent_target(),
                num_chunks,
                options,
                &codec_concurrency,
            );

            let store_chunk = |chunk_indices: Vec<u64>| {
                let chunk_subset = self.chunk_subset(&chunk_indices).unwrap(); // FIXME: unwrap
                let overlap = array_subset.overlap(&chunk_subset).unwrap(); // FIXME: unwrap
                let chunk_subset_in_array_subset =
                    overlap.relative_to(array_subset.start()).unwrap();
                let array_subset_in_chunk_subset =
                    overlap.relative_to(chunk_subset.start()).unwrap();
                let chunk_subset_bytes = subset_bytes
                    .extract_array_subset(
                        &chunk_subset_in_array_subset,
                        array_subset.shape(),
                        self.data_type(),
                    )
                    .unwrap(); // FIXME: unwrap
                let options = options.clone();
                async move {
                    self.async_store_chunk_subset_opt(
                        &chunk_indices,
                        &array_subset_in_chunk_subset,
                        chunk_subset_bytes,
                        &options,
                    )
                    .await
                }
            };

            futures::stream::iter(&chunks.indices())
                .map(Ok)
                .try_for_each_concurrent(Some(chunk_concurrent_limit), store_chunk)
                .await?;
        }
        Ok(())
    }

    /// Async variant of [`store_array_subset_elements_opt`](Array::store_array_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset_elements_opt<T: Element + Send + Sync>(
        &self,
        array_subset: &ArraySubset,
        subset_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let subset_bytes = T::into_array_bytes(self.data_type(), subset_elements)?;
        self.async_store_array_subset_opt(array_subset, subset_bytes, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`store_array_subset_ndarray_opt`](Array::store_array_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset_ndarray_opt<
        T: Element + Send + Sync,
        D: ndarray::Dimension,
    >(
        &self,
        subset_start: &[u64],
        subset_array: impl Into<ndarray::Array<T, D>> + Send,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let subset_array: ndarray::Array<T, D> = subset_array.into();
        let subset = ArraySubset::new_with_start_shape(
            subset_start.to_vec(),
            subset_array.shape().iter().map(|u| *u as u64).collect(),
        )?;
        let subset_array = super::ndarray_into_vec(subset_array);
        self.async_store_array_subset_elements_opt(&subset, &subset_array, options)
            .await
    }
}

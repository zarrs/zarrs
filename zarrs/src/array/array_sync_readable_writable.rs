use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs_storage::ReadableStorageTraits;

use crate::{
    array::ArrayBytes,
    array_subset::ArraySubset,
    storage::{ReadableWritableStorageTraits, StorageHandle},
};

use super::{
    codec::{
        ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, CodecOptions, StoragePartialEncoder,
    },
    concurrency::concurrency_chunks_and_codec,
    update_array_bytes, Array, ArrayError, Element,
};

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// Return a read-only instantiation of the array.
    #[must_use]
    pub fn readable(&self) -> Array<dyn ReadableStorageTraits> {
        self.with_storage(self.storage.clone().readable())
    }

    /// Encode `chunk_subset_bytes` and store in `chunk_subset` of the chunk at `chunk_indices` with default codec options.
    ///
    /// Use [`store_chunk_subset_opt`](Array::store_chunk_subset_opt) to control codec options.
    /// Prefer to use [`store_chunk`](Array::store_chunk) where possible, since this function may decode the chunk before updating it and reencoding it.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_subset` is invalid or out of bounds of the chunk,
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if attempting to reference a byte beyond `usize::MAX`.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunk_subset<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_bytes: impl Into<ArrayBytes<'a>>,
    ) -> Result<(), ArrayError> {
        self.store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_bytes,
            &CodecOptions::default(),
        )
    }

    /// Encode `chunk_subset_elements` and store in `chunk_subset` of the chunk at `chunk_indices` with default codec options.
    ///
    /// Use [`store_chunk_subset_elements_opt`](Array::store_chunk_subset_elements_opt) to control codec options.
    /// Prefer to use [`store_chunk_elements`](Array::store_chunk_elements) where possible, since this will decode the chunk before updating it and reencoding it.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the size of  `T` does not match the data type size, or
    ///  - a [`store_chunk_subset`](Array::store_chunk_subset) error condition is met.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunk_subset_elements<T: Element>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.store_chunk_subset_elements_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_elements,
            &CodecOptions::default(),
        )
    }

    #[cfg(feature = "ndarray")]
    /// Encode `chunk_subset_array` and store in `chunk_subset` of the chunk in the subset starting at `chunk_subset_start`.
    ///
    /// Use [`store_chunk_subset_ndarray_opt`](Array::store_chunk_subset_ndarray_opt) to control codec options.
    /// Prefer to use [`store_chunk_ndarray`](Array::store_chunk_ndarray) where possible, since this will decode the chunk before updating it and reencoding it.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if a [`store_chunk_subset_elements`](Array::store_chunk_subset_elements) error condition is met.
    pub fn store_chunk_subset_ndarray<T: Element, D: ndarray::Dimension>(
        &self,
        chunk_indices: &[u64],
        chunk_subset_start: &[u64],
        chunk_subset_array: impl Into<ndarray::Array<T, D>>,
    ) -> Result<(), ArrayError> {
        self.store_chunk_subset_ndarray_opt(
            chunk_indices,
            chunk_subset_start,
            chunk_subset_array,
            &CodecOptions::default(),
        )
    }

    /// Encode `subset_bytes` and store in `array_subset`.
    ///
    /// Use [`store_array_subset_opt`](Array::store_array_subset_opt) to control codec options.
    /// Prefer to use [`store_chunk`](Array::store_chunk) or [`store_chunks`](Array::store_chunks) where possible, since this will decode and encode each chunk intersecting `array_subset`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the dimensionality of `array_subset` does not match the chunk grid dimensionality
    ///  - the length of `subset_bytes` does not match the expected length governed by the shape of the array subset and the data type size,
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_array_subset<'a>(
        &self,
        array_subset: &ArraySubset,
        subset_bytes: impl Into<ArrayBytes<'a>>,
    ) -> Result<(), ArrayError> {
        self.store_array_subset_opt(array_subset, subset_bytes, &CodecOptions::default())
    }

    /// Encode `subset_elements` and store in `array_subset`.
    ///
    /// Use [`store_array_subset_elements_opt`](Array::store_array_subset_elements_opt) to control codec options.
    /// Prefer to use [`store_chunk_elements`](Array::store_chunk_elements) or [`store_chunks_elements`](Array::store_chunks_elements) where possible, since this will decode and encode each chunk intersecting `array_subset`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the size of `T` does not match the data type size, or
    ///  - a [`store_array_subset`](Array::store_array_subset) error condition is met.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_array_subset_elements<T: Element>(
        &self,
        array_subset: &ArraySubset,
        subset_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.store_array_subset_elements_opt(
            array_subset,
            subset_elements,
            &CodecOptions::default(),
        )
    }

    #[cfg(feature = "ndarray")]
    /// Encode `subset_array` and store in the array subset starting at `subset_start`.
    ///
    /// Use [`store_array_subset_ndarray_opt`](Array::store_array_subset_ndarray_opt) to control codec options.
    /// Prefer to use [`store_chunk_ndarray`](Array::store_chunk_ndarray) or [`store_chunks_ndarray`](Array::store_chunks_ndarray) where possible, since this will decode and encode each chunk intersecting `array_subset`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if a [`store_array_subset_elements`](Array::store_array_subset_elements) error condition is met.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_array_subset_ndarray<T: Element, D: ndarray::Dimension>(
        &self,
        subset_start: &[u64],
        subset_array: impl Into<ndarray::Array<T, D>>,
    ) -> Result<(), ArrayError> {
        self.store_array_subset_ndarray_opt(subset_start, subset_array, &CodecOptions::default())
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    /// Explicit options version of [`store_chunk_subset`](Array::store_chunk_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunk_subset_opt<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_bytes: impl Into<ArrayBytes<'a>>,
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
            self.store_chunk_opt(chunk_indices, chunk_subset_bytes, options)
        } else {
            let chunk_subset_bytes = chunk_subset_bytes.into();
            chunk_subset_bytes.validate(chunk_subset.num_elements(), self.data_type().size())?;

            // Lock the chunk
            // let key = self.chunk_key(chunk_indices);
            // let mutex = self.storage.mutex(&key)?;
            // let _lock = mutex.lock();

            if options.experimental_partial_encoding() {
                let partial_encoder = self.partial_encoder(chunk_indices, options)?;
                Ok(partial_encoder.partial_encode(chunk_subset, &chunk_subset_bytes, options)?)
            } else {
                // Decode the entire chunk
                let chunk_bytes_old = self.retrieve_chunk_opt(chunk_indices, options)?;
                chunk_bytes_old.validate(chunk_shape.iter().product(), self.data_type().size())?;

                // Update the chunk
                let chunk_bytes_new = update_array_bytes(
                    chunk_bytes_old,
                    &chunk_shape,
                    chunk_subset,
                    &chunk_subset_bytes,
                    self.data_type().size(),
                )?;

                // Store the updated chunk
                self.store_chunk_opt(chunk_indices, chunk_bytes_new, options)
            }
        }
    }

    /// Explicit options version of [`store_chunk_subset_elements`](Array::store_chunk_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunk_subset_elements_opt<T: Element>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_subset_bytes = T::into_array_bytes(self.data_type(), chunk_subset_elements)?;
        self.store_chunk_subset_opt(chunk_indices, chunk_subset, chunk_subset_bytes, options)
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`store_chunk_subset_ndarray`](Array::store_chunk_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunk_subset_ndarray_opt<T: Element, D: ndarray::Dimension>(
        &self,
        chunk_indices: &[u64],
        chunk_subset_start: &[u64],
        chunk_subset_array: impl Into<ndarray::Array<T, D>>,
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
        self.store_chunk_subset_elements_opt(chunk_indices, &subset, &chunk_subset_array, options)
    }

    /// Explicit options version of [`store_array_subset`](Array::store_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    #[allow(clippy::too_many_lines)]
    pub fn store_array_subset_opt<'a>(
        &self,
        array_subset: &ArraySubset,
        subset_bytes: impl Into<ArrayBytes<'a>>,
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
                self.store_chunk_opt(chunk_indices, subset_bytes, options)?;
            } else {
                // Store the chunk subset
                self.store_chunk_subset_opt(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    subset_bytes,
                    options,
                )?;
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

            let store_chunk = |chunk_indices: Vec<u64>| -> Result<(), ArrayError> {
                let chunk_subset_in_array = self.chunk_subset(&chunk_indices)?;
                let overlap = array_subset.overlap(&chunk_subset_in_array)?;
                let chunk_subset_in_array_subset = overlap.relative_to(array_subset.start())?;
                let chunk_subset_bytes = subset_bytes.extract_array_subset(
                    &chunk_subset_in_array_subset,
                    array_subset.shape(),
                    self.data_type(),
                )?;
                let array_subset_in_chunk_subset =
                    overlap.relative_to(chunk_subset_in_array.start())?;
                self.store_chunk_subset_opt(
                    &chunk_indices,
                    &array_subset_in_chunk_subset,
                    chunk_subset_bytes,
                    &options,
                )
            };

            let indices = chunks.indices();
            rayon_iter_concurrent_limit::iter_concurrent_limit!(
                chunk_concurrent_limit,
                indices,
                try_for_each,
                store_chunk
            )?;
        }
        Ok(())
    }

    /// Explicit options version of [`store_array_subset_elements`](Array::store_array_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_array_subset_elements_opt<T: Element>(
        &self,
        array_subset: &ArraySubset,
        subset_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let subset_bytes = T::into_array_bytes(self.data_type(), subset_elements)?;
        self.store_array_subset_opt(array_subset, subset_bytes, options)
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`store_array_subset_ndarray`](Array::store_array_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_array_subset_ndarray_opt<T: Element, D: ndarray::Dimension>(
        &self,
        subset_start: &[u64],
        subset_array: impl Into<ndarray::Array<T, D>>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let subset_array: ndarray::Array<T, D> = subset_array.into();
        let subset = ArraySubset::new_with_start_shape(
            subset_start.to_vec(),
            subset_array.shape().iter().map(|u| *u as u64).collect(),
        )?;
        let subset_array = super::ndarray_into_vec(subset_array);
        self.store_array_subset_elements_opt(&subset, &subset_array, options)
    }

    /// Initialises a partial encoder for the chunk at `chunk_indices`.
    ///
    /// Only one partial encoder should be created for a chunk at a time because:
    /// - partial encoders can hold internal state that may become out of sync, and
    /// - parallel writing to the same chunk [may result in data loss](#parallel-writing).
    ///
    /// Partial encoding with [`ArrayPartialEncoderTraits::partial_encode`] will use parallelism internally where possible.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if initialisation of the partial encoder fails.
    pub fn partial_encoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));

        let chunk_representation = self.chunk_array_representation(chunk_indices)?;

        // Input/output
        let storage_transformer = self
            .storage_transformers()
            .create_readable_writable_transformer(storage_handle)?;
        let input_output_handle = Arc::new(StoragePartialEncoder::new(
            storage_transformer,
            self.chunk_key(chunk_indices),
        ));

        Ok(self.codecs.clone().partial_encoder(
            input_output_handle,
            &chunk_representation,
            options,
        )?)
    }
}

use futures::{StreamExt, TryStreamExt};

use super::{
    Array, ArrayError, ArrayIndicesTinyVec, Element, IntoArrayBytes,
    array_bytes::update_array_bytes,
    codec::{
        ArrayToBytesCodecTraits, AsyncArrayPartialEncoderTraits, CodecOptions, CodecTraits,
        StoragePartialEncoder,
    },
    concurrency::concurrency_chunks_and_codec,
};
use crate::storage::AsyncReadableStorageTraits;
use crate::storage::{MaybeSend, MaybeSync};
use crate::{array_subset::ArraySubset, storage::AsyncReadableWritableStorageTraits};

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
        chunk_subset_data: impl IntoArrayBytes<'a> + MaybeSend,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_data,
            &self.codec_options,
        )
        .await
    }

    #[deprecated(since = "0.23.0", note = "Use async_store_chunk_subset() instead")]
    /// Async variant of [`store_chunk_subset_elements`](Array::store_chunk_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_elements<T: Element + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_elements,
            &self.codec_options,
        )
        .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(since = "0.23.0", note = "Use async_store_chunk_subset() instead")]
    /// Async variant of [`store_chunk_subset_ndarray`](Array::store_chunk_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_ndarray<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset_start: &[u64],
        chunk_subset_array: &ndarray::ArrayRef<T, D>,
    ) -> Result<(), ArrayError> {
        let chunk_subset_start = ArraySubset::new_with_start_shape(
            chunk_subset_start.to_vec(),
            chunk_subset_array
                .shape()
                .iter()
                .map(|&x| x as u64)
                .collect(),
        )?;
        self.async_store_chunk_subset_opt(
            chunk_indices,
            &chunk_subset_start,
            chunk_subset_array.as_standard_layout().to_owned(),
            &self.codec_options,
        )
        .await
    }

    /// Async variant of [`store_array_subset`](Array::store_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_array_subset<'a>(
        &self,
        array_subset: &ArraySubset,
        subset_data: impl IntoArrayBytes<'a> + MaybeSend,
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_opt(array_subset, subset_data, &self.codec_options)
            .await
    }

    #[deprecated(since = "0.23.0", note = "Use async_store_array_subset() instead")]
    /// Async variant of [`store_array_subset_elements`](Array::store_array_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_array_subset_elements<T: Element + MaybeSend + MaybeSync>(
        &self,
        array_subset: &ArraySubset,
        subset_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_opt(array_subset, subset_elements, &self.codec_options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(since = "0.23.0", note = "Use async_store_array_subset()  instead")]
    /// Async variant of [`store_array_subset_ndarray`](Array::store_array_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_array_subset_ndarray<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        subset_start: &[u64],
        subset_array: &ndarray::ArrayRef<T, D>,
    ) -> Result<(), ArrayError> {
        let subset = ArraySubset::new_with_start_shape(
            subset_start.to_vec(),
            subset_array.shape().iter().map(|&x| x as u64).collect(),
        )?;
        self.async_store_array_subset_opt(
            &subset,
            subset_array.as_standard_layout().to_owned(),
            &self.codec_options,
        )
        .await
    }

    /// Async variant of [`compact_chunk`](Array::compact_chunk).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_compact_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<bool, ArrayError> {
        let chunk_bytes = self.async_retrieve_encoded_chunk(chunk_indices).await?;
        if let Some(chunk_bytes) = chunk_bytes {
            let chunk_bytes: Vec<u8> = chunk_bytes.into();
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            if let Some(compacted_bytes) = self.codecs.compact(
                chunk_bytes.into(),
                &chunk_shape,
                self.data_type(),
                self.fill_value(),
                options,
            )? {
                // SAFETY: The compacted bytes are already encoded
                unsafe {
                    self.async_store_encoded_chunk(
                        chunk_indices,
                        bytes::Bytes::from(compacted_bytes.into_owned()),
                    )
                    .await?;
                }
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
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
        chunk_subset_data: impl IntoArrayBytes<'a> + MaybeSend,
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
            self.async_store_chunk_opt(chunk_indices, chunk_subset_data, options)
                .await
        } else {
            let chunk_subset_bytes = chunk_subset_data.into_array_bytes(self.data_type())?;
            chunk_subset_bytes.validate(chunk_subset.num_elements(), self.data_type())?;

            // Lock the chunk
            // let key = self.chunk_key(chunk_indices);
            // let mutex = self.storage.mutex(&key).await?;
            // let _lock = mutex.lock();

            if options.experimental_partial_encoding()
                && self.codecs.partial_encoder_capability().partial_encode
                && self.storage.supports_set_partial()
            {
                let partial_encoder = self.async_partial_encoder(chunk_indices, options).await?;
                debug_assert!(
                    partial_encoder.supports_partial_encode(),
                    "partial encoder is misrepresenting its capabilities"
                );
                partial_encoder
                    .partial_encode(chunk_subset, &chunk_subset_bytes, options)
                    .await?;
                Ok(())
            } else {
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
    }

    #[deprecated(since = "0.23.0", note = "Use async_store_chunk_subset_opt() instead")]
    /// Async variant of [`store_chunk_subset_elements_opt`](Array::store_chunk_subset_elements_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_elements_opt<T: Element + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        chunk_subset_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_elements,
            options,
        )
        .await
    }

    #[deprecated(since = "0.23.0", note = "Use async_store_chunk_subset_opt()  instead")]
    /// Async variant of [`store_chunk_subset_ndarray_opt`](Array::store_chunk_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_store_chunk_subset_ndarray_opt<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset_start: &[u64],
        chunk_subset_array: &ndarray::ArrayRef<T, D>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_subset = ArraySubset::new_with_start_shape(
            chunk_subset_start.to_vec(),
            chunk_subset_array
                .shape()
                .iter()
                .map(|&x| x as u64)
                .collect(),
        )?;
        let chunk_subset_array = chunk_subset_array
            .as_standard_layout()
            .to_owned()
            .into_array_bytes(self.data_type())?;
        self.async_store_chunk_subset_opt(chunk_indices, &chunk_subset, chunk_subset_array, options)
            .await
    }

    /// Async variant of [`store_array_subset_opt`](Array::store_array_subset_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    #[allow(clippy::too_many_lines)]
    pub async fn async_store_array_subset_opt<'a>(
        &self,
        array_subset: &ArraySubset,
        subset_data: impl IntoArrayBytes<'a> + MaybeSend,
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
                self.async_store_chunk_opt(chunk_indices, subset_data, options)
                    .await?;
            } else {
                // Store the chunk subset
                self.async_store_chunk_subset_opt(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    subset_data,
                    options,
                )
                .await?;
            }
        } else {
            let subset_bytes = subset_data.into_array_bytes(self.data_type())?;
            subset_bytes.validate(array_subset.num_elements(), self.data_type())?;

            // Calculate chunk/codec concurrency
            let chunk_shape = self.chunk_shape(&vec![0; self.dimensionality()])?;
            let codec_concurrency =
                self.recommended_codec_concurrency(&chunk_shape, self.data_type())?;
            let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                options.concurrent_target(),
                num_chunks,
                options,
                &codec_concurrency,
            );

            let store_chunk = |chunk_indices: ArrayIndicesTinyVec| {
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

    #[deprecated(since = "0.23.0", note = "Use async_store_array_subset_opt() instead")]
    /// Async variant of [`store_array_subset_elements_opt`](Array::store_array_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset_elements_opt<T: Element + MaybeSend + MaybeSync>(
        &self,
        array_subset: &ArraySubset,
        subset_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.async_store_array_subset_opt(array_subset, subset_elements, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(since = "0.23.0", note = "Use async_store_array_subset_opt()  instead")]
    /// Async variant of [`store_array_subset_ndarray_opt`](Array::store_array_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset_ndarray_opt<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        subset_start: &[u64],
        subset_array: &ndarray::ArrayRef<T, D>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let subset = ArraySubset::new_with_start_shape(
            subset_start.to_vec(),
            subset_array.shape().iter().map(|&x| x as u64).collect(),
        )?;
        let subset_array = subset_array
            .as_standard_layout()
            .to_owned()
            .into_array_bytes(self.data_type())?;
        self.async_store_array_subset_opt(&subset, subset_array, options)
            .await
    }

    /// Initialises an asynchronous partial encoder for the chunk at `chunk_indices`.
    ///
    /// Only one partial encoder should be created for a chunk at a time because:
    /// - partial encoders can hold internal state that may become out of sync, and
    /// - parallel writing to the same chunk [may result in data loss](#parallel-writing).
    ///
    /// Partial encoding with [`AsyncArrayPartialEncoderTraits::partial_encode`] will use parallelism internally where possible.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if initialisation of the partial encoder fails.
    pub async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<std::sync::Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError> {
        use std::sync::Arc;

        use crate::storage::StorageHandle;

        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));

        let chunk_shape = self.chunk_shape(chunk_indices)?;

        // Input/output
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_writable_transformer(storage_handle)
            .await?;
        let input_output_handle = Arc::new(StoragePartialEncoder::new(
            storage_transformer,
            self.chunk_key(chunk_indices),
        ));

        Ok(self
            .codecs
            .clone()
            .async_partial_encoder(
                input_output_handle,
                &chunk_shape,
                self.data_type(),
                self.fill_value(),
                options,
            )
            .await?)
    }
}

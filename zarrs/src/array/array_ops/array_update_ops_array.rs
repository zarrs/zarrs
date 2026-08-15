use inherent::inherent;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;

use super::super::concurrency::concurrency_chunks_and_codec;
use super::{ArrayUpdateOps, *};
use crate::IntoConcurrentLimitIterator;
use crate::array::{ArrayBytes, ArrayIndicesTinyVec, ArraySubsetTraits, update_array_bytes};
use zarrs_codec::{ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, CodecTraits};
use zarrs_storage::StorageHandle;

#[inherent]
impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> ArrayUpdateOps
    for Array<TStorage>
{
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
    ) -> Result<(), ArrayError> {
        self.store_chunk_subset_with_options(
            chunk_indices,
            chunk_subset,
            chunk_subset_data,
            self.codec_options(),
        )
    }

    #[allow(clippy::missing_errors_doc)]
    #[allow(clippy::too_many_lines)]
    pub fn store_array_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        // Validation
        if array_subset.dimensionality() != self.shape().len() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        // Find the chunks intersecting this array subset
        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        };
        let num_chunks = chunks.num_elements_usize();
        if num_chunks == 1 {
            let chunk_indices = chunks.start();
            let chunk_subset = self.chunk_subset(chunk_indices)?;
            if array_subset == chunk_subset {
                // A fast path if the array subset matches the chunk subset
                // This skips the internal decoding occurring in store_chunk_subset
                self.store_chunk_with_options(chunk_indices, subset_data, options)?;
            } else {
                // Store the chunk subset
                self.store_chunk_subset_with_options(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    subset_data,
                    options,
                )?;
            }
        } else {
            let subset_bytes = subset_data.into_array_bytes(self.data_type())?;
            subset_bytes.validate(array_subset.num_elements(), self.data_type())?;
            // Calculate chunk/codec concurrency
            let chunk_shape = self.chunk_shape(&vec![0; self.dimensionality()])?;
            let codec_concurrency = recommended_codec_concurrency(self, &chunk_shape)?;
            let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                options.concurrent_target(),
                num_chunks,
                options,
                &codec_concurrency,
            );

            let store_chunk = |chunk_indices: ArrayIndicesTinyVec| -> Result<(), ArrayError> {
                let chunk_subset_in_array = self.chunk_subset(&chunk_indices)?;
                let overlap = array_subset.overlap(&chunk_subset_in_array)?;
                let chunk_subset_in_array_subset = overlap.relative_to(&array_subset.start())?;
                let chunk_subset_bytes = subset_bytes.extract_array_subset(
                    &chunk_subset_in_array_subset,
                    &array_subset.shape(),
                    self.data_type(),
                )?;
                let chunk_subset_in_chunk = overlap.relative_to(chunk_subset_in_array.start())?;
                self.store_chunk_subset_with_options(
                    &chunk_indices,
                    &chunk_subset_in_chunk,
                    chunk_subset_bytes,
                    &options,
                )
            };

            let indices = chunks.indices();
            indices
                .concurrent_limit(chunk_concurrent_limit)
                .try_for_each(store_chunk)?;
        }

        Ok(())
    }

    pub fn compact_chunk(&self, chunk_indices: &[u64]) -> Result<bool, ArrayError> {
        let options = self.codec_options();
        let chunk_bytes = self.retrieve_encoded_chunk(chunk_indices)?;
        if let Some(chunk_bytes) = chunk_bytes {
            if let Some(compacted_bytes) = self.codecs_bound.compact(
                chunk_bytes.into(),
                &self.chunk_shape(chunk_indices)?,
                options,
            )? {
                // SAFETY: The compacted bytes are already encoded
                unsafe {
                    self.store_encoded_chunk(
                        chunk_indices,
                        bytes::Bytes::from(compacted_bytes.into_owned()),
                    )?;
                }
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    pub fn readable(&self) -> Array<dyn ReadableStorageTraits> {
        self.with_storage(self.storage.clone().readable())
    }

    pub fn partial_encoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        self.partial_encoder_with_options(chunk_indices, self.codec_options())
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    pub(in crate::array) fn store_chunk_subset_with_options<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
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
                chunk_subset.to_array_subset(),
                chunk_indices.to_vec(),
                chunk_shape,
            ));
        }

        if chunk_subset.shape() == chunk_shape && chunk_subset.start().iter().all(|&x| x == 0) {
            self.store_chunk_with_options(chunk_indices, chunk_subset_data, options)
        } else {
            let chunk_subset_bytes = chunk_subset_data.into_array_bytes(self.data_type())?;
            chunk_subset_bytes.validate(chunk_subset.num_elements(), self.data_type())?;

            // Lock the chunk
            // let key = self.chunk_key(chunk_indices);
            // let mutex = self.storage.mutex(&key)?;
            // let _lock = mutex.lock();

            if options.experimental_partial_encoding()
                && self.codecs.partial_encoder_capability().partial_encode
                && self.storage.supports_set_partial()
            {
                let partial_encoder = self.partial_encoder_with_options(chunk_indices, options)?;
                debug_assert!(
                    partial_encoder.supports_partial_encode(),
                    "partial encoder is misrepresenting its capabilities"
                );
                Ok(partial_encoder.partial_encode(chunk_subset, &chunk_subset_bytes, options)?)
            } else {
                let chunk_bytes_old: ArrayBytes<'static> =
                    self.retrieve_chunk_with_options(chunk_indices, options)?;
                chunk_bytes_old.validate(chunk_shape.iter().product(), self.data_type())?;

                let chunk_bytes_new = update_array_bytes(
                    chunk_bytes_old,
                    &chunk_shape,
                    chunk_subset,
                    &chunk_subset_bytes,
                    self.data_type().size(),
                )?;

                self.store_chunk_with_options(chunk_indices, chunk_bytes_new, options)
            }
        }
    }

    pub(in crate::array) fn partial_encoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));

        // Input/output
        let storage_transformer = self
            .storage_transformers()
            .create_readable_writable_transformer(storage_handle)?;
        let input_handle = Arc::new((storage_transformer, self.chunk_key(chunk_indices)));
        Ok(self.codecs_bound().partial_encoder(
            input_handle,
            &self.chunk_shape(chunk_indices)?,
            options,
        )?)
    }
}

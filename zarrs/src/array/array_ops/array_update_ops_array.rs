use inherent::inherent;
use std::sync::Arc;

use super::{ArrayUpdateOps, *};
use crate::array::{ArrayBytes, ArraySubsetTraits, update_array_bytes};
use zarrs_codec::{ArrayPartialEncoderTraits, ArrayToBytesCodecTraits};

#[inherent]
impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> ArrayUpdateOps
    for Array<TStorage>
{
    #[allow(clippy::missing_errors_doc)]
    pub fn store_partial_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.store_partial_chunk_with_options(
            chunk_indices,
            indexer,
            indexer_data,
            self.codec_options(),
            resources,
        )
    }

    #[allow(clippy::missing_errors_doc)]
    #[allow(clippy::too_many_lines)]
    pub fn store_array_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        resources: &Resources,
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
                // This skips the internal decoding occurring in store_partial_chunk
                self.store_chunk_with_options(chunk_indices, subset_data, options, resources)?;
            } else {
                // Store the chunk subset
                self.store_partial_chunk_with_options(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    subset_data,
                    options,
                    resources,
                )?;
            }
        } else {
            let subset_bytes = subset_data.into_array_bytes(self.data_type())?;
            subset_bytes.validate(array_subset.num_elements(), self.data_type())?;
            self.store_array_subset_planned(
                &chunks,
                &array_subset.to_array_subset(),
                &subset_bytes,
                options,
                resources,
            )?;
        }

        Ok(())
    }

    pub fn compact_chunk(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<bool, ArrayError> {
        let options = self.codec_options();
        let chunk_bytes = self.retrieve_encoded_chunk(chunk_indices)?;
        if let Some(chunk_bytes) = chunk_bytes {
            if let Some(compacted_bytes) = self.codecs_bound.compact(
                chunk_bytes.into(),
                &self.chunk_shape(chunk_indices)?,
                options,
                resources,
            )? {
                // SAFETY: The compacted bytes are already encoded
                unsafe {
                    self.store_encoded_chunk(chunk_indices, compacted_bytes)?;
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
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        let _ = resources;
        self.planned_partial_encoder(chunk_indices)
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    pub(in crate::array) fn store_partial_chunk_with_options<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        if let Some(chunk_subset) = indexer.as_array_subset() {
            if !chunk_subset.inbounds_shape(&chunk_shape) {
                return Err(ArrayError::InvalidChunkSubset(
                    chunk_subset.to_array_subset(),
                    chunk_indices.to_vec(),
                    chunk_shape,
                ));
            }
            if super::subset_is_whole_chunk(chunk_subset, &chunk_shape) {
                return self.store_chunk_with_options(
                    chunk_indices,
                    indexer_data,
                    options,
                    resources,
                );
            }
        }

        let indexer_bytes = indexer_data.into_array_bytes(self.data_type())?;
        indexer_bytes.validate(indexer.len(), self.data_type())?;

        // Lock the chunk
        // let key = self.chunk_key(chunk_indices);
        // let mutex = self.storage.mutex(&key)?;
        // let _lock = mutex.lock();

        if self.partial_encodes() {
            return self.store_partial_chunk_planned(
                chunk_indices,
                indexer,
                indexer_bytes,
                options,
                resources,
            );
        }
        {
            let chunk_bytes_old: ArrayBytes<'static> =
                self.retrieve_chunk_with_options(chunk_indices, options, resources)?;
            chunk_bytes_old.validate(chunk_shape.iter().product(), self.data_type())?;

            let chunk_bytes_new = update_array_bytes(
                chunk_bytes_old,
                &chunk_shape,
                indexer,
                &indexer_bytes,
                self.data_type().size(),
            )?;

            self.store_chunk_with_options(chunk_indices, chunk_bytes_new, options, resources)
        }
    }
}

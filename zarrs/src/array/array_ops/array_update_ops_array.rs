use crate::array::CowBytes;
use inherent::inherent;
use std::sync::Arc;

use super::{ArrayUpdateOps, *};
use crate::array::ArraySubsetTraits;
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
                Box::new(array_subset.to_array_subset()),
                self.shape().to_vec(),
            ));
        }

        // Find the chunks intersecting this array subset
        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                Box::new(array_subset.to_array_subset()),
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
                resources.compute(),
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

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        encoded: CowBytes<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.store_encoded_subchunk_planned(
            0,
            subchunk_indices,
            Some(encoded.into_bytes()),
            resources,
        )
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        encoded: CowBytes<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.store_encoded_subchunk_planned(
            level,
            subchunk_indices,
            Some(encoded.into_bytes()),
            resources,
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_subchunk(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.store_encoded_subchunk_planned(0, subchunk_indices, None, resources)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.store_encoded_subchunk_planned(level, subchunk_indices, None, resources)
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
                    Box::new(chunk_subset.to_array_subset()),
                    chunk_indices.into(),
                    chunk_shape,
                ));
            }
            if chunk_subset.is_whole(&chunk_shape) {
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

        self.store_partial_chunk_planned(chunk_indices, indexer, indexer_bytes, options, resources)
    }
}

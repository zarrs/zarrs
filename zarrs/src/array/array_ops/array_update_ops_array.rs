use inherent::inherent;
use std::sync::Arc;

use bytes::Bytes;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;

use super::super::concurrency::concurrency_chunks_and_codec;
use super::{
    Array, ArrayError, ArrayOps, ArrayUpdateOps, CodecOptions, Indexer, IntoArrayBytes,
    ReadableStorageTraits, ReadableWritableStorageTraits, WritableStorageTraits,
    recommended_codec_concurrency,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::codec::{InMemoryBytesEncoder, InMemoryBytesUpdate};
use crate::array::{ArrayBytes, ArrayIndices, ArraySubset, ArraySubsetTraits, update_array_bytes};
use zarrs_codec::{ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, CodecError, CodecTraits};
use zarrs_storage::ReadableWritableStorage;

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
    ) -> Result<(), ArrayError> {
        self.store_partial_chunk_with_options(
            chunk_indices,
            indexer,
            indexer_data,
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
                self.store_chunk_with_options(chunk_indices, subset_data, options)?;
            } else {
                // Store the chunk subset
                self.store_partial_chunk_with_options(
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

            let store_chunk = |chunk_indices: ArrayIndices| -> Result<(), ArrayError> {
                let chunk_subset_in_array = self.chunk_subset(&chunk_indices)?;
                let overlap = array_subset.overlap(&chunk_subset_in_array)?;
                let chunk_subset_in_array_subset = overlap.relative_to(&array_subset.start())?;
                let chunk_subset_bytes = subset_bytes.extract_array_subset(
                    &chunk_subset_in_array_subset,
                    &array_subset.shape(),
                    self.data_type(),
                )?;
                let chunk_subset_in_chunk = overlap.relative_to(chunk_subset_in_array.start())?;
                self.store_partial_chunk_with_options(
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

    #[allow(clippy::missing_errors_doc)]
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

    #[allow(clippy::must_use_candidate)]
    pub fn readable(&self) -> Array<dyn ReadableStorageTraits> {
        self.with_storage(self.storage.clone().readable())
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        bytes: zarrs_codec::CowBytes<'_>,
    ) -> Result<(), ArrayError>;

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        bytes: zarrs_codec::CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let (chunk_indices, subset) =
            super::subchunk_chunk_and_local_subset(self, level, subchunk_indices)?;
        // SAFETY: The caller upholds the safety requirements.
        unsafe { self.store_encoded_subchunk_in_chunk(level, &chunk_indices, &subset, bytes) }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_subchunk(&self, subchunk_indices: &[u64]) -> Result<(), ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn erase_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<(), ArrayError> {
        let (chunk_indices, subset) =
            super::subchunk_chunk_and_local_subset(self, level, subchunk_indices)?;
        self.erase_subchunk_in_chunk(level, &chunk_indices, &subset)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_encoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        self.partial_encoder_with_options(chunk_indices, self.codec_options())
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// Store the encoded subchunk at the chunk-local `subset` of the `level` subchunk grid of the chunk at
    /// `chunk_indices`, as located by `subchunk_chunk_and_local_subset`.
    ///
    /// # Safety
    /// See [`store_encoded_subchunk_at_level`](ArrayUpdateOps::store_encoded_subchunk_at_level).
    pub(super) unsafe fn store_encoded_subchunk_in_chunk(
        &self,
        level: usize,
        chunk_indices: &[u64],
        subset: &ArraySubset,
        bytes: zarrs_codec::CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        self.update_encoded_subchunk(level, chunk_indices, subset, |encoder, indices, options| {
            // SAFETY: The caller upholds the safety requirements.
            unsafe { encoder.store_encoded_subchunk_at_level(level, indices, bytes, options) }
        })
    }

    /// Erase the subchunk at the chunk-local `subset` of the `level` subchunk grid of the chunk at `chunk_indices`, as
    /// located by `subchunk_chunk_and_local_subset`.
    pub(super) fn erase_subchunk_in_chunk(
        &self,
        level: usize,
        chunk_indices: &[u64],
        subset: &ArraySubset,
    ) -> Result<(), ArrayError> {
        self.update_encoded_subchunk(level, chunk_indices, subset, |encoder, indices, options| {
            encoder.erase_subchunk_at_level(level, indices, options)
        })
    }

    /// Update the encoded subchunk at the chunk-local `subset` of the `level` subchunk grid with the partial encoder of
    /// the chunk at `chunk_indices`.
    ///
    /// If the store does not support partial writes, the encoded chunk is updated in memory and stored whole if it
    /// changed.
    fn update_encoded_subchunk(
        &self,
        level: usize,
        chunk_indices: &[u64],
        subset: &ArraySubset,
        update: impl FnOnce(
            &dyn ArrayPartialEncoderTraits,
            &[u64],
            &CodecOptions,
        ) -> Result<(), zarrs_codec::CodecError>,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        let storage = self.readable_writable_storage()?;
        if storage.supports_set_partial() {
            let encoder = self.partial_encoder_with_storage(storage, chunk_indices, options)?;
            let local_indices =
                super::local_subchunk_indices(encoder.as_ref(), level, subset, options)?;
            update(encoder.as_ref(), &local_indices, options)?;
            Ok(())
        } else {
            let existing = self.retrieve_encoded_chunk(chunk_indices)?;
            let in_memory = InMemoryChunkUpdate::new(self, chunk_indices, level, subset, existing)?;
            update(in_memory.encoder.as_ref(), &in_memory.indices, options)?;
            match in_memory.finish() {
                // SAFETY: The bytes were encoded by the codecs of the array.
                InMemoryBytesUpdate::Stored(updated) => unsafe {
                    self.store_encoded_chunk(chunk_indices, updated.into())
                },
                InMemoryBytesUpdate::Erased => self.erase_chunk(chunk_indices),
                // The chunk is unchanged
                InMemoryBytesUpdate::Unchanged => Ok(()),
            }
        }
    }

    pub(in crate::array) fn store_partial_chunk_with_options<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        if let Some(chunk_subset) = indexer.as_array_subset() {
            if !chunk_subset.inbounds_shape(&chunk_shape) {
                return Err(ArrayError::InvalidChunkSubset(
                    Box::new(chunk_subset.to_array_subset()),
                    chunk_indices.to_vec(),
                    chunk_shape,
                ));
            }
        } else {
            indexer.validate(&chunk_shape).map_err(CodecError::from)?;
        }
        if indexer.covers_array(&chunk_shape) {
            return self.store_chunk_with_options(chunk_indices, indexer_data, options);
        }

        let indexer_bytes = indexer_data.into_array_bytes(self.data_type())?;
        indexer_bytes.validate(indexer.len(), self.data_type())?;
        if indexer.is_empty() {
            return Ok(());
        }

        // Lock the chunk
        // let key = self.chunk_key(chunk_indices);
        // let mutex = self.storage.mutex(&key)?;
        // let _lock = mutex.lock();

        let partial_encoding_storage = if options.experimental_partial_encoding()
            && self.codecs.partial_encoder_capability().partial_encode
        {
            let storage = self.readable_writable_storage()?;
            storage.supports_set_partial().then_some(storage)
        } else {
            None
        };

        if let Some(storage) = partial_encoding_storage {
            // A write of only the fill value to a missing chunk leaves it missing.
            // The read-modify-write path does this (and erases a chunk that becomes all fill) in `store_chunk_with_options`.
            if !options.store_empty_chunks()
                && indexer_bytes.is_fill_value(self.fill_value())
                && storage.size_key(&self.chunk_key(chunk_indices)?)?.is_none()
            {
                return Ok(());
            }

            let partial_encoder = self.partial_encoder_with_options(chunk_indices, options)?;
            debug_assert!(
                partial_encoder.supports_partial_encode(),
                "partial encoder is misrepresenting its capabilities"
            );
            Ok(partial_encoder.partial_encode(indexer, &indexer_bytes, options)?)
        } else {
            let chunk_bytes_old: ArrayBytes<'static> =
                self.retrieve_chunk_with_options(chunk_indices, options)?;
            chunk_bytes_old.validate(chunk_shape.iter().product(), self.data_type())?;

            let chunk_bytes_new = update_array_bytes(
                chunk_bytes_old,
                &chunk_shape,
                indexer,
                &indexer_bytes,
                self.data_type().size(),
            )?;

            self.store_chunk_with_options(chunk_indices, chunk_bytes_new, options)
        }
    }

    pub(in crate::array) fn partial_encoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        self.partial_encoder_with_storage(self.readable_writable_storage()?, chunk_indices, options)
    }

    /// Initialise the partial encoder of the chunk at `chunk_indices` in `storage`, the storage of the array with its
    /// storage transformers applied.
    fn partial_encoder_with_storage(
        &self,
        storage: ReadableWritableStorage,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let input_handle = Arc::new((storage, chunk_key));
        Ok(self.codecs_bound().partial_encoder(
            input_handle,
            &self.chunk_shape(chunk_indices)?,
            options,
        )?)
    }
}

/// An encoded chunk updated in memory by a partial encoder, for a store without partial writes.
pub(super) struct InMemoryChunkUpdate {
    buffer: Arc<InMemoryBytesEncoder>,
    /// The partial encoder of the chunk in memory.
    pub(super) encoder: Arc<dyn ArrayPartialEncoderTraits>,
    /// The indices of the subchunk in the `level` chunk-local subchunk grid of [`encoder`](Self::encoder).
    pub(super) indices: ArrayIndices,
}

impl InMemoryChunkUpdate {
    /// Create a partial encoder of the `existing` encoded chunk at `chunk_indices`, and locate the subchunk at the
    /// chunk-local `subset` in its `level` subchunk grid.
    pub(super) fn new<A: ArrayOps + ?Sized>(
        array: &A,
        chunk_indices: &[u64],
        level: usize,
        subset: &ArraySubset,
        existing: Option<Bytes>,
    ) -> Result<Self, ArrayError> {
        let options = array.codec_options();
        let buffer = Arc::new(InMemoryBytesEncoder::new(existing));
        let encoder = array.codecs_bound().partial_encoder(
            buffer.clone(),
            &array.chunk_shape(chunk_indices)?,
            options,
        )?;
        let indices = super::local_subchunk_indices(encoder.as_ref(), level, subset, options)?;
        Ok(Self {
            buffer,
            encoder,
            indices,
        })
    }

    /// Finish the update, returning the update of the encoded chunk to store.
    pub(super) fn finish(self) -> InMemoryBytesUpdate {
        drop(self.encoder);
        self.buffer.take_update()
    }
}

use inherent::inherent;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::super::super::concurrency::concurrency_chunks_and_codec;
use super::super::*;
use super::ArrayWriteOps;
use crate::array::{ArrayIndicesTinyVec, ChunkShapeTraits};
use crate::iter_concurrent_limit;
use crate::node::{meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use zarrs_codec::ArrayToBytesCodecTraits;
use zarrs_storage::{Bytes, StorageHandle};

#[inherent]
impl<TStorage: ?Sized + WritableStorageTraits + 'static> ArrayWriteOps for Array<TStorage> {
    pub fn store_metadata(&self) -> Result<(), StorageError>;

    pub fn store_metadata_opt(&self, options: &ArrayMetadataOptions) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_writable_transformer(storage_handle)?;

        // Get the metadata with options applied and store
        let metadata = self.metadata_opt(options);

        // Store the metadata
        let path = self.path();
        match metadata {
            ArrayMetadata::V3(metadata) => {
                let key = meta_key_v3(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_transformer.set(&key, json.into())
            }
            ArrayMetadata::V2(metadata) => {
                let mut metadata = metadata.clone();

                if !metadata.attributes.is_empty() {
                    // Store .zattrs
                    let key = meta_key_v2_attributes(path);
                    let json = serde_json::to_vec_pretty(&metadata.attributes).map_err(|err| {
                        StorageError::InvalidMetadata(key.clone(), err.to_string())
                    })?;
                    storage_transformer.set(&meta_key_v2_attributes(path), json.into())?;

                    metadata.attributes = serde_json::Map::default();
                }

                // Store .zarray
                let key = meta_key_v2_array(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_transformer.set(&key, json.into())
            }
        }
    }

    pub fn erase_metadata(&self) -> Result<(), StorageError>;

    pub fn erase_metadata_opt(&self, options: MetadataEraseVersion) -> Result<(), StorageError> {
        let storage_handle = StorageHandle::new(self.storage.clone());
        match options {
            MetadataEraseVersion::Default => match self.metadata {
                ArrayMetadata::V3(_) => storage_handle.erase(&meta_key_v3(self.path())),
                ArrayMetadata::V2(_) => {
                    storage_handle.erase(&meta_key_v2_array(self.path()))?;
                    storage_handle.erase(&meta_key_v2_attributes(self.path()))
                }
            },
            MetadataEraseVersion::All => {
                storage_handle.erase(&meta_key_v3(self.path()))?;
                storage_handle.erase(&meta_key_v2_array(self.path()))?;
                storage_handle.erase(&meta_key_v2_attributes(self.path()))
            }
            MetadataEraseVersion::V3 => storage_handle.erase(&meta_key_v3(self.path())),
            MetadataEraseVersion::V2 => {
                storage_handle.erase(&meta_key_v2_array(self.path()))?;
                storage_handle.erase(&meta_key_v2_attributes(self.path()))
            }
        }
    }

    pub fn store_chunk<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError>;

    pub fn store_chunk_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_bytes = chunk_data.into_array_bytes(self.data_type())?;

        // Validation
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        chunk_bytes.validate(chunk_shape.num_elements_u64(), self.data_type())?;

        let is_fill_value =
            !options.store_empty_chunks() && chunk_bytes.is_fill_value(self.fill_value());
        if is_fill_value {
            self.erase_chunk(chunk_indices)?;
        } else {
            let chunk_encoded = self
                .codecs()
                .encode(
                    chunk_bytes,
                    &chunk_shape,
                    self.data_type(),
                    self.fill_value(),
                    options,
                )
                .map_err(ArrayError::CodecError)?;
            let chunk_encoded = Bytes::from(chunk_encoded.into_owned());
            unsafe { self.store_encoded_chunk(chunk_indices, chunk_encoded) }?;
        }
        Ok(())
    }

    pub fn store_chunks<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError>;

    pub fn store_chunks_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let chunks_bytes = chunks_data.into_array_bytes(self.data_type())?;
                chunks_bytes.validate(0, self.data_type())?;
            }
            1 => {
                let chunk_indices = chunks.start();
                self.store_chunk_opt(&chunk_indices, chunks_data, options)?;
            }
            _ => {
                let chunks_bytes = chunks_data.into_array_bytes(self.data_type())?;
                let array_subset = self.chunks_subset(chunks)?;
                chunks_bytes.validate(array_subset.num_elements(), self.data_type())?;

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

                let store_chunk = |chunk_indices: ArrayIndicesTinyVec| -> Result<(), ArrayError> {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_bytes = chunks_bytes.extract_array_subset(
                        &chunk_subset.relative_to(array_subset.start())?,
                        array_subset.shape(),
                        self.data_type(),
                    )?;
                    self.store_chunk_opt(&chunk_indices, chunk_bytes, &options)
                };

                let indices = chunks.indices();
                iter_concurrent_limit!(chunk_concurrent_limit, indices, try_for_each, store_chunk)?;
            }
        }

        Ok(())
    }

    pub fn erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_writable_transformer(storage_handle)?;
        storage_transformer.erase(&self.chunk_key(chunk_indices))
    }

    pub fn erase_chunks(&self, chunks: &dyn ArraySubsetTraits) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_writable_transformer(storage_handle)?;
        let erase_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            storage_transformer.erase(&self.chunk_key(&chunk_indices))
        };

        #[cfg(not(target_arch = "wasm32"))]
        chunks.indices().into_par_iter().try_for_each(erase_chunk)?;
        #[cfg(target_arch = "wasm32")]
        chunks.indices().into_iter().try_for_each(erase_chunk)?;

        Ok(())
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: bytes::Bytes,
    ) -> Result<(), ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_writable_transformer(storage_handle)?;
        storage_transformer.set(&self.chunk_key(chunk_indices), encoded_chunk_bytes)?;

        Ok(())
    }
}

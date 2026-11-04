use inherent::inherent;
use zarrs_codec::CowBytes;

use super::{ArrayWriteOps, *};
use crate::array::ChunkShapeTraits;
use crate::node::{meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use zarrs_codec::CodecError;

#[inherent]
impl<TStorage: ?Sized + WritableStorageTraits + 'static> ArrayWriteOps for Array<TStorage> {
    pub fn store_metadata(&self) -> Result<(), StorageError> {
        let storage_transformer = self.writable_storage()?;

        // Get the metadata with options applied and store
        let metadata = self.metadata_opt();

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
                let mut metadata = metadata;

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

    pub fn erase_metadata(&self) -> Result<(), StorageError> {
        let options = self.metadata_erase_version();
        let storage_handle = self.writable_storage()?;
        match options {
            MetadataEraseVersion::Default => match &*self.metadata {
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
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.store_chunk_with_options(chunk_indices, chunk_data, self.codec_options(), resources)
    }

    pub fn store_chunks<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let chunks_bytes = chunks_data.into_array_bytes(self.data_type())?;
                chunks_bytes.validate(0, self.data_type())?;
            }
            1 => {
                let chunk_indices = chunks.start();
                self.store_chunk_with_options(&chunk_indices, chunks_data, options, resources)?;
            }
            _ => {
                let chunks_bytes = chunks_data.into_array_bytes(self.data_type())?;
                let array_subset = self.chunks_subset(chunks)?;
                chunks_bytes.validate(array_subset.num_elements(), self.data_type())?;
                self.store_chunks_planned(
                    chunks,
                    &array_subset,
                    &chunks_bytes,
                    options,
                    resources,
                )?;
            }
        }

        Ok(())
    }

    pub fn erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), ArrayError> {
        let storage_transformer = self.writable_storage()?;
        Ok(storage_transformer.erase(&self.chunk_key(chunk_indices)?)?)
    }

    pub fn erase_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        chunks
            .validate(self.chunk_grid_shape())
            .map_err(CodecError::from)?;
        self.erase_chunks_planned(chunks, resources)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let storage_transformer = self.writable_storage()?;
        storage_transformer.set(&self.chunk_key(chunk_indices)?, encoded_chunk_bytes)?;

        Ok(())
    }
}

impl<TStorage: ?Sized + WritableStorageTraits + 'static> Array<TStorage> {
    pub(in crate::array) fn store_chunk_with_options<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_bytes = chunk_data.into_array_bytes(self.data_type())?;

        // Validation
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        chunk_bytes.validate(chunk_shape.num_elements_u64(), self.data_type())?;
        self.store_chunk_planned(chunk_indices, chunk_bytes, options, resources)
    }
}

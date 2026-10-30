use inherent::inherent;
use zarrs_codec::CowBytes;

use futures::{StreamExt, TryStreamExt};

use super::super::concurrency::concurrency_chunks_and_codec;
use super::{
    Array, ArrayError, ArrayMetadata, ArraySubsetTraits, AsyncArrayWriteOps,
    AsyncWritableStorageTraits, CodecOptions, Indexer, IntoArrayBytes, MaybeSend,
    MetadataEraseVersion, StorageError, recommended_codec_concurrency,
};
use crate::array::{ArrayIndices, ChunkShapeTraits};
use crate::node::{meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use zarrs_codec::{ArrayToBytesCodecTraits, CodecError};

#[cfg(feature = "async")]
#[inherent]
impl<TStorage: ?Sized + AsyncWritableStorageTraits + 'static> AsyncArrayWriteOps
    for Array<TStorage>
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata(&self) -> Result<(), StorageError> {
        let storage_transformer = self.async_writable_storage().await?;

        // Get the metadata with options applied and store
        let metadata = self.metadata_opt();

        // Store the metadata
        let path = self.path();
        match metadata {
            ArrayMetadata::V3(metadata) => {
                let key = meta_key_v3(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_transformer.set(&key, json.into()).await
            }
            ArrayMetadata::V2(metadata) => {
                let mut metadata = metadata.clone();

                if !metadata.attributes.is_empty() {
                    // Store .zattrs
                    let key = meta_key_v2_attributes(path);
                    let json = serde_json::to_vec_pretty(&metadata.attributes).map_err(|err| {
                        StorageError::InvalidMetadata(key.clone(), err.to_string())
                    })?;
                    storage_transformer
                        .set(&meta_key_v2_attributes(path), json.into())
                        .await?;

                    metadata.attributes = serde_json::Map::default();
                }

                // Store .zarray
                let key = meta_key_v2_array(path);
                let json = serde_json::to_vec_pretty(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key.clone(), err.to_string()))?;
                storage_transformer.set(&key, json.into()).await
            }
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata(&self) -> Result<(), StorageError> {
        let options = self.metadata_erase_version();
        let storage_transformer = self.async_writable_storage().await?;
        match options {
            MetadataEraseVersion::Default => match &*self.metadata {
                ArrayMetadata::V3(_) => storage_transformer.erase(&meta_key_v3(self.path())).await,
                ArrayMetadata::V2(_) => {
                    storage_transformer
                        .erase(&meta_key_v2_array(self.path()))
                        .await?;
                    storage_transformer
                        .erase(&meta_key_v2_attributes(self.path()))
                        .await
                }
            },
            MetadataEraseVersion::All => {
                storage_transformer.erase(&meta_key_v3(self.path())).await?;
                storage_transformer
                    .erase(&meta_key_v2_array(self.path()))
                    .await?;
                storage_transformer
                    .erase(&meta_key_v2_attributes(self.path()))
                    .await
            }
            MetadataEraseVersion::V3 => storage_transformer.erase(&meta_key_v3(self.path())).await,
            MetadataEraseVersion::V2 => {
                storage_transformer
                    .erase(&meta_key_v2_array(self.path()))
                    .await?;
                storage_transformer
                    .erase(&meta_key_v2_attributes(self.path()))
                    .await
            }
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_with_options(chunk_indices, chunk_data, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let subset_bytes = chunks_data.into_array_bytes(self.data_type())?;
                subset_bytes.validate(0, self.data_type())?;
            }
            1 => {
                let chunk_indices = chunks.start();
                self.async_store_chunk_with_options(&chunk_indices, chunks_data, options)
                    .await?;
            }
            _ => {
                let subset_bytes = chunks_data.into_array_bytes(self.data_type())?;
                let array_subset = self.chunks_subset(chunks)?;
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

                let store_chunk = |chunk_indices: ArrayIndices| {
                    let chunk_subset = self.chunk_subset(&chunk_indices).unwrap(); // FIXME: unwrap
                    let chunk_bytes = subset_bytes
                        .extract_array_subset(
                            &chunk_subset.relative_to(array_subset.start()).unwrap(), // FIXME: unwrap
                            array_subset.shape(),
                            self.data_type(),
                        )
                        .unwrap(); // FIXME: unwrap
                    async move {
                        self.async_store_chunk_with_options(&chunk_indices, chunk_bytes, &options)
                            .await
                    }
                };
                futures::stream::iter(&chunks.indices())
                    .map(Ok)
                    .try_for_each_concurrent(Some(chunk_concurrent_limit), store_chunk)
                    .await?;
            }
        }

        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_writable_storage().await?;
        Ok(storage_transformer.erase(&chunk_key).await?)
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunks(&self, chunks: &dyn Indexer) -> Result<(), ArrayError> {
        chunks
            .validate(self.chunk_grid_shape())
            .map_err(CodecError::from)?;
        let storage_transformer = self.async_writable_storage().await?;
        let erase_chunk = |chunk_indices: ArrayIndices| {
            let storage_transformer = storage_transformer.clone();
            async move {
                Ok::<_, ArrayError>(
                    storage_transformer
                        .erase(&self.chunk_key(&chunk_indices)?)
                        .await?,
                )
            }
        };
        futures::stream::iter(chunks.iter_indices())
            .map(Ok)
            .try_for_each_concurrent(None, erase_chunk)
            .await
    }
    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub async unsafe fn async_store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: CowBytes<'_>,
    ) -> Result<(), ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_writable_storage().await?;
        storage_transformer
            .set(&chunk_key, encoded_chunk_bytes)
            .await?;
        Ok(())
    }
}

impl<TStorage: ?Sized + AsyncWritableStorageTraits + 'static> Array<TStorage> {
    pub(in crate::array) async fn async_store_chunk_with_options<
        'a,
        T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_bytes = chunk_data.into_array_bytes(self.data_type())?;

        let chunk_shape = self.chunk_shape(chunk_indices)?;
        chunk_bytes.validate(chunk_shape.num_elements_u64(), self.data_type())?;

        let is_fill_value =
            !options.store_empty_chunks() && chunk_bytes.is_fill_value(self.fill_value());
        if is_fill_value {
            self.async_erase_chunk(chunk_indices).await?;
        } else {
            let chunk_encoded = self
                .codecs_bound()
                .encode(chunk_bytes, &chunk_shape, options)
                .map_err(ArrayError::CodecError)?;
            unsafe { self.async_store_encoded_chunk(chunk_indices, chunk_encoded) }.await?;
        }
        Ok(())
    }
}

use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};

use super::concurrency::concurrency_chunks_and_codec;
use super::{
    Array, ArrayError, ArrayIndicesTinyVec, ArrayMetadata, ArrayMetadataOptions, ChunkShapeTraits,
    Element, IntoArrayBytes,
};
use crate::array::ArraySubsetTraits;
use crate::config::MetadataEraseVersion;
use crate::node::{meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use zarrs_codec::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs_storage::{
    AsyncWritableStorageTraits, Bytes, MaybeSend, MaybeSync, StorageError, StorageHandle,
};

impl<TStorage: ?Sized + AsyncWritableStorageTraits + 'static> Array<TStorage> {
    /// Async variant of [`store_metadata`](Array::store_metadata).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata(&self) -> Result<(), StorageError> {
        self.async_store_metadata_opt(&self.metadata_options).await
    }

    /// Async variant of [`store_metadata_opt`](Array::store_metadata_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata_opt(
        &self,
        options: &ArrayMetadataOptions,
    ) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_writable_transformer(storage_handle)
            .await?;

        // Get the metadata with options applied and store
        let metadata = self.metadata_opt(options);

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

    /// Async variant of [`store_chunk`](Array::store_chunk).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_data: impl IntoArrayBytes<'a> + MaybeSend,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_opt(chunk_indices, chunk_data, &CodecOptions::default())
            .await
    }

    #[deprecated(since = "0.23.0", note = "Use async_store_chunk() instead")]
    /// Async variant of [`store_chunk_elements`](Array::store_chunk_elements).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_elements<T: Element + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_opt(chunk_indices, chunk_elements, &CodecOptions::default())
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(since = "0.23.0", note = "Use async_store_chunk()  instead")]
    /// Async variant of [`store_chunk_ndarray`](Array::store_chunk_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_ndarray<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_array: &ndarray::ArrayRef<T, D>,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_opt(
            chunk_indices,
            chunk_array.as_standard_layout().into_owned(),
            &CodecOptions::default(),
        )
        .await
    }

    /// Async variant of [`store_chunks`](Array::store_chunks).
    #[allow(clippy::missing_errors_doc)]
    #[allow(clippy::similar_names)]
    pub async fn async_store_chunks<'a>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: impl IntoArrayBytes<'a> + MaybeSend,
    ) -> Result<(), ArrayError> {
        self.async_store_chunks_opt(chunks, chunks_data, &CodecOptions::default())
            .await
    }

    #[deprecated(since = "0.23.0", note = "Use async_store_chunks() instead")]
    /// Async variant of [`store_chunks_elements`](Array::store_chunks_elements).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks_elements<T: Element + MaybeSend + MaybeSync>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.async_store_chunks_opt(chunks, chunks_elements, &CodecOptions::default())
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(since = "0.23.0", note = "Use async_store_chunks()  instead")]
    /// Async variant of [`store_chunks_ndarray`](Array::store_chunks_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks_ndarray<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_array: &ndarray::ArrayRef<T, D>,
    ) -> Result<(), ArrayError> {
        self.async_store_chunks_opt(
            chunks,
            chunks_array.as_standard_layout().into_owned(),
            &CodecOptions::default(),
        )
        .await
    }

    /// Async variant of [`erase_metadata`](Array::erase_metadata).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata(&self) -> Result<(), StorageError> {
        self.async_erase_metadata_opt(self.metadata_erase_version)
            .await
    }

    /// Async variant of [`erase_metadata_opt`](Array::erase_metadata_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata_opt(
        &self,
        options: MetadataEraseVersion,
    ) -> Result<(), StorageError> {
        let storage_handle = StorageHandle::new(self.storage.clone());
        match options {
            MetadataEraseVersion::Default => match self.metadata {
                ArrayMetadata::V3(_) => storage_handle.erase(&meta_key_v3(self.path())).await,
                ArrayMetadata::V2(_) => {
                    storage_handle
                        .erase(&meta_key_v2_array(self.path()))
                        .await?;
                    storage_handle
                        .erase(&meta_key_v2_attributes(self.path()))
                        .await
                }
            },
            MetadataEraseVersion::All => {
                storage_handle.erase(&meta_key_v3(self.path())).await?;
                storage_handle
                    .erase(&meta_key_v2_array(self.path()))
                    .await?;
                storage_handle
                    .erase(&meta_key_v2_attributes(self.path()))
                    .await
            }
            MetadataEraseVersion::V3 => storage_handle.erase(&meta_key_v3(self.path())).await,
            MetadataEraseVersion::V2 => {
                storage_handle
                    .erase(&meta_key_v2_array(self.path()))
                    .await?;
                storage_handle
                    .erase(&meta_key_v2_attributes(self.path()))
                    .await
            }
        }
    }

    /// Async variant of [`erase_chunk`](Array::erase_chunk).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_writable_transformer(storage_handle)
            .await?;
        storage_transformer
            .erase(&self.chunk_key(chunk_indices))
            .await
    }

    /// Async variant of [`erase_chunks`](Array::erase_chunks).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_writable_transformer(storage_handle)
            .await?;
        let erase_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            let storage_transformer = storage_transformer.clone();
            async move {
                storage_transformer
                    .erase(&self.chunk_key(&chunk_indices))
                    .await
            }
        };
        futures::stream::iter(chunks.indices().into_iter())
            .map(Ok)
            .try_for_each_concurrent(None, erase_chunk)
            .await
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    /// Async variant of [`store_chunk_opt`](Array::store_chunk_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_opt<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_data: impl IntoArrayBytes<'a> + MaybeSend,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_bytes = chunk_data.into_array_bytes(self.data_type())?;

        // Validation
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        chunk_bytes.validate(chunk_shape.num_elements_u64(), self.data_type())?;

        let is_fill_value =
            !options.store_empty_chunks() && chunk_bytes.is_fill_value(self.fill_value());
        if is_fill_value {
            self.async_erase_chunk(chunk_indices).await?;
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
            unsafe { self.async_store_encoded_chunk(chunk_indices, chunk_encoded) }.await?;
        }
        Ok(())
    }

    /// Async variant of [`store_encoded_chunk`](Array::store_encoded_chunk)
    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub async unsafe fn async_store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: Bytes,
    ) -> Result<(), ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_writable_transformer(storage_handle)
            .await?;
        storage_transformer
            .set(&self.chunk_key(chunk_indices), encoded_chunk_bytes)
            .await?;
        Ok(())
    }

    #[deprecated(since = "0.23.0", note = "Use async_store_chunk_opt() instead")]
    /// Async variant of [`store_chunk_elements_opt`](Array::store_chunk_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_elements_opt<T: Element + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let bytes = T::to_array_bytes(self.data_type(), chunk_elements)?;
        self.async_store_chunk_opt(chunk_indices, bytes, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(since = "0.23.0", note = "Use async_store_chunk_opt()  instead")]
    /// Async variant of [`store_chunk_ndarray_opt`](Array::store_chunk_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_ndarray_opt<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_array: &ndarray::ArrayRef<T, D>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape_usize(chunk_indices)?;
        if chunk_array.shape() == chunk_shape {
            self.async_store_chunk_opt(
                chunk_indices,
                chunk_array.as_standard_layout().to_owned(),
                options,
            )
            .await
        } else {
            Err(ArrayError::InvalidDataShape(
                chunk_array.shape().to_vec(),
                chunk_shape,
            ))
        }
    }

    /// Async variant of [`store_chunks_opt`](Array::store_chunks_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    #[allow(clippy::similar_names)]
    pub async fn async_store_chunks_opt<'a>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: impl IntoArrayBytes<'a> + MaybeSend,
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
                self.async_store_chunk_opt(&chunk_indices, chunks_data, options)
                    .await?;
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

                let store_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                    let chunk_subset = self.chunk_subset(&chunk_indices).unwrap(); // FIXME: unwrap
                    let chunk_bytes = chunks_bytes
                        .extract_array_subset(
                            &chunk_subset.relative_to(array_subset.start()).unwrap(), // FIXME: unwrap
                            array_subset.shape(),
                            self.data_type(),
                        )
                        .unwrap(); // FIXME: unwrap
                    async move {
                        self.async_store_chunk_opt(&chunk_indices, chunk_bytes, &options)
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

    #[deprecated(since = "0.23.0", note = "Use async_store_chunks_opt() instead")]
    /// Async variant of [`store_chunks_elements_opt`](Array::store_chunks_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks_elements_opt<T: Element + MaybeSend + MaybeSync>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunks_bytes = T::to_array_bytes(self.data_type(), chunks_elements)?;
        self.async_store_chunks_opt(chunks, chunks_bytes, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(since = "0.23.0", note = "Use async_store_chunks_opt()  instead")]
    /// Async variant of [`store_chunks_ndarray_opt`](Array::store_chunks_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks_ndarray_opt<
        T: Element + MaybeSend + MaybeSync,
        D: ndarray::Dimension,
    >(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_array: &ndarray::ArrayRef<T, D>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunks_subset = self.chunks_subset(chunks)?;
        let chunks_shape = chunks_subset.shape_usize();
        if chunks_array.shape() == chunks_shape {
            self.async_store_chunks_opt(
                chunks,
                chunks_array.as_standard_layout().to_owned(),
                options,
            )
            .await
        } else {
            Err(ArrayError::InvalidDataShape(
                chunks_array.shape().to_vec(),
                chunks_shape,
            ))
        }
    }
}

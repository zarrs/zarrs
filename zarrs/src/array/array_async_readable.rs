use std::borrow::Cow;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use unsafe_cell_slice::UnsafeCellSlice;

use super::concurrency::concurrency_chunks_and_codec;
use super::element::ElementOwned;
use super::{
    Array, ArrayBytes, ArrayBytesFixedDisjointView, ArrayCreateError, ArrayError,
    ArrayIndicesTinyVec, ArrayMetadata, ArrayMetadataV2, ArrayMetadataV3, ChunkShapeTraits,
    DataType, FromArrayBytes,
};
use crate::array::ArraySubsetTraits;
use crate::config::MetadataRetrieveVersion;
use crate::node::{NodePath, meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use crate::storage::{
    AsyncReadableStorageTraits, Bytes, MaybeSend, MaybeSync, StorageError, StorageHandle,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
    AsyncStoragePartialDecoder, CodecError, CodecOptions, build_nested_optional_target,
    copy_fill_value_into, merge_chunks_vlen, merge_chunks_vlen_optional, optional_nesting_depth,
};

impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
    /// Async variant of [`open`](Array::open).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_open(
        storage: Arc<TStorage>,
        path: &str,
    ) -> Result<Array<TStorage>, ArrayCreateError> {
        Self::async_open_opt(storage, path, &MetadataRetrieveVersion::Default).await
    }

    /// Async variant of [`open_opt`](Array::open_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_open_opt(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Array<TStorage>, ArrayCreateError> {
        let metadata = Self::async_open_metadata(storage.clone(), path, version).await?;
        Self::validate_metadata(&metadata)?;
        Self::new_with_metadata(storage, path, metadata)
    }

    async fn async_open_metadata(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<ArrayMetadata, ArrayCreateError> {
        let node_path = NodePath::new(path)?;

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V3 = version {
            // Try V3
            let key_v3 = meta_key_v3(&node_path);
            if let Some(metadata) = storage.get(&key_v3).await? {
                let metadata: ArrayMetadataV3 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v3, err.to_string()))?;
                return Ok(ArrayMetadata::V3(metadata));
            }
        }

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V2 = version {
            // Try V2
            let key_v2 = meta_key_v2_array(&node_path);
            if let Some(metadata) = storage.get(&key_v2).await? {
                let mut metadata: ArrayMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v2, err.to_string()))?;

                let attributes_key = meta_key_v2_attributes(&node_path);
                let attributes = storage.get(&attributes_key).await?;
                if let Some(attributes) = attributes {
                    metadata.attributes = serde_json::from_slice(&attributes).map_err(|err| {
                        StorageError::InvalidMetadata(attributes_key, err.to_string())
                    })?;
                }

                return Ok(ArrayMetadata::V2(metadata));
            }
        }

        Err(ArrayCreateError::MissingMetadata)
    }

    /// Async variant of [`retrieve_chunk_if_exists`](Array::retrieve_chunk_if_exists).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError> {
        self.async_retrieve_chunk_if_exists_opt(chunk_indices, &self.codec_options)
            .await
    }

    #[deprecated(since = "0.23.0", note = "Use async_retrieve_chunk_if_exists instead")]
    /// Async variant of [`retrieve_chunk_elements_if_exists`](Array::retrieve_chunk_elements_if_exists).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_elements_if_exists<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Vec<T>>, ArrayError> {
        self.async_retrieve_chunk_if_exists_opt(chunk_indices, &self.codec_options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_if_exists::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_ndarray_if_exists`](Array::retrieve_chunk_ndarray_if_exists).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_ndarray_if_exists<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ndarray::ArrayD<T>>, ArrayError> {
        self.async_retrieve_chunk_if_exists_opt(chunk_indices, &self.codec_options)
            .await
    }

    /// Retrieve the encoded bytes of a chunk.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    #[allow(clippy::missing_panics_doc)]
    pub async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;

        storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .await
    }

    /// Async variant of [`retrieve_chunk`](Array::retrieve_chunk).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunk_opt(chunk_indices, &self.codec_options)
            .await
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_elements`](Array::retrieve_chunk_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunk_opt(chunk_indices, &self.codec_options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_ndarray`](Array::retrieve_chunk_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunk_opt(chunk_indices, &self.codec_options)
            .await
    }

    /// Async variant of [`retrieve_chunks`](Array::retrieve_chunks).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunks_opt(chunks, &self.codec_options)
            .await
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunks::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunks_elements`](Array::retrieve_chunks_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunks_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunks_opt(chunks, &self.codec_options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunks::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunks_ndarray`](Array::retrieve_chunks_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunks_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunks_opt(chunks, &self.codec_options)
            .await
    }

    /// Async variant of [`retrieve_chunk_subset`](Array::retrieve_chunk_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &self.codec_options)
            .await
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_subset::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_subset_elements`](Array::retrieve_chunk_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &self.codec_options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_subset::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_subset_ndarray`](Array::retrieve_chunk_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &self.codec_options)
            .await
    }

    /// Async variant of [`retrieve_array_subset`](Array::retrieve_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_array_subset_opt(array_subset, &self.codec_options)
            .await
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_array_subset::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_array_subset_elements`](Array::retrieve_array_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_array_subset_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_array_subset_opt(array_subset, &self.codec_options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_array_subset::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_array_subset_ndarray`](Array::retrieve_array_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_array_subset_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_array_subset_opt(array_subset, &self.codec_options)
            .await
    }

    /// Async variant of [`partial_decoder`](Array::partial_decoder).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        self.async_partial_decoder_opt(chunk_indices, &self.codec_options)
            .await
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    /// Async variant of [`retrieve_chunk_if_exists_opt`](Array::retrieve_chunk_if_exists_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_if_exists_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<T>, ArrayError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .await
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = self
                .codecs()
                .decode(
                    Cow::Owned(chunk_encoded.into()),
                    &chunk_shape,
                    self.data_type(),
                    self.fill_value(),
                    options,
                )
                .map_err(ArrayError::CodecError)?;
            bytes.validate(chunk_shape.num_elements_u64(), self.data_type())?;
            Ok(Some(T::from_array_bytes(
                bytes.into_owned(),
                bytemuck::must_cast_slice(&chunk_shape),
                self.data_type(),
            )?))
        } else {
            Ok(None)
        }
    }

    /// Async variant of [`retrieve_chunk_opt`](Array::retrieve_chunk_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if let Some(chunk) = self
            .async_retrieve_chunk_if_exists_opt::<T>(chunk_indices, options)
            .await?
        {
            Ok(chunk)
        } else {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = ArrayBytes::new_fill_value(
                self.data_type(),
                chunk_shape.num_elements_u64(),
                self.fill_value(),
            )
            .map_err(CodecError::from)
            .map_err(ArrayError::from)?;
            T::from_array_bytes(
                bytes,
                bytemuck::must_cast_slice(&chunk_shape),
                self.data_type(),
            )
        }
    }

    /// Async variant of [`retrieve_chunk_into`](Array::retrieve_chunk_into).
    async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .await
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            self.codecs()
                .decode_into(
                    Cow::Owned(chunk_encoded.into()),
                    &chunk_shape,
                    self.data_type(),
                    self.fill_value(),
                    output_target,
                    options,
                )
                .map_err(ArrayError::CodecError)
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError)
        }
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_if_exists_opt::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_elements_if_exists_opt`](Array::retrieve_chunk_elements_if_exists_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_elements_if_exists_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Vec<T>>, ArrayError> {
        self.async_retrieve_chunk_if_exists_opt(chunk_indices, options)
            .await
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_opt::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_elements_opt`](Array::retrieve_chunk_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunk_opt(chunk_indices, options).await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_if_exists_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_ndarray_if_exists_opt`](Array::retrieve_chunk_ndarray_if_exists_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_ndarray_if_exists_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ndarray::ArrayD<T>>, ArrayError> {
        self.async_retrieve_chunk_if_exists_opt(chunk_indices, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_ndarray_opt`](Array::retrieve_chunk_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunk_opt(chunk_indices, options).await
    }

    /// Retrieve the encoded bytes of the chunks in `chunks`.
    ///
    /// The chunks are in order of the chunk indices returned by `chunks.indices().into_iter()`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    #[allow(clippy::missing_panics_doc)]
    pub async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Bytes>>, StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;

        let retrieve_encoded_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            let storage_transformer = storage_transformer.clone();
            async move {
                storage_transformer
                    .get(&self.chunk_key(&chunk_indices))
                    .await
            }
        };

        let indices = chunks.indices();
        let futures = indices.into_iter().map(retrieve_encoded_chunk);
        futures::stream::iter(futures)
            .buffered(options.concurrent_target())
            .try_collect()
            .await
    }

    /// Async variant of [`retrieve_chunks_opt`](Array::retrieve_chunks_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunks_opt<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if chunks.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                chunks.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        let array_subset = self.chunks_subset(chunks)?;
        self.async_retrieve_array_subset_opt(&array_subset, options)
            .await
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunks_opt::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunks_elements_opt`](Array::retrieve_chunks_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunks_opt(chunks, options).await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunks_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunks_ndarray_opt`](Array::retrieve_chunks_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunks_opt(chunks, options).await
    }

    /// Helper method to retrieve multiple chunks with variable-length data types (async).
    /// Also handles optional data types with variable-length inner types (including nested optionals).
    async fn async_retrieve_multi_chunk_variable(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let nesting_depth = optional_nesting_depth(data_type);
        let array_subset_start = array_subset.start();
        let array_subset_shape = array_subset.shape();

        if nesting_depth > 0 {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let array_subset_start = &array_subset_start;
                async move {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                    Ok::<_, ArrayError>((
                        self.async_retrieve_chunk_subset_opt::<ArrayBytes>(
                            &chunk_indices,
                            &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                            options,
                        )
                        .await?
                        .into_optional()?,
                        chunk_subset_overlap.relative_to(array_subset_start)?,
                    ))
                }
            };

            let chunk_bytes_and_subsets: Vec<_> = futures::stream::iter(chunks.indices().iter())
                .map(|chunk_indices| retrieve_chunk(chunk_indices.clone()))
                .buffered(chunk_concurrent_limit)
                .try_collect()
                .await?;

            Ok(ArrayBytes::Optional(merge_chunks_vlen_optional(
                chunk_bytes_and_subsets,
                &array_subset_shape,
                nesting_depth,
            )?))
        } else {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let array_subset_start = &array_subset_start;
                async move {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                    Ok::<_, ArrayError>((
                        self.async_retrieve_chunk_subset_opt::<ArrayBytes>(
                            &chunk_indices,
                            &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                            options,
                        )
                        .await?
                        .into_variable()?,
                        chunk_subset_overlap.relative_to(array_subset_start)?,
                    ))
                }
            };

            let chunk_bytes_and_subsets: Vec<_> = futures::stream::iter(chunks.indices().iter())
                .map(|chunk_indices| retrieve_chunk(chunk_indices.clone()))
                .buffered(chunk_concurrent_limit)
                .try_collect()
                .await?;

            Ok(ArrayBytes::Variable(merge_chunks_vlen(
                chunk_bytes_and_subsets,
                &array_subset_shape,
            )?))
        }
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types (async).
    /// Also handles optional data types with fixed-length inner types.
    async fn async_retrieve_multi_chunk_fixed(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        // Allocate data buffer and mask buffers for each level of optional nesting
        let data_type_size = data_type
            .fixed_size()
            .expect("data_type must have fixed size");
        let num_elements = array_subset.num_elements_usize();
        let size_output = num_elements * data_type_size;
        let nesting_depth = optional_nesting_depth(data_type);
        let mut data_output = Vec::with_capacity(size_output);
        let mut mask_outputs: Vec<Vec<u8>> = (0..nesting_depth)
            .map(|_| Vec::with_capacity(num_elements))
            .collect();

        {
            let data_output_slice =
                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut data_output);
            let mask_output_slices: Vec<_> = mask_outputs
                .iter_mut()
                .map(UnsafeCellSlice::new_from_vec_with_spare_capacity)
                .collect();
            let mask_output_slices = mask_output_slices.as_slice();
            let array_subset_start = array_subset.start();
            let array_subset_shape = array_subset.shape();

            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let array_subset_start = &array_subset_start;
                let array_subset_shape = &array_subset_shape;
                async move {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                    let chunk_subset_in_array =
                        chunk_subset_overlap.relative_to(array_subset_start)?;

                    let mut data_view = unsafe {
                        // SAFETY: chunks represent disjoint array subsets
                        ArrayBytesFixedDisjointView::new(
                            data_output_slice,
                            data_type_size,
                            array_subset_shape,
                            chunk_subset_in_array.clone(),
                        )?
                    };

                    // Create mask views for each nesting level
                    let mut mask_views: Vec<ArrayBytesFixedDisjointView<'_>> = mask_output_slices
                        .iter()
                        .map(|mask_slice| unsafe {
                            // SAFETY: chunks represent disjoint array subsets
                            ArrayBytesFixedDisjointView::new(
                                *mask_slice,
                                1, // 1 byte per element for mask
                                array_subset_shape,
                                chunk_subset_in_array.clone(),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    // Build the nested decode target
                    let target =
                        build_nested_optional_target(&mut data_view, mask_views.as_mut_slice());

                    self.async_retrieve_chunk_subset_into(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        target,
                        options,
                    )
                    .await?;
                    Ok::<_, ArrayError>(())
                }
            };

            futures::stream::iter(&chunks.indices())
                .map(Ok)
                .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
                .await?;
        }

        unsafe { data_output.set_len(size_output) };
        for mask in &mut mask_outputs {
            unsafe { mask.set_len(num_elements) };
        }

        // Build nested ArrayBytes with masks (innermost first, so reverse order)
        let mut array_bytes = ArrayBytes::new_flen(data_output);
        for mask in mask_outputs.into_iter().rev() {
            array_bytes = array_bytes.with_optional_mask(mask);
        }
        Ok(array_bytes)
    }

    /// Async variant of [`retrieve_array_subset_opt`](Array::retrieve_array_subset_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_array_subset_opt<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if array_subset.dimensionality() != self.dimensionality() {
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

        // Retrieve chunk bytes
        let num_chunks = chunks.num_elements_usize();
        let array_subset_shape = array_subset.shape();
        match num_chunks {
            0 => {
                let bytes = ArrayBytes::new_fill_value(
                    self.data_type(),
                    array_subset.num_elements(),
                    self.fill_value(),
                )
                .map_err(CodecError::from)
                .map_err(ArrayError::from)?;
                T::from_array_bytes(bytes, &array_subset_shape, self.data_type())
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    self.async_retrieve_chunk_opt(chunk_indices, options).await
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.async_retrieve_chunk_subset_opt(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        options,
                    )
                    .await
                }
            }
            _ => {
                // Calculate chunk/codec concurrency
                let chunk_shape = self.chunk_shape(chunks.start())?;
                let codec_concurrency =
                    self.recommended_codec_concurrency(&chunk_shape, self.data_type())?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                // Delegate to appropriate helper based on data type size
                let bytes = if self.data_type().is_fixed() {
                    self.async_retrieve_multi_chunk_fixed(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )
                    .await?
                } else {
                    self.async_retrieve_multi_chunk_variable(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )
                    .await?
                };
                T::from_array_bytes(bytes.into_owned(), &array_subset_shape, self.data_type())
            }
        }
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_array_subset_opt::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_array_subset_elements_opt`](Array::retrieve_array_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_elements_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_array_subset_opt(array_subset, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_array_subset_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_array_subset_ndarray_opt`](Array::retrieve_array_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_ndarray_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_array_subset_opt(array_subset, options)
            .await
    }

    /// Async variant of [`retrieve_chunk_subset_opt`](Array::retrieve_chunk_subset_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if !chunk_subset.inbounds_shape(chunk_shape_u64) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        let chunk_subset_shape = chunk_subset.shape();
        if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset_shape.as_ref() == chunk_shape_u64
        {
            // Fast path if `chunk_subset` encompasses the whole chunk
            self.async_retrieve_chunk_opt(chunk_indices, options).await
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_async_readable_transformer(storage_handle)
                .await?;
            let input_handle = Arc::new(AsyncStoragePartialDecoder::new(
                storage_transformer,
                self.chunk_key(chunk_indices),
            ));
            let bytes = self
                .codecs
                .clone()
                .async_partial_decoder(
                    input_handle,
                    &chunk_shape,
                    self.data_type(),
                    self.fill_value(),
                    options,
                )
                .await?
                .partial_decode(chunk_subset, options)
                .await?
                .into_owned();
            bytes.validate(chunk_subset.num_elements(), self.data_type())?;
            T::from_array_bytes(bytes, &chunk_subset_shape, self.data_type())
        }
    }

    async fn async_retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if !chunk_subset.inbounds_shape(chunk_shape_u64) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape().as_ref() == chunk_shape_u64
        {
            // Fast path if `chunk_subset` encompasses the whole chunk
            self.async_retrieve_chunk_into(chunk_indices, output_target, options)
                .await
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_async_readable_transformer(storage_handle)
                .await?;
            let input_handle = Arc::new(AsyncStoragePartialDecoder::new(
                storage_transformer,
                self.chunk_key(chunk_indices),
            ));

            self.codecs
                .clone()
                .async_partial_decoder(
                    input_handle,
                    &chunk_shape,
                    self.data_type(),
                    self.fill_value(),
                    options,
                )
                .await?
                .partial_decode_into(chunk_subset, output_target, options)
                .await?;
            Ok(())
        }
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_subset_opt::<Vec<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_subset_elements_opt`](Array::retrieve_chunk_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset_elements_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_chunk_subset_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Async variant of [`retrieve_chunk_subset_ndarray_opt`](Array::retrieve_chunk_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset_ndarray_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, options)
            .await
    }

    /// Async variant of [`partial_decoder_opt`](Array::partial_decoder_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_decoder_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;
        let input_handle = Arc::new(AsyncStoragePartialDecoder::new(
            storage_transformer,
            self.chunk_key(chunk_indices),
        ));
        Ok(self
            .codecs
            .clone()
            .async_partial_decoder(
                input_handle,
                &self.chunk_shape(chunk_indices)?,
                self.data_type(),
                self.fill_value(),
                options,
            )
            .await?)
    }
}

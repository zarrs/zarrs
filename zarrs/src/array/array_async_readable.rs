use std::{borrow::Cow, sync::Arc};

use futures::{StreamExt, TryStreamExt};
use unsafe_cell_slice::UnsafeCellSlice;

#[cfg(feature = "ndarray")]
use super::elements_to_ndarray;
use super::{
    array_bytes::{
        build_nested_optional_target, copy_fill_value_into, merge_chunks_vlen,
        optional_nesting_depth,
    },
    codec::{
        ArrayBytesDecodeIntoTarget, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
        AsyncStoragePartialDecoder, CodecError, CodecOptions,
    },
    concurrency::concurrency_chunks_and_codec,
    element::ElementOwned,
    Array, ArrayBytes, ArrayBytesFixedDisjointView, ArrayCreateError, ArrayError, ArrayMetadata,
    ArrayMetadataV2, ArrayMetadataV3, DataType,
};
use crate::storage::{MaybeSend, MaybeSync};
use crate::{
    array_subset::ArraySubset,
    config::MetadataRetrieveVersion,
    node::{meta_key_v2_array, meta_key_v2_attributes, meta_key_v3, NodePath},
    storage::{AsyncReadableStorageTraits, Bytes, StorageError, StorageHandle},
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
    pub async fn async_retrieve_chunk_if_exists(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytes<'_>>, ArrayError> {
        self.async_retrieve_chunk_if_exists_opt(chunk_indices, &CodecOptions::default())
            .await
    }

    /// Async variant of [`retrieve_chunk_elements_if_exists`](Array::retrieve_chunk_elements_if_exists).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_elements_if_exists<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Vec<T>>, ArrayError> {
        self.async_retrieve_chunk_elements_if_exists_opt(chunk_indices, &CodecOptions::default())
            .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunk_ndarray_if_exists`](Array::retrieve_chunk_ndarray_if_exists).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_ndarray_if_exists<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ndarray::ArrayD<T>>, ArrayError> {
        self.async_retrieve_chunk_ndarray_if_exists_opt(chunk_indices, &CodecOptions::default())
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
    pub async fn async_retrieve_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        self.async_retrieve_chunk_opt(chunk_indices, &CodecOptions::default())
            .await
    }

    /// Async variant of [`retrieve_chunk_elements`](Array::retrieve_chunk_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunk_elements_opt(chunk_indices, &CodecOptions::default())
            .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunk_ndarray`](Array::retrieve_chunk_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunk_ndarray_opt(chunk_indices, &CodecOptions::default())
            .await
    }

    /// Async variant of [`retrieve_chunks`](Array::retrieve_chunks).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunks(
        &self,
        chunks: &ArraySubset,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        self.async_retrieve_chunks_opt(chunks, &CodecOptions::default())
            .await
    }

    /// Async variant of [`retrieve_chunks_elements`](Array::retrieve_chunks_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunks_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &ArraySubset,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunks_elements_opt(chunks, &CodecOptions::default())
            .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunks_ndarray`](Array::retrieve_chunks_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunks_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &ArraySubset,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunks_ndarray_opt(chunks, &CodecOptions::default())
            .await
    }

    /// Async variant of [`retrieve_chunk_subset`](Array::retrieve_chunk_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        self.async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &CodecOptions::default())
            .await
    }

    /// Async variant of [`retrieve_chunk_subset_elements`](Array::retrieve_chunk_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_chunk_subset_elements_opt(
            chunk_indices,
            chunk_subset,
            &CodecOptions::default(),
        )
        .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunk_subset_ndarray`](Array::retrieve_chunk_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_chunk_subset_ndarray_opt(
            chunk_indices,
            chunk_subset,
            &CodecOptions::default(),
        )
        .await
    }

    /// Async variant of [`retrieve_array_subset`](Array::retrieve_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_array_subset(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        self.async_retrieve_array_subset_opt(array_subset, &CodecOptions::default())
            .await
    }

    /// Async variant of [`retrieve_array_subset_elements`](Array::retrieve_array_subset_elements).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_array_subset_elements<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_array_subset_elements_opt(array_subset, &CodecOptions::default())
            .await
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_array_subset_ndarray`](Array::retrieve_array_subset_ndarray).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_array_subset_ndarray<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_array_subset_ndarray_opt(array_subset, &CodecOptions::default())
            .await
    }

    /// Async variant of [`partial_decoder`](Array::partial_decoder).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        self.async_partial_decoder_opt(chunk_indices, &CodecOptions::default())
            .await
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    /// Async variant of [`retrieve_chunk_if_exists_opt`](Array::retrieve_chunk_if_exists_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_if_exists_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytes<'_>>, ArrayError> {
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
            let chunk_encoded: Vec<u8> = chunk_encoded.into();
            let chunk_representation = self.chunk_array_representation(chunk_indices)?;
            let bytes = self
                .codecs()
                .decode(Cow::Owned(chunk_encoded), &chunk_representation, options)
                .map_err(ArrayError::CodecError)?;
            bytes.validate(
                chunk_representation.num_elements(),
                chunk_representation.data_type(),
            )?;
            Ok(Some(bytes))
        } else {
            Ok(None)
        }
    }

    /// Async variant of [`retrieve_chunk_opt`](Array::retrieve_chunk_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let chunk = self
            .async_retrieve_chunk_if_exists_opt(chunk_indices, options)
            .await?;
        if let Some(chunk) = chunk {
            Ok(chunk)
        } else {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            ArrayBytes::new_fill_value(
                self.data_type(),
                chunk_shape.num_elements_u64(),
                self.fill_value(),
            )
            .map_err(CodecError::from)
            .map_err(ArrayError::from)
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
            let chunk_encoded: Vec<u8> = chunk_encoded.into();
            let chunk_representation = self.chunk_array_representation(chunk_indices)?;
            self.codecs()
                .decode_into(
                    Cow::Owned(chunk_encoded),
                    &chunk_representation,
                    output_target,
                    options,
                )
                .map_err(ArrayError::CodecError)
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError)
        }
    }

    /// Async variant of [`retrieve_chunk_elements_if_exists_opt`](Array::retrieve_chunk_elements_if_exists_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_elements_if_exists_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Vec<T>>, ArrayError> {
        if let Some(bytes) = self
            .async_retrieve_chunk_if_exists_opt(chunk_indices, options)
            .await?
        {
            let elements = T::from_array_bytes(self.data_type(), bytes)?;
            Ok(Some(elements))
        } else {
            Ok(None)
        }
    }

    /// Async variant of [`retrieve_chunk_elements_opt`](Array::retrieve_chunk_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        let bytes = self
            .async_retrieve_chunk_opt(chunk_indices, options)
            .await?;
        let elements = T::from_array_bytes(self.data_type(), bytes)?;
        Ok(elements)
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunk_ndarray_if_exists_opt`](Array::retrieve_chunk_ndarray_if_exists_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_ndarray_if_exists_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ndarray::ArrayD<T>>, ArrayError> {
        // validate_element_size::<T>(self.data_type())?; in // async_retrieve_chunk_elements_if_exists
        let shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        let elements = self
            .async_retrieve_chunk_elements_if_exists_opt(chunk_indices, options)
            .await?;
        if let Some(elements) = elements {
            Ok(Some(elements_to_ndarray(&shape, elements)?))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunk_ndarray_opt`](Array::retrieve_chunk_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        // validate_element_size::<T>(self.data_type())?; // in async_retrieve_chunk_elements
        let shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        let elements = self
            .async_retrieve_chunk_elements_opt(chunk_indices, options)
            .await?;
        elements_to_ndarray(&shape, elements)
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
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Bytes>>, StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;

        let retrieve_encoded_chunk = |chunk_indices: Vec<u64>| {
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
    pub async fn async_retrieve_chunks_opt(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if chunks.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                chunks.clone(),
                self.shape().to_vec(),
            ));
        }

        let array_subset = self.chunks_subset(chunks)?;
        self.async_retrieve_array_subset_opt(&array_subset, options)
            .await
    }

    /// Async variant of [`retrieve_chunks_elements_opt`](Array::retrieve_chunks_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        let bytes = self.async_retrieve_chunks_opt(chunks, options).await?;
        let elements = T::from_array_bytes(self.data_type(), bytes)?;
        Ok(elements)
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunks_ndarray_opt`](Array::retrieve_chunks_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        let elements = self
            .async_retrieve_chunks_elements_opt(chunks, options)
            .await?;
        elements_to_ndarray(array_subset.shape(), elements)
    }

    /// Helper method to retrieve multiple chunks with variable-length data types (async).
    /// Also handles optional data types with variable-length inner types (will error).
    async fn async_retrieve_multi_chunk_variable(
        &self,
        array_subset: &ArraySubset,
        chunks: &ArraySubset,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if data_type.is_optional() {
            // Optional data type with variable-length inner type is not supported
            return Err(ArrayError::CodecError(CodecError::Other(
                "Optional data type with variable-length inner type is not supported in multi-chunk retrieval".to_string(),
            )));
        }

        let retrieve_chunk = |chunk_indices: Vec<u64>| {
            let options = options.clone();
            async move {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                Ok::<_, ArrayError>((
                    self.async_retrieve_chunk_subset_opt(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        &options,
                    )
                    .await?,
                    chunk_subset_overlap.relative_to(array_subset.start())?,
                ))
            }
        };

        let chunk_bytes_and_subsets: Vec<_> = futures::stream::iter(chunks.indices().iter())
            .map(|chunk_indices| retrieve_chunk(chunk_indices.clone()))
            .buffered(chunk_concurrent_limit)
            .try_collect()
            .await?;

        Ok(merge_chunks_vlen(
            chunk_bytes_and_subsets,
            array_subset.shape(),
        )?)
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types (async).
    /// Also handles optional data types with fixed-length inner types.
    async fn async_retrieve_multi_chunk_fixed(
        &self,
        array_subset: &ArraySubset,
        chunks: &ArraySubset,
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

            let retrieve_chunk = |chunk_indices: Vec<u64>| {
                let options = options.clone();
                async move {
                    let chunk_subset = self.chunk_subset(&chunk_indices)?;
                    let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                    let chunk_subset_in_array =
                        chunk_subset_overlap.relative_to(array_subset.start())?;

                    let mut data_view = unsafe {
                        // SAFETY: chunks represent disjoint array subsets
                        ArrayBytesFixedDisjointView::new(
                            data_output_slice,
                            data_type_size,
                            array_subset.shape(),
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
                                array_subset.shape(),
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
                        &options,
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
    pub async fn async_retrieve_array_subset_opt(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if array_subset.dimensionality() != self.dimensionality() {
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

        // Retrieve chunk bytes
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => ArrayBytes::new_fill_value(
                self.data_type(),
                array_subset.num_elements(),
                self.fill_value(),
            )
            .map_err(CodecError::from)
            .map_err(ArrayError::from),
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if &chunk_subset == array_subset {
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
                let chunk_representation =
                    self.chunk_array_representation(&vec![0; self.dimensionality()])?;
                let codec_concurrency =
                    self.recommended_codec_concurrency(&chunk_representation)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                // Delegate to appropriate helper based on data type size
                let data_type = chunk_representation.data_type();
                if data_type.is_fixed() {
                    self.async_retrieve_multi_chunk_fixed(
                        array_subset,
                        &chunks,
                        data_type,
                        chunk_concurrent_limit,
                        &options,
                    )
                    .await
                } else {
                    self.async_retrieve_multi_chunk_variable(
                        array_subset,
                        &chunks,
                        data_type,
                        chunk_concurrent_limit,
                        &options,
                    )
                    .await
                }
            }
        }
    }

    /// Async variant of [`retrieve_array_subset_elements_opt`](Array::retrieve_array_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_elements_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        let bytes = self
            .async_retrieve_array_subset_opt(array_subset, options)
            .await?;
        let elements = T::from_array_bytes(self.data_type(), bytes)?;
        Ok(elements)
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_array_subset_ndarray_opt`](Array::retrieve_array_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_ndarray_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements = self
            .async_retrieve_array_subset_elements_opt(array_subset, options)
            .await?;
        elements_to_ndarray(array_subset.shape(), elements)
    }

    /// Async variant of [`retrieve_chunk_subset_opt`](Array::retrieve_chunk_subset_opt).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub async fn async_retrieve_chunk_subset_opt(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let chunk_representation = self.chunk_array_representation(chunk_indices)?;
        if !chunk_subset.inbounds_shape(&chunk_representation.shape_u64()) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.clone(),
                self.shape().to_vec(),
            ));
        }

        let bytes = if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape() == chunk_representation.shape_u64()
        {
            // Fast path if `chunk_subset` encompasses the whole chunk
            self.async_retrieve_chunk_opt(chunk_indices, options)
                .await?
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
                .async_partial_decoder(input_handle, &chunk_representation, options)
                .await?
                .partial_decode(chunk_subset, options)
                .await?
                .into_owned()
        };
        bytes.validate(chunk_subset.num_elements(), self.data_type())?;
        Ok(bytes)
    }

    async fn async_retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_representation = self.chunk_array_representation(chunk_indices)?;
        if !chunk_subset.inbounds_shape(&chunk_representation.shape_u64()) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.clone(),
                self.shape().to_vec(),
            ));
        }

        if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape() == chunk_representation.shape_u64()
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
                .async_partial_decoder(input_handle, &chunk_representation, options)
                .await?
                .partial_decode_into(chunk_subset, output_target, options)
                .await?;
            Ok(())
        }
    }

    /// Async variant of [`retrieve_chunk_subset_elements_opt`](Array::retrieve_chunk_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset_elements_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        let bytes = self
            .async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, options)
            .await?;
        let elements = T::from_array_bytes(self.data_type(), bytes)?;
        Ok(elements)
    }

    #[cfg(feature = "ndarray")]
    /// Async variant of [`retrieve_chunk_subset_ndarray_opt`](Array::retrieve_chunk_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset_ndarray_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements = self
            .async_retrieve_chunk_subset_elements_opt(chunk_indices, chunk_subset, options)
            .await?;
        elements_to_ndarray(chunk_subset.shape(), elements)
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
        let chunk_representation = self.chunk_array_representation(chunk_indices)?;
        Ok(self
            .codecs
            .clone()
            .async_partial_decoder(input_handle, &chunk_representation, options)
            .await?)
    }
}

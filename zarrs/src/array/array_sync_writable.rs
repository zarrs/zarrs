use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{
    codec::{ArrayToBytesCodecTraits, CodecOptions},
    concurrency::concurrency_chunks_and_codec,
    Array, ArrayError, ArrayMetadata, ArrayMetadataOptions, Element,
};
use crate::iter_concurrent_limit;
use crate::{
    array::ArrayBytes,
    array_subset::ArraySubset,
    config::{global_config, MetadataEraseVersion},
    node::{meta_key_v2_array, meta_key_v2_attributes, meta_key_v3},
    storage::{Bytes, StorageError, StorageHandle, WritableStorageTraits},
};

impl<TStorage: ?Sized + WritableStorageTraits + 'static> Array<TStorage> {
    /// Store metadata with default [`ArrayMetadataOptions`].
    ///
    /// The metadata is created with [`Array::metadata_opt`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
    pub fn store_metadata(&self) -> Result<(), StorageError> {
        self.store_metadata_opt(&ArrayMetadataOptions::default())
    }

    /// Store metadata with non-default [`ArrayMetadataOptions`].
    ///
    /// The metadata is created with [`Array::metadata_opt`].
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
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

    /// Encode `chunk_bytes` and store at `chunk_indices`.
    ///
    /// Use [`store_chunk_opt`](Array::store_chunk_opt) to control codec options.
    /// A chunk composed entirely of the fill value will not be written to the store.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_indices` are invalid,
    ///  - the length of `chunk_bytes` is not equal to the expected length (the product of the number of elements in the chunk and the data type size in bytes),
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    pub fn store_chunk<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_bytes: impl Into<ArrayBytes<'a>>,
    ) -> Result<(), ArrayError> {
        self.store_chunk_opt(chunk_indices, chunk_bytes, &CodecOptions::default())
    }

    /// Encode `chunk_elements` and store at `chunk_indices`.
    ///
    /// Use [`store_chunk_elements_opt`](Array::store_chunk_elements_opt) to control codec options.
    /// A chunk composed entirely of the fill value will not be written to the store.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the size of  `T` does not match the data type size, or
    ///  - a [`store_chunk`](Array::store_chunk) error condition is met.
    pub fn store_chunk_elements<T: Element>(
        &self,
        chunk_indices: &[u64],
        chunk_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.store_chunk_elements_opt(chunk_indices, chunk_elements, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    /// Encode `chunk_array` and store at `chunk_indices`.
    ///
    /// Use [`store_chunk_ndarray_opt`](Array::store_chunk_ndarray_opt) to control codec options.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the shape of the array does not match the shape of the chunk,
    ///  - a [`store_chunk_elements`](Array::store_chunk_elements) error condition is met.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunk_ndarray<T: Element, D: ndarray::Dimension>(
        &self,
        chunk_indices: &[u64],
        chunk_array: &ndarray::ArrayRef<T, D>,
    ) -> Result<(), ArrayError> {
        self.store_chunk_ndarray_opt(chunk_indices, chunk_array, &CodecOptions::default())
    }

    /// Encode `chunks_bytes` and store at the chunks with indices represented by the `chunks` array subset.
    ///
    /// Use [`store_chunks_opt`](Array::store_chunks_opt) to control codec options.
    /// A chunk composed entirely of the fill value will not be written to the store.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunks` are invalid,
    ///  - the length of `chunk_bytes` is not equal to the expected length (the product of the number of elements in the chunks and the data type size in bytes),
    ///  - there is a codec encoding error, or
    ///  - an underlying store error.
    #[allow(clippy::similar_names)]
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunks<'a>(
        &self,
        chunks: &ArraySubset,
        chunks_bytes: impl Into<ArrayBytes<'a>>,
    ) -> Result<(), ArrayError> {
        self.store_chunks_opt(chunks, chunks_bytes, &CodecOptions::default())
    }

    /// Encode `chunks_elements` and store at the chunks with indices represented by the `chunks` array subset.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the size of  `T` does not match the data type size, or
    ///  - a [`store_chunks`](Array::store_chunks) error condition is met.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunks_elements<T: Element>(
        &self,
        chunks: &ArraySubset,
        chunks_elements: &[T],
    ) -> Result<(), ArrayError> {
        self.store_chunks_elements_opt(chunks, chunks_elements, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    /// Encode `chunks_array` and store at the chunks with indices represented by the `chunks` array subset.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the shape of the array does not match the shape of the chunks,
    ///  - a [`store_chunks_elements`](Array::store_chunks_elements) error condition is met.
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunks_ndarray<T: Element, D: ndarray::Dimension>(
        &self,
        chunks: &ArraySubset,
        chunks_array: &ndarray::ArrayRef<T, D>,
    ) -> Result<(), ArrayError> {
        self.store_chunks_ndarray_opt(chunks, chunks_array, &CodecOptions::default())
    }

    /// Erase the metadata with default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    pub fn erase_metadata(&self) -> Result<(), StorageError> {
        let erase_version = global_config().metadata_erase_version();
        self.erase_metadata_opt(erase_version)
    }

    /// Erase the metadata with non-default [`MetadataEraseVersion`] options.
    ///
    /// Succeeds if the metadata does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
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

    /// Erase the chunk at `chunk_indices`.
    ///
    /// Succeeds if the chunk does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    pub fn erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_writable_transformer(storage_handle)?;
        storage_transformer.erase(&self.chunk_key(chunk_indices))
    }

    /// Erase the chunks in `chunks`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    pub fn erase_chunks(&self, chunks: &ArraySubset) -> Result<(), StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_writable_transformer(storage_handle)?;
        let erase_chunk =
            |chunk_indices: Vec<u64>| storage_transformer.erase(&self.chunk_key(&chunk_indices));

        #[cfg(not(target_arch = "wasm32"))]
        chunks.indices().into_par_iter().try_for_each(erase_chunk)?;
        #[cfg(target_arch = "wasm32")]
        chunks.indices().into_iter().try_for_each(erase_chunk)?;

        Ok(())
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    /// Explicit options version of [`store_chunk`](Array::store_chunk).
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_opt<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_bytes: impl Into<ArrayBytes<'a>>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_bytes = chunk_bytes.into();

        // Validation
        let chunk_array_representation = self.chunk_array_representation(chunk_indices)?;
        chunk_bytes.validate(
            chunk_array_representation.num_elements(),
            chunk_array_representation.data_type(),
        )?;

        let is_fill_value =
            !options.store_empty_chunks() && chunk_bytes.is_fill_value(self.fill_value());
        if is_fill_value {
            self.erase_chunk(chunk_indices)?;
        } else {
            let chunk_encoded = self
                .codecs()
                .encode(chunk_bytes, &chunk_array_representation, options)
                .map_err(ArrayError::CodecError)?;
            let chunk_encoded = Bytes::from(chunk_encoded.into_owned());
            unsafe { self.store_encoded_chunk(chunk_indices, chunk_encoded) }?;
        }
        Ok(())
    }

    /// Store `encoded_chunk_bytes` at `chunk_indices`
    ///
    /// # Safety
    /// The responsibility is on the caller to ensure the chunk is encoded correctly
    ///
    /// # Errors
    /// Returns [`StorageError`] if there is an underlying store error.
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

    /// Explicit options version of [`store_chunk_elements`](Array::store_chunk_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_elements_opt<T: Element>(
        &self,
        chunk_indices: &[u64],
        chunk_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_bytes = T::into_array_bytes(self.data_type(), chunk_elements)?;
        self.store_chunk_opt(chunk_indices, chunk_bytes, options)
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`store_chunk_ndarray`](Array::store_chunk_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_ndarray_opt<T: Element, D: ndarray::Dimension>(
        &self,
        chunk_indices: &[u64],
        chunk_array: &ndarray::ArrayRef<T, D>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape_usize(chunk_indices)?;
        if chunk_array.shape() == chunk_shape {
            let chunk_array = super::ndarray_to_cow(chunk_array);
            self.store_chunk_elements_opt(chunk_indices, &chunk_array, options)
        } else {
            Err(ArrayError::InvalidDataShape(
                chunk_array.shape().to_vec(),
                chunk_shape,
            ))
        }
    }

    /// Explicit options version of [`store_chunks`](Array::store_chunks).
    #[allow(clippy::similar_names)]
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn store_chunks_opt<'a>(
        &self,
        chunks: &ArraySubset,
        chunks_bytes: impl Into<ArrayBytes<'a>>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let chunks_bytes = chunks_bytes.into();
                chunks_bytes.validate(0, self.data_type())?;
            }
            1 => {
                let chunk_indices = chunks.start();
                self.store_chunk_opt(chunk_indices, chunks_bytes, options)?;
            }
            _ => {
                let chunks_bytes = chunks_bytes.into();
                let array_subset = self.chunks_subset(chunks)?;
                chunks_bytes.validate(array_subset.num_elements(), self.data_type())?;

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

                let store_chunk = |chunk_indices: Vec<u64>| -> Result<(), ArrayError> {
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

    /// Explicit options version of [`store_chunks_elements`](Array::store_chunks_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunks_elements_opt<T: Element>(
        &self,
        chunks: &ArraySubset,
        chunks_elements: &[T],
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunks_bytes = T::into_array_bytes(self.data_type(), chunks_elements)?;
        self.store_chunks_opt(chunks, chunks_bytes, options)
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`store_chunks_ndarray`](Array::store_chunks_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunks_ndarray_opt<T: Element, D: ndarray::Dimension>(
        &self,
        chunks: &ArraySubset,
        chunks_array: &ndarray::ArrayRef<T, D>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunks_subset = self.chunks_subset(chunks)?;
        let chunks_shape = chunks_subset.shape_usize();
        if chunks_array.shape() == chunks_shape {
            let chunks_array = super::ndarray_to_cow(chunks_array);
            self.store_chunks_elements_opt(chunks, &chunks_array, options)
        } else {
            Err(ArrayError::InvalidDataShape(
                chunks_array.shape().to_vec(),
                chunks_shape,
            ))
        }
    }
}

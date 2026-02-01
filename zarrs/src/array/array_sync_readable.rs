use std::borrow::Cow;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use unsafe_cell_slice::UnsafeCellSlice;

use super::concurrency::concurrency_chunks_and_codec;
use super::element::ElementOwned;
use super::{
    Array, ArrayBytesFixedDisjointView, ArrayCreateError, ArrayError, ArrayIndicesTinyVec,
    ArrayMetadata, ArrayMetadataV3, DataType, FromArrayBytes,
};
use crate::array::{ArrayBytes, ArrayMetadataV2, ArraySubset, ArraySubsetTraits};
use crate::config::MetadataRetrieveVersion;
use crate::iter_concurrent_limit;
use crate::node::{NodePath, meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayBytesOptional, ArrayBytesVariableLength,
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions,
    StoragePartialDecoder, copy_fill_value_into,
};

use super::array_bytes_internal::{
    build_nested_optional_target, merge_chunks_vlen, merge_chunks_vlen_optional,
    optional_nesting_depth,
};
use zarrs_storage::{ReadableStorageTraits, StorageError, StorageHandle};

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Array<TStorage> {
    /// Open an existing array in `storage` at `path` with default [`MetadataRetrieveVersion`].
    /// The metadata is read from the store.
    ///
    /// # Errors
    /// Returns [`ArrayCreateError`] if there is a storage error or any metadata is invalid.
    pub fn open(storage: Arc<TStorage>, path: &str) -> Result<Self, ArrayCreateError> {
        Self::open_opt(storage, path, &MetadataRetrieveVersion::Default)
    }

    /// Open an existing array in `storage` at `path` with non-default [`MetadataRetrieveVersion`].
    /// The metadata is read from the store.
    ///
    /// # Errors
    /// Returns [`ArrayCreateError`] if there is a storage error or any metadata is invalid.
    pub fn open_opt(
        storage: Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<Self, ArrayCreateError> {
        let metadata = Self::open_metadata(&storage, path, version)?;
        Self::validate_metadata(&metadata)?;
        Self::new_with_metadata(storage, path, metadata)
    }

    fn open_metadata(
        storage: &Arc<TStorage>,
        path: &str,
        version: &MetadataRetrieveVersion,
    ) -> Result<ArrayMetadata, ArrayCreateError> {
        let node_path = NodePath::new(path)?;

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V3 = version {
            // Try V3
            let key_v3 = meta_key_v3(&node_path);
            if let Some(metadata) = storage.get(&key_v3)? {
                let metadata: ArrayMetadataV3 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v3, err.to_string()))?;
                return Ok(ArrayMetadata::V3(metadata));
            }
        }

        if let MetadataRetrieveVersion::Default | MetadataRetrieveVersion::V2 = version {
            // Try V2
            let key_v2 = meta_key_v2_array(&node_path);
            if let Some(metadata) = storage.get(&key_v2)? {
                let mut metadata: ArrayMetadataV2 = serde_json::from_slice(&metadata)
                    .map_err(|err| StorageError::InvalidMetadata(key_v2, err.to_string()))?;

                let attributes_key = meta_key_v2_attributes(&node_path);
                let attributes = storage.get(&attributes_key)?;
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

    /// Read and decode the chunk at `chunk_indices` into its bytes if it exists with default codec options.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_indices` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of elements in the chunk exceeds `usize::MAX`.
    pub fn retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError> {
        self.retrieve_chunk_if_exists_opt(chunk_indices, &CodecOptions::default())
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_if_exists::<Vec<T>>() instead"
    )]
    /// Read and decode the chunk at `chunk_indices` into a vector of its elements if it exists with default codec options.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the size of `T` does not match the data type size,
    ///  - the decoded bytes cannot be transmuted,
    ///  - `chunk_indices` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    pub fn retrieve_chunk_elements_if_exists<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Vec<T>>, ArrayError> {
        self.retrieve_chunk_if_exists_opt(chunk_indices, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_if_exists::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the chunk at `chunk_indices` into an [`ndarray::ArrayD`] if it exists.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the size of `T` does not match the data type size,
    ///  - the decoded bytes cannot be transmuted,
    ///  - the chunk indices are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Will panic if a chunk dimension is larger than `usize::MAX`.
    pub fn retrieve_chunk_ndarray_if_exists<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ndarray::ArrayD<T>>, ArrayError> {
        self.retrieve_chunk_if_exists_opt(chunk_indices, &CodecOptions::default())
    }

    /// Retrieve the encoded bytes of a chunk.
    ///
    /// # Errors
    /// Returns an [`StorageError`] if there is an underlying store error.
    #[allow(clippy::missing_panics_doc)]
    pub fn retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;

        storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .map(|maybe_bytes| maybe_bytes.map(Into::into))
    }

    /// Read and decode the chunk at `chunk_indices` into its bytes or the fill value if it does not exist with default codec options.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - `chunk_indices` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of elements in the chunk exceeds `usize::MAX`.
    pub fn retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<T, ArrayError> {
        self.retrieve_chunk_opt(chunk_indices, &CodecOptions::default())
    }

    #[deprecated(since = "0.23.0", note = "Use retrieve_chunk::<Vec<T>>() instead")]
    /// Read and decode the chunk at `chunk_indices` into a vector of its elements or the fill value if it does not exist.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - the size of `T` does not match the data type size,
    ///  - the decoded bytes cannot be transmuted,
    ///  - `chunk_indices` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    pub fn retrieve_chunk_elements<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Vec<T>, ArrayError> {
        self.retrieve_chunk_opt(chunk_indices, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the chunk at `chunk_indices` into an [`ndarray::ArrayD`]. It is filled with the fill value if it does not exist.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the size of `T` does not match the data type size,
    ///  - the decoded bytes cannot be transmuted,
    ///  - the chunk indices are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Will panic if a chunk dimension is larger than `usize::MAX`.
    pub fn retrieve_chunk_ndarray<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunk_opt(chunk_indices, &CodecOptions::default())
    }

    /// Retrieve the encoded bytes of the chunks in `chunks`.
    ///
    /// The chunks are in order of the chunk indices returned by `chunks.indices().into_iter()`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    pub fn retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;

        let retrieve_encoded_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            storage_transformer
                .get(&self.chunk_key(&chunk_indices))
                .map(|maybe_bytes| maybe_bytes.map(Into::into))
        };

        let indices = chunks.indices();
        iter_concurrent_limit!(
            options.concurrent_target(),
            indices,
            map,
            retrieve_encoded_chunk
        )
        .collect()
    }

    /// Read and decode the chunks at `chunks` into their bytes.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if
    ///  - any chunk indices in `chunks` are invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if the number of array elements in the chunk exceeds `usize::MAX`.
    pub fn retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.retrieve_chunks_opt(chunks, &CodecOptions::default())
    }

    #[deprecated(since = "0.23.0", note = "Use retrieve_chunks::<Vec<T>>() instead")]
    /// Read and decode the chunks at `chunks` into a vector of their elements.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if any chunk indices in `chunks` are invalid or an error condition in [`Array::retrieve_chunks_opt`].
    ///
    /// # Panics
    /// Panics if the number of array elements in the chunks exceeds `usize::MAX`.
    pub fn retrieve_chunks_elements<T: ElementOwned>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<T>, ArrayError> {
        self.retrieve_chunks_opt(chunks, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunks::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the chunks at `chunks` into an [`ndarray::ArrayD`].
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if any chunk indices in `chunks` are invalid or an error condition in [`Array::retrieve_chunks_elements_opt`].
    ///
    /// # Panics
    /// Panics if the number of array elements in the chunks exceeds `usize::MAX`.
    pub fn retrieve_chunks_ndarray<T: ElementOwned>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunks_opt(chunks, &CodecOptions::default())
    }

    /// Read and decode the `chunk_subset` of the chunk at `chunk_indices` into its bytes.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the chunk indices are invalid,
    ///  - the chunk subset is invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Will panic if the number of elements in `chunk_subset` is `usize::MAX` or larger.
    pub fn retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &CodecOptions::default())
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_subset::<Vec<T>>() instead"
    )]
    /// Read and decode the `chunk_subset` of the chunk at `chunk_indices` into its elements.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the chunk indices are invalid,
    ///  - the chunk subset is invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    pub fn retrieve_chunk_subset_elements<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<Vec<T>, ArrayError> {
        self.retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_subset::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the `chunk_subset` of the chunk at `chunk_indices` into an [`ndarray::ArrayD`].
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the chunk indices are invalid,
    ///  - the chunk subset is invalid,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Will panic if the number of elements in `chunk_subset` is `usize::MAX` or larger.
    pub fn retrieve_chunk_subset_ndarray<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &CodecOptions::default())
    }

    /// Read and decode the `array_subset` of array into its bytes.
    ///
    /// Out-of-bounds elements will have the fill value.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the `array_subset` dimensionality does not match the chunk grid dimensionality,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Panics if attempting to reference a byte beyond `usize::MAX`.
    pub fn retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.retrieve_array_subset_opt(array_subset, &CodecOptions::default())
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_array_subset::<Vec<T>>() instead"
    )]
    /// Read and decode the `array_subset` of array into a vector of its elements.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - the size of `T` does not match the data type size,
    ///  - the decoded bytes cannot be transmuted,
    ///  - an array subset is invalid or out of bounds of the array,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    pub fn retrieve_array_subset_elements<T: ElementOwned>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<Vec<T>, ArrayError> {
        self.retrieve_array_subset_opt(array_subset, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_array_subset::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the `array_subset` of array into an [`ndarray::ArrayD`].
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if:
    ///  - an array subset is invalid or out of bounds of the array,
    ///  - there is a codec decoding error, or
    ///  - an underlying store error.
    ///
    /// # Panics
    /// Will panic if any dimension in `chunk_subset` is `usize::MAX` or larger.
    pub fn retrieve_array_subset_ndarray<T: ElementOwned>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_array_subset_opt(array_subset, &CodecOptions::default())
    }

    /// Initialises a partial decoder for the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if initialisation of the partial decoder fails.
    pub fn partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        self.partial_decoder_opt(chunk_indices, &CodecOptions::default())
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    /// Explicit options version of [`retrieve_chunk_if_exists`](Array::retrieve_chunk_if_exists).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_if_exists_opt<T: FromArrayBytes>(
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
            .create_readable_transformer(storage_handle)?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
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
            Ok(Some(T::from_array_bytes(
                bytes.into_owned(),
                bytemuck::must_cast_slice(&chunk_shape),
                self.data_type(),
            )?))
        } else {
            Ok(None)
        }
    }

    /// Explicit options version of [`retrieve_chunk`](Array::retrieve_chunk).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if let Some(chunk) = self.retrieve_chunk_if_exists_opt::<T>(chunk_indices, options)? {
            Ok(chunk)
        } else {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = ArrayBytes::new_fill_value(
                self.data_type(),
                chunk_shape.iter().map(|&x| x.get()).product::<u64>(),
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

    fn retrieve_chunk_into(
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
            .create_readable_transformer(storage_handle)?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            self.codecs()
                .decode_into(
                    Cow::Owned(chunk_encoded.into()),
                    &self.chunk_shape(chunk_indices)?,
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
        note = "Use retrieve_chunk_if_exists_opt::<Vec<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_chunk_elements_if_exists`](Array::retrieve_chunk_elements_if_exists).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_elements_if_exists_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Vec<T>>, ArrayError> {
        if let Some(bytes) =
            self.retrieve_chunk_if_exists_opt::<ArrayBytes<'static>>(chunk_indices, options)?
        {
            Ok(Some(T::from_array_bytes(self.data_type(), bytes)?))
        } else {
            Ok(None)
        }
    }

    #[deprecated(since = "0.23.0", note = "Use retrieve_chunk_opt::<Vec<T>>() instead")]
    /// Explicit options version of [`retrieve_chunk_elements`](Array::retrieve_chunk_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_elements_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        Ok(T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunk_opt::<ArrayBytes<'static>>(chunk_indices, options)?,
        )?)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_if_exists_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_chunk_ndarray_if_exists`](Array::retrieve_chunk_ndarray_if_exists).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_ndarray_if_exists_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ndarray::ArrayD<T>>, ArrayError> {
        self.retrieve_chunk_if_exists_opt(chunk_indices, options)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_chunk_ndarray`](Array::retrieve_chunk_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_ndarray_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunk_opt(chunk_indices, options)
    }

    /// Explicit options version of [`retrieve_chunks`](Array::retrieve_chunks).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks_opt<T: FromArrayBytes>(
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
        self.retrieve_array_subset_opt(&array_subset, options)
    }

    #[deprecated(since = "0.23.0", note = "Use retrieve_chunks_opt::<Vec<T>>() instead")]
    /// Explicit options version of [`retrieve_chunks_elements`](Array::retrieve_chunks_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks_elements_opt<T: ElementOwned>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        Ok(T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunks_opt::<ArrayBytes<'static>>(chunks, options)?,
        )?)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunks_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_chunks_ndarray`](Array::retrieve_chunks_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks_ndarray_opt<T: ElementOwned>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunks_opt(chunks, options)
    }

    /// Helper method to retrieve multiple chunks with variable-length data types.
    /// Also handles optional data types with variable-length inner types (including nested optionals).
    fn retrieve_multi_chunk_variable(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let nesting_depth = optional_nesting_depth(data_type);

        // Retrieve all the chunks
        let chunk_indices = chunks.indices();
        if nesting_depth > 0 {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| -> Result<
                (ArrayBytesOptional<'static>, ArraySubset),
                ArrayError,
            > {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                Ok((
                    self.retrieve_chunk_subset_opt::<ArrayBytes<'static>>(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        options,
                    )?
                    .into_optional()?,
                    chunk_subset_overlap.relative_to(&array_subset.start())?,
                ))
            };
            let chunk_bytes_and_subsets =
                iter_concurrent_limit!(chunk_concurrent_limit, chunk_indices, map, retrieve_chunk)
                    .collect::<Result<Vec<_>, _>>()?;
            Ok(ArrayBytes::Optional(merge_chunks_vlen_optional(
                chunk_bytes_and_subsets,
                &array_subset.shape(),
                nesting_depth,
            )?))
        } else {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| -> Result<
                (ArrayBytesVariableLength<'static>, ArraySubset),
                ArrayError,
            > {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                Ok((
                    self.retrieve_chunk_subset_opt::<ArrayBytes<'static>>(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        options,
                    )?
                    .into_variable()?,
                    chunk_subset_overlap.relative_to(&array_subset.start())?,
                ))
            };
            let chunk_bytes_and_subsets =
                iter_concurrent_limit!(chunk_concurrent_limit, chunk_indices, map, retrieve_chunk)
                    .collect::<Result<Vec<_>, _>>()?;
            Ok(ArrayBytes::Variable(merge_chunks_vlen(
                chunk_bytes_and_subsets,
                &array_subset.shape(),
            )))
        }
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types.
    /// Also handles optional data types with fixed-length inner types (including nested optionals).
    fn retrieve_multi_chunk_fixed(
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

            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                let chunk_subset_in_array =
                    chunk_subset_overlap.relative_to(&array_subset.start())?;

                let array_subset_shape = array_subset.shape();
                let mut data_view = unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        data_output_slice,
                        data_type_size,
                        &array_subset_shape,
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
                            &array_subset_shape,
                            chunk_subset_in_array.clone(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Build the nested decode target
                let target =
                    build_nested_optional_target(&mut data_view, mask_views.as_mut_slice());

                self.retrieve_chunk_subset_into(
                    &chunk_indices,
                    &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                    target,
                    options,
                )?;
                Ok::<_, ArrayError>(())
            };

            let indices = chunks.indices();
            iter_concurrent_limit!(
                chunk_concurrent_limit,
                indices,
                try_for_each,
                retrieve_chunk
            )?;
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

    /// Explicit options version of [`retrieve_array_subset`](Array::retrieve_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn retrieve_array_subset_opt<T: FromArrayBytes>(
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
        match num_chunks {
            0 => {
                let bytes = ArrayBytes::new_fill_value(
                    self.data_type(),
                    array_subset.num_elements(),
                    self.fill_value(),
                )
                .map_err(CodecError::from)
                .map_err(ArrayError::from)?;
                T::from_array_bytes(bytes, &array_subset.shape(), self.data_type())
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    self.retrieve_chunk_opt(chunk_indices, options)
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.retrieve_chunk_subset_opt(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        options,
                    )
                }
            }
            _ => {
                let chunk_shape = self.chunk_shape(chunks.start())?;

                // Calculate chunk/codec concurrency
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
                    self.retrieve_multi_chunk_fixed(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )?
                } else {
                    self.retrieve_multi_chunk_variable(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )?
                };
                T::from_array_bytes(bytes.into_owned(), &array_subset.shape(), self.data_type())
            }
        }
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_array_subset_opt::<Vec<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_array_subset_elements`](Array::retrieve_array_subset_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_elements_opt<T: ElementOwned>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        Ok(T::from_array_bytes(
            self.data_type(),
            self.retrieve_array_subset_opt::<ArrayBytes<'static>>(array_subset, options)?,
        )?)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_array_subset_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_array_subset_ndarray`](Array::retrieve_array_subset_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_ndarray_opt<T: ElementOwned>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_array_subset_opt(array_subset, options)
    }

    /// Explicit options version of [`retrieve_chunk_subset`](Array::retrieve_chunk_subset).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_opt<T: FromArrayBytes>(
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

        let bytes = if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape() == chunk_shape_u64
        {
            // Fast path if `chunk_subset` encompasses the whole chunk
            return self.retrieve_chunk_opt(chunk_indices, options);
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_readable_transformer(storage_handle)?;
            let input_handle = Arc::new(StoragePartialDecoder::new(
                storage_transformer,
                self.chunk_key(chunk_indices),
            ));

            self.codecs
                .clone()
                .partial_decoder(
                    input_handle,
                    &chunk_shape,
                    self.data_type(),
                    self.fill_value(),
                    options,
                )?
                .partial_decode(chunk_subset, options)?
                .into_owned()
        };
        bytes.validate(chunk_subset.num_elements(), self.data_type())?;
        T::from_array_bytes(bytes, &chunk_subset.shape(), self.data_type())
    }

    fn retrieve_chunk_subset_into(
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

        if chunk_subset.start().iter().all(|&o| o == 0) && chunk_subset.shape() == chunk_shape_u64 {
            // Fast path if `chunk_subset` encompasses the whole chunk
            self.retrieve_chunk_into(chunk_indices, output_target, options)
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_readable_transformer(storage_handle)?;
            let input_handle = Arc::new(StoragePartialDecoder::new(
                storage_transformer,
                self.chunk_key(chunk_indices),
            ));
            self.codecs
                .clone()
                .partial_decoder(
                    input_handle,
                    &chunk_shape,
                    self.data_type(),
                    self.fill_value(),
                    options,
                )?
                .partial_decode_into(chunk_subset, output_target, options)?;
            Ok(())
        }
    }

    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_subset_opt::<Vec<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_chunk_subset_elements`](Array::retrieve_chunk_subset_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_elements_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        Ok(T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunk_subset_opt::<ArrayBytes<'static>>(
                chunk_indices,
                chunk_subset,
                options,
            )?,
        )?)
    }

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use retrieve_chunk_subset_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Explicit options version of [`retrieve_chunk_subset_ndarray`](Array::retrieve_chunk_subset_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_ndarray_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunk_subset_opt(chunk_indices, chunk_subset, options)
    }

    /// Explicit options version of [`partial_decoder`](Array::partial_decoder).
    #[allow(clippy::missing_errors_doc)]
    pub fn partial_decoder_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;
        let input_handle = Arc::new(StoragePartialDecoder::new(
            storage_transformer,
            self.chunk_key(chunk_indices),
        ));
        Ok(self.codecs.clone().partial_decoder(
            input_handle,
            &self.chunk_shape(chunk_indices)?,
            self.data_type(),
            self.fill_value(),
            options,
        )?)
    }
}

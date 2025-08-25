use std::{borrow::Cow, sync::Arc};

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;

use crate::{
    array::{ArrayBytes, ArrayMetadataV2},
    array_subset::ArraySubset,
    config::MetadataRetrieveVersion,
    node::{meta_key_v2_array, meta_key_v2_attributes, meta_key_v3, NodePath},
    storage::{ReadableStorageTraits, StorageError, StorageHandle},
};

use super::{
    array_bytes::{copy_fill_value_into, merge_chunks_vlen},
    codec::{
        ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecOptions, StoragePartialDecoder,
    },
    concurrency::concurrency_chunks_and_codec,
    element::ElementOwned,
    Array, ArrayBytesFixedDisjointView, ArrayCreateError, ArrayError, ArrayMetadata,
    ArrayMetadataV3, ArraySize, DataTypeSize,
};

#[cfg(feature = "ndarray")]
use super::elements_to_ndarray;

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
    pub fn retrieve_chunk_if_exists(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytes<'_>>, ArrayError> {
        self.retrieve_chunk_if_exists_opt(chunk_indices, &CodecOptions::default())
    }

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
        self.retrieve_chunk_elements_if_exists_opt(chunk_indices, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
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
        self.retrieve_chunk_ndarray_if_exists_opt(chunk_indices, &CodecOptions::default())
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
            .map(|maybe_bytes| maybe_bytes.map(|bytes| bytes.to_vec()))
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
    pub fn retrieve_chunk(&self, chunk_indices: &[u64]) -> Result<ArrayBytes<'_>, ArrayError> {
        self.retrieve_chunk_opt(chunk_indices, &CodecOptions::default())
    }

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
        self.retrieve_chunk_elements_opt(chunk_indices, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
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
        self.retrieve_chunk_ndarray_opt(chunk_indices, &CodecOptions::default())
    }

    /// Retrieve the encoded bytes of the chunks in `chunks`.
    ///
    /// The chunks are in order of the chunk indices returned by `chunks.indices().into_iter()`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying store error.
    pub fn retrieve_encoded_chunks(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;

        let retrieve_encoded_chunk = |chunk_indices: Vec<u64>| {
            storage_transformer
                .get(&self.chunk_key(&chunk_indices))
                .map(|maybe_bytes| maybe_bytes.map(|bytes| bytes.to_vec()))
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
    pub fn retrieve_chunks(&self, chunks: &ArraySubset) -> Result<ArrayBytes<'_>, ArrayError> {
        self.retrieve_chunks_opt(chunks, &CodecOptions::default())
    }

    /// Read and decode the chunks at `chunks` into a vector of their elements.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if any chunk indices in `chunks` are invalid or an error condition in [`Array::retrieve_chunks_opt`].
    ///
    /// # Panics
    /// Panics if the number of array elements in the chunks exceeds `usize::MAX`.
    pub fn retrieve_chunks_elements<T: ElementOwned>(
        &self,
        chunks: &ArraySubset,
    ) -> Result<Vec<T>, ArrayError> {
        self.retrieve_chunks_elements_opt(chunks, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
    /// Read and decode the chunks at `chunks` into an [`ndarray::ArrayD`].
    ///
    /// # Errors
    /// Returns an [`ArrayError`] if any chunk indices in `chunks` are invalid or an error condition in [`Array::retrieve_chunks_elements_opt`].
    ///
    /// # Panics
    /// Panics if the number of array elements in the chunks exceeds `usize::MAX`.
    pub fn retrieve_chunks_ndarray<T: ElementOwned>(
        &self,
        chunks: &ArraySubset,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunks_ndarray_opt(chunks, &CodecOptions::default())
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
    pub fn retrieve_chunk_subset(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        self.retrieve_chunk_subset_opt(chunk_indices, chunk_subset, &CodecOptions::default())
    }

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
        chunk_subset: &ArraySubset,
    ) -> Result<Vec<T>, ArrayError> {
        self.retrieve_chunk_subset_elements_opt(
            chunk_indices,
            chunk_subset,
            &CodecOptions::default(),
        )
    }

    #[cfg(feature = "ndarray")]
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
        chunk_subset: &ArraySubset,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_chunk_subset_ndarray_opt(
            chunk_indices,
            chunk_subset,
            &CodecOptions::default(),
        )
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
    pub fn retrieve_array_subset(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        self.retrieve_array_subset_opt(array_subset, &CodecOptions::default())
    }

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
        array_subset: &ArraySubset,
    ) -> Result<Vec<T>, ArrayError> {
        self.retrieve_array_subset_elements_opt(array_subset, &CodecOptions::default())
    }

    #[cfg(feature = "ndarray")]
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
        array_subset: &ArraySubset,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.retrieve_array_subset_ndarray_opt(array_subset, &CodecOptions::default())
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
    pub fn retrieve_chunk_if_exists_opt(
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
            .create_readable_transformer(storage_handle)?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            let chunk_encoded: Vec<u8> = chunk_encoded.into();
            let chunk_representation = self.chunk_array_representation(chunk_indices)?;
            let bytes = self
                .codecs()
                .decode(Cow::Owned(chunk_encoded), &chunk_representation, options)
                .map_err(ArrayError::CodecError)?;
            Ok(Some(bytes))
        } else {
            Ok(None)
        }
    }

    /// Explicit options version of [`retrieve_chunk`](Array::retrieve_chunk).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let chunk = self.retrieve_chunk_if_exists_opt(chunk_indices, options)?;
        if let Some(chunk) = chunk {
            Ok(chunk)
        } else {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let array_size =
                ArraySize::new(self.data_type().size(), chunk_shape.num_elements_u64());
            Ok(ArrayBytes::new_fill_value(array_size, self.fill_value()))
        }
    }

    fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
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
            let chunk_encoded: Vec<u8> = chunk_encoded.into();
            let chunk_representation = self.chunk_array_representation(chunk_indices)?;
            self.codecs()
                .decode_into(
                    Cow::Owned(chunk_encoded),
                    &chunk_representation,
                    output_view,
                    options,
                )
                .map_err(ArrayError::CodecError)
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_view)
                .map_err(ArrayError::CodecError)
        }
    }

    /// Explicit options version of [`retrieve_chunk_elements_if_exists`](Array::retrieve_chunk_elements_if_exists).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_elements_if_exists_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Vec<T>>, ArrayError> {
        if let Some(bytes) = self.retrieve_chunk_if_exists_opt(chunk_indices, options)? {
            Ok(Some(T::from_array_bytes(self.data_type(), bytes)?))
        } else {
            Ok(None)
        }
    }

    /// Explicit options version of [`retrieve_chunk_elements`](Array::retrieve_chunk_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_elements_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunk_opt(chunk_indices, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`retrieve_chunk_ndarray_if_exists`](Array::retrieve_chunk_ndarray_if_exists).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_ndarray_if_exists_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ndarray::ArrayD<T>>, ArrayError> {
        let shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        let elements = self.retrieve_chunk_elements_if_exists_opt::<T>(chunk_indices, options)?;
        if let Some(elements) = elements {
            Ok(Some(elements_to_ndarray(&shape, elements)?))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`retrieve_chunk_ndarray`](Array::retrieve_chunk_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_ndarray_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        elements_to_ndarray(
            &shape,
            self.retrieve_chunk_elements_opt::<T>(chunk_indices, options)?,
        )
    }

    /// Explicit options version of [`retrieve_chunks`](Array::retrieve_chunks).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks_opt(
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
        self.retrieve_array_subset_opt(&array_subset, options)
    }

    /// Explicit options version of [`retrieve_chunks_elements`](Array::retrieve_chunks_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks_elements_opt<T: ElementOwned>(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(self.data_type(), self.retrieve_chunks_opt(chunks, options)?)
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`retrieve_chunks_ndarray`](Array::retrieve_chunks_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks_ndarray_opt<T: ElementOwned>(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        let elements = self.retrieve_chunks_elements_opt::<T>(chunks, options)?;
        elements_to_ndarray(array_subset.shape(), elements)
    }

    /// Explicit options version of [`retrieve_array_subset`](Array::retrieve_array_subset).
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    #[allow(clippy::too_many_lines)]
    pub fn retrieve_array_subset_opt(
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
            0 => {
                let array_size =
                    ArraySize::new(self.data_type().size(), array_subset.num_elements());
                Ok(ArrayBytes::new_fill_value(array_size, self.fill_value()))
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if &chunk_subset == array_subset {
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
                let chunk_representation =
                    self.chunk_array_representation(&vec![0; self.dimensionality()])?;

                // Calculate chunk/codec concurrency
                let codec_concurrency =
                    self.recommended_codec_concurrency(&chunk_representation)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                match chunk_representation.data_type().size() {
                    DataTypeSize::Variable => {
                        // Retrieve all the chunks
                        let retrieve_chunk = |chunk_indices: Vec<u64>| -> Result<
                            (ArrayBytes<'_>, ArraySubset),
                            ArrayError,
                        > {
                            let chunk_subset = self.chunk_subset(&chunk_indices)?;
                            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                            Ok((
                                self.retrieve_chunk_subset_opt(
                                    &chunk_indices,
                                    &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                                    &options,
                                )?,
                                chunk_subset_overlap.relative_to(array_subset.start())?,
                            ))
                        };
                        let chunk_indices = chunks.indices();
                        let chunk_bytes_and_subsets = iter_concurrent_limit!(
                            chunk_concurrent_limit,
                            chunk_indices,
                            map,
                            retrieve_chunk
                        )
                        .collect::<Result<Vec<_>, _>>()?;

                        Ok(merge_chunks_vlen(
                            chunk_bytes_and_subsets,
                            array_subset.shape(),
                        )?)
                    }
                    DataTypeSize::Fixed(data_type_size) => {
                        // Allocate the output
                        let size_output = array_subset.num_elements_usize() * data_type_size;
                        if size_output == 0 {
                            return Ok(ArrayBytes::new_flen(vec![]));
                        }
                        let mut output = Vec::with_capacity(size_output);

                        {
                            let output =
                                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
                            let retrieve_chunk = |chunk_indices: Vec<u64>| {
                                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                                let mut output_view = unsafe {
                                    // SAFETY: chunks represent disjoint array subsets
                                    ArrayBytesFixedDisjointView::new(
                                        output,
                                        data_type_size,
                                        array_subset.shape(),
                                        chunk_subset_overlap.relative_to(array_subset.start())?,
                                    )?
                                };
                                self.retrieve_chunk_subset_into(
                                    &chunk_indices,
                                    &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                                    &mut output_view,
                                    &options,
                                )?;
                                // let chunk_subset_bytes = self.retrieve_chunk_subset_opt(
                                //     &chunk_indices,
                                //     &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                                //     &options,
                                // )?;
                                // update_bytes_flen(
                                //     &output,
                                //     array_subset.shape(),
                                //     &chunk_subset_bytes.into_fixed()?,
                                //     &chunk_subset_overlap.relative_to(array_subset.start())?,
                                //     data_type_size,
                                // );
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
                        unsafe { output.set_len(size_output) };
                        Ok(ArrayBytes::from(output))
                    }
                }
            }
        }
    }

    /// Explicit options version of [`retrieve_array_subset_elements`](Array::retrieve_array_subset_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_elements_opt<T: ElementOwned>(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_array_subset_opt(array_subset, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`retrieve_array_subset_ndarray`](Array::retrieve_array_subset_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_ndarray_opt<T: ElementOwned>(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements = self.retrieve_array_subset_elements_opt::<T>(array_subset, options)?;
        elements_to_ndarray(array_subset.shape(), elements)
    }

    /// Explicit options version of [`retrieve_chunk_subset`](Array::retrieve_chunk_subset).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_opt(
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
            self.retrieve_chunk_opt(chunk_indices, options)?
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
                .partial_decoder(input_handle, &chunk_representation, options)?
                .partial_decode(chunk_subset, options)?
                .into_owned()
        };
        bytes.validate(chunk_subset.num_elements(), self.data_type().size())?;
        Ok(bytes)
    }

    fn retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
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
            self.retrieve_chunk_into(chunk_indices, output_view, options)
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
                .partial_decoder(input_handle, &chunk_representation, options)?
                .partial_decode_into(chunk_subset, output_view, options)?;
            Ok(())
        }
    }

    /// Explicit options version of [`retrieve_chunk_subset_elements`](Array::retrieve_chunk_subset_elements).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_elements_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunk_subset_opt(chunk_indices, chunk_subset, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    /// Explicit options version of [`retrieve_chunk_subset_ndarray`](Array::retrieve_chunk_subset_ndarray).
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_ndarray_opt<T: ElementOwned>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements =
            self.retrieve_chunk_subset_elements_opt::<T>(chunk_indices, chunk_subset, options)?;
        elements_to_ndarray(chunk_subset.shape(), elements)
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
        let chunk_representation = self.chunk_array_representation(chunk_indices)?;
        Ok(self
            .codecs
            .clone()
            .partial_decoder(input_handle, &chunk_representation, options)?)
    }
}

/// Macro shared by both WASM- and non-WASM `chunk_cache_lru` implementations.
#[macro_export]
macro_rules! impl_ChunkCacheLruEncoded {
    () => {
        fn retrieve_chunk(
            &self,
            chunk_indices: &[u64],
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk_encoded = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(self
                        .array
                        .retrieve_encoded_chunk(chunk_indices)?
                        .map(|chunk| Arc::new(Cow::Owned(chunk))))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;

            if let Some(chunk_encoded) = chunk_encoded.as_ref() {
                let chunk_representation = self.array.chunk_array_representation(chunk_indices)?;
                let bytes = self
                    .array
                    .codecs()
                    .decode(Cow::Borrowed(chunk_encoded), &chunk_representation, options)
                    .map_err(ArrayError::CodecError)?;
                bytes.validate(
                    chunk_representation.num_elements(),
                    chunk_representation.data_type().size(),
                )?;
                Ok(Arc::new(bytes.into_owned()))
            } else {
                let chunk_shape = self.array.chunk_shape(chunk_indices)?;
                let array_size = ArraySize::new(
                    self.array.data_type().size(),
                    chunk_shape.num_elements_u64(),
                );
                Ok(Arc::new(ArrayBytes::new_fill_value(
                    array_size,
                    self.array.fill_value(),
                )))
            }
        }

        fn retrieve_chunk_subset(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk_encoded: ChunkCacheTypeEncoded = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(self
                        .array
                        .retrieve_encoded_chunk(chunk_indices)?
                        .map(|chunk| Arc::new(Cow::Owned(chunk))))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;

            if let Some(chunk_encoded) = chunk_encoded {
                let chunk_representation = self.array.chunk_array_representation(chunk_indices)?;
                Ok(self
                    .array
                    .codecs()
                    .partial_decoder(chunk_encoded, &chunk_representation, options)?
                    .partial_decode(chunk_subset, options)?
                    .into_owned()
                    .into())
            } else {
                let array_size =
                    ArraySize::new(self.array.data_type().size(), chunk_subset.num_elements());
                Ok(Arc::new(ArrayBytes::new_fill_value(
                    array_size,
                    self.array.fill_value(),
                )))
            }
        }
    };
}

/// Macro shared by both WASM- and non-WASM `chunk_cache_lru` implementations.
#[macro_export]
macro_rules! impl_ChunkCacheLruDecoded {
    () => {
        fn retrieve_chunk(
            &self,
            chunk_indices: &[u64],
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            self.try_get_or_insert_with(chunk_indices.to_vec(), || {
                Ok(Arc::new(
                    self.array
                        .retrieve_chunk_opt(chunk_indices, options)?
                        .into_owned(),
                ))
            })
            .map_err(|err| {
                // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                Arc::try_unwrap(err).unwrap_or_else(|err| {
                    ArrayError::StorageError(StorageError::from(err.to_string()))
                })
            })
        }

        fn retrieve_chunk_subset(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let chunk = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    Ok(Arc::new(
                        self.array
                            .retrieve_chunk_opt(chunk_indices, options)?
                            .into_owned(),
                    ))
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            let chunk_representation = self.array.chunk_array_representation(chunk_indices)?;
            Ok(chunk
                .extract_array_subset(
                    chunk_subset,
                    &chunk_representation.shape_u64(),
                    self.array.data_type(),
                )?
                .into_owned()
                .into())
        }
    };
}

/// Macro shared by both WASM- and non-WASM `chunk_cache_lru` implementations.
#[macro_export]
macro_rules! impl_ChunkCacheLruPartialDecoder {
    () => {
        fn retrieve_chunk(
            &self,
            chunk_indices: &[u64],
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let partial_decoder = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    self.array.partial_decoder(chunk_indices)
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            let chunk_shape =
                $crate::array::chunk_shape_to_array_shape(&self.array.chunk_shape(chunk_indices)?);
            Ok(partial_decoder
                .partial_decode(&ArraySubset::new_with_shape(chunk_shape), options)?
                .into_owned()
                .into())
        }

        fn retrieve_chunk_subset(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &$crate::array::codec::CodecOptions,
        ) -> Result<ChunkCacheTypeDecoded, ArrayError> {
            let partial_decoder = self
                .try_get_or_insert_with(chunk_indices.to_vec(), || {
                    self.array.partial_decoder(chunk_indices)
                })
                .map_err(|err| {
                    // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                    Arc::try_unwrap(err).unwrap_or_else(|err| {
                        ArrayError::StorageError(StorageError::from(err.to_string()))
                    })
                })?;
            Ok(partial_decoder
                .partial_decode(chunk_subset, options)?
                .into_owned()
                .into())
        }
    };
}

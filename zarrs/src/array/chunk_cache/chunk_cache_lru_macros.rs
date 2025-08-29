

#[macro_export]
macro_rules! impl_ChunkCacheLruPartialDecoder {
    () => {
        fn retrieve_chunk(
            &self,
            chunk_indices: &[u64],
            options: &crate::array::codec::CodecOptions,
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
                crate::array::chunk_shape_to_array_shape(&self.array.chunk_shape(chunk_indices)?);
            Ok(partial_decoder
                .partial_decode(&ArraySubset::new_with_shape(chunk_shape), options)?
                .into_owned()
                .into())
        }

        fn retrieve_chunk_subset(
            &self,
            chunk_indices: &[u64],
            chunk_subset: &ArraySubset,
            options: &crate::array::codec::CodecOptions,
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
use inherent::inherent;
use std::sync::Arc;

use super::{AsyncArrayReadOps, *};
use zarrs_codec::{ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits, CodecError};
use zarrs_storage::{Bytes, StorageHandle};

#[cfg(feature = "async")]
#[inherent]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> AsyncArrayReadOps
    for Array<TStorage>
{
    pub async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunk_with_options(chunk_indices, self.codec_options(), resources)
            .await
    }

    pub async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_into_with_options(
            chunk_indices,
            output_target,
            self.codec_options(),
            resources,
        )
        .await
    }

    pub async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    pub async fn async_retrieve_partial_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_partial_chunk_with_options(
            chunk_indices,
            indexer,
            self.codec_options(),
            resources,
        )
        .await
    }

    pub async fn async_retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_partial_chunk_into_with_options(
            chunk_indices,
            indexer,
            output_target,
            self.codec_options(),
            resources,
        )
        .await
    }

    pub async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();

        let bytes = self
            .async_retrieve_array_subset_planned(array_subset, options, resources)
            .await?;
        T::from_array_bytes(bytes, &array_subset.shape(), self.data_type())
    }

    pub async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError> {
        let options = self.codec_options();
        let _ = options;
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_async_readable_transformer(storage_handle)
            .await?;

        Ok(storage_transformer
            .get(&self.chunk_key(chunk_indices)?)
            .await?)
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<EncodedSubchunk>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<EncodedSubchunk>, ArrayError> {
        self.async_retrieve_encoded_subchunk_planned(level, subchunk_indices, resources)
            .await
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    pub async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        self.async_retrieve_chunk_if_exists_with_options(
            chunk_indices,
            self.codec_options(),
            resources,
        )
        .await
    }

    pub async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<Vec<Option<Bytes>>, ArrayError> {
        chunks
            .validate(self.chunk_grid_shape())
            .map_err(CodecError::from)?;
        self.async_retrieve_encoded_chunks_planned(chunks, resources)
            .await
    }

    pub async fn async_retrieve_subchunk<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    pub async fn async_retrieve_subchunks<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    pub async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_array_subset_into_planned(
            array_subset,
            output_target,
            self.codec_options(),
            resources,
        )
        .await
    }

    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        self.async_partial_decoder_with_options(chunk_indices, self.codec_options(), resources)
            .await
    }

    pub async fn async_local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<ChunkGrid>, ArrayError>;

    pub async fn async_local_subchunk_grid_at_level(
        &self,
        level: usize,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<ChunkGrid>, ArrayError>;
}

impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
    pub(in crate::array) async fn async_retrieve_chunk_with_options<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let bytes = self
            .async_retrieve_chunk_planned(chunk_indices, &chunk_shape, options, resources)
            .await?;
        T::from_array_bytes(
            bytes,
            bytemuck::must_cast_slice(&chunk_shape),
            self.data_type(),
        )
    }

    pub(in crate::array) async fn async_retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let (chunk_shape, indexer) = self.whole_chunk_indexer(chunk_indices)?;
        self.async_retrieve_partial_chunk_into_planned(
            chunk_indices,
            &chunk_shape,
            &indexer,
            output_target,
            options,
            resources,
        )
        .await
    }

    pub(in crate::array) async fn async_retrieve_partial_chunk_with_options<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if let Some(chunk_subset) = indexer.as_array_subset()
            && !chunk_subset.inbounds_shape(chunk_shape_u64)
        {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                chunk_shape_u64.to_vec(),
            ));
        }

        // Whole chunks are recognised by the plan
        let bytes = self
            .async_retrieve_partial_chunk_planned(
                chunk_indices,
                &chunk_shape,
                indexer,
                options,
                resources,
            )
            .await?;
        T::from_array_bytes(bytes, &indexer.output_shape(), self.data_type())
    }

    pub(in crate::array) async fn async_retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if let Some(chunk_subset) = indexer.as_array_subset()
            && !chunk_subset.inbounds_shape(chunk_shape_u64)
        {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                chunk_shape_u64.to_vec(),
            ));
        }

        // Whole chunks are recognised by the plan
        self.async_retrieve_partial_chunk_into_planned(
            chunk_indices,
            &chunk_shape,
            indexer,
            output_target,
            options,
            resources,
        )
        .await
    }

    pub(in crate::array) async fn async_retrieve_chunk_if_exists_with_options<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }

        let shape = self.chunk_shape(chunk_indices)?;
        self.async_retrieve_chunk_if_exists_planned(chunk_indices, options, resources)
            .await?
            .map(|bytes| {
                T::from_array_bytes(bytes, bytemuck::must_cast_slice(&shape), self.data_type())
            })
            .transpose()
    }

    pub(in crate::array) async fn async_partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        self.async_planned_partial_decoder(chunk_indices, options, resources)
            .await
    }
}

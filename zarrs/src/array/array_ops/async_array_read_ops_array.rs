use inherent::inherent;
use std::sync::Arc;
use zarrs_codec::{CowBytes, Resources};

use futures::{StreamExt, TryStreamExt};

use super::super::ArrayIndices;
use super::super::array_bytes_internal::{FixedDecodeBuffers, merge_chunks};
use super::super::concurrency::concurrency_chunks_and_codec;
use super::async_array_read_ops_common::{AsyncRetrieveInto, retrieve_multi_chunk_fixed_into};
use super::{
    Array, ArrayError, ArraySubsetTraits, AsyncArrayReadOps, AsyncReadableStorageTraits, ChunkGrid,
    CodecOptions, DataType, FromArrayBytes, Indexer, recommended_codec_concurrency,
};
use crate::array::{ArrayBytes, ChunkShapeTraits};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
    CodecError, InvalidNumberOfElementsError, copy_fill_value_into,
};
use zarrs_storage::Bytes;

#[cfg(feature = "async")]
#[inherent]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> AsyncArrayReadOps
    for Array<TStorage>
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunk_with_options(chunk_indices, self.codec_options(), resources)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
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

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
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

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
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

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        if array_subset.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                Box::new(array_subset.to_array_subset()),
                self.shape().to_vec(),
            ));
        }

        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                Box::new(array_subset.to_array_subset()),
                self.shape().to_vec(),
            ));
        };

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
                    self.async_retrieve_chunk(chunk_indices, resources).await
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.async_retrieve_partial_chunk(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        resources,
                    )
                    .await
                }
            }
            _ => {
                let chunk_shape = self.chunk_shape(chunks.start())?;
                let codec_concurrency = recommended_codec_concurrency(self, &chunk_shape)?;
                let (chunk_concurrent_limit, resources) = concurrency_chunks_and_codec(
                    num_chunks,
                    self.chunk_concurrent_minimum,
                    resources,
                    &codec_concurrency,
                );

                let bytes = if self.data_type().is_fixed() {
                    self.async_retrieve_multi_chunk_fixed(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        options,
                        &resources,
                    )
                    .await?
                } else {
                    self.async_retrieve_multi_chunk_variable(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        options,
                        &resources,
                    )
                    .await?
                };
                T::from_array_bytes(bytes.into_owned(), &array_subset_shape, self.data_type())
            }
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_storage().await?;

        Ok(storage_transformer.get(&chunk_key).await?)
    }

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    #[allow(clippy::missing_errors_doc)]
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

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<Vec<Option<Bytes>>, ArrayError> {
        chunks
            .validate(self.chunk_grid_shape())
            .map_err(CodecError::from)?;
        let storage_transformer = self.async_readable_storage().await?;

        let retrieve_encoded_chunk = |chunk_indices: ArrayIndices| {
            let storage_transformer = storage_transformer.clone();
            async move {
                Ok::<_, ArrayError>(
                    storage_transformer
                        .get(&self.chunk_key(&chunk_indices)?)
                        .await?,
                )
            }
        };

        let futures = chunks.iter_indices().map(retrieve_encoded_chunk);
        let mut values = Vec::with_capacity(futures.size_hint().0);
        let mut results = futures::stream::iter(futures)
            // A zero target is unconstrained, but buffered(0) never makes progress.
            .buffered(match resources.compute_target() {
                0 => usize::MAX,
                target => target,
            });

        while let Some(value) = results.try_next().await? {
            values.push(value);
        }
        Ok(values)
    }

    #[allow(clippy::missing_errors_doc)]
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

    #[allow(clippy::missing_errors_doc)]
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

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        super::async_array_read_ops_common::retrieve_array_subset_into(
            self,
            self,
            array_subset,
            output_target,
            self.chunk_concurrent_minimum,
            self.codec_options(),
            resources,
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        self.async_partial_decoder_with_options(chunk_indices, self.codec_options(), resources)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<ChunkGrid>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
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
        let chunk = self
            .async_retrieve_chunk_if_exists_with_options::<T>(chunk_indices, options, resources)
            .await?;
        super::chunk_or_fill_value(self, chunk_indices, chunk)
    }

    pub(in crate::array) async fn async_retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        if output_target.num_elements() != chunk_shape.num_elements_u64() {
            return Err(ArrayError::CodecError(
                InvalidNumberOfElementsError::new(
                    output_target.num_elements(),
                    chunk_shape.num_elements_u64(),
                )
                .into(),
            ));
        }
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_storage().await?;
        let chunk_encoded = storage_transformer
            .get(&chunk_key)
            .await
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            self.codecs_bound()
                .decode_into(
                    CowBytes::Shared(chunk_encoded),
                    &chunk_shape,
                    output_target,
                    options,
                    resources.compute(),
                )
                .map_err(ArrayError::CodecError)
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError)
        }
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
                Box::new(chunk_subset.to_array_subset()),
                chunk_shape_u64.to_vec(),
            ));
        }
        if indexer.covers_array(chunk_shape_u64) {
            return self
                .async_retrieve_chunk_with_options(chunk_indices, options, resources)
                .await;
        }

        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_storage().await?;
        let input_handle = Arc::new((storage_transformer, chunk_key));
        let bytes = self
            .codecs_bound()
            .async_partial_decoder(input_handle, &chunk_shape, options, resources)
            .await?
            .partial_decode(indexer, options, resources)
            .await?
            .into_owned();
        bytes.validate(indexer.len(), self.data_type())?;
        T::from_array_bytes(bytes, &indexer.output_shape(), self.data_type())
    }

    pub(in crate::array) async fn async_retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if let Some(chunk_subset) = indexer.as_array_subset()
            && !chunk_subset.inbounds_shape(chunk_shape_u64)
        {
            return Err(ArrayError::InvalidArraySubset(
                Box::new(chunk_subset.to_array_subset()),
                chunk_shape_u64.to_vec(),
            ));
        }
        if indexer.covers_array(chunk_shape_u64) {
            return self
                .async_retrieve_chunk_into_with_options(
                    chunk_indices,
                    output_target,
                    options,
                    resources,
                )
                .await;
        }

        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_storage().await?;
        let input_handle = Arc::new((storage_transformer, chunk_key));
        self.codecs_bound()
            .async_partial_decoder(input_handle, &chunk_shape, options, resources)
            .await?
            .partial_decode_into(indexer, output_target, options, resources)
            .await?;
        Ok(())
    }

    pub(in crate::array) async fn async_retrieve_chunk_if_exists_with_options<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_storage().await?;
        let chunk_encoded = storage_transformer
            .get(&chunk_key)
            .await
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = self
                .codecs_bound()
                .decode(
                    CowBytes::Shared(chunk_encoded),
                    &chunk_shape,
                    options,
                    resources.compute(),
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

    pub(in crate::array) async fn async_partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_storage().await?;
        let input_handle = Arc::new((storage_transformer, chunk_key));
        Ok(self
            .codecs_bound()
            .async_partial_decoder(
                input_handle,
                &self.chunk_shape(chunk_indices)?,
                options,
                resources,
            )
            .await?)
    }
    /// Helper method to retrieve multiple chunks with variable-length data types (async).
    /// Also handles optional data types with variable-length inner types (including nested optionals).
    pub(crate) async fn async_retrieve_multi_chunk_variable(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let array_subset_start = array_subset.start();
        let retrieve_chunk = |chunk_indices: ArrayIndices| {
            let array_subset_start = &array_subset_start;
            async move {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                Ok::<_, ArrayError>((
                    self.async_retrieve_partial_chunk_with_options::<ArrayBytes>(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        options,
                        resources,
                    )
                    .await?,
                    chunk_subset_overlap.relative_to(array_subset_start)?,
                ))
            }
        };
        let chunk_bytes_and_subsets = {
            let indices = chunks.indices();
            let indices = indices.iter();
            let mut values = Vec::with_capacity(indices.size_hint().0);
            let mut results = futures::stream::iter(indices)
                .map(retrieve_chunk)
                .buffered(chunk_concurrent_limit);

            while let Some(value) = results.try_next().await? {
                values.push(value);
            }
            values
        };
        Ok(merge_chunks(
            chunk_bytes_and_subsets,
            &array_subset.shape(),
            data_type,
        )?)
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types (async).
    /// Also handles optional data types with fixed-length inner types.
    pub(crate) async fn async_retrieve_multi_chunk_fixed(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let mut buffers = FixedDecodeBuffers::new(data_type, &array_subset.shape())
            .expect("data_type must have fixed size");
        retrieve_multi_chunk_fixed_into(
            self,
            self,
            array_subset,
            chunks,
            chunk_concurrent_limit,
            &buffers.target()?,
            options,
            resources,
        )
        .await?;
        // SAFETY: the chunks cover the array subset, so every element has been written
        Ok(unsafe { buffers.into_array_bytes() })
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> AsyncRetrieveInto
    for Array<TStorage>
{
    async fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_into_with_options(
            chunk_indices,
            output_target,
            options,
            resources,
        )
        .await
    }

    async fn retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_partial_chunk_into_with_options(
            chunk_indices,
            indexer,
            output_target,
            options,
            resources,
        )
        .await
    }
}

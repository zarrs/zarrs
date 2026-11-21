use inherent::inherent;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::Resources;

use futures::{StreamExt, TryStreamExt};

use super::super::ArrayIndices;
use super::super::array_bytes_internal::{FixedDecodeBuffers, merge_chunks};
use super::super::concurrency::concurrency_chunks_and_codec;
use super::async_array_read_ops_common::{AsyncRetrieveInto, retrieve_multi_chunk_fixed_into};
use super::{
    Array, ArrayError, ArraySubsetTraits, AsyncArrayReadOps, AsyncReadableStorageTraits, ChunkGrid,
    CodecOptions, FromArrayBytes, Indexer, recommended_codec_concurrency,
};
use crate::array::{ArrayBytes, ArraySubset, ChunkShapeTraits};
use zarrs_codec::{
    ArrayBytesTarget, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits, CodecError,
    InvalidNumberOfElementsError, copy_fill_value_into,
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
        let options = self.codec_options();
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::async_retrieve_in_order(
            self.data_type(),
            self.element_order(),
            bytemuck::must_cast_slice(&chunk_shape),
            async |target| {
                self.async_retrieve_chunk_into_with_options(
                    chunk_indices,
                    target,
                    options,
                    resources,
                )
                .await
            },
            async || {
                self.async_retrieve_chunk_with_shape(
                    chunk_indices,
                    &chunk_shape,
                    options,
                    resources,
                )
                .await
            },
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
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
        let options = self.codec_options();
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        super::async_retrieve_in_order(
            self.data_type(),
            super::indexer_element_order(indexer, self.element_order()),
            &indexer.output_shape(),
            async |target| {
                self.async_retrieve_partial_chunk_into_with_shape(
                    chunk_indices,
                    &chunk_shape,
                    indexer,
                    target,
                    options,
                    resources,
                )
                .await
            },
            async || {
                self.async_retrieve_partial_chunk_with_shape(
                    chunk_indices,
                    &chunk_shape,
                    indexer,
                    options,
                    resources,
                )
                .await
            },
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
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
        let chunks = super::array_subset_chunks(self, array_subset)?;
        let array_subset_shape = array_subset.shape();
        super::async_retrieve_in_order(
            self.data_type(),
            self.element_order(),
            &array_subset_shape,
            async |target| {
                self.async_retrieve_array_subset_into(array_subset, target, resources)
                    .await
            },
            async || {
                self.async_retrieve_array_subset_c_order(
                    array_subset,
                    &array_subset_shape,
                    &chunks,
                    resources,
                )
                .await
            },
        )
        .await
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
        let Some(chunk_encoded) = self.async_retrieve_encoded_chunk(chunk_indices).await? else {
            return Ok(None);
        };
        let options = self.codec_options();
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        // One retrieval decodes the chunk, and takes it so that its bytes can be decoded in place if they are unique
        let chunk_encoded = std::cell::Cell::new(Some(chunk_encoded));
        let take_encoded = || chunk_encoded.take().expect("one retrieval");
        super::retrieve_in_order(
            self.data_type(),
            self.element_order(),
            bytemuck::must_cast_slice(&chunk_shape),
            |target| {
                self.decode_chunk_into(take_encoded(), &chunk_shape, target, options, resources)
            },
            || self.decode_chunk(take_encoded(), &chunk_shape, options, resources),
        )
        .map(Some)
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
        output_target: &mut ArrayBytesTarget<'_>,
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
    /// Retrieve `array_subset`, which spans `chunks`, with elements in C order.
    ///
    /// `chunks` are the chunks intersecting `array_subset` (see [`array_subset_chunks`](super::array_subset_chunks)).
    async fn async_retrieve_array_subset_c_order(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        array_subset_shape: &[u64],
        chunks: &ArraySubset,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        let options = self.codec_options();
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => super::fill_value_bytes(
                self.data_type(),
                self.fill_value(),
                array_subset.num_elements(),
            ),
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    self.async_retrieve_chunk_with_options(chunk_indices, options, resources)
                        .await
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.async_retrieve_partial_chunk_with_options(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        options,
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

                let bytes = if let Some(buffers) =
                    FixedDecodeBuffers::new(self.data_type(), array_subset_shape)
                {
                    self.async_retrieve_multi_chunk_fixed(
                        array_subset,
                        chunks,
                        buffers,
                        chunk_concurrent_limit,
                        options,
                        &resources,
                    )
                    .await?
                } else {
                    self.async_retrieve_multi_chunk_variable(
                        array_subset,
                        chunks,
                        array_subset_shape,
                        chunk_concurrent_limit,
                        options,
                        &resources,
                    )
                    .await?
                };
                bytes.validate(array_subset.num_elements(), self.data_type())?;
                Ok(bytes.into_owned())
            }
        }
    }

    /// Retrieve the chunk at `chunk_indices` in C order, or its fill value if it does not exist.
    pub(in crate::array) async fn async_retrieve_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        self.async_retrieve_chunk_with_shape(
            chunk_indices,
            &self.chunk_shape(chunk_indices)?,
            options,
            resources,
        )
        .await
    }

    /// Retrieve the chunk at `chunk_indices` with `chunk_shape` in C order, or its fill value if it does not exist.
    async fn async_retrieve_chunk_with_shape(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        if let Some(chunk_encoded) = self.async_retrieve_encoded_chunk(chunk_indices).await? {
            self.decode_chunk(chunk_encoded, chunk_shape, options, resources)
        } else {
            super::fill_value_bytes(
                self.data_type(),
                self.fill_value(),
                chunk_shape.num_elements_u64(),
            )
        }
    }

    pub(in crate::array) async fn async_retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
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
        if let Some(chunk_encoded) = self.async_retrieve_encoded_chunk(chunk_indices).await? {
            self.decode_chunk_into(
                chunk_encoded,
                &chunk_shape,
                output_target,
                options,
                resources,
            )
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError)
        }
    }

    /// Retrieve the elements of `indexer` in the chunk at `chunk_indices` in C order.
    pub(in crate::array) async fn async_retrieve_partial_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        self.async_retrieve_partial_chunk_with_shape(
            chunk_indices,
            &chunk_shape,
            indexer,
            options,
            resources,
        )
        .await
    }

    /// Retrieve the elements of `indexer` in the chunk at `chunk_indices` with `chunk_shape` in C order.
    ///
    /// `indexer` must have been validated against `chunk_shape` (see [`validate_partial_chunk_subset`](super::validate_partial_chunk_subset)).
    async fn async_retrieve_partial_chunk_with_shape(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        if indexer.covers_array(bytemuck::must_cast_slice(chunk_shape)) {
            return self
                .async_retrieve_chunk_with_shape(chunk_indices, chunk_shape, options, resources)
                .await;
        }

        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_storage().await?;
        let input_handle = Arc::new((storage_transformer, chunk_key));
        let bytes = self
            .codecs_bound()
            .async_partial_decoder(input_handle, chunk_shape, options, resources)
            .await?
            .partial_decode(indexer, options, resources)
            .await?
            .into_owned();
        bytes.validate(indexer.len(), self.data_type())?;
        Ok(bytes)
    }

    pub(in crate::array) async fn async_retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        self.async_retrieve_partial_chunk_into_with_shape(
            chunk_indices,
            &chunk_shape,
            indexer,
            output_target,
            options,
            resources,
        )
        .await
    }

    /// Retrieve the elements of `indexer` in the chunk at `chunk_indices` with `chunk_shape` into `output_target`.
    ///
    /// `indexer` must have been validated against `chunk_shape` (see [`validate_partial_chunk_subset`](super::validate_partial_chunk_subset)).
    async fn async_retrieve_partial_chunk_into_with_shape(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        if indexer.covers_array(bytemuck::must_cast_slice(chunk_shape)) {
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
            .async_partial_decoder(input_handle, chunk_shape, options, resources)
            .await?
            .partial_decode_into(indexer, output_target, options, resources)
            .await?;
        Ok(())
    }

    /// Retrieve the chunk at `chunk_indices` in C order if it exists.
    pub(in crate::array) async fn async_retrieve_chunk_if_exists_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<ArrayBytes<'static>>, ArrayError> {
        let Some(chunk_encoded) = self.async_retrieve_encoded_chunk(chunk_indices).await? else {
            return Ok(None);
        };
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        self.decode_chunk(chunk_encoded, &chunk_shape, options, resources)
            .map(Some)
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
        array_subset_shape: &[u64],
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
                    self.async_retrieve_partial_chunk_with_options(
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
            array_subset_shape,
            self.data_type(),
        )?)
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types (async).
    /// Also handles optional data types with fixed-length inner types.
    pub(crate) async fn async_retrieve_multi_chunk_fixed(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        mut buffers: FixedDecodeBuffers,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        retrieve_multi_chunk_fixed_into(
            self,
            array_subset,
            chunks,
            chunk_concurrent_limit,
            &buffers.target()?,
            |chunk_indices, chunk_subset, overlap, mut target| async move {
                self.async_retrieve_partial_chunk_into_with_options(
                    &chunk_indices,
                    &overlap.relative_to(chunk_subset.start())?,
                    &mut target,
                    options,
                    resources,
                )
                .await
            },
        )
        .await?;
        // SAFETY: decoding covers the array subset and satisfies the target's output requirements
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
        output_target: &mut ArrayBytesTarget<'_>,
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
        output_target: &mut ArrayBytesTarget<'_>,
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

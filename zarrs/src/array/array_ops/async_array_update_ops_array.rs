use crate::array::ElementOrder;
use inherent::inherent;
use std::sync::Arc;
use zarrs_codec::Resources;

use futures::{StreamExt, TryStreamExt};

use super::super::concurrency::concurrency_chunks_and_codec;
use super::{
    Array, ArrayError, ArraySubsetTraits, AsyncArrayUpdateOps, AsyncReadableStorageTraits,
    AsyncReadableWritableStorageTraits, AsyncWritableStorageTraits, CodecOptions, Indexer,
    IntoArrayBytes, MaybeSend, recommended_codec_concurrency,
};
use crate::array::{ArrayIndices, update_array_bytes};
use zarrs_codec::{
    ArrayToBytesCodecTraits, AsyncArrayPartialEncoderTraits, CodecError, CodecTraits,
};

#[cfg(feature = "async")]
#[inherent]
impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> AsyncArrayUpdateOps
    for Array<TStorage>
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_partial_chunk<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_store_partial_chunk_with_options(
            chunk_indices,
            indexer,
            indexer_data,
            self.codec_options(),
            resources,
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_array_subset<'a, T: IntoArrayBytes<'a> + MaybeSend>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let options = self.codec_options();
        // Validation
        if array_subset.dimensionality() != self.shape().len() {
            return Err(ArrayError::InvalidArraySubset(
                Box::new(array_subset.to_array_subset()),
                self.shape().to_vec(),
            ));
        }

        // Find the chunks intersecting this array subset
        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                Box::new(array_subset.to_array_subset()),
                self.shape().to_vec(),
            ));
        };
        let num_chunks = chunks.num_elements_usize();
        if num_chunks == 1 {
            let chunk_indices = chunks.start();
            let chunk_subset = self.chunk_subset(chunk_indices)?;
            if chunk_subset == array_subset {
                // A fast path if the array subset matches the chunk subset
                // This skips the internal decoding occurring in store_partial_chunk
                self.async_store_chunk_with_options(chunk_indices, subset_data, options, resources)
                    .await?;
            } else {
                // Store the chunk subset
                self.async_store_partial_chunk_with_options(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    subset_data,
                    options,
                    resources,
                )
                .await?;
            }
        } else {
            let subset_bytes = subset_data.into_array_bytes(self.data_type(), ElementOrder::C)?;
            subset_bytes.validate(array_subset.num_elements(), self.data_type())?;

            // Calculate chunk/codec concurrency
            let chunk_shape = self.chunk_shape(&vec![0; self.dimensionality()])?;
            let codec_concurrency = recommended_codec_concurrency(self, &chunk_shape)?;
            let (chunk_concurrent_limit, resources) = concurrency_chunks_and_codec(
                num_chunks,
                self.chunk_concurrent_minimum,
                resources,
                &codec_concurrency,
            );

            let array_subset_start = array_subset.start();
            let array_subset_shape = array_subset.shape();
            let resources = &resources;
            let store_chunk = |chunk_indices: ArrayIndices| {
                let chunk_subset = self.chunk_subset(&chunk_indices).unwrap(); // FIXME: unwrap
                let overlap = array_subset.overlap(&chunk_subset).unwrap(); // FIXME: unwrap
                let chunk_subset_in_array_subset =
                    overlap.relative_to(&array_subset_start).unwrap();
                let array_subset_in_chunk_subset =
                    overlap.relative_to(chunk_subset.start()).unwrap();
                let chunk_subset_bytes = subset_bytes
                    .extract_array_subset(
                        &chunk_subset_in_array_subset,
                        &array_subset_shape,
                        self.data_type(),
                    )
                    .unwrap(); // FIXME: unwrap
                async move {
                    self.async_store_partial_chunk_with_options(
                        &chunk_indices,
                        &array_subset_in_chunk_subset,
                        chunk_subset_bytes,
                        options,
                        resources,
                    )
                    .await
                }
            };

            futures::stream::iter(&chunks.indices())
                .map(Ok)
                .try_for_each_concurrent(Some(chunk_concurrent_limit), store_chunk)
                .await?;
        }
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_compact_chunk(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<bool, ArrayError> {
        let options = self.codec_options();
        let chunk_bytes = self.async_retrieve_encoded_chunk(chunk_indices).await?;
        if let Some(chunk_bytes) = chunk_bytes {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            if let Some(compacted_bytes) = self.codecs_bound.compact(
                chunk_bytes.into(),
                &chunk_shape,
                options,
                resources.compute(),
            )? {
                unsafe {
                    self.async_store_encoded_chunk(chunk_indices, compacted_bytes)
                        .await?;
                }
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }
    #[allow(clippy::must_use_candidate)]
    pub fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits> {
        self.with_storage(self.storage.clone().readable())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError> {
        self.async_partial_encoder_with_options(chunk_indices, self.codec_options(), resources)
            .await
    }
}

impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> Array<TStorage> {
    pub(in crate::array) async fn async_store_partial_chunk_with_options<
        'a,
        T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_data: T,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices)?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        if let Some(chunk_subset) = indexer.as_array_subset() {
            if !chunk_subset.inbounds_shape(&chunk_shape) {
                return Err(ArrayError::InvalidChunkSubset(
                    Box::new(chunk_subset.to_array_subset()),
                    chunk_indices.to_vec(),
                    chunk_shape,
                ));
            }
        } else {
            indexer.validate(&chunk_shape).map_err(CodecError::from)?;
        }
        if indexer.covers_array(&chunk_shape) {
            return self
                .async_store_chunk_with_options(chunk_indices, indexer_data, options, resources)
                .await;
        }

        let indexer_bytes = indexer_data.into_array_bytes(self.data_type(), ElementOrder::C)?;
        indexer_bytes.validate(indexer.len(), self.data_type())?;
        if indexer.is_empty() {
            return Ok(());
        }

        let partial_encoding_storage = if options.experimental_partial_encoding()
            && self.codecs.partial_encoder_capability().partial_encode
        {
            let storage = self.async_readable_writable_storage().await?;
            storage.supports_set_partial().then_some(storage)
        } else {
            None
        };

        if let Some(storage) = partial_encoding_storage {
            // A write of only the fill value to a missing chunk leaves it missing.
            // The read-modify-write path does this (and erases a chunk that becomes all fill) in `async_store_chunk_with_options`.
            if !options.store_empty_chunks()
                && indexer_bytes.is_fill_value(self.fill_value())
                && storage
                    .size_key(&self.chunk_key(chunk_indices)?)
                    .await?
                    .is_none()
            {
                return Ok(());
            }

            let partial_encoder = self
                .async_partial_encoder_with_options(chunk_indices, options, resources)
                .await?;
            debug_assert!(
                partial_encoder.supports_partial_encode(),
                "partial encoder is misrepresenting its capabilities"
            );
            partial_encoder
                .partial_encode(indexer, &indexer_bytes, options, resources)
                .await?;
            Ok(())
        } else {
            let chunk_bytes_old = self
                .async_retrieve_chunk_with_options(chunk_indices, options, resources)
                .await?;

            let chunk_bytes_new = update_array_bytes(
                chunk_bytes_old,
                &chunk_shape,
                indexer,
                &indexer_bytes,
                self.data_type().size(),
            )?;

            self.async_store_chunk_with_options(chunk_indices, chunk_bytes_new, options, resources)
                .await
        }
    }

    pub(in crate::array) async fn async_partial_encoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;

        // Input/output
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.async_readable_writable_storage().await?;
        let input_output_handle = Arc::new((storage_transformer, chunk_key));
        Ok(self
            .codecs_bound()
            .async_partial_encoder(input_output_handle, &chunk_shape, options, resources)
            .await?)
    }
}

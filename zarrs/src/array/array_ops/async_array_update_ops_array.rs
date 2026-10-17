use inherent::inherent;
use std::sync::Arc;

use super::{AsyncArrayUpdateOps, *};
use zarrs_codec::{ArrayToBytesCodecTraits, AsyncArrayPartialEncoderTraits};

#[cfg(feature = "async")]
#[inherent]
impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> AsyncArrayUpdateOps
    for Array<TStorage>
{
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
            let subset_bytes = subset_data.into_array_bytes(self.data_type())?;
            subset_bytes.validate(array_subset.num_elements(), self.data_type())?;
            self.async_store_array_subset_planned(
                &chunks,
                &array_subset.to_array_subset(),
                subset_bytes,
                options,
                resources,
            )
            .await?;
        }
        Ok(())
    }

    pub async fn async_compact_chunk(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<bool, ArrayError> {
        let options = self.codec_options();
        let chunk_bytes = self.async_retrieve_encoded_chunk(chunk_indices).await?;
        if let Some(chunk_bytes) = chunk_bytes {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            if let Some(compacted_bytes) =
                self.codecs_bound
                    .compact(chunk_bytes.into(), &chunk_shape, options, resources)?
            {
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
    pub fn async_readable(&self) -> Array<dyn AsyncReadableStorageTraits> {
        self.with_storage(self.storage.clone().readable())
    }

    pub async fn async_partial_encoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError> {
        let _ = resources;
        self.async_planned_partial_encoder(chunk_indices).await
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
                    chunk_subset.to_array_subset(),
                    chunk_indices.to_vec(),
                    chunk_shape,
                ));
            }
            if super::subset_is_whole_chunk(chunk_subset, &chunk_shape) {
                return self
                    .async_store_chunk_with_options(chunk_indices, indexer_data, options, resources)
                    .await;
            }
        }

        let indexer_bytes = indexer_data.into_array_bytes(self.data_type())?;
        indexer_bytes.validate(indexer.len(), self.data_type())?;

        self.async_store_partial_chunk_planned(
            chunk_indices,
            indexer,
            indexer_bytes,
            options,
            resources,
        )
        .await
    }
}

use std::collections::HashMap;
use std::sync::Arc;

use futures::lock::Mutex as AsyncMutex;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use zarrs_chunk_grid::{ChunkGrid, ChunkGridTraits};

use super::{
    ShardingCodecOptions, ShardingIndexLocation, nested_local_subchunk_grids, sharding_index_shape,
    subchunk_updates,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::codec::array_to_bytes::sharding::{
    calculate_chunks_per_shard, compute_index_encoded_size, subchunk_grid,
};
use crate::array::{
    ArrayBytes, ChunkShape, ChunkShapeTraits, CodecChainBound, CowBytes, DataType,
    transmute_to_bytes,
};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderSubchunkingTraits,
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits, CodecError, CodecOptions, Resources, update_array_bytes,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

pub(crate) struct AsyncShardingPartialEncoder {
    input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
    subchunk_shape: ChunkShape,
    chunk_grid: RegularBoundedChunkGrid,
    inner_codecs: Arc<CodecChainBound>,
    index_codecs: Arc<CodecChainBound>,
    index_location: ShardingIndexLocation,
    index_shape: ChunkShape,
    shard_index: Arc<AsyncMutex<Option<Vec<u64>>>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl AsyncShardingPartialEncoder {
    /// Create a new partial encoder for the sharding codec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn new(
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shard_shape: &[std::num::NonZeroU64],
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: Arc<CodecChainBound>,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        resources: &Resources,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(shard_shape, &subchunk_shape)?;
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());

        // Decode the index
        let shard_index = super::decode_shard_index_async_partial_decoder(
            input_output_handle.as_ref(),
            &index_codecs,
            index_location,
            shard_shape,
            &subchunk_shape,
            options,
            resources,
        )
        .await?
        .unwrap_or_else(|| {
            let num_chunks =
                usize::try_from(chunks_per_shard.iter().map(|x| x.get()).product::<u64>()).unwrap();
            vec![u64::MAX; num_chunks * 2]
        });

        let chunk_grid = subchunk_grid(shard_shape, &subchunk_shape)?;
        Ok(Self {
            input_output_handle,
            subchunk_shape,
            chunk_grid,
            inner_codecs,
            index_codecs,
            index_location,
            index_shape,
            shard_index: Arc::new(AsyncMutex::new(Some(shard_index))),
            sharding_options,
        })
    }

    async fn loaded_shard_index<'a>(
        &self,
        index: &'a mut Option<Vec<u64>>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<&'a mut Vec<u64>, CodecError> {
        if index.is_none() {
            let shape: ChunkShape = self
                .chunk_grid
                .array_shape()
                .iter()
                .map(|&length| std::num::NonZeroU64::new(length).unwrap())
                .collect();
            *index = Some(
                super::decode_shard_index_async_partial_decoder(
                    self.input_output_handle.as_ref(),
                    &self.index_codecs,
                    self.index_location,
                    &shape,
                    &self.subchunk_shape,
                    options,
                    resources,
                )
                .await?
                .unwrap_or_else(|| vec![u64::MAX; self.index_shape.num_elements_usize()]),
            );
        }
        Ok(index.as_mut().expect("loaded above"))
    }

    /// Append the encoded `updated_subchunks` to the shard and replace their entries in the shard index.
    ///
    /// A [`None`] subchunk is removed. The cached `shard_index_guard` is only updated once the shard is written.
    /// If the input/output handle does not support partial encoding, the shard is rewritten compactly instead.
    #[allow(clippy::too_many_lines)]
    async fn commit_subchunk_updates(
        &self,
        shard_index_guard: &mut Option<Vec<u64>>,
        updated_subchunks: Vec<(u64, Option<CowBytes<'_>>)>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        let mut shard_index = self.loaded_shard_index(shard_index_guard, options, resources).await?.clone();
        // Rewrite the retained subchunks too if the shard is rewritten entirely anyway (e.g. in memory)
        //  Otherwise superseded subchunks accumulate in the shard
        let mut updated_subchunks = updated_subchunks;
        if !self.input_output_handle.supports_partial_encode() {
            let (retained, byte_ranges) =
                super::retained_subchunks(&shard_index, &updated_subchunks);
            if !retained.is_empty() {
                let retained_encoded = self
                    .input_output_handle
                    .partial_decode_many(Box::new(byte_ranges.into_iter()), options, resources)
                    .await?
                    .ok_or_else(super::missing_retained_subchunks)?;
                updated_subchunks.extend(
                    std::iter::zip(retained, retained_encoded)
                        .map(|(index, encoded)| (index, Some(encoded.into_static()))),
                );
            }
        }


        // Get the maximum offset of existing encoded chunks
        let max_data_offset = shard_index
            .iter()
            .tuples()
            .map(|(&offset, &size)| {
                if offset == u64::MAX && size == u64::MAX {
                    Ok(0)
                } else {
                    offset
                        .checked_add(size)
                        .ok_or_else(super::invalid_shard_index)
                }
            })
            .try_fold(0, |max, end| end.map(|end| end.max(max)))?;

        // Check if the shard can be entirely rewritten instead of appended
        //  This occurs if the shard index is empty if all of the intersected subchunks are removed
        for (subchunk_index, _) in &updated_subchunks {
            shard_index[usize::try_from(subchunk_index * 2).unwrap()] = u64::MAX;
            shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()] = u64::MAX;
        }
        let rewrite = shard_index.par_iter().all(|&x| x == u64::MAX);
        let max_data_offset = if rewrite { 0 } else { max_data_offset };

        // Get the offset for new data
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &self.index_shape)?;
        let offset_new_chunks = match self.index_location {
            ShardingIndexLocation::Start => max_data_offset.max(index_encoded_size),
            ShardingIndexLocation::End => max_data_offset,
        };

        // Validate the entire append before changing the cached index or storage.
        let appended_size = updated_subchunks
            .iter()
            .try_fold(0u64, |size, (_, encoded)| {
                size.checked_add(encoded.as_ref().map_or(0, |bytes| bytes.len() as u64))
                    .ok_or_else(super::invalid_shard_index)
            })?;
        let data_end = offset_new_chunks
            .checked_add(appended_size)
            .ok_or_else(super::invalid_shard_index)?;
        if self.index_location == ShardingIndexLocation::End {
            data_end
                .checked_add(index_encoded_size)
                .ok_or_else(super::invalid_shard_index)?;
        }

        // Update the shard index
        {
            let mut offset_append = offset_new_chunks;
            for (subchunk_index, subchunk_encoded) in &updated_subchunks {
                if let Some(subchunk_encoded) = subchunk_encoded {
                    let len = subchunk_encoded.len() as u64;
                    shard_index[usize::try_from(subchunk_index * 2).unwrap()] = offset_append;
                    shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()] = len;
                    offset_append += len;
                } else {
                    shard_index[usize::try_from(subchunk_index * 2).unwrap()] = u64::MAX;
                    shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()] = u64::MAX;
                }
            }
        }

        // A failed write may have changed storage; reload its index before reuse.
        *shard_index_guard = None;
        if shard_index.par_iter().all(|&x| x == u64::MAX) {
            // Erase the shard if all chunks are empty
            self.input_output_handle.erase().await?;
        } else {
            // Encode the updated shard index
            let shard_index_bytes: CowBytes = transmute_to_bytes(shard_index.as_slice()).into();
            let encoded_array_index = self
                .index_codecs
                .encode(
                    shard_index_bytes.into(),
                    &self.index_shape,
                    options,
                    resources.compute(),
                )?
                .into_static();

            // Get the total size of the encoded subchunks
            let encoded_subchunks_size = updated_subchunks
                .iter()
                .filter_map(|(_, subchunk_encoded)| {
                    subchunk_encoded.as_ref().map(zarrs_codec::CowBytes::len)
                })
                .sum::<usize>();

            // Get the suffix write size
            let suffix_write_size = match self.index_location {
                ShardingIndexLocation::Start => encoded_subchunks_size,
                ShardingIndexLocation::End => encoded_subchunks_size + encoded_array_index.len(),
            };

            // Concatenate the updated subchunks
            let mut encoded_output = Vec::with_capacity(suffix_write_size);
            for (_, subchunk_encoded) in updated_subchunks {
                if let Some(subchunk_encoded) = subchunk_encoded {
                    encoded_output.extend_from_slice(&subchunk_encoded);
                }
            }

            if rewrite {
                // Replace the shard rather than erasing it first, so a failure leaves it intact
                let shard = match self.index_location {
                    ShardingIndexLocation::Start => {
                        debug_assert_eq!(offset_new_chunks, encoded_array_index.len() as u64);
                        let mut shard =
                            Vec::with_capacity(encoded_array_index.len() + encoded_output.len());
                        shard.extend_from_slice(&encoded_array_index);
                        shard.extend_from_slice(&encoded_output);
                        shard
                    }
                    ShardingIndexLocation::End => {
                        debug_assert_eq!(offset_new_chunks, 0);
                        encoded_output.extend_from_slice(&encoded_array_index);
                        encoded_output
                    }
                };
                self.input_output_handle
                    .encode(CowBytes::from(shard), options, resources)
                    .await?;
            } else {
                // Write the encoded index and updated subchunks
                match self.index_location {
                    ShardingIndexLocation::Start => {
                        self.input_output_handle
                            .partial_encode_many(
                                Box::new(
                                    [
                                        (0, encoded_array_index),
                                        (offset_new_chunks, CowBytes::from(encoded_output)),
                                    ]
                                    .into_iter(),
                                ),
                                options,
                                resources,
                            )
                            .await?;
                    }
                    ShardingIndexLocation::End => {
                        encoded_output.extend_from_slice(&encoded_array_index);
                        self.input_output_handle
                            .partial_encode_many(
                                Box::new(
                                    [(offset_new_chunks, CowBytes::from(encoded_output))]
                                        .into_iter(),
                                ),
                                options,
                                resources,
                            )
                            .await?;
                    }
                }
            }
        }
        *shard_index_guard = Some(shard_index);
        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderSubchunkingTraits for AsyncShardingPartialEncoder {
    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        nested_local_subchunk_grids(ChunkGrid::new(self.chunk_grid.clone()), &self.inner_codecs)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialEncoder {
    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
            + self.index_shape.num_elements_usize() * size_of::<u64>()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let handle: Arc<dyn AsyncBytesPartialDecoderTraits> = self.input_output_handle.clone();
        let mut shard_index = self.shard_index.lock().await;
        let shard_index = self
            .loaded_shard_index(&mut shard_index, options, resources)
            .await?;
        super::sharding_partial_decoder_async::partial_decode(
            &handle,
            &self.chunk_grid,
            &self.subchunk_shape,
            &self.inner_codecs,
            Some(shard_index.as_slice()),
            indexer,
            options,
            resources,
        )
        .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialEncoderTraits for AsyncShardingPartialEncoder {
    async fn erase(&self) -> Result<(), super::CodecError> {
        let mut shard_index = self.shard_index.lock().await;
        let result = self.input_output_handle.erase().await;
        if result.is_ok() {
            if let Some(index) = shard_index.as_mut() {
                index.fill(u64::MAX);
            } else {
                *shard_index = Some(vec![u64::MAX; self.index_shape.num_elements_usize()]);
            }
        } else {
            *shard_index = None;
        }
        result
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
    async fn partial_encode(
        &self,
        chunk_subset_indexer: &dyn crate::array::Indexer,
        chunk_subset_bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
        resources: &Resources,
    ) -> Result<(), super::CodecError> {
        let data_type = self.inner_codecs.data_type();
        let fill_value = self
            .inner_codecs
            .fill_value()
            .ok_or(CodecError::UndefinedFillValue)?;
        let mut shard_index_guard = self.shard_index.lock().await;
        let shard_index = self
            .loaded_shard_index(&mut shard_index_guard, options, resources)
            .await?;

        // Validate the indexer and bytes
        chunk_subset_indexer.validate(self.chunk_grid.array_shape())?;
        chunk_subset_bytes.validate(chunk_subset_indexer.len(), data_type)?;
        if chunk_subset_indexer.is_empty() {
            return Ok(());
        }

        // Split the update into the intersected subchunks
        let updates = subchunk_updates(
            &self.chunk_grid,
            &self.subchunk_shape,
            chunk_subset_indexer,
            chunk_subset_bytes,
            data_type,
        )?;

        // Get the byte ranges of the subchunks that need to be retrieved
        //   This only includes chunks that are not entirely replaced by the update.
        //   Sorting byte ranges may improves store retrieve efficiency in some cases
        let (subchunks_straddling, byte_ranges): (Vec<u64>, Vec<ByteRange>) = updates
            .iter()
            .filter(|update| !update.fully_covered)
            .filter_map(|update| {
                let (offset, size) = super::subchunk_offset_size(
                    shard_index,
                    usize::try_from(update.subchunk_index).unwrap(),
                )?;
                Some((
                    update.subchunk_index,
                    ByteRange::FromStart(offset, Some(size)),
                ))
            })
            .sorted_by_key(|(_, byte_range)| *byte_range)
            .unzip();

        // Read the straddling subchunks
        let mut subchunks_encoded: HashMap<u64, Vec<u8>> = self
            .input_output_handle
            .partial_decode_many(Box::new(byte_ranges.into_iter()), options, resources)
            .await?
            .map(|bytes| {
                std::iter::zip(
                    subchunks_straddling,
                    bytes.into_iter().map(CowBytes::into_vec),
                )
                .collect()
            })
            .unwrap_or_default();

        // Decode, update and encode all of the intersecting subchunks
        let updates = updates
            .into_iter()
            .map(|update| {
                let subchunk_encoded = subchunks_encoded.remove(&update.subchunk_index);
                (update, subchunk_encoded)
            })
            .collect::<Vec<_>>();

        let (subchunk_concurrent_limit, subchunk_resources) =
            super::get_concurrent_target_and_resources(
                &self.inner_codecs,
                &self.subchunk_shape,
                updates.len(),
                resources.compute(),
            )?;
        let updated_subchunks = updates
            .concurrent_limit(subchunk_concurrent_limit)
            .map(|(update, subchunk_encoded)| {
                let subchunk_decoded = if let Some(subchunk_encoded) = subchunk_encoded {
                    self.inner_codecs
                        .decode(
                            CowBytes::from(subchunk_encoded),
                            &update.subchunk_shape,
                            options,
                            &subchunk_resources,
                        )?
                        .into_owned()
                } else {
                    ArrayBytes::new_fill_value(
                        data_type,
                        update.subchunk_shape.num_elements_u64(),
                        fill_value,
                    )?
                };
                let subchunk_updated = update_array_bytes(
                    subchunk_decoded,
                    bytemuck::cast_slice(&update.subchunk_shape),
                    update.indexer.as_ref(),
                    &update.bytes,
                    data_type.size(),
                )?;
                if subchunk_updated.is_fill_value(fill_value) {
                    Ok((update.subchunk_index, None))
                } else {
                    let subchunk_encoded = self
                        .inner_codecs
                        .encode(
                            subchunk_updated,
                            &update.subchunk_shape,
                            options,
                            &subchunk_resources,
                        )?
                        .into_static();
                    Ok((update.subchunk_index, Some(subchunk_encoded)))
                }
            })
            .collect::<Result<Vec<_>, CodecError>>()?;

        self.commit_subchunk_updates(
            &mut shard_index_guard,
            updated_subchunks,
            options,
            resources,
        )
        .await
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

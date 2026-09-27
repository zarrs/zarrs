use std::collections::HashMap;
use std::sync::Arc;

use futures::lock::Mutex as AsyncMutex;
use itertools::Itertools;
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use zarrs_chunk_grid::ChunkGrid;

use super::{
    ShardingCodecOptions, ShardingIndexLocation, nested_local_subchunk_grids, sharding_index_shape,
    subchunk_updates,
};
use crate::array::chunk_grid::{RegularBoundedChunkGrid, RegularChunkGrid};
use crate::array::codec::array_to_bytes::sharding::{
    calculate_chunks_per_shard, compute_index_encoded_size,
};
use crate::array::{
    ArrayBytes, ChunkShape, ChunkShapeTraits, CodecChainBound, CowBytes, DataType,
    transmute_to_bytes,
};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderSubchunkingTraits,
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits, CodecError, CodecOptions, update_array_bytes,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

pub(crate) struct AsyncShardingPartialEncoder {
    input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    chunk_grid: RegularChunkGrid,
    inner_codecs: Arc<CodecChainBound>,
    index_codecs: Arc<CodecChainBound>,
    index_location: ShardingIndexLocation,
    index_shape: ChunkShape,
    shard_index: Arc<AsyncMutex<Vec<u64>>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl AsyncShardingPartialEncoder {
    /// Create a new partial encoder for the sharding codec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn new(
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: Arc<CodecChainBound>,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(&shard_shape, &subchunk_shape)?;
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());

        // Decode the index
        let shard_index = super::decode_shard_index_async_partial_decoder(
            input_output_handle.as_ref(),
            &index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )
        .await?
        .unwrap_or_else(|| {
            let num_chunks =
                usize::try_from(chunks_per_shard.iter().map(|x| x.get()).product::<u64>()).unwrap();
            vec![u64::MAX; num_chunks * 2]
        });

        let chunk_grid = RegularChunkGrid::new(
            bytemuck::must_cast_slice(shard_shape.as_slice()).to_vec(),
            subchunk_shape.clone(),
        )
        .map_err(|err| CodecError::from(err.to_string()))?;
        Ok(Self {
            input_output_handle,
            shard_shape,
            subchunk_shape,
            chunk_grid,
            inner_codecs,
            index_codecs,
            index_location,
            index_shape,
            shard_index: Arc::new(AsyncMutex::new(shard_index)),
            sharding_options,
        })
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderSubchunkingTraits for AsyncShardingPartialEncoder {
    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        let shard_shape = bytemuck::must_cast_slice(&self.shard_shape).to_vec();
        let subchunk_grid = ChunkGrid::new(
            RegularBoundedChunkGrid::new(shard_shape, self.subchunk_shape.clone())
                .map_err(|err| CodecError::Other(err.to_string()))?,
        );
        nested_local_subchunk_grids(subchunk_grid, &self.inner_codecs)
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
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let handle: Arc<dyn AsyncBytesPartialDecoderTraits> = self.input_output_handle.clone();
        let shard_index = self.shard_index.lock().await;
        super::sharding_partial_decoder_async::partial_decode(
            &handle,
            &self.shard_shape,
            &self.subchunk_shape,
            &self.inner_codecs,
            Some(shard_index.as_slice()),
            indexer,
            options,
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
        self.input_output_handle.erase().await
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
    async fn partial_encode(
        &self,
        chunk_subset_indexer: &dyn crate::array::Indexer,
        chunk_subset_bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let data_type = self.inner_codecs.data_type();
        let fill_value = self.inner_codecs.fill_value();
        let mut shard_index = self.shard_index.lock().await;

        let chunks_per_shard = calculate_chunks_per_shard(&self.shard_shape, &self.subchunk_shape)?;
        let chunks_per_shard = chunks_per_shard.to_array_shape();

        // Get the maximum offset of existing encoded chunks
        let max_data_offset = shard_index
            .iter()
            .tuples()
            .map(|(&offset, &size)| {
                if offset == u64::MAX && size == u64::MAX {
                    0
                } else {
                    offset + size
                }
            })
            .max()
            .expect("shards cannot be empty");

        let subchunk_fill_value = || {
            ArrayBytes::new_fill_value(
                self.inner_codecs.data_type(),
                self.subchunk_shape.num_elements_u64(),
                self.inner_codecs.fill_value(),
            )
        };

        // Validate the indexer and bytes
        chunk_subset_indexer.validate(bytemuck::cast_slice(&self.shard_shape))?;
        chunk_subset_bytes.validate(chunk_subset_indexer.len(), data_type)?;
        if chunk_subset_indexer.is_empty() {
            return Ok(());
        }

        // Split the update into the intersected subchunks
        let updates = subchunk_updates(
            &self.chunk_grid,
            &chunks_per_shard,
            &self.subchunk_shape,
            chunk_subset_indexer,
            chunk_subset_bytes,
            data_type,
        )?;
        let subchunks_intersected: Vec<u64> =
            updates.iter().map(|update| update.subchunk_index).collect();

        // Get all the subchunks that need to be retrieved
        //   This only includes chunks that are not entirely replaced by the update.
        let subchunks_indices: Vec<u64> = updates
            .iter()
            .filter(|update| !update.fully_covered)
            .map(|update| update.subchunk_index)
            .collect();

        // Get the byte ranges of the straddling subchunk indices
        //   Sorting byte ranges may improves store retrieve efficiency in some cases
        #[cfg(not(target_arch = "wasm32"))]
        let iterator = subchunks_indices.into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let iterator = subchunks_indices.into_iter();

        let (subchunks_indices, byte_ranges): (Vec<_>, Vec<_>) = iterator
            .filter_map(|subchunk_index| {
                let offset = shard_index[usize::try_from(subchunk_index * 2).unwrap()];
                let size = shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()];
                if offset == u64::MAX && size == u64::MAX {
                    None
                } else {
                    Some((subchunk_index, ByteRange::FromStart(offset, Some(size))))
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .sorted_by_key(|(_, byte_range)| *byte_range)
            .unzip();

        // Read the straddling subchunks
        let subchunks_encoded = self
            .input_output_handle
            .partial_decode_many(Box::new(byte_ranges.into_iter()), options)
            .await?
            .map(|bytes| {
                bytes
                    .into_iter()
                    .map(CowBytes::into_vec)
                    .collect::<Vec<_>>()
            });

        // Decode the straddling subchunks
        let mut subchunks_decoded: HashMap<_, _> =
            if let Some(subchunks_encoded) = subchunks_encoded {
                #[cfg(not(target_arch = "wasm32"))]
                let iterator = subchunks_indices.into_par_iter();
                #[cfg(target_arch = "wasm32")]
                let iterator = subchunks_indices.into_iter();

                let subchunks_encoded = iterator
                    .zip(subchunks_encoded)
                    .map(|(subchunk_index, subchunk_encoded)| {
                        Ok((
                            subchunk_index,
                            self.inner_codecs.decode(
                                CowBytes::from(subchunk_encoded),
                                &self.subchunk_shape,
                                options,
                            )?,
                        ))
                    })
                    .collect::<Result<Vec<_>, CodecError>>()?;
                HashMap::from_iter(subchunks_encoded)
            } else {
                HashMap::new()
            };

        // Update all of the intersecting subchunks
        let updates = updates
            .into_iter()
            .map(|update| {
                let subchunk_decoded = subchunks_decoded.remove(&update.subchunk_index);
                (update, subchunk_decoded)
            })
            .collect::<Vec<_>>();

        #[cfg(not(target_arch = "wasm32"))]
        let iterator = updates.into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let iterator = updates.into_iter();

        let subchunks_decoded = iterator
            .map(|(update, subchunk_decoded)| {
                let subchunk_decoded = if let Some(subchunk_decoded) = subchunk_decoded {
                    subchunk_decoded.into_owned()
                } else {
                    subchunk_fill_value()?
                };
                let subchunk_updated = update_array_bytes(
                    subchunk_decoded,
                    bytemuck::cast_slice(&self.subchunk_shape),
                    update.indexer.as_ref(),
                    &update.bytes,
                    data_type.size(),
                )?;
                Ok((update.subchunk_index, subchunk_updated))
            })
            .collect::<Result<Vec<_>, CodecError>>()?;

        // Encode the updated subchunks
        #[cfg(not(target_arch = "wasm32"))]
        let iterator = subchunks_decoded.into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let iterator = subchunks_decoded.into_iter();

        let updated_subchunks = iterator
            .map(|(subchunk_index, subchunk_decoded)| {
                if subchunk_decoded.is_fill_value(fill_value) {
                    Ok((subchunk_index, None))
                } else {
                    let subchunk_encoded = self
                        .inner_codecs
                        .encode(subchunk_decoded, &self.subchunk_shape, options)?
                        .into_static();
                    Ok((subchunk_index, Some(subchunk_encoded)))
                }
            })
            .collect::<Result<Vec<_>, CodecError>>()?;

        // Check if the shard can be entirely rewritten instead of appended
        //  This occurs if the shard index is empty if all of the intersected subchunks are removed
        for subchunk_index in &subchunks_intersected {
            shard_index[usize::try_from(subchunk_index * 2).unwrap()] = u64::MAX;
            shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()] = u64::MAX;
        }
        let max_data_offset = if shard_index.par_iter().all(|&x| x == u64::MAX) {
            self.input_output_handle.erase().await?;
            0
        } else {
            max_data_offset
        };

        // Get the offset for new data
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &self.index_shape)?;
        let offset_new_chunks = match self.index_location {
            ShardingIndexLocation::Start => max_data_offset.max(index_encoded_size),
            ShardingIndexLocation::End => max_data_offset,
        };

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

        if shard_index.par_iter().all(|&x| x == u64::MAX) {
            // Erase the shard if all chunks are empty
            self.input_output_handle.erase().await?;
        } else {
            // Encode the updated shard index
            let shard_index_bytes: CowBytes = transmute_to_bytes(shard_index.as_slice()).into();
            let encoded_array_index = self
                .index_codecs
                .encode(shard_index_bytes.into(), &self.index_shape, options)?
                .into_static();

            // Get the total size of the encoded subchunks
            let encoded_subchunks_size = updated_subchunks
                .iter()
                .filter_map(|(_, subchunk_encoded)| {
                    subchunk_encoded.as_ref().map(|bytes| bytes.len())
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
                        )
                        .await?;
                }
                ShardingIndexLocation::End => {
                    encoded_output.extend_from_slice(&encoded_array_index);
                    self.input_output_handle
                        .partial_encode_many(
                            Box::new(
                                [(offset_new_chunks, CowBytes::from(encoded_output))].into_iter(),
                            ),
                            options,
                        )
                        .await?;
                }
            }
        }
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

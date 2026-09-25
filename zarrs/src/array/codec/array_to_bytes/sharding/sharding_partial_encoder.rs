use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use itertools::Itertools;
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::IntoParallelIterator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use zarrs_data_type::FillValue;

use super::{ShardingCodecOptions, ShardingIndexLocation, sharding_index_shape};
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::array_to_bytes::sharding::{
    calculate_chunks_per_shard, compute_index_encoded_size,
};
use crate::array::{
    ArrayBytes, ArrayBytesRaw, ChunkShape, ChunkShapeTraits, CodecChain, DataType,
    transmute_to_bytes,
};
use zarrs_codec::{
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToBytesCodecTraits,
    BytesPartialEncoderTraits, CodecError, CodecOptions, update_array_bytes,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

pub(crate) struct ShardingPartialEncoder {
    input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
    shard_shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
    subchunk_shape: ChunkShape,
    chunk_grid: RegularChunkGrid,
    inner_codecs: Arc<CodecChain>,
    index_codecs: Arc<CodecChain>,
    index_location: ShardingIndexLocation,
    index_shape: ChunkShape,
    shard_index: Arc<Mutex<Vec<u64>>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl ShardingPartialEncoder {
    /// Create a new partial encoder for the sharding codec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        data_type: DataType,
        fill_value: FillValue,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: Arc<CodecChain>,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(&shard_shape, &subchunk_shape)?;
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());

        // Decode the index
        let shard_index = super::decode_shard_index_partial_decoder(
            input_output_handle.clone().into_dyn_decoder().as_ref(),
            &index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )?
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
            data_type,
            fill_value,
            subchunk_shape,
            chunk_grid,
            inner_codecs,
            index_codecs,
            index_location,
            index_shape,
            shard_index: Arc::new(Mutex::new(shard_index)),
            sharding_options,
        })
    }
}

impl ArrayPartialDecoderTraits for ShardingPartialEncoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.shard_index.lock().unwrap().len()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        super::sharding_partial_decoder_sync::partial_decode(
            &self.input_output_handle.clone().into_dyn_decoder(),
            &self.data_type,
            &self.fill_value,
            &self.shard_shape,
            &self.subchunk_shape,
            &self.inner_codecs,
            Some(self.shard_index.lock().unwrap().as_slice()),
            indexer,
            options,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl ArrayPartialEncoderTraits for ShardingPartialEncoder {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), super::CodecError> {
        let mut index = self.shard_index.lock().unwrap();
        self.input_output_handle.erase()?;
        index.fill(u64::MAX);
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
    fn partial_encode(
        &self,
        chunk_subset_indexer: &dyn crate::array::Indexer,
        chunk_subset_bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        chunk_subset_indexer.validate(bytemuck::must_cast_slice(&self.shard_shape))?;
        let num_elements = chunk_subset_indexer.len();
        chunk_subset_bytes.validate(num_elements, &self.data_type)?;
        if num_elements == 0 {
            return Ok(());
        }
        let mut shard_index_guard = self.shard_index.lock().unwrap();
        let mut shard_index = shard_index_guard.clone();

        let chunks_per_shard = calculate_chunks_per_shard(&self.shard_shape, &self.subchunk_shape)?;
        let chunks_per_shard = chunks_per_shard.to_array_shape();

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
        let (subchunk_concurrent_limit, subchunk_options) =
            super::get_concurrent_target_and_codec_options(
                &self.inner_codecs,
                &self.data_type,
                &self.subchunk_shape,
                &chunks_per_shard,
                options,
            )?;

        // Split each selection into one ordered update per affected subchunk.
        let updates = super::subchunk_updates(
            &self.chunk_grid,
            &chunks_per_shard,
            &self.subchunk_shape,
            chunk_subset_indexer,
            chunk_subset_bytes,
            &self.data_type,
        )?;
        let subchunks_intersected: Vec<_> = updates.iter().map(|u| u.subchunk_index).collect();
        let subchunks_indices: Vec<_> = updates
            .iter()
            .filter(|u| !u.fully_covered)
            .map(|u| u.subchunk_index)
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
        let subchunks_encoded = if byte_ranges.is_empty() {
            None
        } else {
            self.input_output_handle
                .partial_decode_many(Box::new(byte_ranges.into_iter()), options)?
                .map(|bytes| bytes.into_iter().map(Cow::into_owned).collect::<Vec<_>>())
        };

        // Decode the straddling subchunks
        let mut subchunks_decoded: HashMap<_, _> =
            if let Some(subchunks_encoded) = subchunks_encoded {
                let subchunks_encoded = crate::iter_concurrent_limit!(
                    subchunk_concurrent_limit,
                    subchunks_indices
                        .into_iter()
                        .zip(subchunks_encoded)
                        .collect::<Vec<_>>(),
                    map,
                    |(subchunk_index, subchunk_encoded)| {
                        Ok((
                            subchunk_index,
                            self.inner_codecs.decode(
                                Cow::Owned(subchunk_encoded),
                                &self.subchunk_shape,
                                &self.data_type,
                                &self.fill_value,
                                &subchunk_options,
                            )?,
                        ))
                    }
                )
                .collect::<Result<Vec<_>, CodecError>>()?;
                HashMap::from_iter(subchunks_encoded)
            } else {
                HashMap::new()
            };

        // Every update owns its subchunk; duplicates retain their input order.
        let updates = updates
            .into_iter()
            .map(|update| {
                let decoded = subchunks_decoded.remove(&update.subchunk_index);
                (update, decoded)
            })
            .collect::<Vec<_>>();
        let subchunks_decoded =
            crate::iter_concurrent_limit!(subchunk_concurrent_limit, updates, map, |(
                update,
                decoded,
            )| {
                let decoded = match decoded {
                    Some(decoded) => decoded.into_owned(),
                    None => ArrayBytes::new_fill_value(
                        &self.data_type,
                        self.subchunk_shape.num_elements_u64(),
                        &self.fill_value,
                    )?,
                };
                let updated = update_array_bytes(
                    decoded,
                    bytemuck::cast_slice(&self.subchunk_shape),
                    update.indexer.as_ref(),
                    &update.bytes,
                    self.data_type.size(),
                )?;
                Ok((update.subchunk_index, updated))
            })
            .collect::<Result<Vec<_>, CodecError>>()?;

        // Encode the updated subchunks
        let updated_subchunks = crate::iter_concurrent_limit!(
            subchunk_concurrent_limit,
            subchunks_decoded,
            map,
            |(subchunk_index, subchunk_decoded)| {
                if subchunk_decoded.is_fill_value(&self.fill_value) {
                    Ok((subchunk_index, None))
                } else {
                    let subchunk_encoded = self
                        .inner_codecs
                        .encode(
                            subchunk_decoded,
                            &self.subchunk_shape,
                            &self.data_type,
                            &self.fill_value,
                            &subchunk_options,
                        )?
                        .into_owned();
                    Ok((subchunk_index, Some(subchunk_encoded)))
                }
            }
        )
        .collect::<Result<Vec<_>, CodecError>>()?;

        // Check if the shard can be entirely rewritten instead of appended
        //  This occurs if the shard index is empty if all of the intersected subchunks are removed
        for subchunk_index in &subchunks_intersected {
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

        let appended_size = updated_subchunks
            .iter()
            .try_fold(0u64, |size, (_, bytes)| {
                size.checked_add(bytes.as_ref().map_or(0, |bytes| bytes.len() as u64))
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

        if shard_index.par_iter().all(|&x| x == u64::MAX) {
            // Erase the shard if all chunks are empty
            self.input_output_handle.erase()?;
        } else {
            // Encode the updated shard index
            let shard_index_bytes: ArrayBytesRaw =
                transmute_to_bytes(shard_index.as_slice()).into();
            let encoded_array_index = self
                .index_codecs
                .encode(
                    shard_index_bytes.into(),
                    &self.index_shape,
                    &crate::array::data_type::uint64(),
                    &FillValue::from(u64::MAX),
                    options,
                )?
                .into_owned();

            // Get the total size of the encoded subchunks
            let encoded_subchunks_size = updated_subchunks
                .iter()
                .filter_map(|(_, subchunk_encoded)| subchunk_encoded.as_ref().map(Vec::len))
                .try_fold(0usize, |size, len| {
                    size.checked_add(len).ok_or_else(super::invalid_shard_index)
                })?;

            // Get the suffix write size
            let suffix_write_size = match self.index_location {
                ShardingIndexLocation::Start => encoded_subchunks_size,
                ShardingIndexLocation::End => encoded_subchunks_size
                    .checked_add(encoded_array_index.len())
                    .ok_or_else(super::invalid_shard_index)?,
            };

            // Concatenate the updated subchunks
            let mut encoded_output = Vec::with_capacity(suffix_write_size);
            for (_, subchunk_encoded) in updated_subchunks {
                if let Some(subchunk_encoded) = subchunk_encoded {
                    encoded_output.extend(subchunk_encoded);
                }
            }

            if rewrite {
                self.input_output_handle.erase()?;
            }
            // Write the encoded index and updated subchunks
            match self.index_location {
                ShardingIndexLocation::Start => {
                    self.input_output_handle.partial_encode_many(
                        Box::new(
                            [
                                (0, Cow::Owned(encoded_array_index)),
                                (offset_new_chunks, Cow::Owned(encoded_output)),
                            ]
                            .into_iter(),
                        ),
                        options,
                    )?;
                }
                ShardingIndexLocation::End => {
                    encoded_output.extend(encoded_array_index);
                    self.input_output_handle.partial_encode_many(
                        Box::new([(offset_new_chunks, Cow::Owned(encoded_output))].into_iter()),
                        options,
                    )?;
                }
            }
        }
        *shard_index_guard = shard_index;
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

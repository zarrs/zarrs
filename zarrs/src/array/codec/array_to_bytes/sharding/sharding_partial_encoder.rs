use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use itertools::Itertools;
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use zarrs_data_type::FillValue;

use super::{ShardingIndexLocation, sharding_index_shape};
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::array_to_bytes::sharding::{
    calculate_chunks_per_shard, compute_index_encoded_size,
};
use crate::array::{
    ArrayBytes, ArrayBytesRaw, ArrayIndicesTinyVec, ChunkShape, ChunkShapeTraits, CodecChain,
    DataType, IndexerError, ravel_indices, transmute_to_bytes,
};
use zarrs_codec::{
    ArrayBytesExt, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToBytesCodecTraits,
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
        self.input_output_handle.erase()
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
    fn partial_encode(
        &self,
        chunk_subset_indexer: &dyn crate::array::Indexer,
        chunk_subset_bytes: &ArrayBytes<'_>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let mut shard_index = self.shard_index.lock().unwrap();

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

        let get_subchunks = |chunk_subset| self.chunk_grid.chunks_in_array_subset(chunk_subset);
        let subchunk_fill_value = || {
            ArrayBytes::new_fill_value(
                &self.data_type,
                self.subchunk_shape.num_elements_u64(),
                &self.fill_value,
            )
        };

        // Get all the subchunks that need to be retrieved
        //   This only includes chunks that straddle chunk subsets.
        //   Chunks that are entirely within a chunk subset are entirely replaced and are not read.
        let mut subchunks_intersected = HashSet::<u64>::new();
        let mut subchunks_indices = HashSet::<u64>::new();

        let Some(chunk_subset_indexer) = chunk_subset_indexer.as_array_subset() else {
            // TODO: Add support for generic indexers
            return Err(CodecError::from(
                "sharding_indexed does not yet support partial encoding with generic indexers",
            ));
        };

        // Check the subset is within the chunk shape
        if chunk_subset_indexer
            .end_exc()
            .iter()
            .zip(&self.shard_shape)
            .any(|(a, b)| *a > b.get())
        {
            Err(IndexerError::new_oob(
                chunk_subset_indexer.end_exc(),
                bytemuck::cast_slice(&self.shard_shape).to_vec(),
            ))?;
        }

        // Get the iterator over the subchunks
        let subchunks = get_subchunks(chunk_subset_indexer)?;
        let subchunks = subchunks.indices();

        // Get all the subchunks intersected
        subchunks_intersected.extend(subchunks.iter().map(
            |subchunk_indices: ArrayIndicesTinyVec| {
                ravel_indices(&subchunk_indices, &chunks_per_shard).expect("inbounds chunk")
            },
        ));

        // Get all the subchunks that need to be updated
        let chunk_subset_start = chunk_subset_indexer.start();
        let chunk_subset_end_exc = chunk_subset_indexer.end_exc();
        subchunks_indices.extend(subchunks.iter().filter_map(
            |subchunk_indices: ArrayIndicesTinyVec| {
                let subchunk_subset = self
                    .chunk_grid
                    .subset(&subchunk_indices)
                    .expect("matching dimensionality");

                // Check if the subchunk straddles the chunk subset
                if subchunk_subset
                    .start()
                    .iter()
                    .zip(chunk_subset_start.iter())
                    .any(|(a, b)| a < b)
                    || subchunk_subset
                        .end_exc()
                        .iter()
                        .zip(chunk_subset_end_exc.iter())
                        .any(|(a, b)| *a > *b)
                {
                    let subchunk_index = ravel_indices(&subchunk_indices, &chunks_per_shard)
                        .expect("inbounds chunk");
                    Some(subchunk_index)
                } else {
                    None
                }
            },
        ));

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
            .partial_decode_many(Box::new(byte_ranges.into_iter()), options)?
            .map(|bytes| bytes.into_iter().map(Cow::into_owned).collect::<Vec<_>>());

        // Decode the straddling subchunks
        let subchunks_decoded: HashMap<_, _> = if let Some(subchunks_encoded) = subchunks_encoded {
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
                            Cow::Owned(subchunk_encoded),
                            &self.subchunk_shape,
                            &self.data_type,
                            &self.fill_value,
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
        let subchunks_decoded = Arc::new(Mutex::new(subchunks_decoded));
        let subchunks = get_subchunks(chunk_subset_indexer)?;

        #[cfg(not(target_arch = "wasm32"))]
        let iterator = subchunks.indices().into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let mut iterator = subchunks.indices().into_iter();

        let chunk_subset_start = chunk_subset_indexer.start();
        let chunk_subset_shape = chunk_subset_indexer.shape();
        iterator.try_for_each(|subchunk_indices: ArrayIndicesTinyVec| {
            // Extract the subchunk bytes that overlap with the chunk subset
            let subchunk_index =
                ravel_indices(&subchunk_indices, &chunks_per_shard).expect("inbounds chunk");
            let subchunk_subset = self
                .chunk_grid
                .subset(&subchunk_indices)
                .expect("matching dimensionality");
            let subchunk_subset_overlap = chunk_subset_indexer.overlap(&subchunk_subset).unwrap();
            let subchunk_bytes = chunk_subset_bytes.extract_array_subset(
                &subchunk_subset_overlap
                    .relative_to(&chunk_subset_start)
                    .unwrap(),
                &chunk_subset_shape,
                &self.data_type,
            )?;

            // Decode the subchunk
            let subchunk_decoded = if let Some(subchunk_decoded) =
                subchunks_decoded.lock().unwrap().remove(&subchunk_index)
            {
                subchunk_decoded.into_owned()
            } else {
                subchunk_fill_value()?
            };

            // Update the subchunk
            let subchunk_updated = update_array_bytes(
                subchunk_decoded,
                bytemuck::cast_slice(&self.subchunk_shape),
                &subchunk_subset_overlap
                    .relative_to(subchunk_subset.start())
                    .unwrap(),
                &subchunk_bytes,
                self.data_type.size(),
            )?;
            subchunks_decoded
                .lock()
                .unwrap()
                .insert(subchunk_index, subchunk_updated);

            Ok::<_, CodecError>(())
        })?;
        let subchunks_decoded = Arc::try_unwrap(subchunks_decoded)
            .expect("subchunks_decoded should have one strong reference")
            .into_inner()
            .expect("subchunks_decoded should not be poisoned");

        // Encode the updated subchunks
        #[cfg(not(target_arch = "wasm32"))]
        let iterator = subchunks_decoded.into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let iterator = subchunks_decoded.into_iter();

        let updated_subchunks = iterator
            .map(|(subchunk_index, subchunk_decoded)| {
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
                            options,
                        )?
                        .into_owned();
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
            self.input_output_handle.erase()?;
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
                    encoded_output.extend(subchunk_encoded);
                }
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
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

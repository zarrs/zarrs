use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::lock::Mutex as AsyncMutex;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use zarrs_chunk_grid::{ChunkGrid, ChunkGridTraits};

use super::{
    ShardingCodecOptions, ShardingIndexLocation, nested_local_subchunk_grids, sharding_index_shape,
    subchunk_updates,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::codec::array_to_bytes::sharding::{calculate_chunks_per_shard, subchunk_grid};
use crate::array::{ArrayBytes, ChunkShape, ChunkShapeTraits, CodecChainBound, CowBytes, DataType};
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
        let mut shard_index = self
            .loaded_shard_index(shard_index_guard, options, resources)
            .await?
            .clone();
        let max_data_offset = shard_index
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|entry| **entry != [u64::MAX; 2])
            .try_fold(0, |max, &[offset, size]| {
                offset
                    .checked_add(size)
                    .map(|end| max.max(end))
                    .ok_or_else(super::invalid_shard_index)
            })?;
        let shard_size = if self.input_output_handle.supports_partial_encode()
            && self.index_location == ShardingIndexLocation::End
            && shard_index.iter().any(|&entry| entry != u64::MAX)
        {
            self.input_output_handle.size(options, resources).await?
        } else {
            None
        };
        let write = super::encode_shard_update(
            &mut shard_index,
            max_data_offset,
            shard_size,
            updated_subchunks,
            &self.index_codecs,
            &self.index_shape,
            self.index_location,
            options,
            resources.compute(),
        )?;
        // Reload after an interrupted write, including cancellation of an async write.
        *shard_index_guard = None;
        let result = match write {
            super::ShardWrite::Erase => self.input_output_handle.erase().await,
            super::ShardWrite::Replace(bytes) => {
                self.input_output_handle
                    .encode(bytes.into(), options, resources)
                    .await
            }
            super::ShardWrite::Partial(values) => {
                self.input_output_handle
                    .partial_encode_many(Box::new(values.into_iter()), options, resources)
                    .await
            }
        };
        *shard_index_guard = result.is_ok().then_some(shard_index);
        result
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

        // A shard that cannot be partially encoded (e.g. behind a bytes-to-bytes codec) is replaced.
        //   Its subchunks that are not updated are kept, so they are read with the straddling subchunks.
        let replace = !self.input_output_handle.supports_partial_encode();
        let subchunks_kept: Vec<u64> = if replace {
            let subchunks_updated: HashSet<u64> =
                updates.iter().map(|update| update.subchunk_index).collect();
            (0..shard_index.len() as u64 / 2)
                .filter(|subchunk_index| !subchunks_updated.contains(subchunk_index))
                .collect()
        } else {
            vec![]
        };

        // Get the byte ranges of the subchunks that need to be retrieved
        //   This only includes chunks that are not entirely replaced by the update.
        //   Sorting byte ranges may improves store retrieve efficiency in some cases
        let (subchunks_read, byte_ranges): (Vec<u64>, Vec<ByteRange>) = updates
            .iter()
            .filter(|update| !update.fully_covered)
            .map(|update| update.subchunk_index)
            .chain(subchunks_kept.iter().copied())
            .filter_map(|subchunk_index| {
                let (offset, size) = super::subchunk_offset_size(
                    shard_index,
                    usize::try_from(subchunk_index).unwrap(),
                )?;
                Some((subchunk_index, ByteRange::FromStart(offset, Some(size))))
            })
            .sorted_by_key(|(_, byte_range)| *byte_range)
            .unzip();

        // Read the straddling and kept subchunks
        let mut subchunks_encoded: HashMap<u64, Vec<u8>> = self
            .input_output_handle
            .partial_decode_many(Box::new(byte_ranges.into_iter()), options, resources)
            .await?
            .map(|bytes| {
                std::iter::zip(subchunks_read, bytes.into_iter().map(CowBytes::into_vec)).collect()
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
        let mut updated_subchunks = updates
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

        // Preserve the encoded bytes of subchunks retained by a whole-shard rewrite.
        updated_subchunks.extend(subchunks_kept.into_iter().filter_map(|subchunk_index| {
            let bytes = subchunks_encoded.remove(&subchunk_index)?;
            Some((subchunk_index, Some(CowBytes::from(bytes))))
        }));

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

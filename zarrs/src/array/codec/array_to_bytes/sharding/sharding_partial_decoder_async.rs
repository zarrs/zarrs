use super::sharding_partial_decoder_common::{
    coalesce_chunks, collect_chunk_indices, group_read_concurrent_limit, ready_chunks,
};
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::unravel_index;
use futures::{StreamExt, TryStreamExt};
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridTraits;

use super::{
    ShardingCodecOptions, ShardingIndexLocation, nested_local_subchunk_grids, subchunk_grid,
};
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArraySubsetTraits, ChunkGrid, ChunkShape,
    CodecChainBound, CowBytes, DataType, DataTypeSize, Indexer, ravel_indices,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayToBytesCodecTraits,
    AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits,
    AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits, CodecError, CodecOptions,
    InvalidNumberOfElementsError, decode_into_array_bytes_target,
};
use zarrs_plugin::ExtensionAliasesV3;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Asynchronous partial decoder for the sharding codec.
pub struct AsyncShardingPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: RegularBoundedChunkGrid,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl AsyncShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub async fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: &CodecChainBound,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
        let shard_index = super::decode_shard_index_async_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )
        .await?;

        Ok(Self {
            input_handle,
            subchunk_grid: subchunk_grid(&shard_shape, &subchunk_shape)?,
            subchunk_shape,
            inner_codecs,
            shard_index,
            sharding_options,
        })
    }

    /// Retrieve the byte range of an encoded subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub fn subchunk_byte_range(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, CodecError> {
        super::subchunk_byte_range(
            self.shard_index.as_deref(),
            &self.subchunk_grid,
            chunk_indices,
        )
    }

    /// Retrieve the encoded bytes of a subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub async fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<CowBytes<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
                .await
        } else {
            Ok(None)
        }
    }
}

pub(crate) async fn partial_decode(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    indexer.validate(subchunk_grid.array_shape())?;

    let data_type = inner_codecs.data_type();
    if data_type.is_optional() {
        return Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            super::ShardingCodec::aliases_v3().default_name.to_string(),
        ));
    }

    let Some(subset) = indexer.as_array_subset() else {
        return partial_decode_indexer(
            input_handle,
            subchunk_grid,
            subchunk_shape,
            inner_codecs,
            shard_index,
            indexer,
            options,
        )
        .await;
    };

    match data_type.size() {
        DataTypeSize::Fixed(_data_type_size) => {
            partial_decode_fixed_array_subset(
                input_handle,
                subchunk_grid,
                subchunk_shape,
                inner_codecs,
                shard_index,
                subset,
                options,
            )
            .await
        }
        DataTypeSize::Variable => {
            partial_decode_variable_array_subset(
                input_handle,
                subchunk_grid,
                subchunk_shape,
                inner_codecs,
                shard_index,
                subset,
                options,
            )
            .await
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderSubchunkingTraits for AsyncShardingPartialDecoder {
    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        nested_local_subchunk_grids(
            ChunkGrid::new(self.subchunk_grid.clone()),
            &self.inner_codecs,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
            + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        partial_decode(
            &self.input_handle,
            &self.subchunk_grid,
            &self.subchunk_shape,
            &self.inner_codecs,
            self.shard_index.as_deref(),
            indexer,
            options,
        )
        .await
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }
        if let DataTypeSize::Fixed(_) = self.inner_codecs.data_type().size()
            && let ArrayBytesDecodeIntoTarget::Fixed(output_view) = output_target
        {
            indexer.validate(self.subchunk_grid.array_shape())?;
            if let Some(subset) = indexer.as_array_subset() {
                partial_decode_fixed_array_subset_into(
                    &self.input_handle,
                    &self.subchunk_grid,
                    &self.subchunk_shape,
                    &self.inner_codecs,
                    self.shard_index.as_deref(),
                    subset,
                    options,
                    output_view,
                )
                .await
            } else {
                partial_decode_fixed_indexer_into(
                    &self.input_handle,
                    &self.subchunk_grid,
                    &self.subchunk_shape,
                    &self.inner_codecs,
                    self.shard_index.as_deref(),
                    indexer,
                    options,
                    output_view,
                )
                .await
            }
        } else {
            let decoded_value = self.partial_decode(indexer, options).await?;
            decode_into_array_bytes_target(&decoded_value, output_target)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

async fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
    inner_codecs
        .clone()
        .async_partial_decoder(
            Arc::new(AsyncByteIntervalPartialDecoder::new(
                input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            subchunk_shape,
            options,
        )
        .await
        .map_err(|err| {
            if let CodecError::InvalidByteRangeError(_) = err {
                CodecError::Other(
                    "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                        .to_string(),
                )
            } else {
                err
            }
        })
}

#[expect(clippy::too_many_arguments)]
async fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    output_view: &mut ArrayBytesFixedDisjointView<'_>,
) -> Result<(), CodecError> {
    let fill_value = inner_codecs.fill_value();
    if array_subset.len() != output_view.num_elements() {
        return Err(InvalidNumberOfElementsError::new(
            array_subset.len(),
            output_view.num_elements(),
        )
        .into());
    }
    let Some(shard_index) = shard_index else {
        return output_view
            .fill(fill_value.as_ne_bytes())
            .map_err(CodecError::from);
    };
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        super::num_subchunks(subchunk_grid),
        options,
    )?;
    let array_subset_start = array_subset.start();
    let chunks = subchunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let mut subchunks = Vec::with_capacity(chunks.num_elements_usize());
    for chunk_indices in chunks.indices() {
        let shard_index_idx =
            ravel_indices(&chunk_indices, subchunk_grid.grid_shape()).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset_size = super::subchunk_offset_size(shard_index, shard_index_idx);
        let chunk_subset = subchunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;
        let chunk_relative = chunk_subset_overlap.relative_to(&array_subset_start)?;
        let chunk_output_overlap_subset = chunk_relative.offset(output_view.subset().start())?;
        subchunks.push((
            offset_size,
            chunk_subset,
            chunk_subset_overlap,
            chunk_output_overlap_subset,
        ));
    }

    use futures::{StreamExt, TryStreamExt};
    let decoded_subchunks = futures::stream::iter(subchunks)
        .map(
            |(offset_size, chunk_subset, chunk_subset_overlap, output_subset)| {
                let options = &options;
                async move {
                    if let Some((offset, size)) = offset_size {
                        let inner_partial_decoder = get_subchunk_partial_decoder(
                            input_handle,
                            &chunk_subset.chunk_shape().expect("nonempty subchunk"),
                            inner_codecs,
                            options,
                            offset,
                            size,
                        )
                        .await?;
                        let decoded = inner_partial_decoder
                            .partial_decode(
                                &chunk_subset_overlap
                                    .relative_to(chunk_subset.start())
                                    .unwrap(),
                                options,
                            )
                            .await?
                            .into_fixed()?
                            .into_vec();
                        Ok((Some(decoded), output_subset))
                    } else {
                        Ok::<_, CodecError>((None, output_subset))
                    }
                }
            },
        )
        .buffer_unordered(subchunk_concurrent_limit)
        .try_collect::<Vec<_>>()
        .await?;

    for (decoded, output_subset) in decoded_subchunks {
        // SAFETY: subchunks are disjoint array subsets and each view is dropped before the next.
        let mut subchunk_view = unsafe { output_view.subdivide(output_subset)? };
        if let Some(decoded) = decoded {
            subchunk_view.copy_from_slice(&decoded)?;
        } else {
            subchunk_view.fill(fill_value.as_ne_bytes())?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
async fn partial_decode_fixed_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard = subchunk_grid.grid_shape();
    let shard_chunk_grid = subchunk_grid;

    let chunk_indices_1d = collect_chunk_indices(shard_chunk_grid, array_subset, chunks_per_shard)?;
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;
    let num_groups = coalesced_groups.len();
    let group_read_limit = group_read_concurrent_limit(options, num_groups);
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape)?;
    let array_subset_start = array_subset.start();
    let array_subset_shape = array_subset.shape();

    // Stage 1: Read all groups in parallel.
    let read_group = |(group_idx, group): (
        usize,
        &super::sharding_partial_decoder_common::CoalescedGroup,
    )| {
        let start = group.start;
        let total_len = group.total_len;
        async move {
            Ok::<(usize, CowBytes<'static>), CodecError>((
                group_idx,
                input_handle
                    .partial_decode(ByteRange::FromStart(start, Some(total_len)), options)
                    .await?
                    .ok_or_else(|| {
                        CodecError::Other("Shard does not exist during partial decode.".to_string())
                    })?
                    .into_bytes()
                    .into(),
            ))
        }
    };
    let mut loaded_groups = futures::stream::iter(coalesced_groups.iter().enumerate())
        .map(read_group)
        .buffer_unordered(group_read_limit)
        .try_collect::<Vec<_>>()
        .await?;
    loaded_groups.sort_unstable_by_key(|(group_idx, _)| *group_idx);
    let loaded_groups = loaded_groups
        .into_iter()
        .map(|(_, bytes)| bytes)
        .collect::<Vec<_>>();

    // Stage 2: Decode all chunks in one globally balanced workload.
    let ready_chunks = ready_chunks(&coalesced_groups);
    let (chunk_concurrent_limit, codec_options) = concurrency_chunks_and_codec(
        options.concurrent_target(),
        ready_chunks.len(),
        options,
        &codec_concurrency,
    );
    let decode_chunk = |(group_idx, chunk_idx): (usize, usize)| {
        let group = &coalesced_groups[group_idx];
        let coalesced_bytes = loaded_groups[group_idx].clone();
        let chunk_indices_1d = &chunk_indices_1d;
        let codec_options = &codec_options;
        let shard_chunk_grid = &shard_chunk_grid;
        let array_subset_start = &array_subset_start;
        async move {
            let pos = group.chunks[chunk_idx];
            let idx = chunk_indices_1d[pos];
            let i = usize::try_from(idx).unwrap();
            let offset = shard_index[i * 2];
            let size = shard_index[i * 2 + 1];
            let chunk_indices_nd =
                unravel_index(idx, chunks_per_shard).expect("inbounds chunk index");
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices_nd)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");
            let overlap = array_subset.overlap(&chunk_subset)?;
            let subchunk_shape = chunk_subset.chunk_shape().expect("nonempty subchunk");
            let subchunk_shape = subchunk_shape.as_slice();
            let subchunk_num_elements: u64 = subchunk_shape.iter().map(|d| d.get()).product();
            let overlap_in_chunk = overlap.relative_to(chunk_subset.start())?;
            let start = usize::try_from(offset - group.start).unwrap();
            let end = start + usize::try_from(size).unwrap();
            let decoded = if overlap.num_elements() == subchunk_num_elements {
                inner_codecs
                    .decode(
                        coalesced_bytes.slice(start..end),
                        subchunk_shape,
                        codec_options,
                    )?
                    .into_fixed()?
                    .into_static()
            } else {
                let coalesced_input: Arc<dyn AsyncBytesPartialDecoderTraits> =
                    Arc::new(coalesced_bytes);
                get_subchunk_partial_decoder(
                    &coalesced_input,
                    subchunk_shape,
                    inner_codecs,
                    codec_options,
                    offset - group.start,
                    size,
                )
                .await?
                .partial_decode(&overlap_in_chunk, codec_options)
                .await?
                .into_fixed()?
                .into_static()
            };
            Ok::<_, CodecError>((decoded, overlap.relative_to(array_subset_start)?))
        }
    };
    let decoded_chunks = futures::stream::iter(ready_chunks)
        .map(decode_chunk)
        .buffer_unordered(chunk_concurrent_limit)
        .try_collect::<Vec<_>>()
        .await?;

    let shard_size = array_subset.num_elements_usize() * data_type_size;
    let mut shard = vec![0; shard_size];
    let shard_slice = unsafe_cell_slice::UnsafeCellSlice::new(shard.as_mut_slice());
    for pos in fill_indices {
        let chunk_indices_nd =
            unravel_index(chunk_indices_1d[pos], chunks_per_shard).expect("inbounds chunk index");
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices_nd)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let overlap = array_subset
            .overlap(&chunk_subset)?
            .relative_to(&array_subset_start)?;
        let mut output_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                shard_slice,
                data_type_size,
                &array_subset_shape,
                overlap,
            )?
        };
        output_view
            .fill(fill_value.as_ne_bytes())
            .map_err(CodecError::from)?;
    }
    for (decoded, overlap) in decoded_chunks {
        let mut output_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                shard_slice,
                data_type_size,
                &array_subset_shape,
                overlap,
            )?
        };
        output_view
            .copy_from_slice(&decoded)
            .expect("chunk subset bytes are the correct length");
    }
    Ok(ArrayBytes::from(shard))
}

async fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard = subchunk_grid.grid_shape();
    let shard_chunk_grid = subchunk_grid;
    let chunk_indices_1d = collect_chunk_indices(shard_chunk_grid, array_subset, chunks_per_shard)?;
    let num_chunks = chunk_indices_1d.len();
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;
    let num_groups = coalesced_groups.len();
    let group_read_limit = group_read_concurrent_limit(options, num_groups);
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape)?;
    let array_subset_start = array_subset.start();

    // Stage 1: Read all groups in parallel.
    let read_group = |(group_idx, group): (
        usize,
        &super::sharding_partial_decoder_common::CoalescedGroup,
    )| {
        let start = group.start;
        let total_len = group.total_len;
        async move {
            Ok::<(usize, CowBytes<'static>), CodecError>((
                group_idx,
                input_handle
                    .partial_decode(ByteRange::FromStart(start, Some(total_len)), options)
                    .await?
                    .ok_or_else(|| {
                        CodecError::Other("Shard does not exist during partial decode.".to_string())
                    })?
                    .into_bytes()
                    .into(),
            ))
        }
    };
    let mut loaded_groups = futures::stream::iter(coalesced_groups.iter().enumerate())
        .map(read_group)
        .buffer_unordered(group_read_limit)
        .try_collect::<Vec<_>>()
        .await?;
    loaded_groups.sort_unstable_by_key(|(group_idx, _)| *group_idx);
    let loaded_groups = loaded_groups
        .into_iter()
        .map(|(_, bytes)| bytes)
        .collect::<Vec<_>>();

    // Stage 2: Decode all chunks in one globally balanced workload.
    let ready_chunks = ready_chunks(&coalesced_groups);
    let (chunk_concurrent_limit, codec_options) = concurrency_chunks_and_codec(
        options.concurrent_target(),
        ready_chunks.len(),
        options,
        &codec_concurrency,
    );
    let decode_chunk = |(group_idx, chunk_idx): (usize, usize)| {
        let group = &coalesced_groups[group_idx];
        let coalesced_bytes = loaded_groups[group_idx].clone();
        let chunk_indices_1d = &chunk_indices_1d;
        let codec_options = &codec_options;
        let shard_chunk_grid = &shard_chunk_grid;
        let array_subset_start = &array_subset_start;
        async move {
            let pos = group.chunks[chunk_idx];
            let idx = chunk_indices_1d[pos];
            let i = usize::try_from(idx).unwrap();
            let offset = shard_index[i * 2];
            let size = shard_index[i * 2 + 1];
            let chunk_indices_nd =
                unravel_index(idx, chunks_per_shard).expect("inbounds chunk index");
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices_nd)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");
            let overlap = array_subset.overlap(&chunk_subset)?;
            let subchunk_shape = chunk_subset.chunk_shape().expect("nonempty subchunk");
            let subchunk_shape = subchunk_shape.as_slice();
            let subchunk_num_elements: u64 = subchunk_shape.iter().map(|d| d.get()).product();
            let overlap_in_chunk = overlap.relative_to(chunk_subset.start())?;
            let start = usize::try_from(offset - group.start).unwrap();
            let end = start + usize::try_from(size).unwrap();
            let decoded = if overlap.num_elements() == subchunk_num_elements {
                inner_codecs
                    .decode(
                        coalesced_bytes.slice(start..end),
                        subchunk_shape,
                        codec_options,
                    )?
                    .into_owned()
                    .into_variable()?
            } else {
                let coalesced_input: Arc<dyn AsyncBytesPartialDecoderTraits> =
                    Arc::new(coalesced_bytes);
                get_subchunk_partial_decoder(
                    &coalesced_input,
                    subchunk_shape,
                    inner_codecs,
                    codec_options,
                    offset - group.start,
                    size,
                )
                .await?
                .partial_decode(&overlap_in_chunk, codec_options)
                .await?
                .into_owned()
                .into_variable()?
            };
            Ok::<_, CodecError>((pos, decoded, overlap.relative_to(array_subset_start)?))
        }
    };
    let decoded_chunks = futures::stream::iter(ready_chunks)
        .map(decode_chunk)
        .buffer_unordered(chunk_concurrent_limit)
        .try_collect::<Vec<_>>()
        .await?;

    let mut results = vec![None; num_chunks];
    for pos in fill_indices {
        let chunk_indices_nd =
            unravel_index(chunk_indices_1d[pos], chunks_per_shard).expect("inbounds chunk index");
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices_nd)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let overlap = array_subset
            .overlap(&chunk_subset)?
            .relative_to(&array_subset_start)?;
        let decoded = ArrayBytes::new_fill_value(data_type, overlap.num_elements(), fill_value)?
            .into_variable()?;
        results[pos] = Some((decoded, overlap));
    }
    for (pos, decoded, overlap) in decoded_chunks {
        results[pos] = Some((decoded, overlap));
    }
    let chunk_bytes_and_subsets = results
        .into_iter()
        .map(|result| result.expect("all chunks decoded"))
        .collect();
    Ok(ArrayBytes::Variable(merge_chunks_vlen(
        chunk_bytes_and_subsets,
        &array_subset.shape(),
    )))
}

async fn partial_decode_indexer(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let groups = super::group_indices_by_subchunk(subchunk_grid, subchunk_shape, indexer);
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        groups.len(),
        options,
    )?;
    use futures::{StreamExt, TryStreamExt};

    let options = &options;
    let decoded = futures::stream::iter(groups)
        .map(|(subchunk_index, group)| async move {
            let shard_index_idx = usize::try_from(subchunk_index).unwrap();
            let bytes = if let Some((offset, size)) =
                super::subchunk_offset_size(shard_index, shard_index_idx)
            {
                let decoder = get_subchunk_partial_decoder(
                    input_handle,
                    &group.subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
                .await?;
                decoder.partial_decode(&group, options).await?.into_owned()
            } else {
                ArrayBytes::new_fill_value(data_type, group.len(), fill_value)?
            };
            Ok::<_, CodecError>((group.positions, bytes))
        })
        .buffer_unordered(subchunk_concurrent_limit)
        .try_collect::<Vec<_>>()
        .await?;
    super::merge_indexer_subchunks(decoded, usize::try_from(indexer.len()).unwrap(), data_type)
}

/// Decode the elements of `indexer` into `output_view` by scattering each subchunk group directly.
///
/// The `indexer` must be validated against the shard shape.
#[expect(clippy::too_many_arguments)]
async fn partial_decode_fixed_indexer_into<'a>(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
    output_view: &'a mut ArrayBytesFixedDisjointView<'a>,
) -> Result<(), CodecError> {
    use futures::{StreamExt, TryStreamExt};

    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = shard_index else {
        return output_view
            .fill(fill_value.as_ne_bytes())
            .map_err(CodecError::from);
    };
    let groups = super::group_indices_by_subchunk(subchunk_grid, subchunk_shape, indexer);
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        groups.len(),
        options,
    )?;
    let options = &options;
    let subchunk_decoder = async |subchunk_index: u64, subchunk_shape: &[NonZeroU64]| {
        let shard_index_idx = usize::try_from(subchunk_index).unwrap();
        if let Some((offset, size)) = super::subchunk_offset_size(shard_index, shard_index_idx) {
            get_subchunk_partial_decoder(
                input_handle,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            )
            .await
            .map(Some)
        } else {
            Ok(None)
        }
    };

    if groups.len() == 1 {
        // Positions within a group are ascending and cover every element, so decode in place
        let (subchunk_index, group) = groups.into_iter().next().expect("one group");
        return if let Some(decoder) =
            subchunk_decoder(subchunk_index, &group.subchunk_shape).await?
        {
            decoder
                .partial_decode_into(
                    &group,
                    ArrayBytesDecodeIntoTarget::Fixed(output_view),
                    options,
                )
                .await
        } else {
            output_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from)
        };
    }

    futures::stream::iter(groups)
        .map(|(subchunk_index, group)| async move {
            let bytes = if let Some(decoder) =
                subchunk_decoder(subchunk_index, &group.subchunk_shape).await?
            {
                Some(
                    decoder
                        .partial_decode(&group, options)
                        .await?
                        .into_owned()
                        .into_fixed()?,
                )
            } else {
                None
            };
            Ok::<_, CodecError>((group.positions, bytes))
        })
        .buffer_unordered(subchunk_concurrent_limit)
        .try_for_each(|(positions, bytes)| {
            let result = if let Some(bytes) = bytes {
                output_view.copy_elements_from_slice(&positions, &bytes)
            } else {
                output_view.fill_elements(&positions, fill_value.as_ne_bytes())
            };
            futures::future::ready(result)
        })
        .await
}

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_data_type::FillValue;

use super::sharding_partial_decoder_common::{
    coalesce_chunks, collect_chunk_indices, group_read_concurrent_limit, ready_chunks,
};
use super::{ShardingCodecOptions, ShardingIndexLocation, calculate_chunks_per_shard};
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::CodecChain;
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesOffsets, ArrayBytesRaw, ArrayIndices,
    ArraySubsetTraits, ChunkShape, ChunkShapeTraits, DataType, DataTypeSize,
    IncompatibleDimensionalityError, Indexer, IndexerError, ravel_indices, unravel_index,
};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
    AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits, CodecError, CodecOptions,
};
use zarrs_plugin::ExtensionAliasesV3;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Asynchronous partial decoder for the sharding codec.
pub(crate) struct AsyncShardingPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: DataType,
    fill_value: FillValue,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl AsyncShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub(crate) async fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        data_type: DataType,
        fill_value: FillValue,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<AsyncShardingPartialDecoder, CodecError> {
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
            data_type,
            fill_value,
            shard_shape,
            subchunk_shape,
            inner_codecs,
            shard_index,
            sharding_options,
        })
    }

    /// Retrieve the byte range of an encoded subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub(crate) fn subchunk_byte_range(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, CodecError> {
        super::subchunk_byte_range(
            self.shard_index.as_deref(),
            &self.shard_shape,
            &self.subchunk_shape,
            chunk_indices,
        )
    }

    /// Retrieve the encoded bytes of a subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub(crate) async fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
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

#[expect(clippy::too_many_arguments)]
pub(crate) async fn partial_decode(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    if indexer.dimensionality() != shard_shape.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            indexer.dimensionality(),
            shard_shape.len(),
        )
        .into());
    }

    if data_type.is_optional() {
        return Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            super::ShardingCodec::aliases_v3().default_name.to_string(),
        ));
    }

    match data_type.size() {
        DataTypeSize::Fixed(_data_type_size) => {
            if let Some(subset) = indexer.as_array_subset() {
                partial_decode_fixed_array_subset(
                    input_handle,
                    data_type,
                    fill_value,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    subset,
                    options,
                )
                .await
            } else {
                partial_decode_fixed_indexer(
                    input_handle,
                    data_type,
                    fill_value,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    indexer,
                    options,
                )
                .await
            }
        }
        DataTypeSize::Variable => {
            if let Some(subset) = indexer.as_array_subset() {
                partial_decode_variable_array_subset(
                    input_handle,
                    data_type,
                    fill_value,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    subset,
                    options,
                )
                .await
            } else {
                partial_decode_variable_indexer(
                    input_handle,
                    data_type,
                    fill_value,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    indexer,
                    options,
                )
                .await
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
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
            &self.data_type,
            &self.fill_value,
            &self.shard_shape,
            &self.subchunk_shape,
            &self.inner_codecs,
            self.shard_index.as_deref(),
            indexer,
            options,
        )
        .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[expect(clippy::too_many_arguments)]
async fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
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
            data_type,
            fill_value,
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

#[allow(clippy::too_many_lines)]
#[expect(clippy::too_many_arguments)]
async fn partial_decode_fixed_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let subchunk_shape_u64: &[u64] = bytemuck::must_cast_slice(subchunk_shape);
    let subchunk_num_elements: u64 = subchunk_shape_u64.iter().product();
    let chunk_indices_1d =
        collect_chunk_indices(&shard_chunk_grid, array_subset, &chunks_per_shard)?;
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;
    let num_groups = coalesced_groups.len();
    let group_read_limit = group_read_concurrent_limit(options, num_groups);
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape, data_type)?;
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
            Ok::<_, CodecError>((
                group_idx,
                Arc::new(
                    input_handle
                        .partial_decode(ByteRange::FromStart(start, Some(total_len)), options)
                        .await?
                        .ok_or_else(|| {
                            CodecError::Other(
                                "Shard does not exist during partial decode.".to_string(),
                            )
                        })?
                        .into_owned(),
                ),
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
        let coalesced_bytes = Arc::clone(&loaded_groups[group_idx]);
        let chunk_indices_1d = &chunk_indices_1d;
        let chunks_per_shard = &chunks_per_shard;
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
            let overlap_in_chunk = overlap.relative_to(chunk_subset.start())?;
            let start = usize::try_from(offset - group.start).unwrap();
            let end = start + usize::try_from(size).unwrap();
            let decoded = if overlap.num_elements() == subchunk_num_elements {
                inner_codecs
                    .decode(
                        Cow::Borrowed(&coalesced_bytes[start..end]),
                        subchunk_shape,
                        data_type,
                        fill_value,
                        codec_options,
                    )?
                    .into_fixed()?
                    .into_owned()
            } else {
                let coalesced_input: Arc<dyn AsyncBytesPartialDecoderTraits> = coalesced_bytes;
                get_subchunk_partial_decoder(
                    &coalesced_input,
                    data_type,
                    fill_value,
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
                .into_owned()
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
            unravel_index(chunk_indices_1d[pos], &chunks_per_shard).expect("inbounds chunk index");
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

#[expect(clippy::too_many_arguments)]
async fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .expect("matching dimensionality");
    let subchunk_shape_u64: &[u64] = bytemuck::must_cast_slice(subchunk_shape);
    let subchunk_num_elements: u64 = subchunk_shape_u64.iter().product();
    let chunk_indices_1d =
        collect_chunk_indices(&shard_chunk_grid, array_subset, &chunks_per_shard)?;
    let num_chunks = chunk_indices_1d.len();
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;
    let num_groups = coalesced_groups.len();
    let group_read_limit = group_read_concurrent_limit(options, num_groups);
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape, data_type)?;
    let array_subset_start = array_subset.start();

    // Stage 1: Read all groups in parallel.
    let read_group = |(group_idx, group): (
        usize,
        &super::sharding_partial_decoder_common::CoalescedGroup,
    )| {
        let start = group.start;
        let total_len = group.total_len;
        async move {
            Ok::<_, CodecError>((
                group_idx,
                Arc::new(
                    input_handle
                        .partial_decode(ByteRange::FromStart(start, Some(total_len)), options)
                        .await?
                        .ok_or_else(|| {
                            CodecError::Other(
                                "Shard does not exist during partial decode.".to_string(),
                            )
                        })?
                        .into_owned(),
                ),
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
        let coalesced_bytes = Arc::clone(&loaded_groups[group_idx]);
        let chunk_indices_1d = &chunk_indices_1d;
        let chunks_per_shard = &chunks_per_shard;
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
            let overlap_in_chunk = overlap.relative_to(chunk_subset.start())?;
            let start = usize::try_from(offset - group.start).unwrap();
            let end = start + usize::try_from(size).unwrap();
            let decoded = if overlap.num_elements() == subchunk_num_elements {
                inner_codecs
                    .decode(
                        Cow::Borrowed(&coalesced_bytes[start..end]),
                        subchunk_shape,
                        data_type,
                        fill_value,
                        codec_options,
                    )?
                    .into_owned()
                    .into_variable()?
            } else {
                let coalesced_input: Arc<dyn AsyncBytesPartialDecoderTraits> = coalesced_bytes;
                get_subchunk_partial_decoder(
                    &coalesced_input,
                    data_type,
                    fill_value,
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
            unravel_index(chunk_indices_1d[pos], &chunks_per_shard).expect("inbounds chunk index");
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

#[expect(clippy::too_many_arguments)]
async fn partial_decode_fixed_indexer(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    // let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
    //     &inner_codecs,
    //     &chunk_representation,
    //     &chunks_per_shard,
    //     options,
    // )?;
    let options = &options;

    let output_len = usize::try_from(indexer.len() * data_type_size as u64).unwrap();
    let mut output: Vec<u8> = Vec::with_capacity(output_len);

    #[cfg(not(target_arch = "wasm32"))]
    let subchunk_partial_decoders = moka::future::Cache::new(chunks_per_shard.iter().product());
    #[cfg(target_arch = "wasm32")]
    let subchunk_partial_decoders = quick_cache::sync::Cache::new(
        usize::try_from(chunks_per_shard.iter().product::<u64>()).unwrap(),
    );

    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != shard_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indices.len(),
                shard_shape.len(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i / cs)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard)
            .ok_or_else(|| IndexerError::new_oob(chunk_index, chunks_per_shard.clone()))?;

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = subchunk_partial_decoders
            .entry(chunk_index_1d)
            .or_try_insert_with(get_subchunk_partial_decoder(
                input_handle,
                data_type,
                fill_value,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            ))
            .await
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = subchunk_partial_decoders
            .get_or_insert_async(&chunk_index_1d, async {
                get_subchunk_partial_decoder(
                    input_handle,
                    data_type,
                    fill_value,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
                .await
            })
            .await?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let element_bytes = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)
            .await?
            .into_fixed()
            .expect("fixed data");
        output.extend_from_slice(&element_bytes);
    }

    debug_assert_eq!(output.len(), output_len);

    Ok(output.into())
}

#[expect(clippy::too_many_arguments)]
async fn partial_decode_variable_indexer(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    // let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
    //     &inner_codecs,
    //     &chunk_representation,
    //     &chunks_per_shard,
    //     options,
    // )?;
    let options = &options;

    let offsets_len = usize::try_from(indexer.len() + 1).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(offsets_len);
    offsets.push(0);

    #[cfg(not(target_arch = "wasm32"))]
    let subchunk_partial_decoders = moka::future::Cache::new(chunks_per_shard.iter().product());
    #[cfg(target_arch = "wasm32")]
    let subchunk_partial_decoders = quick_cache::sync::Cache::new(
        usize::try_from(chunks_per_shard.iter().product::<u64>()).unwrap(),
    );

    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != shard_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indices.len(),
                shard_shape.len(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i / cs)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard)
            .ok_or_else(|| IndexerError::new_oob(chunk_index, chunks_per_shard.clone()))?;

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = subchunk_partial_decoders
            .entry(chunk_index_1d)
            .or_try_insert_with(get_subchunk_partial_decoder(
                input_handle,
                data_type,
                fill_value,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            ))
            .await
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = subchunk_partial_decoders
            .get_or_insert_async(&chunk_index_1d, async {
                get_subchunk_partial_decoder(
                    input_handle,
                    data_type,
                    fill_value,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
                .await
            })
            .await?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let (element_bytes, element_offsets) = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)
            .await?
            .into_variable()?
            .into_parts();
        debug_assert_eq!(element_offsets.len(), 2);
        bytes.extend_from_slice(&element_bytes);
        offsets.push(bytes.len());
    }

    Ok(ArrayBytes::new_vlen(
        bytes,
        ArrayBytesOffsets::new(offsets)?,
    )?)
}

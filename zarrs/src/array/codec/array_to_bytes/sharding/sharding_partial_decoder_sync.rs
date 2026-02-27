use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_data_type::FillValue;

use super::{ShardingIndexLocation, calculate_chunks_per_shard};
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::CodecChain;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesOffsets, ArrayBytesRaw, ArrayIndices,
    ArraySubset, ArraySubsetTraits, ChunkShape, ChunkShapeTraits, DataType, DataTypeSize,
    IncompatibleDimensionalityError, Indexer, IndexerError, ravel_indices,
};
use zarrs_codec::{
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, ByteIntervalPartialDecoder,
    BytesPartialDecoderTraits, CodecError, CodecOptions,
};
use zarrs_plugin::ExtensionAliasesV3;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Partial decoder for the sharding codec.
pub(crate) struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    data_type: DataType,
    fill_value: FillValue,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
}

impl ShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        data_type: DataType,
        fill_value: FillValue,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let shard_index = super::decode_shard_index_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )?;

        Ok(Self {
            input_handle,
            data_type,
            fill_value,
            shard_shape,
            subchunk_shape,
            inner_codecs,
            shard_index,
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
    pub(crate) fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
        } else {
            Ok(None)
        }
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn partial_decode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
            }
        }
    }
}

impl ArrayPartialDecoderTraits for ShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
            + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    fn partial_decode(
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
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[expect(clippy::too_many_arguments)]
fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
    inner_codecs
        .clone()
        .partial_decoder(
            Arc::new(ByteIntervalPartialDecoder::new(
                input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            subchunk_shape,
            data_type,
            fill_value,
            options,
        )
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

/// Metadata for one inner chunk, collected before parallel I/O.
struct ChunkInfo {
    /// The overlap with the requested array subset, relative to the output array subset origin.
    chunk_subset_overlap_in_output: ArraySubset,
    /// The overlap with the requested array subset, relative to the inner chunk origin.
    chunk_subset_overlap_in_chunk: ArraySubset,
    /// Byte offset of the encoded inner chunk in the shard. `u64::MAX` = fill value.
    offset: u64,
    /// Byte length of the encoded inner chunk.
    size: u64,
}

/// A set of byte-adjacent inner chunks that can be read in a single I/O call.
struct CoalescedGroup {
    /// Byte offset of the first byte in the shard for this group.
    start: u64,
    /// Total byte length of the coalesced read.
    total_len: u64,
    /// Indices into the `chunk_infos` slice, in ascending byte-offset order.
    chunks: Vec<usize>,
}

/// Sort inner chunks by byte offset and merge exactly-adjacent ranges.
///
/// Returns coalesced I/O groups and a separate list of fill-value chunk indices.
fn coalesce_chunks(chunk_infos: &[ChunkInfo]) -> (Vec<CoalescedGroup>, Vec<usize>) {
    let mut fill_indices: Vec<usize> = Vec::new();
    let mut io_indices: Vec<usize> = chunk_infos
        .iter()
        .enumerate()
        .filter_map(|(i, info)| {
            if info.offset == u64::MAX {
                fill_indices.push(i);
                None
            } else {
                Some(i)
            }
        })
        .collect();

    io_indices.sort_by_key(|&i| chunk_infos[i].offset);

    let mut groups: Vec<CoalescedGroup> = Vec::new();
    for idx in io_indices {
        let info = &chunk_infos[idx];
        if let Some(last) = groups.last_mut()
            && last.start + last.total_len == info.offset
        {
            last.total_len += info.size;
            last.chunks.push(idx);
        } else {
            groups.push(CoalescedGroup {
                start: info.offset,
                total_len: info.size,
                chunks: vec![idx],
            });
        }
    }

    (groups, fill_indices)
}

#[expect(clippy::too_many_arguments)]
#[expect(clippy::too_many_lines)]
fn partial_decode_fixed_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        data_type,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;

    let array_subset_size = array_subset.num_elements_usize() * data_type_size;
    let mut out_array_subset = vec![0u8; array_subset_size];
    let out_array_subset_slice = UnsafeCellSlice::new(out_array_subset.as_mut_slice());

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let array_subset_start = array_subset.start();
    let array_subset_shape = array_subset.shape();
    let subchunk_shape_u64: &[u64] = bytemuck::must_cast_slice(subchunk_shape);

    // Phase 1: Collect chunk metadata serially from the shard index.
    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    let mut chunk_infos: Vec<ChunkInfo> = Vec::with_capacity(chunks.num_elements_usize());
    for chunk_indices in chunks.indices() {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;

        chunk_infos.push(ChunkInfo {
            chunk_subset_overlap_in_output: chunk_subset_overlap
                .relative_to(&array_subset_start)
                .unwrap(),
            chunk_subset_overlap_in_chunk: chunk_subset_overlap
                .relative_to(chunk_subset.start())
                .unwrap(),
            offset,
            size,
        });
    }

    // Phase 2: Sort by byte offset and merge adjacent ranges into coalesced groups.
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_infos);

    // Phase 3: Process fill chunks and I/O groups all in parallel.
    //   - Fill chunks: replicate the fill element bytes into each disjoint output region.
    //   - I/O groups: one coalesced read per group, then inner chunks decoded in a nested
    //     parallel loop.
    //
    // The outer iterator covers fill_indices.len() + coalesced_groups.len() items:
    //   k < fill_indices.len()  → fill chunk at fill_indices[k]
    //   k >= fill_indices.len() → I/O group at coalesced_groups[k - fill_indices.len()]
    let fill_element_bytes = fill_value.as_ne_bytes();
    let num_fill = fill_indices.len();
    let num_groups = coalesced_groups.len();
    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        (0..num_fill + num_groups),
        try_for_each,
        |k: usize| -> Result<(), CodecError> {
            if k < num_fill {
                // Fill-value chunk: replicate fill bytes into the output region.
                let ci = fill_indices[k];
                let info = &chunk_infos[ci];
                let decoded = fill_element_bytes
                    .repeat(info.chunk_subset_overlap_in_output.num_elements_usize());
                let mut output_view = unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        out_array_subset_slice,
                        data_type_size,
                        &array_subset_shape,
                        info.chunk_subset_overlap_in_output.clone(),
                    )?
                };
                output_view
                    .copy_from_slice(&decoded)
                    .map_err(CodecError::from)
            } else {
                // Coalesced I/O group: one read, then decode inner chunks in parallel.
                let group = &coalesced_groups[k - num_fill];
                let coalesced_bytes: Arc<Vec<u8>> = Arc::new(
                    input_handle
                        .partial_decode(
                            ByteRange::FromStart(group.start, Some(group.total_len)),
                            &options,
                        )?
                        .ok_or_else(|| {
                            CodecError::Other(
                                "Shard does not exist during partial decode.".to_string(),
                            )
                        })?
                        .into_owned(),
                );

                let num_chunks_in_group = group.chunks.len();
                crate::iter_concurrent_limit!(
                    subchunk_concurrent_limit,
                    (0..num_chunks_in_group),
                    try_for_each,
                    |j: usize| -> Result<(), CodecError> {
                        let ci = group.chunks[j];
                        let info = &chunk_infos[ci];
                        let start = usize::try_from(info.offset - group.start).unwrap();
                        let end = start + usize::try_from(info.size).unwrap();
                        let decoded =
                            if info.chunk_subset_overlap_in_chunk.shape() == subchunk_shape_u64 {
                                // The overlap is the full inner chunk: decode directly.
                                inner_codecs
                                    .decode(
                                        Cow::Borrowed(&coalesced_bytes[start..end]),
                                        subchunk_shape,
                                        data_type,
                                        fill_value,
                                        &options,
                                    )?
                                    .into_fixed()?
                            } else {
                                let coalesced_bytes: Arc<dyn BytesPartialDecoderTraits> =
                                    coalesced_bytes.clone();
                                get_subchunk_partial_decoder(
                                    &coalesced_bytes,
                                    data_type,
                                    fill_value,
                                    subchunk_shape,
                                    inner_codecs,
                                    &options,
                                    info.offset - group.start,
                                    info.size,
                                )?
                                .partial_decode(&info.chunk_subset_overlap_in_chunk, &options)?
                                .into_owned()
                                .into_fixed()?
                            };
                        let mut output_view = unsafe {
                            // SAFETY: chunks represent disjoint array subsets
                            ArrayBytesFixedDisjointView::new(
                                out_array_subset_slice,
                                data_type_size,
                                &array_subset_shape,
                                info.chunk_subset_overlap_in_output.clone(),
                            )?
                        };
                        output_view
                            .copy_from_slice(&decoded)
                            .map_err(CodecError::from)
                    }
                )
            }
        }
    )?;

    Ok(ArrayBytes::from(out_array_subset))
}

#[expect(clippy::too_many_arguments)]
#[expect(clippy::too_many_lines)]
fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = &shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        data_type,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;
    let options = &options;

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .expect("matching dimensionality");

    let array_subset_start = array_subset.start();
    let subchunk_shape_u64: &[u64] = bytemuck::must_cast_slice(subchunk_shape);

    // Phase 1: Collect chunk metadata serially.
    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    let num_chunks = chunks.num_elements_usize();
    let mut chunk_infos: Vec<ChunkInfo> = Vec::with_capacity(num_chunks);
    for chunk_indices in chunks.indices() {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;

        chunk_infos.push(ChunkInfo {
            chunk_subset_overlap_in_output: chunk_subset_overlap
                .relative_to(&array_subset_start)
                .unwrap(),
            chunk_subset_overlap_in_chunk: chunk_subset_overlap
                .relative_to(chunk_subset.start())
                .unwrap(),
            offset,
            size,
        });
    }

    // Phase 2: Sort and coalesce.
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_infos);

    // Phase 3: Decode each group in parallel; write results into a pre-allocated vec
    // indexed by original chunk order (required for merge_chunks_vlen ordering).
    let mut results: Vec<Option<ArrayBytes<'static>>> = (0..num_chunks).map(|_| None).collect();
    let results_slice = UnsafeCellSlice::new(results.as_mut_slice());

    let num_fill = fill_indices.len();
    let num_groups = coalesced_groups.len();
    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        (0..num_fill + num_groups),
        try_for_each,
        |k: usize| -> Result<(), CodecError> {
            if k < num_fill {
                let ci = fill_indices[k];
                let info = &chunk_infos[ci];
                let decoded = ArrayBytes::new_fill_value(
                    data_type,
                    info.chunk_subset_overlap_in_output.num_elements(),
                    fill_value,
                )?
                .into_variable()?;
                // SAFETY: each ci is unique across all groups
                unsafe {
                    *results_slice.index_mut(ci) = Some(ArrayBytes::Variable(decoded));
                }
                Ok(())
            } else {
                let group = &coalesced_groups[k - num_fill];
                let coalesced_bytes: Arc<Vec<u8>> = Arc::new(
                    input_handle
                        .partial_decode(
                            ByteRange::FromStart(group.start, Some(group.total_len)),
                            options,
                        )?
                        .ok_or_else(|| {
                            CodecError::Other(
                                "Shard does not exist during partial decode.".to_string(),
                            )
                        })?
                        .into_owned(),
                );

                let num_chunks_in_group = group.chunks.len();
                crate::iter_concurrent_limit!(
                    subchunk_concurrent_limit,
                    (0..num_chunks_in_group),
                    try_for_each,
                    |j: usize| -> Result<(), CodecError> {
                        let ci = group.chunks[j];
                        let info = &chunk_infos[ci];
                        let start = usize::try_from(info.offset - group.start).unwrap();
                        let end = start + usize::try_from(info.size).unwrap();
                        let decoded =
                            if info.chunk_subset_overlap_in_chunk.shape() == subchunk_shape_u64 {
                                // The overlap is the full inner chunk: decode directly.
                                inner_codecs
                                    .decode(
                                        Cow::Borrowed(&coalesced_bytes[start..end]),
                                        subchunk_shape,
                                        data_type,
                                        fill_value,
                                        options,
                                    )?
                                    .into_owned()
                                    .into_variable()?
                            } else {
                                let coalesced_bytes: Arc<dyn BytesPartialDecoderTraits> =
                                    coalesced_bytes.clone();
                                get_subchunk_partial_decoder(
                                    &coalesced_bytes,
                                    data_type,
                                    fill_value,
                                    subchunk_shape,
                                    inner_codecs,
                                    options,
                                    info.offset - group.start,
                                    info.size,
                                )?
                                .partial_decode(&info.chunk_subset_overlap_in_chunk, options)?
                                .into_owned()
                                .into_variable()?
                            };
                        // SAFETY: each ci is unique across all groups
                        unsafe {
                            *results_slice.index_mut(ci) = Some(ArrayBytes::Variable(decoded));
                        }
                        Ok(())
                    }
                )
            }
        }
    )?;

    let chunk_bytes_and_subsets = results
        .into_iter()
        .zip(
            chunk_infos
                .iter()
                .map(|i| i.chunk_subset_overlap_in_output.clone()),
        )
        .map(|(bytes, subset)| {
            bytes
                .expect("all chunks decoded")
                .into_variable()
                .map(|v| (v, subset))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, &array_subset.shape());
    Ok(ArrayBytes::Variable(out_array_subset))
}

#[expect(clippy::too_many_arguments)]
fn partial_decode_fixed_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
    let Some(shard_index) = &shard_index else {
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
    let subchunk_partial_decoders = moka::sync::Cache::new(chunks_per_shard.iter().product());
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
            .or_try_insert_with(|| {
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
            })
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder =
            subchunk_partial_decoders.get_or_insert_with(&chunk_index_1d, || {
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
            })?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let element_bytes = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)?
            .into_fixed()
            .expect("fixed data");
        output.extend_from_slice(&element_bytes);
    }

    debug_assert_eq!(output.len(), output_len);

    Ok(output.into())
}

#[expect(clippy::too_many_arguments)]
fn partial_decode_variable_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = &shard_index else {
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
    let subchunk_partial_decoders = moka::sync::Cache::new(chunks_per_shard.iter().product());
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
            .or_try_insert_with(|| {
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
            })
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder =
            subchunk_partial_decoders.get_or_insert_with(&chunk_index_1d, || {
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
            })?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let (element_bytes, element_offsets) = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)?
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

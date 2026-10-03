#![allow(clippy::similar_names)]
use super::sharding_partial_decoder_common::{
    coalesce_chunks, collect_chunk_indices, group_read_concurrent_limit, ready_chunks,
};
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::unravel_index;

use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::{ArraySubset, ChunkGridTraits};

use super::{
    ShardingCodecOptions, ShardingIndexLocation, nested_local_subchunk_grids, subchunk_grid,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArraySubsetTraits, ChunkGrid, ChunkShape,
    CodecChainBound, CowBytes, DataType, DataTypeSize, Indexer,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderSubchunkingTraits,
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, ByteIntervalPartialDecoder,
    BytesPartialDecoderTraits, CodecError, CodecOptions, InvalidNumberOfElementsError,
    decode_into_array_bytes_target,
};
use zarrs_plugin::ExtensionAliasesV3;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Partial decoder for the sharding codec.
pub struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: RegularBoundedChunkGrid,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl ShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: &CodecChainBound,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
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
    pub fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<CowBytes<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
        } else {
            Ok(None)
        }
    }
}

pub(crate) fn partial_decode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    indexer.validate(subchunk_grid.array_shape())?;

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
        );
    };

    match data_type.size() {
        DataTypeSize::Fixed(data_type_size) => {
            let array_shape = subset.shape();
            let array_subset_size = subset.num_elements_usize() * data_type_size;
            let mut out_array_subset = vec![0; array_subset_size];
            let out_array_subset_slice = UnsafeCellSlice::new(out_array_subset.as_mut_slice());
            let mut output_view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    out_array_subset_slice,
                    data_type_size,
                    &array_shape,
                    ArraySubset::new_with_shape(array_shape.to_vec()),
                )?
            };
            partial_decode_fixed_array_subset_into(
                input_handle,
                subchunk_grid,
                subchunk_shape,
                inner_codecs,
                shard_index,
                subset,
                options,
                &mut output_view,
            )?;
            Ok(ArrayBytes::from(out_array_subset))
        }
        DataTypeSize::Variable => partial_decode_variable_array_subset(
            input_handle,
            subchunk_grid,
            subchunk_shape,
            inner_codecs,
            shard_index,
            subset,
            options,
        ),
    }
}

impl ArrayPartialDecoderSubchunkingTraits for ShardingPartialDecoder {
    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        nested_local_subchunk_grids(
            ChunkGrid::new(self.subchunk_grid.clone()),
            &self.inner_codecs,
        )
    }
}

impl ArrayPartialDecoderTraits for ShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
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
            &self.subchunk_grid,
            &self.subchunk_shape,
            &self.inner_codecs,
            self.shard_index.as_deref(),
            indexer,
            options,
        )
    }

    fn partial_decode_into(
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
        if let DataTypeSize::Fixed(_data_type_size) = self.inner_codecs.data_type().size()
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
            }
        } else {
            let decoded_value = self.partial_decode(indexer, options)?;
            decode_into_array_bytes_target(&decoded_value, output_target)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
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

#[expect(clippy::too_many_arguments)]
#[expect(clippy::too_many_lines)]
fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
    let chunks_per_shard = subchunk_grid.grid_shape();
    let shard_chunk_grid = subchunk_grid;

    // Phase 1: Collect 1-D chunk indices for all inner chunks overlapping the subset.
    let chunk_indices_1d = collect_chunk_indices(shard_chunk_grid, array_subset, chunks_per_shard)?;

    // Phase 2: Sort by byte offset and merge adjacent ranges into coalesced groups.
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;

    let num_groups = coalesced_groups.len();
    let group_read_limit = group_read_concurrent_limit(options, num_groups);
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape)?;

    // Helper: compute the overlap of chunk `chunk_indices_nd` with `array_subset`,
    // relative to subset origin.
    let chunk_output_overlap_subset =
        |chunk_indices_nd: &[u64]| -> Result<ArraySubset, CodecError> {
            let chunk_subset = shard_chunk_grid
                .subset(chunk_indices_nd)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");
            let overlap = array_subset.overlap(&chunk_subset)?;
            overlap
                .relative_to(&array_subset.start())
                .map_err(CodecError::from)
        };

    // Phase 3a: Fill chunks in parallel (disjoint output regions, no I/O).
    let fill_element_bytes = fill_value.as_ne_bytes();
    let num_fill = fill_indices.len();
    (0..num_fill)
        .concurrent_limit(options.concurrent_target())
        .try_for_each(|f: usize| -> Result<(), CodecError> {
            let chunk_indices_nd =
                unravel_index(chunk_indices_1d[fill_indices[f]], chunks_per_shard)
                    .expect("inbounds chunk index");
            let overlap = chunk_output_overlap_subset(&chunk_indices_nd)?;
            // SAFETY: chunks represent disjoint array subsets
            let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
                unsafe { output_view.subdivide(overlap.offset(output_view.subset().start())?)? };
            subchunk_view
                .fill(fill_element_bytes)
                .map_err(CodecError::from)
        })?;

    // Phase 3b: Read all groups in parallel.
    let array_subset_start = array_subset.start();
    let loaded_groups = (&coalesced_groups)
        .concurrent_limit(group_read_limit)
        .map(|group| -> Result<CowBytes<'static>, CodecError> {
            Ok(input_handle
                .partial_decode(
                    ByteRange::FromStart(group.start, Some(group.total_len)),
                    options,
                )?
                .ok_or_else(|| {
                    CodecError::Other("Shard does not exist during partial decode.".to_string())
                })?
                .into_bytes()
                .into())
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Phase 3c: Decode all chunks in one shared Rayon workload.
    let ready_chunks = ready_chunks(&coalesced_groups);
    let (chunk_concurrent_limit, codec_options) = concurrency_chunks_and_codec(
        options.concurrent_target(),
        ready_chunks.len(),
        options,
        &codec_concurrency,
    );
    let decode_chunk = |(group_idx, chunk_idx): (usize, usize)| -> Result<(), CodecError> {
        let group = &coalesced_groups[group_idx];
        let coalesced_bytes = &loaded_groups[group_idx];
        let pos = group.chunks[chunk_idx];
        let idx = chunk_indices_1d[pos];
        let i = usize::try_from(idx).unwrap();
        let offset = shard_index[i * 2];
        let size = shard_index[i * 2 + 1];
        let chunk_indices_nd = unravel_index(idx, chunks_per_shard).expect("inbounds chunk index");
        let overlap = chunk_output_overlap_subset(&chunk_indices_nd)?;
        let subchunk_shape = shard_chunk_grid
            .chunk_shape(&chunk_indices_nd)?
            .expect("subchunk within shard");
        let subchunk_shape = subchunk_shape.as_slice();
        let subchunk_num_elements: u64 = subchunk_shape.iter().map(|d| d.get()).product();
        // SAFETY: chunks represent disjoint array subsets
        let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
            unsafe { output_view.subdivide(overlap.offset(output_view.subset().start())?)? };
        let start = usize::try_from(offset - group.start).unwrap();
        let end = start + usize::try_from(size).unwrap();
        if overlap.num_elements() == subchunk_num_elements {
            // Fast path: the overlap covers the full subchunk — decode directly.
            inner_codecs.decode_into(
                coalesced_bytes.slice(start..end),
                subchunk_shape,
                ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
                &codec_options,
            )
        } else {
            // Slow path: partial subchunk
            // Map the overlap to coordinates within the clipped subchunk.
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices_nd)?
                .expect("subchunk within shard");
            let chunk_subset_overlap_in_chunk = overlap
                .offset(&array_subset_start)?
                .relative_to(chunk_subset.start())?;
            let coalesced_bytes_arc: CowBytes<'static> = coalesced_bytes.clone();
            get_subchunk_partial_decoder(
                &(Arc::new(coalesced_bytes_arc) as Arc<dyn BytesPartialDecoderTraits>),
                subchunk_shape,
                inner_codecs,
                &codec_options,
                offset - group.start,
                size,
            )?
            .partial_decode_into(
                &chunk_subset_overlap_in_chunk,
                ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
                &codec_options,
            )
        }
    };
    ready_chunks
        .concurrent_limit(chunk_concurrent_limit)
        .try_for_each(decode_chunk)?;
    Ok(())
}

#[expect(clippy::too_many_lines)]
fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = &shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard = subchunk_grid.grid_shape();
    let shard_chunk_grid = subchunk_grid;

    // Phase 1: Collect 1-D chunk indices for all inner chunks overlapping the subset.
    let chunk_indices_1d = collect_chunk_indices(shard_chunk_grid, array_subset, chunks_per_shard)?;
    let num_chunks = chunk_indices_1d.len();

    // Phase 2: Sort and coalesce.
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;

    let num_groups = coalesced_groups.len();
    let group_read_limit = group_read_concurrent_limit(options, num_groups);
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape)?;

    // Helper: compute the overlap of chunk `chunk_indices_nd` with `array_subset`,
    // relative to subset origin.
    let chunk_overlap_in_output = |chunk_indices_nd: &[u64]| -> Result<ArraySubset, CodecError> {
        let chunk_subset = shard_chunk_grid
            .subset(chunk_indices_nd)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let overlap = array_subset.overlap(&chunk_subset)?;
        Ok(overlap.relative_to(&array_subset.start()).unwrap())
    };

    // Phase 3: Decode each group; write results (bytes + overlap subset) into a
    // pre-allocated vec indexed by original chunk order (required for merge_chunks_vlen
    // ordering). Storing the overlap avoids recomputing it in the final collection pass.
    let mut results: Vec<Option<(ArrayBytes<'static>, ArraySubset)>> = vec![None; num_chunks];
    let results_slice = UnsafeCellSlice::new(results.as_mut_slice());

    // Phase 3a: Fill chunks in parallel (no I/O).
    let num_fill = fill_indices.len();
    (0..num_fill)
        .concurrent_limit(options.concurrent_target())
        .try_for_each(|f: usize| -> Result<(), CodecError> {
            let pos = fill_indices[f];
            let chunk_indices_nd = unravel_index(chunk_indices_1d[pos], chunks_per_shard)
                .expect("inbounds chunk index");
            let overlap = chunk_overlap_in_output(&chunk_indices_nd)?;
            let decoded =
                ArrayBytes::new_fill_value(data_type, overlap.num_elements(), fill_value)?
                    .into_variable()?;
            // SAFETY: fill_indices holds unique positions into chunk_indices_1d
            unsafe {
                *results_slice.index_mut(pos) = Some((ArrayBytes::Variable(decoded), overlap));
            }
            Ok(())
        })?;

    // Phase 3b: Read all groups in parallel.
    let array_subset_start = array_subset.start();
    let loaded_groups = (&coalesced_groups)
        .concurrent_limit(group_read_limit)
        .map(|group| -> Result<CowBytes<'static>, CodecError> {
            Ok(input_handle
                .partial_decode(
                    ByteRange::FromStart(group.start, Some(group.total_len)),
                    options,
                )?
                .ok_or_else(|| {
                    CodecError::Other("Shard does not exist during partial decode.".to_string())
                })?
                .into_bytes()
                .into())
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Phase 3c: Decode all chunks in one shared Rayon workload.
    let ready_chunks = ready_chunks(&coalesced_groups);
    let (chunk_concurrent_limit, codec_options) = concurrency_chunks_and_codec(
        options.concurrent_target(),
        ready_chunks.len(),
        options,
        &codec_concurrency,
    );
    let decode_chunk = |(group_idx, chunk_idx): (usize, usize)| -> Result<(), CodecError> {
        let group = &coalesced_groups[group_idx];
        let coalesced_bytes = &loaded_groups[group_idx];
        let pos = group.chunks[chunk_idx];
        let idx = chunk_indices_1d[pos];
        let i = usize::try_from(idx).unwrap();
        let offset = shard_index[i * 2];
        let size = shard_index[i * 2 + 1];
        // Compute chunk_indices_nd once; reused for both overlap and slow path.
        let chunk_indices_nd = unravel_index(idx, chunks_per_shard).expect("inbounds chunk index");
        let overlap = chunk_overlap_in_output(&chunk_indices_nd)?;
        let subchunk_shape = shard_chunk_grid
            .chunk_shape(&chunk_indices_nd)?
            .expect("subchunk within shard");
        let subchunk_shape = subchunk_shape.as_slice();
        let subchunk_num_elements: u64 = subchunk_shape.iter().map(|d| d.get()).product();
        let start = usize::try_from(offset - group.start).unwrap();
        let end = start + usize::try_from(size).unwrap();
        let decoded = if overlap.num_elements() == subchunk_num_elements {
            // Fast path: the overlap covers the full subchunk — decode directly.
            inner_codecs
                .decode(
                    coalesced_bytes.slice(start..end),
                    subchunk_shape,
                    &codec_options,
                )?
                .into_owned()
                .into_variable()?
        } else {
            // Slow path: partial subchunk
            // Map the overlap to coordinates within the clipped subchunk.
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices_nd)?
                .expect("subchunk within shard");
            let chunk_subset_overlap_in_chunk = overlap
                .offset(&array_subset_start)?
                .relative_to(chunk_subset.start())?;
            let coalesced_bytes_arc: CowBytes<'static> = coalesced_bytes.clone();
            get_subchunk_partial_decoder(
                &(Arc::new(coalesced_bytes_arc) as Arc<dyn BytesPartialDecoderTraits>),
                subchunk_shape,
                inner_codecs,
                &codec_options,
                offset - group.start,
                size,
            )?
            .partial_decode(&chunk_subset_overlap_in_chunk, &codec_options)?
            .into_owned()
            .into_variable()?
        };
        // SAFETY: group.chunks holds unique positions into chunk_indices_1d
        unsafe {
            *results_slice.index_mut(pos) = Some((ArrayBytes::Variable(decoded), overlap));
        }
        Ok(())
    };
    ready_chunks
        .concurrent_limit(chunk_concurrent_limit)
        .try_for_each(decode_chunk)?;

    let chunk_bytes_and_subsets = results
        .into_iter()
        .map(|r| {
            let (bytes, s) = r.expect("all chunks decoded");
            let v = bytes.into_variable()?;
            Ok((v, s))
        })
        .collect::<Result<Vec<_>, CodecError>>()?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, &array_subset.shape());
    Ok(ArrayBytes::Variable(out_array_subset))
}

fn partial_decode_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
    let decoded = groups
        .into_iter()
        .collect::<Vec<_>>()
        .concurrent_limit(subchunk_concurrent_limit)
        .map(|(subchunk_index, group)| {
            let shard_index_idx = usize::try_from(subchunk_index).unwrap();
            let bytes = if let Some((offset, size)) =
                super::subchunk_offset_size(shard_index, shard_index_idx)
            {
                let decoder = get_subchunk_partial_decoder(
                    input_handle,
                    &group.subchunk_shape,
                    inner_codecs,
                    &options,
                    offset,
                    size,
                )?;
                decoder.partial_decode(&group, &options)?.into_owned()
            } else {
                ArrayBytes::new_fill_value(data_type, group.len(), fill_value)?
            };
            Ok::<_, CodecError>((group.positions, bytes))
        })
        .collect::<Result<Vec<_>, _>>()?;
    super::merge_indexer_subchunks(decoded, usize::try_from(indexer.len()).unwrap(), data_type)
}

/// Decode the elements of `indexer` into `output_view` by scattering each subchunk group directly.
///
/// The `indexer` must be validated against the shard shape.
#[expect(clippy::too_many_arguments)]
fn partial_decode_fixed_indexer_into<'a>(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
    output_view: &'a mut ArrayBytesFixedDisjointView<'a>,
) -> Result<(), CodecError> {
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
    let subchunk_decoder = |subchunk_index: u64, subchunk_shape: &[NonZeroU64]| {
        let shard_index_idx = usize::try_from(subchunk_index).unwrap();
        super::subchunk_offset_size(shard_index, shard_index_idx)
            .map(|(offset, size)| {
                get_subchunk_partial_decoder(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    &options,
                    offset,
                    size,
                )
            })
            .transpose()
    };

    if groups.len() == 1 {
        // Positions within a group are ascending and cover every element, so decode in place
        let (subchunk_index, group) = groups.into_iter().next().expect("one group");
        return if let Some(decoder) = subchunk_decoder(subchunk_index, &group.subchunk_shape)? {
            decoder.partial_decode_into(
                &group,
                ArrayBytesDecodeIntoTarget::Fixed(output_view),
                &options,
            )
        } else {
            output_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from)
        };
    }

    let decoded = groups
        .into_iter()
        .collect::<Vec<_>>()
        .concurrent_limit(subchunk_concurrent_limit)
        .map(|(subchunk_index, group)| {
            let bytes = subchunk_decoder(subchunk_index, &group.subchunk_shape)?
                .map(|decoder| {
                    Ok::<_, CodecError>(
                        decoder
                            .partial_decode(&group, &options)?
                            .into_owned()
                            .into_fixed()?,
                    )
                })
                .transpose()?;
            Ok::<_, CodecError>((group.positions, bytes))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (positions, bytes) in decoded {
        if let Some(bytes) = bytes {
            output_view.copy_elements_from_slice(&positions, &bytes)?;
        } else {
            output_view.fill_elements(&positions, fill_value.as_ne_bytes())?;
        }
    }
    Ok(())
}

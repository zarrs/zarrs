#![allow(clippy::similar_names)]

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ArraySubset;
use zarrs_data_type::FillValue;

use super::{ShardingIndexLocation, calculate_chunks_per_shard};
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::CodecChain;
use crate::array::concurrency::calc_concurrency_outer_inner;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesOffsets, ArrayBytesRaw, ArrayIndices,
    ArraySubsetTraits, ChunkShape, ChunkShapeTraits, DataType, DataTypeSize,
    IncompatibleDimensionalityError, Indexer, IndexerError, ravel_indices, unravel_index,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecError, CodecOptions, InvalidNumberOfElementsError, RecommendedConcurrency, decode_into_array_bytes_target
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
        DataTypeSize::Fixed(data_type_size) => {
            if let Some(subset) = indexer.as_array_subset() {
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
                    data_type,
                    fill_value,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    subset,
                    options,
                    &mut output_view,
                )?;
                Ok(ArrayBytes::from(out_array_subset))
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
        if let DataTypeSize::Fixed(_data_type_size) = &self.data_type.size()
            && let Some(subset) = indexer.as_array_subset()
            && let ArrayBytesDecodeIntoTarget::Fixed(output_view) = output_target
        {
            partial_decode_fixed_array_subset_into(
                &self.input_handle,
                &self.data_type,
                &self.fill_value,
                &self.shard_shape,
                &self.subchunk_shape,
                &self.inner_codecs,
                self.shard_index.as_deref(),
                subset,
                options,
                output_view,
            )
        } else {
            let decoded_value = self.partial_decode(indexer, options)?;
            decode_into_array_bytes_target(&decoded_value, output_target)
        }
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

/// A set of byte-adjacent inner chunks that can be read in a single I/O call.
struct CoalescedGroup {
    /// Byte offset of the first byte in the shard for this group.
    start: u64,
    /// Total byte length of the coalesced read.
    total_len: u64,
    /// Positions into `chunk_indices_1d` in ascending byte-offset order.
    chunks: Vec<usize>,
}

/// Collect the 1-D ravelled indices of all inner chunks overlapping `array_subset`.
fn collect_chunk_indices(
    shard_chunk_grid: &RegularChunkGrid,
    array_subset: &dyn ArraySubsetTraits,
    chunks_per_shard: &[u64],
) -> Result<Vec<u64>, CodecError> {
    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    let mut chunk_indices = Vec::with_capacity(chunks.num_elements_usize());
    for chunk_indices_nd in chunks.indices() {
        let idx = ravel_indices(&chunk_indices_nd, chunks_per_shard).expect("inbounds chunk");
        chunk_indices.push(idx);
    }
    Ok(chunk_indices)
}

/// Sort inner chunks by byte offset and merge exactly-adjacent ranges.
///
/// Returns coalesced groups and fill-value positions, both as positions into `chunk_indices_1d`.
///
/// # Errors
/// Returns an error if a shard index entry has only one of `offset`/`size` equal to `u64::MAX`,
/// which indicates a corrupted shard index.
fn coalesce_chunks(
    chunk_indices_1d: &[u64],
    shard_index: &[u64],
) -> Result<(Vec<CoalescedGroup>, Vec<usize>), CodecError> {
    let mut fill_positions: Vec<usize> = Vec::new();
    let mut io_positions: Vec<usize> = Vec::new();
    for (pos, &idx) in chunk_indices_1d.iter().enumerate() {
        let i = usize::try_from(idx).unwrap();
        let offset = shard_index[i * 2];
        let size = shard_index[i * 2 + 1];
        match (offset == u64::MAX, size == u64::MAX) {
            (true, true) => fill_positions.push(pos),
            (false, false) => io_positions.push(pos),
            _ => {
                return Err(CodecError::Other(
                    "Shard index entry has mismatched sentinel values; the shard may be corrupted."
                        .to_string(),
                ));
            }
        }
    }

    io_positions.sort_by_key(|&pos| {
        let i = usize::try_from(chunk_indices_1d[pos]).unwrap();
        shard_index[i * 2]
    });

    let mut groups: Vec<CoalescedGroup> = Vec::new();
    for pos in io_positions {
        let i = usize::try_from(chunk_indices_1d[pos]).unwrap();
        let offset = shard_index[i * 2];
        let size = shard_index[i * 2 + 1];
        if let Some(last) = groups.last_mut()
            && last.start + last.total_len == offset
        {
            last.total_len += size;
            last.chunks.push(pos);
        } else {
            groups.push(CoalescedGroup {
                start: offset,
                total_len: size,
                chunks: vec![pos],
            });
        }
    }

    Ok((groups, fill_positions))
}

#[expect(clippy::too_many_arguments)]
fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    output_view: &mut ArrayBytesFixedDisjointView<'_>,
) -> Result<(), CodecError> {
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
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        data_type,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;
    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let array_subset_shape = array_subset.shape();
    let subchunk_shape_u64: &[u64] = bytemuck::must_cast_slice(subchunk_shape);

    // Phase 1: Collect 1-D chunk indices for all inner chunks overlapping the subset.
    let chunk_indices_1d =
        collect_chunk_indices(&shard_chunk_grid, array_subset, &chunks_per_shard)?;

    // Phase 2: Sort by byte offset and merge adjacent ranges into coalesced groups.
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;

    // Concurrency: split the budget across three levels — groups, chunks-per-group, codec.
    //   Step 1: group vs. (chunk+codec).
    //   `chunk_concurrent_minimum` sets the floor so that even with many tiny groups the
    //   per-group budget does not collapse to 1; the ceiling is clamped to `num_groups`.
    let num_groups = coalesced_groups.len();
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape, data_type)?;
    let group_concurrent_minimum = std::cmp::min(options.chunk_concurrent_minimum(), num_groups);
    let group_concurrent_maximum = std::cmp::max(options.chunk_concurrent_minimum(), num_groups);
    let (group_concurrent_limit, chunk_budget) = calc_concurrency_outer_inner(
        options.concurrent_target(),
        &RecommendedConcurrency::new(group_concurrent_minimum..group_concurrent_maximum),
        &codec_concurrency,
    );
    let chunk_and_codec_options = options.with_concurrent_target(chunk_budget);
    //   Step 2: chunk vs. codec (reuses existing helper).
    let (chunk_concurrent_limit, codec_options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        data_type,
        subchunk_shape,
        &chunks_per_shard,
        &chunk_and_codec_options,
    )?;

    // Helper: compute the overlap of chunk `chunk_indices_nd` with `array_subset`,
    // relative to subset origin.
    let chunk_output_overlap_subset = |chunk_indices_nd: &[u64]| -> Result<ArraySubset, CodecError> {
        let chunk_subset = shard_chunk_grid
            .subset(chunk_indices_nd)
            .expect("matching dimensionality");
        let overlap = array_subset.overlap(&chunk_subset)?;
        overlap.relative_to(&array_subset.start()).map_err(CodecError::from)
    };

    // Phase 3a: Fill chunks in parallel (disjoint output regions, no I/O).
    let fill_element_bytes = fill_value.as_ne_bytes();
    let num_fill = fill_indices.len();
    crate::iter_concurrent_limit!(
        options.concurrent_target(),
        (0..num_fill),
        try_for_each,
        |f: usize| -> Result<(), CodecError> {
            let chunk_indices_nd =
                unravel_index(chunk_indices_1d[fill_indices[f]], &chunks_per_shard)
                    .expect("inbounds chunk index");
            let overlap = chunk_output_overlap_subset(&chunk_indices_nd)?;
            // SAFETY: chunks represent disjoint array subsets
            let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
                unsafe { output_view.subdivide(overlap.offset(output_view.subset().start())?)? };
            subchunk_view
                .fill(fill_element_bytes)
                .map_err(CodecError::from)
        }
    )?;

    // Phase 3b: I/O groups in parallel; chunks within each group also in parallel.
    let subchunk_num_elements: u64 = subchunk_shape_u64.iter().product();
    let array_subset_start = array_subset.start();
    let decode_group = |g: usize| -> Result<(), CodecError> {
        let group = &coalesced_groups[g];
        // Hold as Arc so the slow path can share the buffer without copying.
        let coalesced_bytes: Arc<Vec<u8>> = Arc::new(
            input_handle
                .partial_decode(
                    ByteRange::FromStart(group.start, Some(group.total_len)),
                    &options,
                )?
                .ok_or_else(|| {
                    CodecError::Other("Shard does not exist during partial decode.".to_string())
                })?
                .into_owned(),
        );

        let decode_chunk = |j: usize| -> Result<(), CodecError> {
            let pos = group.chunks[j];
            let idx = chunk_indices_1d[pos];
            let i = usize::try_from(idx).unwrap();
            let offset = shard_index[i * 2];
            let size = shard_index[i * 2 + 1];
            let chunk_indices_nd =
                unravel_index(idx, &chunks_per_shard).expect("inbounds chunk index");
            let overlap = chunk_output_overlap_subset(&chunk_indices_nd)?;
            // SAFETY: chunks represent disjoint array subsets
            let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
                unsafe { output_view.subdivide(overlap.offset(output_view.subset().start())?)? };
            let start = usize::try_from(offset - group.start).unwrap();
            let end = start + usize::try_from(size).unwrap();
            if overlap.num_elements() == subchunk_num_elements {
                // Fast path: the overlap covers the full subchunk — decode directly.
                inner_codecs
                    .decode_into(
                        Cow::Borrowed(&coalesced_bytes[start..end]),
                        subchunk_shape,
                        data_type,
                        fill_value,
                        ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
                        &codec_options,
                    )
            } else {
                // Slow path: partial subchunk
                // Compute the overlap region in chunk-local coordinates in a single pass,
                // avoiding two intermediate Vec allocations.
                let chunk_subset_overlap_in_chunk = ArraySubset::new_with_start_shape(
                    std::iter::zip(
                        std::iter::zip(overlap.start().iter(), array_subset_start.iter()),
                        std::iter::zip(&chunk_indices_nd, subchunk_shape),
                    )
                    .map(|((&rel, &abs), (&ci, &cs))| rel + abs - ci * cs.get())
                    .collect(),
                    overlap.shape().to_owned(),
                )
                .expect("valid subset");
                let coalesced_bytes_arc: Arc<Vec<u8>> = Arc::clone(&coalesced_bytes);
                get_subchunk_partial_decoder(
                    &(coalesced_bytes_arc as Arc<dyn BytesPartialDecoderTraits>),
                    data_type,
                    fill_value,
                    subchunk_shape,
                    inner_codecs,
                    &codec_options,
                    offset - group.start,
                    size,
                )?
                .partial_decode_into(&chunk_subset_overlap_in_chunk, ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),&codec_options)
            }
        };

        let num_chunks_in_group = group.chunks.len();
        crate::iter_concurrent_limit!(
            chunk_concurrent_limit,
            (0..num_chunks_in_group),
            try_for_each,
            decode_chunk
        )
    };
    crate::iter_concurrent_limit!(
        group_concurrent_limit,
        (0..num_groups),
        try_for_each,
        decode_group
    )?;
    Ok(())
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

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .expect("matching dimensionality");

    let subchunk_shape_u64: &[u64] = bytemuck::must_cast_slice(subchunk_shape);

    // Phase 1: Collect 1-D chunk indices for all inner chunks overlapping the subset.
    let chunk_indices_1d =
        collect_chunk_indices(&shard_chunk_grid, array_subset, &chunks_per_shard)?;
    let num_chunks = chunk_indices_1d.len();

    // Phase 2: Sort and coalesce.
    let (coalesced_groups, fill_indices) = coalesce_chunks(&chunk_indices_1d, shard_index)?;

    // Concurrency: split the budget across three levels — groups, chunks-per-group, codec.
    //   Step 1: group vs. (chunk+codec).
    let num_groups = coalesced_groups.len();
    let codec_concurrency = inner_codecs.recommended_concurrency(subchunk_shape, data_type)?;
    let group_concurrent_minimum = std::cmp::min(options.chunk_concurrent_minimum(), num_groups);
    let group_concurrent_maximum = std::cmp::max(options.chunk_concurrent_minimum(), num_groups);
    let (group_concurrent_limit, chunk_budget) = calc_concurrency_outer_inner(
        options.concurrent_target(),
        &RecommendedConcurrency::new(group_concurrent_minimum..group_concurrent_maximum),
        &codec_concurrency,
    );
    let chunk_and_codec_options = options.with_concurrent_target(chunk_budget);
    //   Step 2: chunk vs. codec (reuses existing helper).
    let (chunk_concurrent_limit, codec_options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        data_type,
        subchunk_shape,
        &chunks_per_shard,
        &chunk_and_codec_options,
    )?;

    // Helper: compute the overlap of chunk `chunk_indices_nd` with `array_subset`,
    // relative to subset origin.
    let chunk_overlap_in_output = |chunk_indices_nd: &[u64]| -> Result<ArraySubset, CodecError> {
        let chunk_subset = shard_chunk_grid
            .subset(chunk_indices_nd)
            .expect("matching dimensionality");
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
    crate::iter_concurrent_limit!(
        options.concurrent_target(),
        (0..num_fill),
        try_for_each,
        |f: usize| -> Result<(), CodecError> {
            let pos = fill_indices[f];
            let chunk_indices_nd = unravel_index(chunk_indices_1d[pos], &chunks_per_shard)
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
        }
    )?;

    // Phase 3b: I/O groups in parallel; chunks within each group also in parallel.
    let subchunk_num_elements: u64 = subchunk_shape_u64.iter().product();
    let array_subset_start = array_subset.start();
    let decode_group = |g: usize| -> Result<(), CodecError> {
        let group = &coalesced_groups[g];
        // Hold as Arc so the slow path can share the buffer without copying.
        let coalesced_bytes: Arc<Vec<u8>> = Arc::new(
            input_handle
                .partial_decode(
                    ByteRange::FromStart(group.start, Some(group.total_len)),
                    options,
                )?
                .ok_or_else(|| {
                    CodecError::Other("Shard does not exist during partial decode.".to_string())
                })?
                .into_owned(),
        );

        let decode_chunk = |j: usize| -> Result<(), CodecError> {
            let pos = group.chunks[j];
            let idx = chunk_indices_1d[pos];
            let i = usize::try_from(idx).unwrap();
            let offset = shard_index[i * 2];
            let size = shard_index[i * 2 + 1];
            // Compute chunk_indices_nd once; reused for both overlap and slow path.
            let chunk_indices_nd =
                unravel_index(idx, &chunks_per_shard).expect("inbounds chunk index");
            let overlap = chunk_overlap_in_output(&chunk_indices_nd)?;
            let start = usize::try_from(offset - group.start).unwrap();
            let end = start + usize::try_from(size).unwrap();
            let decoded = if overlap.num_elements() == subchunk_num_elements {
                // Fast path: the overlap covers the full subchunk — decode directly.
                inner_codecs
                    .decode(
                        Cow::Borrowed(&coalesced_bytes[start..end]),
                        subchunk_shape,
                        data_type,
                        fill_value,
                        &codec_options,
                    )?
                    .into_owned()
                    .into_variable()?
            } else {
                // Slow path: partial subchunk
                // Compute the overlap region in chunk-local coordinates in a single pass,
                // avoiding two intermediate Vec allocations.
                let chunk_subset_overlap_in_chunk = ArraySubset::new_with_start_shape(
                    std::iter::zip(
                        std::iter::zip(overlap.start().iter(), array_subset_start.iter()),
                        std::iter::zip(&chunk_indices_nd, subchunk_shape),
                    )
                    .map(|((&rel, &abs), (&ci, &cs))| rel + abs - ci * cs.get())
                    .collect(),
                    overlap.shape().to_owned(),
                )
                .expect("valid subset");
                let coalesced_bytes_arc: Arc<Vec<u8>> = Arc::clone(&coalesced_bytes);
                get_subchunk_partial_decoder(
                    &(coalesced_bytes_arc as Arc<dyn BytesPartialDecoderTraits>),
                    data_type,
                    fill_value,
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

        let num_chunks_in_group = group.chunks.len();
        crate::iter_concurrent_limit!(
            chunk_concurrent_limit,
            (0..num_chunks_in_group),
            try_for_each,
            decode_chunk
        )
    };
    crate::iter_concurrent_limit!(
        group_concurrent_limit,
        (0..num_groups),
        try_for_each,
        decode_group
    )?;

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

use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, BytesPartialDecoderTraits, CodecError,
    CodecOptions, InvalidNumberOfElementsError,
};

use super::super::calculate_chunks_per_shard;
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec, ArraySubsetTraits,
    ChunkShapeTraits, CodecChainBound, IncompatibleDimensionalityError, ravel_indices,
};

#[expect(clippy::too_many_arguments)]
pub(super) fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
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
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let (subchunk_concurrent_limit, options) =
        super::super::get_concurrent_target_and_codec_options(
            inner_codecs,
            subchunk_shape,
            &chunks_per_shard,
            options,
        )?;
    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let array_subset_start = array_subset.start();
    let decode_subchunk_subset_into_slice = |chunk_indices: ArrayIndicesTinyVec| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;
        // Calculate the chunk's position in the output view coordinate space
        let chunk_relative = chunk_subset_overlap.relative_to(&array_subset_start)?;
        let chunk_output_overlap_subset = chunk_relative.offset(output_view.subset().start())?;
        // SAFETY: chunks represent disjoint array subsets
        let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
            unsafe { output_view.subdivide(chunk_output_overlap_subset)? };
        if offset == u64::MAX && size == u64::MAX {
            subchunk_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from)
        } else {
            // Partially decode the subchunk
            let inner_partial_decoder = super::get_subchunk_partial_decoder(
                input_handle,
                subchunk_shape,
                inner_codecs,
                &options,
                offset,
                size,
            )?;
            inner_partial_decoder.partial_decode_into(
                &chunk_subset_overlap
                    .relative_to(chunk_subset.start())
                    .unwrap(),
                ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
                &options,
            )
        }
    };

    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        chunks.indices(),
        try_for_each,
        decode_subchunk_subset_into_slice
    )?;
    Ok(())
}

pub(super) fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = &shard_index else {
        return super::super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let (subchunk_concurrent_limit, options) =
        super::super::get_concurrent_target_and_codec_options(
            inner_codecs,
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
    let decode_subchunk_subset = |chunk_indices: ArrayIndicesTinyVec| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;

        let chunk_subset_bytes = if offset == u64::MAX && size == u64::MAX {
            ArrayBytes::new_fill_value(data_type, chunk_subset_overlap.num_elements(), fill_value)?
                .into_variable()?
        } else {
            // Partially decode the subchunk
            let inner_partial_decoder = super::get_subchunk_partial_decoder(
                input_handle,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            )?;
            inner_partial_decoder
                .partial_decode(
                    &chunk_subset_overlap
                        .relative_to(chunk_subset.start())
                        .unwrap(),
                    options,
                )?
                .into_owned()
                .into_variable()?
        };
        Ok::<_, CodecError>((
            chunk_subset_bytes,
            chunk_subset_overlap
                .relative_to(&array_subset_start)
                .unwrap(),
        ))
    };
    // Decode the subchunk subsets
    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let chunk_bytes_and_subsets = crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        chunks.indices(),
        map,
        decode_subchunk_subset
    )
    .collect::<Result<Vec<_>, _>>()?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, &array_subset.shape());
    Ok(ArrayBytes::Variable(out_array_subset))
}

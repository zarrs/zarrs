use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_codec::{ArrayCodecTraits, AsyncBytesPartialDecoderTraits, CodecError, CodecOptions};
use zarrs_data_type::FillValue;

use super::super::calculate_chunks_per_shard;
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec, ArraySubset, ArraySubsetTraits,
    ChunkShapeTraits, CodecChainBound, DataType, IncompatibleDimensionalityError, ravel_indices,
};

#[allow(clippy::too_many_lines)]
pub(super) async fn partial_decode_fixed_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
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
        return super::super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    // Find filled / non filled chunks
    let chunk_info = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard")
        .indices()
        .into_iter()
        .map(|chunk_indices: ArrayIndicesTinyVec| {
            let chunk_index =
                ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
            let chunk_index = usize::try_from(chunk_index).unwrap();

            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");

            // Read the offset/size
            let offset = shard_index[chunk_index * 2];
            let size = shard_index[chunk_index * 2 + 1];
            if offset == u64::MAX && size == u64::MAX {
                (chunk_subset, None)
            } else {
                (chunk_subset, Some((offset, size)))
            }
        })
        .collect::<Vec<_>>();

    let shard_size = array_subset.num_elements_usize() * data_type_size;
    let mut shard = Vec::with_capacity(shard_size);
    let shard_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut shard);

    // Decode unfilled chunks
    let results = futures::future::join_all(
        chunk_info
            .iter()
            .filter_map(|(chunk_subset, offset_size)| {
                offset_size
                    .as_ref()
                    .map(|offset_size| (chunk_subset, offset_size))
            })
            .map(|(chunk_subset, (offset, size))| {
                async move {
                    let inner_partial_decoder = super::get_subchunk_partial_decoder_async(
                        input_handle,
                        subchunk_shape,
                        inner_codecs,
                        options,
                        *offset,
                        *size,
                    )
                    .await?;
                    let chunk_subset_overlap = array_subset.overlap(chunk_subset).unwrap(); // FIXME: unwrap

                    // Partial decoding is actually really slow with the blosc codec! Assume sharded chunks are small, and just decode the whole thing and extract bytes
                    // TODO: Investigate further
                    // let decoded_chunk = partial_decoder
                    //     .partial_decode(&[chunk_subset_overlap.relative_to(chunk_subset.start())?])
                    //     .await?
                    //     .remove(0);

                    let decoded_chunk = inner_partial_decoder
                        .partial_decode(
                            &ArraySubset::new_with_shape(chunk_subset.shape().to_vec()),
                            options,
                        ) // TODO: Adjust options for partial decoding
                        .await?
                        .into_owned();
                    let decoded_chunk = decoded_chunk
                        .extract_array_subset(
                            &chunk_subset_overlap
                                .relative_to(chunk_subset.start())
                                .unwrap(),
                            chunk_subset.shape(),
                            data_type,
                        )?
                        .into_fixed()?
                        .into_owned();
                    Ok::<_, CodecError>((decoded_chunk, chunk_subset_overlap))
                }
            }),
    )
    .await;
    // FIXME: Concurrency limit for futures

    let array_subset_start = array_subset.start();
    let array_subset_shape = array_subset.shape();

    if !results.is_empty() {
        crate::iter_concurrent_limit!(
            options.concurrent_target(),
            results,
            try_for_each,
            |subset_and_decoded_chunk| {
                let (chunk_subset_bytes, chunk_subset_overlap): (Vec<u8>, ArraySubset) =
                    subset_and_decoded_chunk?;
                let mut output_view = unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        shard_slice,
                        data_type_size,
                        &array_subset_shape,
                        chunk_subset_overlap
                            .relative_to(&array_subset_start)
                            .unwrap(),
                    )?
                };
                output_view
                    .copy_from_slice(&chunk_subset_bytes)
                    .expect("chunk subset bytes are the correct length");
                Ok::<_, CodecError>(())
            }
        )?;
    }

    // Write filled chunks
    let filled_chunks = chunk_info
        .iter()
        .filter_map(|(chunk_subset, offset_size)| {
            if offset_size.is_none() {
                Some(chunk_subset)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if !filled_chunks.is_empty() {
        // Write filled chunks
        crate::iter_concurrent_limit!(
            options.concurrent_target(),
            filled_chunks,
            try_for_each,
            |chunk_subset: &ArraySubset| {
                let chunk_subset_overlap = array_subset.overlap(chunk_subset)?;
                let mut output_view = unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        shard_slice,
                        data_type_size,
                        &array_subset_shape,
                        chunk_subset_overlap
                            .relative_to(&array_subset_start)
                            .unwrap(),
                    )?
                };
                output_view
                    .fill(fill_value.as_ne_bytes())
                    .map_err(CodecError::from)
            }
        )?;
    }
    unsafe { shard.set_len(shard_size) };
    Ok(ArrayBytes::from(shard))
}

#[expect(clippy::too_many_arguments)]
pub(super) async fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return super::super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .expect("matching dimensionality");

    let array_subset_start = array_subset.start();
    let decode_subchunk_subset = |chunk_indices: ArrayIndicesTinyVec, chunk_subset: ArraySubset| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let array_subset_start = &array_subset_start;
        async move {
            let offset = shard_index[shard_index_idx * 2];
            let size = shard_index[shard_index_idx * 2 + 1];

            // Get the subset of bytes from the chunk which intersect the array
            let chunk_subset_overlap = array_subset.overlap(&chunk_subset).unwrap(); // FIXME: unwrap

            let chunk_subset_bytes = if offset == u64::MAX && size == u64::MAX {
                ArrayBytes::new_fill_value(
                    data_type,
                    chunk_subset_overlap.num_elements(),
                    fill_value,
                )?
                .into_variable()?
            } else {
                // Partially decode the subchunk
                let inner_partial_decoder = super::get_subchunk_partial_decoder_async(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
                .await?;
                inner_partial_decoder
                    .partial_decode(
                        &chunk_subset_overlap
                            .relative_to(chunk_subset.start())
                            .unwrap(),
                        options,
                    )
                    .await?
                    .into_owned()
                    .into_variable()?
            };
            Ok::<_, CodecError>((
                chunk_subset_bytes,
                chunk_subset_overlap
                    .relative_to(array_subset_start)
                    .unwrap(),
            ))
        }
    };

    // Decode the subchunk subsets
    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let chunk_bytes_and_subsets =
        futures::future::try_join_all(chunks.indices().into_iter().map(|chunk_indices| {
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");
            let decode = &decode_subchunk_subset;
            decode(chunk_indices, chunk_subset)
        }))
        .await?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, &array_subset.shape());
    Ok(ArrayBytes::Variable(out_array_subset))
}

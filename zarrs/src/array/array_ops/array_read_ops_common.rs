use crate::IntoConcurrentLimitIterator;
use crate::array::{
    ArrayError, ArrayIndices, ArrayOps, ArraySubset, ArraySubsetTraits,
    IncompatibleDimensionalityError, Indexer,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, CodecError, CodecOptions, InvalidNumberOfElementsError, Resources,
    copy_fill_value_into,
};
use zarrs_storage::{MaybeSend, MaybeSync};

use super::super::array_bytes_internal::{build_nested_optional_target, extract_target_views};
use super::super::concurrency::concurrency_chunks_and_codec;
use super::recommended_codec_concurrency;

/// Shared implementation of `ArrayReadOps::retrieve_array_subset_into`.
///
#[expect(clippy::too_many_arguments)]
pub(super) fn retrieve_array_subset_into<A, RetrieveChunkInto, RetrieveChunkSubsetInto>(
    array: &A,
    array_subset: &dyn ArraySubsetTraits,
    output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
    chunk_concurrent_minimum: usize,
    options: &CodecOptions,
    resources: &Resources,
    retrieve_chunk_into: RetrieveChunkInto,
    retrieve_partial_chunk_into: RetrieveChunkSubsetInto,
) -> Result<(), ArrayError>
where
    A: ArrayOps + MaybeSync,
    RetrieveChunkInto: for<'a> Fn(
        &[u64],
        &mut ArrayBytesDecodeIntoTarget<'a>,
        &CodecOptions,
        &Resources,
    ) -> Result<(), ArrayError>,
    RetrieveChunkSubsetInto: for<'a> Fn(
            &[u64],
            &dyn Indexer,
            &mut ArrayBytesDecodeIntoTarget<'a>,
            &CodecOptions,
            &Resources,
        ) -> Result<(), ArrayError>
        + MaybeSend
        + MaybeSync,
{
    if array_subset.dimensionality() != array.dimensionality() {
        return Err(ArrayError::InvalidArraySubset(
            Box::new(array_subset.to_array_subset()),
            array.shape().to_vec(),
        ));
    }

    if !array.data_type().is_fixed() {
        return Err(ArrayError::CodecError(CodecError::Other(
            "retrieve_array_subset_into does not support variable-length data types".to_string(),
        )));
    }

    if output_target.num_elements() != array_subset.num_elements() {
        return Err(ArrayError::CodecError(
            InvalidNumberOfElementsError::new(
                output_target.num_elements(),
                array_subset.num_elements(),
            )
            .into(),
        ));
    }

    let Some(chunks) = array.chunks_in_array_subset(array_subset)? else {
        return Err(ArrayError::InvalidArraySubset(
            Box::new(array_subset.to_array_subset()),
            array.shape().to_vec(),
        ));
    };

    let num_chunks = chunks.num_elements_usize();
    match num_chunks {
        0 => copy_fill_value_into(array.data_type(), array.fill_value(), output_target)
            .map_err(ArrayError::CodecError),
        1 => {
            let chunk_indices = chunks.start();
            let chunk_subset = array.chunk_subset(chunk_indices)?;
            if chunk_subset == array_subset {
                retrieve_chunk_into(chunk_indices, output_target, options, resources)
            } else {
                retrieve_partial_chunk_into(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    output_target,
                    options,
                    resources,
                )
            }
        }
        _ => {
            let chunk_shape = array.chunk_shape(chunks.start())?;
            let codec_concurrency = recommended_codec_concurrency(array, &chunk_shape)?;
            let (chunk_concurrent_limit, resources) = concurrency_chunks_and_codec(
                num_chunks,
                chunk_concurrent_minimum,
                resources,
                &codec_concurrency,
            );
            retrieve_multi_chunk_fixed_into(
                array,
                array_subset,
                &chunks,
                chunk_concurrent_limit,
                output_target,
                options,
                &resources,
                &retrieve_partial_chunk_into,
            )
        }
    }
}

#[expect(clippy::too_many_arguments)]
pub(super) fn retrieve_multi_chunk_fixed_into<A, RetrieveChunkSubsetInto>(
    array: &A,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    output_target: &ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
    resources: &Resources,
    retrieve_partial_chunk_into: &RetrieveChunkSubsetInto,
) -> Result<(), ArrayError>
where
    A: ArrayOps + MaybeSync,
    RetrieveChunkSubsetInto: for<'a> Fn(
            &[u64],
            &dyn Indexer,
            &mut ArrayBytesDecodeIntoTarget<'a>,
            &CodecOptions,
            &Resources,
        ) -> Result<(), ArrayError>
        + MaybeSend
        + MaybeSync,
{
    let (data_view_ref, mask_view_refs) = extract_target_views(output_target);
    let array_subset_start = array_subset.start();
    let output_start = data_view_ref.subset().start();

    let retrieve_chunk = |chunk_indices: ArrayIndices| {
        let chunk_subset = array.chunk_subset(&chunk_indices)?;
        let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
        let chunk_subset_in_view =
            subset_in_output(&chunk_subset_overlap, &array_subset_start, output_start)?;

        let data_sub = unsafe {
            // SAFETY: chunks represent disjoint array subsets.
            data_view_ref.subdivide(chunk_subset_in_view.clone())?
        };

        let mut mask_subs = Vec::with_capacity(mask_view_refs.len());
        for mask_view in &mask_view_refs {
            mask_subs.push(unsafe {
                // SAFETY: chunks represent disjoint array subsets.
                mask_view.subdivide(chunk_subset_in_view.clone())
            }?);
        }

        let mut target = build_nested_optional_target(data_sub, mask_subs);
        retrieve_partial_chunk_into(
            &chunk_indices,
            &chunk_subset_overlap.relative_to(chunk_subset.start())?,
            &mut target,
            options,
            resources,
        )?;
        Ok::<_, ArrayError>(())
    };

    chunks
        .indices()
        .concurrent_limit(chunk_concurrent_limit)
        .try_for_each(retrieve_chunk)?;

    Ok(())
}

/// Translate `subset`, which lies within an array subset starting at `array_subset_start`, to an output starting at `output_start`.
///
/// Equivalent to `subset.relative_to(array_subset_start)?.offset(output_start)` without the intermediate subset, which is measurable when reading many small chunks.
pub(super) fn subset_in_output(
    subset: &ArraySubset,
    array_subset_start: &[u64],
    output_start: &[u64],
) -> Result<ArraySubset, ArrayError> {
    for start in [array_subset_start, output_start] {
        if start.len() != subset.dimensionality() {
            return Err(
                IncompatibleDimensionalityError::new(start.len(), subset.dimensionality()).into(),
            );
        }
    }
    let start: Vec<u64> = std::iter::zip(
        subset.start(),
        std::iter::zip(array_subset_start, output_start),
    )
    .map(|(&start, (&array_start, &output_start))| start - array_start + output_start)
    .collect();
    Ok(ArraySubset::new_with_start_shape(start, subset.shape())?)
}

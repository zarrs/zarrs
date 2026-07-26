use crate::array::{
    Array, ArrayBytesFixedDisjointView, ArrayError, ArrayIndicesTinyVec, ArraySubset,
    ArraySubsetTraits,
};
use crate::iter_concurrent_limit;
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, CodecError, CodecOptions, InvalidNumberOfElementsError,
    copy_fill_value_into,
};
use zarrs_storage::{MaybeSend, MaybeSync, ReadableStorageTraits};

use super::super::array_bytes_internal::{build_nested_optional_target, extract_target_views};
use super::super::concurrency::concurrency_chunks_and_codec;

pub(super) fn retrieve_array_subset_into<TStorage, RetrieveChunkInto, RetrieveChunkSubsetInto>(
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    output_target: ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
    retrieve_chunk_into: RetrieveChunkInto,
    retrieve_chunk_subset_into: RetrieveChunkSubsetInto,
) -> Result<(), ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    RetrieveChunkInto:
        for<'a> Fn(&[u64], ArrayBytesDecodeIntoTarget<'a>, &CodecOptions) -> Result<(), ArrayError>,
    RetrieveChunkSubsetInto: for<'a> Fn(
            &[u64],
            &dyn ArraySubsetTraits,
            ArrayBytesDecodeIntoTarget<'a>,
            &CodecOptions,
        ) -> Result<(), ArrayError>
        + MaybeSend
        + MaybeSync,
{
    if array_subset.dimensionality() != array.dimensionality() {
        return Err(ArrayError::InvalidArraySubset(
            array_subset.to_array_subset(),
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
            array_subset.to_array_subset(),
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
                retrieve_chunk_into(chunk_indices, output_target, options)
            } else {
                retrieve_chunk_subset_into(
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    output_target,
                    options,
                )
            }
        }
        _ => {
            let chunk_shape = array.chunk_shape(chunks.start())?;
            let codec_concurrency = array.recommended_codec_concurrency(&chunk_shape)?;
            let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                options.concurrent_target(),
                num_chunks,
                options,
                &codec_concurrency,
            );
            retrieve_multi_chunk_fixed_into(
                array,
                array_subset,
                &chunks,
                chunk_concurrent_limit,
                &output_target,
                &options,
                &retrieve_chunk_subset_into,
            )
        }
    }
}

fn retrieve_multi_chunk_fixed_into<TStorage, RetrieveChunkSubsetInto>(
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    output_target: &ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
    retrieve_chunk_subset_into: &RetrieveChunkSubsetInto,
) -> Result<(), ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    RetrieveChunkSubsetInto: for<'a> Fn(
            &[u64],
            &dyn ArraySubsetTraits,
            ArrayBytesDecodeIntoTarget<'a>,
            &CodecOptions,
        ) -> Result<(), ArrayError>
        + MaybeSend
        + MaybeSync,
{
    let (data_view_ref, mask_view_refs) = extract_target_views(output_target);
    let parent_start = data_view_ref.subset().start().to_vec();

    let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
        let chunk_subset = array.chunk_subset(&chunk_indices)?;
        let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
        let chunk_subset_in_array = chunk_subset_overlap.relative_to(&array_subset.start())?;

        let chunk_start_in_view: Vec<u64> = chunk_subset_in_array
            .start()
            .iter()
            .zip(&parent_start)
            .map(|(&c, &p)| c + p)
            .collect();
        let chunk_subset_in_view = ArraySubset::new_with_start_shape(
            chunk_start_in_view,
            chunk_subset_in_array.shape().to_vec(),
        )?;

        let mut data_sub = unsafe {
            // SAFETY: chunks represent disjoint array subsets.
            data_view_ref.subdivide(chunk_subset_in_view.clone())?
        };

        let mut mask_subs: Vec<ArrayBytesFixedDisjointView<'_>> = mask_view_refs
            .iter()
            .map(|mask_view| unsafe {
                // SAFETY: chunks represent disjoint array subsets.
                mask_view.subdivide(chunk_subset_in_view.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let target = build_nested_optional_target(&mut data_sub, mask_subs.as_mut_slice());
        retrieve_chunk_subset_into(
            &chunk_indices,
            &chunk_subset_overlap.relative_to(chunk_subset.start())?,
            target,
            options,
        )?;
        Ok::<_, ArrayError>(())
    };

    iter_concurrent_limit!(
        chunk_concurrent_limit,
        chunks.indices(),
        try_for_each,
        retrieve_chunk
    )?;

    Ok(())
}

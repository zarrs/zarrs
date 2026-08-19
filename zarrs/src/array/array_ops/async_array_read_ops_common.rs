use futures::{StreamExt, TryStreamExt};

use std::sync::Arc;

use crate::array::{
    ArrayBytesFixedDisjointView, ArrayError, ArrayIndices, ArrayIndicesTinyVec, ArrayOps,
    ArraySubset, ArraySubsetTraits, AsyncArrayReadOps,
};
use zarrs_codec::AsyncArrayPartialDecoderTraits;

use super::{enclosing_subchunk_indices, subchunk_chunk_and_local_subset};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, CodecError, CodecOptions, InvalidNumberOfElementsError,
    copy_fill_value_into,
};
use zarrs_storage::MaybeSync;

use super::super::array_bytes_internal::{build_nested_optional_target, extract_target_views};
use super::super::concurrency::concurrency_chunks_and_codec;
use super::recommended_codec_concurrency;

/// Chunk level `_into` retrieval, as used by [`retrieve_array_subset_into`].
///
/// This is the asynchronous counterpart of the `retrieve_chunk_into` and
/// `retrieve_chunk_subset_into` closure parameters of
/// [`retrieve_array_subset_into`](super::array_read_ops_common::retrieve_array_subset_into). A
/// trait is used rather than closures because the returned futures must be `Send` on non-`wasm32`
/// targets, which the `AsyncFn` traits cannot express.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub(super) trait AsyncRetrieveInto {
    async fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;

    async fn retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError>;
}

pub(super) async fn retrieve_array_subset_into<A, R>(
    array: &A,
    retrieve: &R,
    array_subset: &dyn ArraySubsetTraits,
    output_target: ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
) -> Result<(), ArrayError>
where
    A: ArrayOps + MaybeSync,
    R: AsyncRetrieveInto + ?Sized,
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
                retrieve
                    .retrieve_chunk_into(chunk_indices, output_target, options)
                    .await
            } else {
                retrieve
                    .retrieve_chunk_subset_into(
                        chunk_indices,
                        &array_subset.relative_to(chunk_subset.start())?,
                        output_target,
                        options,
                    )
                    .await
            }
        }
        _ => {
            let chunk_shape = array.chunk_shape(chunks.start())?;
            let codec_concurrency = recommended_codec_concurrency(array, &chunk_shape)?;
            let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                options.concurrent_target(),
                num_chunks,
                options,
                &codec_concurrency,
            );
            retrieve_multi_chunk_fixed_into(
                array,
                retrieve,
                array_subset,
                &chunks,
                chunk_concurrent_limit,
                &output_target,
                &options,
            )
            .await
        }
    }
}

async fn retrieve_multi_chunk_fixed_into<A, R>(
    array: &A,
    retrieve: &R,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    output_target: &ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
) -> Result<(), ArrayError>
where
    A: ArrayOps + MaybeSync,
    R: AsyncRetrieveInto + ?Sized,
{
    let (data_view_ref, mask_view_refs) = extract_target_views(output_target);
    let parent_start = data_view_ref.subset().start().to_vec();
    let array_subset_start = array_subset.start();

    let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
        let array_subset_start = &array_subset_start;
        let parent_start = &parent_start;
        let mask_view_refs = &mask_view_refs;
        async move {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
            let chunk_subset_in_array = chunk_subset_overlap.relative_to(array_subset_start)?;

            let chunk_start_in_view: Vec<u64> = chunk_subset_in_array
                .start()
                .iter()
                .zip(parent_start.iter())
                .map(|(&c, &p)| c + p)
                .collect();
            let chunk_subset_in_view = ArraySubset::new_with_start_shape(
                chunk_start_in_view,
                chunk_subset_in_array.shape().to_vec(),
            )?;

            let mut data_sub = unsafe {
                // SAFETY: chunks represent disjoint array subsets
                data_view_ref.subdivide(chunk_subset_in_view.clone())?
            };

            let mut mask_subs: Vec<ArrayBytesFixedDisjointView<'_>> = mask_view_refs
                .iter()
                .map(|mask_view| unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    mask_view.subdivide(chunk_subset_in_view.clone())
                })
                .collect::<Result<Vec<_>, _>>()?;

            let target = build_nested_optional_target(&mut data_sub, mask_subs.as_mut_slice());

            retrieve
                .retrieve_chunk_subset_into(
                    &chunk_indices,
                    &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                    target,
                    options,
                )
                .await?;
            Ok::<_, ArrayError>(())
        }
    };

    futures::stream::iter(&chunks.indices())
        .map(Ok)
        .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
        .await?;

    Ok(())
}

/// Async variant of
/// [`subchunk_partial_decoder_and_local_indices`](super::array_read_ops_common::subchunk_partial_decoder_and_local_indices).
pub(super) async fn subchunk_partial_decoder_and_local_indices<A>(
    array: &A,
    level: usize,
    subchunk_indices: &[u64],
) -> Result<(Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayIndices), ArrayError>
where
    A: AsyncArrayReadOps + ?Sized,
{
    let (chunk_indices, subchunk_subset) =
        subchunk_chunk_and_local_subset(array, level, subchunk_indices)?;
    let options = array.codec_options();
    let partial_decoder = array.async_partial_decoder(&chunk_indices).await?;
    let local_subchunk_grid = partial_decoder
        .local_subchunk_grid_at_level(level, options)
        .await
        .map_err(ArrayError::CodecError)?
        .ok_or(ArrayError::MissingSubchunkGrid)?;
    let local_indices = enclosing_subchunk_indices(&local_subchunk_grid, &subchunk_subset)?;
    Ok((partial_decoder, local_indices))
}

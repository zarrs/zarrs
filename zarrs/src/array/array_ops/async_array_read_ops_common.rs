use futures::{StreamExt, TryStreamExt};

use crate::array::{ArrayError, ArrayIndices, ArrayOps, ArraySubset, ArraySubsetTraits, Indexer};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, CodecError, CodecOptions, InvalidNumberOfElementsError, Resources,
    copy_fill_value_into, decode_into_array_bytes_target,
};
use zarrs_storage::MaybeSync;

use super::super::array_bytes_internal::FixedDecodeBuffers;
use super::super::concurrency::concurrency_chunks_and_codec;
use super::array_read_ops_common::subset_in_output;
use super::recommended_codec_concurrency;

/// Chunk level `_into` retrieval, as used by [`retrieve_array_subset_into`].
///
/// This is the asynchronous counterpart of the `retrieve_chunk_into` and
/// `retrieve_partial_chunk_into` closure parameters of
/// [`retrieve_array_subset_into`](super::array_read_ops_common::retrieve_array_subset_into). A
/// trait is used rather than closures because the returned futures must be `Send` on non-`wasm32`
/// targets, which the `AsyncFn` traits cannot express.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub(super) trait AsyncRetrieveInto {
    async fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>;

    async fn retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError>;
}

pub(super) async fn retrieve_array_subset_into<A, R>(
    array: &A,
    retrieve: &R,
    array_subset: &dyn ArraySubsetTraits,
    output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
    chunk_concurrent_minimum: usize,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<(), ArrayError>
where
    A: ArrayOps + MaybeSync,
    R: AsyncRetrieveInto + MaybeSync + ?Sized,
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
                retrieve
                    .retrieve_chunk_into(chunk_indices, output_target, options, resources)
                    .await
            } else {
                retrieve
                    .retrieve_partial_chunk_into(
                        chunk_indices,
                        &array_subset.relative_to(chunk_subset.start())?,
                        output_target,
                        options,
                        resources,
                    )
                    .await
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
            // Chunks are placed by their position in the array subset, which needs a target with the array subset shape
            let mut buffers = (output_target.subset().shape() != array_subset.shape().as_ref())
                .then(|| {
                    FixedDecodeBuffers::new(array.data_type(), &array_subset.shape())
                        .expect("fixed data type")
                });
            let buffers_target = buffers
                .as_mut()
                .map(FixedDecodeBuffers::target)
                .transpose()?;
            let resources = &resources;
            retrieve_multi_chunk_fixed_into(
                array,
                array_subset,
                &chunks,
                chunk_concurrent_limit,
                buffers_target.as_ref().unwrap_or(output_target),
                |chunk_indices, chunk_subset, overlap, mut target| async move {
                    retrieve
                        .retrieve_partial_chunk_into(
                            &chunk_indices,
                            &overlap.relative_to(chunk_subset.start())?,
                            &mut target,
                            options,
                            resources,
                        )
                        .await
                },
            )
            .await?;
            drop(buffers_target);
            if let Some(buffers) = buffers {
                // SAFETY: the chunks cover the array subset, so every element has been written
                let bytes = unsafe { buffers.into_array_bytes() };
                decode_into_array_bytes_target(&bytes, output_target)?;
            }
            Ok(())
        }
    }
}

/// Retrieve `array_subset`, which spans `chunks`, into `output_target`.
///
/// This is the asynchronous counterpart of
/// [`retrieve_multi_chunk_fixed_into`](super::array_read_ops_common::retrieve_multi_chunk_fixed_into).
/// The arguments of `retrieve_chunk_overlap_into` are owned so that its future needs no boxing.
pub(super) async fn retrieve_multi_chunk_fixed_into<'a, A, F, Fut>(
    array: &A,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    output_target: &ArrayBytesDecodeIntoTarget<'a>,
    retrieve_chunk_overlap_into: F,
) -> Result<(), ArrayError>
where
    A: ArrayOps + MaybeSync,
    F: Fn(ArrayIndices, ArraySubset, ArraySubset, ArrayBytesDecodeIntoTarget<'a>) -> Fut,
    Fut: Future<Output = Result<(), ArrayError>>,
{
    let array_subset_start = array_subset.start();
    let output_start = output_target.subset().start();

    let retrieve_chunk = |chunk_indices: ArrayIndices| {
        let array_subset_start = &array_subset_start;
        let retrieve_chunk_overlap_into = &retrieve_chunk_overlap_into;
        async move {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
            let chunk_subset_in_view =
                subset_in_output(&chunk_subset_overlap, array_subset_start, output_start)?;

            // SAFETY: chunks represent disjoint array subsets
            let target = unsafe { output_target.subdivide(chunk_subset_in_view) }?;

            retrieve_chunk_overlap_into(chunk_indices, chunk_subset, chunk_subset_overlap, target)
                .await
        }
    };

    futures::stream::iter(&chunks.indices())
        .map(Ok)
        .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
        .await?;

    Ok(())
}

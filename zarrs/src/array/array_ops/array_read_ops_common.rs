use std::num::NonZeroU64;

use crate::IntoConcurrentLimitIterator;
use crate::array::{
    Array, ArrayBytes, ArrayError, ArrayIndices, ArrayOps, ArraySubset, ArraySubsetTraits,
    ChunkShapeTraits, Indexer,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayToBytesCodecTraits, CodecError, CodecOptions, CowBytes,
    InvalidNumberOfElementsError, Resources, copy_fill_value_into, decode_into_array_bytes_target,
};
use zarrs_storage::{Bytes, MaybeSend, MaybeSync};

use super::super::array_bytes_internal::FixedDecodeBuffers;
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
            // Chunks are placed by their position in the array subset, which needs a target with the array subset shape
            let mut buffers = (output_target.shape() != array_subset.shape().as_ref()).then(|| {
                FixedDecodeBuffers::new(array.data_type(), &array_subset.shape())
                    .expect("fixed data type")
            });
            let buffers_target = buffers
                .as_mut()
                .map(FixedDecodeBuffers::target)
                .transpose()?;
            retrieve_multi_chunk_fixed_into(
                array,
                array_subset,
                &chunks,
                chunk_concurrent_limit,
                buffers_target.as_ref().unwrap_or(output_target),
                options,
                &resources,
                &|chunk_indices, chunk_subset, overlap, target, options, resources| {
                    retrieve_partial_chunk_into(
                        chunk_indices,
                        &overlap.relative_to(chunk_subset.start())?,
                        target,
                        options,
                        resources,
                    )
                },
            )?;
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
/// `retrieve_chunk_overlap_into` retrieves the overlap of a chunk with `array_subset` into a subdivision of `output_target`, given the chunk indices, the chunk subset, and the overlap (in array coordinates).
#[expect(clippy::too_many_arguments)]
pub(super) fn retrieve_multi_chunk_fixed_into<A, RetrieveChunkOverlapInto>(
    array: &A,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    output_target: &ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
    resources: &Resources,
    retrieve_chunk_overlap_into: &RetrieveChunkOverlapInto,
) -> Result<(), ArrayError>
where
    A: ArrayOps + MaybeSync,
    RetrieveChunkOverlapInto: for<'a> Fn(
            &[u64],
            &ArraySubset,
            &ArraySubset,
            &mut ArrayBytesDecodeIntoTarget<'a>,
            &CodecOptions,
            &Resources,
        ) -> Result<(), ArrayError>
        + MaybeSend
        + MaybeSync,
{
    let array_subset_start = array_subset.start();

    let retrieve_chunk = |chunk_indices: ArrayIndices| {
        let chunk_subset = array.chunk_subset(&chunk_indices)?;
        let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
        let chunk_subset_in_view = chunk_subset_overlap.relative_to(&array_subset_start)?;

        // SAFETY: chunks represent disjoint array subsets.
        let mut target = unsafe { output_target.subdivide(chunk_subset_in_view) }?;
        retrieve_chunk_overlap_into(
            &chunk_indices,
            &chunk_subset,
            &chunk_subset_overlap,
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

impl<TStorage: ?Sized> Array<TStorage> {
    /// Decode the encoded chunk `chunk_encoded` with `chunk_shape`.
    pub(super) fn decode_chunk(
        &self,
        chunk_encoded: Bytes,
        chunk_shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        let bytes = self
            .codecs_bound()
            .decode(
                CowBytes::Shared(chunk_encoded),
                chunk_shape,
                options,
                resources.compute(),
            )
            .map_err(ArrayError::CodecError)?;
        bytes.validate(chunk_shape.num_elements_u64(), self.data_type())?;
        Ok(bytes.into_owned())
    }

    /// Decode the encoded chunk `chunk_encoded` with `chunk_shape` into `output_target`.
    pub(super) fn decode_chunk_into(
        &self,
        chunk_encoded: Bytes,
        chunk_shape: &[NonZeroU64],
        output_target: &mut ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.codecs_bound()
            .decode_into(
                CowBytes::Shared(chunk_encoded).into(),
                chunk_shape,
                output_target,
                options,
                resources.compute(),
            )
            .map_err(ArrayError::CodecError)
    }
}

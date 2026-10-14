use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_data_type::FillValue;

use super::{ShardingCodecOptions, ShardingIndexLocation, calculate_chunks_per_shard};
use crate::array::array_bytes_internal::{
    FixedDecodeBuffers, build_nested_optional_target, extract_target_views, fill_target,
    merge_chunks_vlen, optional_innermost,
};
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::CodecChain;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesRaw, ArrayIndicesTinyVec, ArraySubset,
    ArraySubsetTraits, ChunkShape, ChunkShapeTraits, DataType, IncompatibleDimensionalityError,
    Indexer, ravel_indices,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
    AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits, CodecError, CodecOptions,
    InvalidNumberOfElementsError, decode_into_array_bytes_target,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Asynchronous partial decoder for the sharding codec.
pub(crate) struct AsyncShardingPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: DataType,
    fill_value: FillValue,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl AsyncShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub(crate) async fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        data_type: DataType,
        fill_value: FillValue,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<AsyncShardingPartialDecoder, CodecError> {
        let shard_index = super::decode_shard_index_async_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )
        .await?;

        Ok(Self {
            input_handle,
            data_type,
            fill_value,
            shard_shape,
            subchunk_shape,
            inner_codecs,
            shard_index,
            sharding_options,
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
    pub(crate) async fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
                .await
        } else {
            Ok(None)
        }
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) async fn partial_decode(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    indexer.validate(bytemuck::must_cast_slice(shard_shape))?;
    if let Some(subset) = indexer.as_array_subset()
        && let Some(mut buffers) = FixedDecodeBuffers::new(data_type, &subset.shape())
    {
        {
            let (mut data, mut masks) = buffers.views()?;
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
                build_nested_optional_target(&mut data, &mut masks),
            )
            .await?;
        }
        // SAFETY: the disjoint decode above initializes every selected element and mask.
        return Ok(unsafe { buffers.into_array_bytes() });
    }
    if data_type.is_optional() || indexer.as_array_subset().is_none() {
        return partial_decode_indexer(
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
        .await;
    }

    let subset = indexer.as_array_subset().expect("rectangular indexer");
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
    .await
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
            + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    async fn partial_decode(
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
        .await
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        indexer.validate(bytemuck::must_cast_slice(&self.shard_shape))?;
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }
        if optional_innermost(&self.data_type).is_fixed()
            && let Some(subset) = indexer.as_array_subset()
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
                output_target,
            )
            .await
        } else {
            let decoded = self.partial_decode(indexer, options).await?;
            decode_into_array_bytes_target(&decoded, output_target)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[expect(clippy::too_many_arguments)]
async fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
    inner_codecs
        .clone()
        .async_partial_decoder(
            Arc::new(AsyncByteIntervalPartialDecoder::new(
                input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            subchunk_shape,
            data_type,
            fill_value,
            options,
        )
        .await
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

#[allow(clippy::too_many_lines)]
#[expect(clippy::too_many_arguments)]
async fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    output_target: ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), CodecError> {
    if array_subset.len() != output_target.num_elements() {
        return Err(InvalidNumberOfElementsError::new(
            array_subset.len(),
            output_target.num_elements(),
        )
        .into());
    }
    let Some(shard_index) = shard_index else {
        return fill_target(output_target, data_type, fill_value);
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

    let (output_view, mask_views) = extract_target_views(&output_target);
    let array_subset_start = array_subset.start();
    let options = &options;
    let chunks_per_shard = &chunks_per_shard;
    let mask_views = &mask_views;
    let shard_chunk_grid = &shard_chunk_grid;
    let array_subset_start = &array_subset_start;
    let decode_subchunk_subset_into_slice = |chunk_indices: ArrayIndicesTinyVec| async move {
        let shard_index_idx =
            ravel_indices(&chunk_indices, chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;
        // Calculate the chunk's position in the output view coordinate space
        let chunk_relative = chunk_subset_overlap.relative_to(array_subset_start)?;
        let chunk_output_overlap_subset = chunk_relative.offset(output_view.subset().start())?;
        // SAFETY: chunks represent disjoint array subsets
        let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
            unsafe { output_view.subdivide(chunk_output_overlap_subset.clone())? };
        let mut mask_views = mask_views
            .iter()
            .map(|view| unsafe {
                // SAFETY: these subchunks select disjoint regions of each mask.
                view.subdivide(chunk_output_overlap_subset.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let target = build_nested_optional_target(&mut subchunk_view, &mut mask_views);
        if offset == u64::MAX && size == u64::MAX {
            fill_target(target, data_type, fill_value)
        } else {
            // Partially decode the subchunk
            let inner_partial_decoder = get_subchunk_partial_decoder(
                input_handle,
                data_type,
                fill_value,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            )
            .await?;
            inner_partial_decoder
                .partial_decode_into(
                    &chunk_subset_overlap
                        .relative_to(chunk_subset.start())
                        .unwrap(),
                    target,
                    options,
                )
                .await
        }
    };

    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    use futures::{StreamExt, TryStreamExt};
    futures::stream::iter(
        chunks
            .indices()
            .into_iter()
            .map(decode_subchunk_subset_into_slice),
    )
    .buffered(subchunk_concurrent_limit.max(1))
    .try_collect::<Vec<_>>()
    .await?;
    Ok(())
}

#[expect(clippy::too_many_arguments)]
async fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
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
                let inner_partial_decoder = get_subchunk_partial_decoder(
                    input_handle,
                    data_type,
                    fill_value,
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
    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    let chunk_bytes_and_subsets =
        futures::future::try_join_all(chunks.indices().into_iter().map(|chunk_indices| {
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality");
            let decode = &decode_subchunk_subset;
            decode(chunk_indices, chunk_subset)
        }))
        .await?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, &array_subset.shape());
    Ok(ArrayBytes::Variable(out_array_subset))
}

#[expect(clippy::too_many_arguments)]
async fn partial_decode_indexer(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let groups = super::group_indices_by_subchunk(&chunks_per_shard, subchunk_shape, indexer);
    let (limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        data_type,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;
    use futures::{StreamExt, TryStreamExt};
    let options = &options;
    let decode = |(subchunk_index, group): (u64, super::SubchunkIndices)| async move {
        let bytes = if let Some((offset, size)) =
            super::subchunk_offset_size(shard_index, usize::try_from(subchunk_index).unwrap())
        {
            let decoder = get_subchunk_partial_decoder(
                input_handle,
                data_type,
                fill_value,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            )
            .await?;
            decoder.partial_decode(&group, options).await?.into_owned()
        } else {
            ArrayBytes::new_fill_value(data_type, group.len(), fill_value)?
        };
        Ok::<_, CodecError>((group.positions, bytes))
    };
    let decoded = futures::stream::iter(groups.into_iter().map(decode))
        .buffered(limit.max(1))
        .try_collect::<Vec<_>>()
        .await?;
    super::merge_indexer_subchunks(decoded, usize::try_from(indexer.len()).unwrap(), data_type)
}

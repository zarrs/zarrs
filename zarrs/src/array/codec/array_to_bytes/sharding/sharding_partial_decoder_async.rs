use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_data_type::FillValue;

use super::{
    ShardingCodecOptions, ShardingIndexLocation, nested_local_subchunk_grids, subchunk_grid,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::array_bytes_internal::{
    FixedDecodeBuffers, build_nested_optional_target, extract_target_views, fill_target,
    merge_chunks, optional_innermost,
};
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec, ArraySubset, ArraySubsetTraits,
    ChunkGrid, ChunkShape, CodecChainBound, CowBytes, DataType, Indexer, ravel_indices,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayToBytesCodecTraits,
    AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits,
    AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits, CodecError, CodecOptions,
    InvalidNumberOfElementsError, decode_into_array_bytes_target,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Asynchronous partial decoder for the sharding codec.
pub struct AsyncShardingPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: RegularBoundedChunkGrid,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl AsyncShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub async fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: &CodecChainBound,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
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
    pub async fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<CowBytes<'_>>, CodecError> {
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

pub(crate) async fn partial_decode(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    indexer.validate(subchunk_grid.array_shape())?;

    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(subset) = indexer.as_array_subset() else {
        return partial_decode_indexer(
            input_handle,
            subchunk_grid,
            subchunk_shape,
            inner_codecs,
            shard_index,
            indexer,
            options,
        )
        .await;
    };

    if !data_type.is_optional() && data_type.is_fixed() {
        return partial_decode_fixed_array_subset(
            input_handle,
            subchunk_grid,
            inner_codecs,
            shard_index,
            subset,
            options,
        )
        .await;
    }
    // Optional data with fixed length inner data: decode each subchunk directly into the output
    if let Some(mut buffers) = FixedDecodeBuffers::new(data_type, &subset.shape()) {
        {
            let (mut data_view, mut mask_views) = buffers.views()?;
            partial_decode_fixed_array_subset_into(
                input_handle,
                subchunk_grid,
                subchunk_shape,
                inner_codecs,
                shard_index,
                subset,
                options,
                build_nested_optional_target(&mut data_view, &mut mask_views),
            )
            .await?;
        }
        // SAFETY: every element of the output (and masks) is written by `partial_decode_fixed_array_subset_into`
        Ok(unsafe { buffers.into_array_bytes() })
    } else {
        partial_decode_merged_array_subset(
            input_handle,
            data_type,
            fill_value,
            subchunk_grid,
            inner_codecs,
            shard_index,
            subset,
            options,
        )
        .await
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderSubchunkingTraits for AsyncShardingPartialDecoder {
    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        nested_local_subchunk_grids(
            ChunkGrid::new(self.subchunk_grid.clone()),
            &self.inner_codecs,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
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
            &self.subchunk_grid,
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
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }
        let data_type = self.inner_codecs.data_type();
        let fixed = optional_innermost(data_type).is_fixed();
        if fixed {
            indexer.validate(self.subchunk_grid.array_shape())?;
        }
        match (fixed, indexer.as_array_subset(), output_target) {
            // Fixed length data (including optional data with fixed length inner data)
            (true, Some(subset), output_target) => {
                partial_decode_fixed_array_subset_into(
                    &self.input_handle,
                    &self.subchunk_grid,
                    &self.subchunk_shape,
                    &self.inner_codecs,
                    self.shard_index.as_deref(),
                    subset,
                    options,
                    output_target,
                )
                .await
            }
            (true, None, ArrayBytesDecodeIntoTarget::Fixed(output_view))
                if !data_type.is_optional() =>
            {
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
                .await
            }
            (_, _, output_target) => {
                let decoded_value = self.partial_decode(indexer, options).await?;
                decode_into_array_bytes_target(&decoded_value, output_target)
            }
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

async fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
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

#[expect(clippy::too_many_arguments)]
async fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    output_target: ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
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
    // Optional data is decoded into views of its inner data and each validity mask
    let (output_view, mask_views) = extract_target_views(&output_target);
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        super::num_subchunks(subchunk_grid),
        options,
    )?;
    let array_subset_start = array_subset.start();
    let chunks = subchunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let mut subchunks = Vec::with_capacity(chunks.num_elements_usize());
    for chunk_indices in chunks.indices() {
        let shard_index_idx =
            ravel_indices(&chunk_indices, subchunk_grid.grid_shape()).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset_size = super::subchunk_offset_size(shard_index, shard_index_idx);
        let chunk_subset = subchunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;
        let chunk_relative = chunk_subset_overlap.relative_to(&array_subset_start)?;
        let chunk_output_overlap_subset = chunk_relative.offset(output_view.subset().start())?;
        subchunks.push((
            offset_size,
            chunk_subset,
            chunk_subset_overlap,
            chunk_output_overlap_subset,
        ));
    }

    use futures::{StreamExt, TryStreamExt};
    let decoded_subchunks = futures::stream::iter(subchunks)
        .map(
            |(offset_size, chunk_subset, chunk_subset_overlap, output_subset)| {
                let options = &options;
                async move {
                    if let Some((offset, size)) = offset_size {
                        let inner_partial_decoder = get_subchunk_partial_decoder(
                            input_handle,
                            &chunk_subset.chunk_shape().expect("nonempty subchunk"),
                            inner_codecs,
                            options,
                            offset,
                            size,
                        )
                        .await?;
                        let decoded = inner_partial_decoder
                            .partial_decode(
                                &chunk_subset_overlap
                                    .relative_to(chunk_subset.start())
                                    .unwrap(),
                                options,
                            )
                            .await?
                            .into_owned();
                        Ok((Some(decoded), output_subset))
                    } else {
                        Ok::<_, CodecError>((None, output_subset))
                    }
                }
            },
        )
        .buffer_unordered(subchunk_concurrent_limit)
        .try_collect::<Vec<_>>()
        .await?;

    for (decoded, output_subset) in decoded_subchunks {
        // SAFETY: subchunks are disjoint array subsets and each view is dropped before the next.
        let mut subchunk_view = unsafe { output_view.subdivide(output_subset.clone())? };
        let mut subchunk_mask_views = mask_views
            .iter()
            .map(|mask_view| unsafe { mask_view.subdivide(output_subset.clone()) })
            .collect::<Result<Vec<_>, _>>()?;
        let subchunk_target =
            build_nested_optional_target(&mut subchunk_view, &mut subchunk_mask_views);
        if let Some(decoded) = decoded {
            decode_into_array_bytes_target(&decoded, subchunk_target)?;
        } else {
            fill_target(subchunk_target, data_type, fill_value)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
async fn partial_decode_fixed_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    // Find filled / non filled chunks
    let chunk_info = subchunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard")
        .indices()
        .into_iter()
        .map(|chunk_indices: ArrayIndicesTinyVec| {
            let chunk_index =
                ravel_indices(&chunk_indices, subchunk_grid.grid_shape()).expect("inbounds chunk");
            let chunk_index = usize::try_from(chunk_index).unwrap();

            let chunk_subset = subchunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");

            // Read the offset/size
            (
                chunk_subset,
                super::subchunk_offset_size(shard_index, chunk_index),
            )
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
                    let inner_partial_decoder = get_subchunk_partial_decoder(
                        input_handle,
                        &chunk_subset.chunk_shape().expect("nonempty subchunk"),
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
                        .into_vec();
                    Ok::<_, CodecError>((decoded_chunk, chunk_subset_overlap))
                }
            }),
    )
    .await;
    // FIXME: Concurrency limit for futures

    let array_subset_start = array_subset.start();
    let array_subset_shape = array_subset.shape();

    if !results.is_empty() {
        results
            .concurrent_limit(options.concurrent_target())
            .try_for_each(|subset_and_decoded_chunk| {
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
            })?;
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
        filled_chunks
            .concurrent_limit(options.concurrent_target())
            .try_for_each(|chunk_subset: &ArraySubset| {
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
            })?;
    }
    unsafe { shard.set_len(shard_size) };
    Ok(ArrayBytes::from(shard))
}

#[expect(clippy::too_many_arguments)]
/// Partially decode an array subset by decoding the overlapping region of each subchunk and merging them.
///
/// This supports any data type, including variable length and optional data types.
async fn partial_decode_merged_array_subset(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    subchunk_grid: &RegularBoundedChunkGrid,
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let array_subset_start = array_subset.start();
    let decode_subchunk_subset = |chunk_indices: ArrayIndicesTinyVec, chunk_subset: ArraySubset| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, subchunk_grid.grid_shape()).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let array_subset_start = &array_subset_start;
        async move {
            let offset_size = super::subchunk_offset_size(shard_index, shard_index_idx);

            // Get the subset of bytes from the chunk which intersect the array
            let chunk_subset_overlap = array_subset.overlap(&chunk_subset).unwrap(); // FIXME: unwrap

            let chunk_subset_bytes = if let Some((offset, size)) = offset_size {
                // Partially decode the subchunk
                let inner_partial_decoder = get_subchunk_partial_decoder(
                    input_handle,
                    &chunk_subset.chunk_shape().expect("nonempty subchunk"),
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
            } else {
                ArrayBytes::new_fill_value(
                    data_type,
                    chunk_subset_overlap.num_elements(),
                    fill_value,
                )?
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
    let chunks = subchunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let chunk_bytes_and_subsets =
        futures::future::try_join_all(chunks.indices().into_iter().map(|chunk_indices| {
            let chunk_subset = subchunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");
            let decode = &decode_subchunk_subset;
            decode(chunk_indices, chunk_subset)
        }))
        .await?;

    // Convert into an array
    merge_chunks(chunk_bytes_and_subsets, &array_subset.shape(), data_type)
}

async fn partial_decode_indexer(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
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
    use futures::{StreamExt, TryStreamExt};

    let options = &options;
    let decoded = futures::stream::iter(groups)
        .map(|(subchunk_index, group)| async move {
            let shard_index_idx = usize::try_from(subchunk_index).unwrap();
            let bytes = if let Some((offset, size)) =
                super::subchunk_offset_size(shard_index, shard_index_idx)
            {
                let decoder = get_subchunk_partial_decoder(
                    input_handle,
                    &group.subchunk_shape,
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
            Ok::<_, CodecError>((bytes, group.positions))
        })
        .buffer_unordered(subchunk_concurrent_limit)
        .try_collect::<Vec<_>>()
        .await?;
    super::merge_indexer_subchunks(decoded, usize::try_from(indexer.len()).unwrap(), data_type)
}

/// Decode the elements of `indexer` into `output_view` by scattering each subchunk group directly.
///
/// The `indexer` must be validated against the shard shape.
#[expect(clippy::too_many_arguments)]
async fn partial_decode_fixed_indexer_into<'a>(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
    output_view: &'a mut ArrayBytesFixedDisjointView<'a>,
) -> Result<(), CodecError> {
    use futures::{StreamExt, TryStreamExt};

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
    let options = &options;
    let subchunk_decoder = async |subchunk_index: u64, subchunk_shape: &[NonZeroU64]| {
        let shard_index_idx = usize::try_from(subchunk_index).unwrap();
        if let Some((offset, size)) = super::subchunk_offset_size(shard_index, shard_index_idx) {
            get_subchunk_partial_decoder(
                input_handle,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            )
            .await
            .map(Some)
        } else {
            Ok(None)
        }
    };

    if groups.len() == 1 {
        // Positions within a group are ascending and cover every element, so decode in place
        let (subchunk_index, group) = groups.into_iter().next().expect("one group");
        return if let Some(decoder) =
            subchunk_decoder(subchunk_index, &group.subchunk_shape).await?
        {
            decoder
                .partial_decode_into(
                    &group,
                    ArrayBytesDecodeIntoTarget::Fixed(output_view),
                    options,
                )
                .await
        } else {
            output_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from)
        };
    }

    futures::stream::iter(groups)
        .map(|(subchunk_index, group)| async move {
            let bytes = if let Some(decoder) =
                subchunk_decoder(subchunk_index, &group.subchunk_shape).await?
            {
                Some(
                    decoder
                        .partial_decode(&group, options)
                        .await?
                        .into_owned()
                        .into_fixed()?,
                )
            } else {
                None
            };
            Ok::<_, CodecError>((group.positions, bytes))
        })
        .buffer_unordered(subchunk_concurrent_limit)
        .try_for_each(|(positions, bytes)| {
            let result = if let Some(bytes) = bytes {
                output_view.copy_elements_from_slice(&positions, &bytes)
            } else {
                output_view.fill_elements(&positions, fill_value.as_ne_bytes())
            };
            futures::future::ready(result)
        })
        .await
}

use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_data_type::FillValue;

use super::{ShardingIndexLocation, calculate_chunks_per_shard};
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::CodecChain;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesOffsets, ArrayBytesRaw, ArrayIndices,
    ArrayIndicesTinyVec, ArraySubset, ArraySubsetTraits, ChunkShape, ChunkShapeTraits, DataType,
    DataTypeSize, IncompatibleDimensionalityError, Indexer, IndexerError, ravel_indices,
};
use zarrs_codec::{
    ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits, AsyncByteIntervalPartialDecoder,
    AsyncBytesPartialDecoderTraits, CodecError, CodecOptions,
};
use zarrs_plugin::ExtensionAliasesV3;
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
        DataTypeSize::Fixed(_data_type_size) => {
            if let Some(subset) = indexer.as_array_subset() {
                partial_decode_fixed_array_subset(
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
                .await
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
                .await
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
                .await
            }
        }
    }
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
async fn partial_decode_fixed_array_subset(
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
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
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
        .indices()
        .into_iter()
        .map(|chunk_indices: ArrayIndicesTinyVec| {
            let chunk_index =
                ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
            let chunk_index = usize::try_from(chunk_index).unwrap();

            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality");

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
                    let inner_partial_decoder = get_subchunk_partial_decoder(
                        input_handle,
                        data_type,
                        fill_value,
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
async fn partial_decode_fixed_indexer(
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
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
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
    let subchunk_partial_decoders = moka::future::Cache::new(chunks_per_shard.iter().product());
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
            .or_try_insert_with(get_subchunk_partial_decoder(
                input_handle,
                data_type,
                fill_value,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            ))
            .await
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = subchunk_partial_decoders
            .get_or_insert_async(&chunk_index_1d, async {
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
                .await
            })
            .await?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let element_bytes = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)
            .await?
            .into_fixed()
            .expect("fixed data");
        output.extend_from_slice(&element_bytes);
    }

    debug_assert_eq!(output.len(), output_len);

    Ok(output.into())
}

#[expect(clippy::too_many_arguments)]
async fn partial_decode_variable_indexer(
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
    let subchunk_partial_decoders = moka::future::Cache::new(chunks_per_shard.iter().product());
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
            .or_try_insert_with(get_subchunk_partial_decoder(
                input_handle,
                data_type,
                fill_value,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            ))
            .await
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = subchunk_partial_decoders
            .get_or_insert_async(&chunk_index_1d, async {
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
                .await
            })
            .await?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let (element_bytes, element_offsets) = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)
            .await?
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

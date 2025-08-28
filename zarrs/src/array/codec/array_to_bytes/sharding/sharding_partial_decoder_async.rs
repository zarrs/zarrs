use std::sync::Arc;

use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

use crate::{
    array::{
        array_bytes::merge_chunks_vlen,
        chunk_grid::RegularChunkGrid,
        codec::{
            ArraySubset, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
            AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits, CodecChain,
            CodecError, CodecOptions,
        },
        ravel_indices, ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndices, ArraySize,
        ChunkRepresentation, ChunkShape, DataType, DataTypeSize, RawBytes, RawBytesOffsets,
    },
    array_subset::IncompatibleDimensionalityError,
    indexer::{IncompatibleIndexerError, Indexer},
};

use super::{calculate_chunks_per_shard, ShardingIndexLocation};

/// Asynchronous partial decoder for the sharding codec.
pub(crate) struct AsyncShardingPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    shard_representation: ChunkRepresentation,
    chunk_representation: ChunkRepresentation,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
}

impl AsyncShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    pub(crate) async fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shard_representation: ChunkRepresentation,
        chunk_shape: &ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
    ) -> Result<AsyncShardingPartialDecoder, CodecError> {
        let shard_index = super::decode_shard_index_async_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            chunk_shape,
            &shard_representation,
            options,
        )
        .await?;
        let chunk_representation = unsafe {
            ChunkRepresentation::new_unchecked(
                chunk_shape.to_vec(),
                shard_representation.data_type().clone(),
                shard_representation.fill_value().clone(),
            )
        };

        Ok(Self {
            input_handle,
            shard_representation,
            chunk_representation,
            inner_codecs,
            shard_index,
        })
    }

    /// Retrieve the byte range of an encoded inner chunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub(crate) fn inner_chunk_byte_range(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, CodecError> {
        super::inner_chunk_byte_range(
            self.shard_index.as_deref(),
            self.shard_representation.shape(),
            self.chunk_representation.shape(),
            chunk_indices,
        )
    }

    /// Retrieve the encoded bytes of an inner chunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub(crate) async fn retrieve_inner_chunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<RawBytes<'_>>, CodecError> {
        let byte_range = self.inner_chunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode_concat(&mut [byte_range].into_iter(), &CodecOptions::default())
                .await
        } else {
            Ok(None)
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.shard_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if indexer.dimensionality() != self.shard_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indexer.dimensionality(),
                self.shard_representation.dimensionality(),
            )
            .into());
        }

        match self.shard_representation.element_size() {
            DataTypeSize::Fixed(data_type_size) => {
                if let Some(subset) = indexer.as_array_subset() {
                    partial_decode_fixed_array_subset(self, subset, options, data_type_size).await
                } else {
                    partial_decode_fixed_indexer(self, indexer, options, data_type_size).await
                }
            }
            DataTypeSize::Variable => {
                if let Some(subset) = indexer.as_array_subset() {
                    partial_decode_variable_array_subset(self, subset, options).await
                } else {
                    partial_decode_variable_indexer(self, indexer, options).await
                }
            }
        }
    }
}

async fn get_inner_chunk_partial_decoder(
    partial_decoder: &AsyncShardingPartialDecoder,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
    partial_decoder
        .inner_codecs
        .clone()
        .async_partial_decoder(
            Arc::new(AsyncByteIntervalPartialDecoder::new(
                partial_decoder.input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            &partial_decoder.chunk_representation,
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
async fn partial_decode_fixed_array_subset(
    partial_decoder: &AsyncShardingPartialDecoder,
    array_subset: &ArraySubset,
    options: &CodecOptions,
    data_type_size: usize,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = &partial_decoder.shard_index else {
        return Ok(super::partial_decode_empty_shard(
            &partial_decoder.shard_representation,
            array_subset,
        ));
    };
    let chunks_per_shard = calculate_chunks_per_shard(
        partial_decoder.shard_representation.shape(),
        partial_decoder.chunk_representation.shape(),
    )?
    .to_array_shape();

    let shard_chunk_grid = RegularChunkGrid::new(
        partial_decoder.shard_representation.shape_u64(),
        partial_decoder.chunk_representation.shape().into(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    // Find filled / non filled chunks
    let chunk_info = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .indices()
        .into_iter()
        .map(|chunk_indices: Vec<u64>| {
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
                    let inner_partial_decoder =
                        get_inner_chunk_partial_decoder(partial_decoder, options, *offset, *size)
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
                            partial_decoder.shard_representation.data_type(),
                        )?
                        .into_fixed()?
                        .into_owned();
                    Ok::<_, CodecError>((decoded_chunk, chunk_subset_overlap))
                }
            }),
    )
    .await;
    // FIXME: Concurrency limit for futures

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
                        array_subset.shape(),
                        chunk_subset_overlap
                            .relative_to(array_subset.start())
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
                        array_subset.shape(),
                        chunk_subset_overlap
                            .relative_to(array_subset.start())
                            .unwrap(),
                    )?
                };
                output_view
                    .fill(
                        partial_decoder
                            .shard_representation
                            .fill_value()
                            .as_ne_bytes(),
                    )
                    .map_err(CodecError::from)
            }
        )?;
    }
    unsafe { shard.set_len(shard_size) };
    Ok(ArrayBytes::from(shard))
}

async fn partial_decode_variable_array_subset(
    partial_decoder: &AsyncShardingPartialDecoder,
    array_subset: &ArraySubset,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = &partial_decoder.shard_index else {
        return Ok(super::partial_decode_empty_shard(
            &partial_decoder.shard_representation,
            array_subset,
        ));
    };
    let chunks_per_shard = calculate_chunks_per_shard(
        partial_decoder.shard_representation.shape(),
        partial_decoder.chunk_representation.shape(),
    )?
    .to_array_shape();

    let shard_chunk_grid = RegularChunkGrid::new(
        partial_decoder.shard_representation.shape_u64(),
        partial_decoder.chunk_representation.shape().into(),
    )
    .expect("matching dimensionality");

    let decode_inner_chunk_subset = |chunk_indices: Vec<u64>, chunk_subset| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let chunk_representation = partial_decoder.chunk_representation.clone();
        async move {
            let offset = shard_index[shard_index_idx * 2];
            let size = shard_index[shard_index_idx * 2 + 1];

            // Get the subset of bytes from the chunk which intersect the array
            let chunk_subset_overlap = array_subset.overlap(&chunk_subset).unwrap(); // FIXME: unwrap

            let chunk_subset_bytes = if offset == u64::MAX && size == u64::MAX {
                let array_size = ArraySize::new(
                    chunk_representation.data_type().size(),
                    chunk_subset_overlap.num_elements(),
                );
                ArrayBytes::new_fill_value(array_size, chunk_representation.fill_value())
            } else {
                // Partially decode the inner chunk
                let inner_partial_decoder =
                    get_inner_chunk_partial_decoder(partial_decoder, options, offset, size).await?;
                inner_partial_decoder
                    .partial_decode(
                        &chunk_subset_overlap
                            .relative_to(chunk_subset.start())
                            .unwrap(),
                        options,
                    )
                    .await?
                    .into_owned()
            };
            Ok::<_, CodecError>((
                chunk_subset_bytes,
                chunk_subset_overlap
                    .relative_to(array_subset.start())
                    .unwrap(),
            ))
        }
    };

    // Decode the inner chunk subsets
    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    let chunk_bytes_and_subsets =
        futures::future::try_join_all(chunks.indices().into_iter().map(|chunk_indices| {
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality");
            decode_inner_chunk_subset(chunk_indices, chunk_subset)
        }))
        .await?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?;
    Ok(out_array_subset)
}

async fn partial_decode_fixed_indexer(
    partial_decoder: &AsyncShardingPartialDecoder,
    indexer: &dyn Indexer,
    options: &CodecOptions,
    data_type_size: usize,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = &partial_decoder.shard_index else {
        return Ok(super::partial_decode_empty_shard(
            &partial_decoder.shard_representation,
            indexer,
        ));
    };
    let chunks_per_shard = calculate_chunks_per_shard(
        partial_decoder.shard_representation.shape(),
        partial_decoder.chunk_representation.shape(),
    )?
    .to_array_shape();
    // let (inner_chunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
    //     &partial_decoder.inner_codecs,
    //     &partial_decoder.chunk_representation,
    //     &chunks_per_shard,
    //     options,
    // )?;
    let options = &options;

    let output_len = usize::try_from(indexer.len() * data_type_size as u64).unwrap();
    let mut output: Vec<u8> = Vec::with_capacity(output_len);

    #[cfg(not(target_arch = "wasm32"))]
    let inner_chunk_partial_decoders = moka::future::Cache::new(chunks_per_shard.iter().product());
    #[cfg(target_arch = "wasm32")]
    let inner_chunk_partial_decoders = quick_cache::sync::Cache::new(chunks_per_shard.iter().product::<u64>() as usize);

    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != partial_decoder.chunk_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indices.len(),
                partial_decoder.chunk_representation.dimensionality(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i / cs)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard).ok_or_else(|| {
            IncompatibleIndexerError::new_oob(chunk_index, chunks_per_shard.clone())
        })?;

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = {
            let inner_partial_decoder_entry = inner_chunk_partial_decoders
                .entry(chunk_index_1d)
                .or_try_insert_with(get_inner_chunk_partial_decoder(
                    partial_decoder,
                    options,
                    offset,
                    size,
                ))
                .await
                .map_err(Arc::unwrap_or_clone)?;
            inner_partial_decoder_entry.value()
        };
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = inner_chunk_partial_decoders
            .get_or_insert_async(&chunk_index_1d, async {
                get_inner_chunk_partial_decoder(partial_decoder, options, offset, size).await
            })
            .await?;

        // Get the element index
        let indices_in_inner_chunk: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let element_bytes = inner_partial_decoder
            .partial_decode(&[indices_in_inner_chunk], options)
            .await?
            .into_fixed()
            .expect("fixed data");
        output.extend_from_slice(&element_bytes);
    }

    debug_assert_eq!(output.len(), output_len);

    Ok(output.into())
}

async fn partial_decode_variable_indexer(
    partial_decoder: &AsyncShardingPartialDecoder,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = &partial_decoder.shard_index else {
        return Ok(super::partial_decode_empty_shard(
            &partial_decoder.shard_representation,
            indexer,
        ));
    };
    let chunks_per_shard = calculate_chunks_per_shard(
        partial_decoder.shard_representation.shape(),
        partial_decoder.chunk_representation.shape(),
    )?
    .to_array_shape();
    // let (inner_chunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
    //     &partial_decoder.inner_codecs,
    //     &partial_decoder.chunk_representation,
    //     &chunks_per_shard,
    //     options,
    // )?;
    let options = &options;

    let offsets_len = usize::try_from(indexer.len() + 1).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(offsets_len);
    offsets.push(0);
    
    #[cfg(not(target_arch = "wasm32"))]
    let inner_chunk_partial_decoders = moka::future::Cache::new(chunks_per_shard.iter().product());
    #[cfg(target_arch = "wasm32")]
    let inner_chunk_partial_decoders = quick_cache::sync::Cache::new(chunks_per_shard.iter().product::<u64>() as usize);
    
    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != partial_decoder.chunk_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indices.len(),
                partial_decoder.chunk_representation.dimensionality(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i / cs)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard).ok_or_else(|| {
            IncompatibleIndexerError::new_oob(chunk_index, chunks_per_shard.clone())
        })?;

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];
        
        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = {
            let inner_partial_decoder_entry = inner_chunk_partial_decoders
                .entry(chunk_index_1d)
                .or_try_insert_with(get_inner_chunk_partial_decoder(
                    partial_decoder,
                    options,
                    offset,
                    size,
                ))
                .await
                .map_err(Arc::unwrap_or_clone)?;

            inner_partial_decoder_entry.value()
        };
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = inner_chunk_partial_decoders
            .get_or_insert_async(&chunk_index_1d, async {
                get_inner_chunk_partial_decoder(partial_decoder, options, offset, size).await
            })
            .await?;

        // Get the element index
        let indices_in_inner_chunk: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let (element_bytes, element_offsets) = inner_partial_decoder
            .partial_decode(&[indices_in_inner_chunk], options)
            .await?
            .into_variable()
            .expect("fixed data");
        debug_assert_eq!(element_offsets.len(), 2);
        bytes.extend_from_slice(&element_bytes);
        offsets.push(bytes.len());
    }

    Ok(ArrayBytes::new_vlen(bytes, RawBytesOffsets::new(offsets)?)?)
}

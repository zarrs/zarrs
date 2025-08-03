use std::sync::Arc;

use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

use crate::array::{
    array_bytes::merge_chunks_vlen,
    chunk_grid::{ChunkGridTraits, RegularChunkGrid},
    codec::{
        ArraySubset, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
        AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits, CodecChain, CodecError,
        CodecOptions,
    },
    ravel_indices, ArrayBytes, ArrayBytesFixedDisjointView, ArraySize, ChunkRepresentation,
    ChunkShape, DataType, DataTypeSize, RawBytes,
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
                .partial_decode_concat(&[byte_range], &CodecOptions::default())
                .await
        } else {
            Ok(None)
        }
    }
}

#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.shard_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if indexer.dimensionality() != self.shard_representation.dimensionality() {
            return Err(CodecError::InvalidArraySubsetDimensionalityError(
                indexer.clone(),
                self.shard_representation.dimensionality(),
            ));
        }

        match self.shard_representation.element_size() {
            DataTypeSize::Fixed(data_type_size) => {
                partial_decode_fixed_array_subset(self, indexer, options, data_type_size).await
            }
            DataTypeSize::Variable => {
                partial_decode_variable_array_subset(self, indexer, options).await
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
    .expect("matching dimensionality");

    // Find filled / non filled chunks
    let chunk_info = shard_chunk_grid
        .chunks_in_array_subset(array_subset)
        .expect("matching dimensionality")
        .expect("chunks_in_array_subset is determinate for a regular chunk grid")
        .indices()
        .into_iter()
        .map(|chunk_indices: Vec<u64>| {
            let chunk_index = ravel_indices(&chunk_indices, &chunks_per_shard);
            let chunk_index = usize::try_from(chunk_index).unwrap();

            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("chunk subset is determinate for a regular chunk grid");

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
        rayon_iter_concurrent_limit::iter_concurrent_limit!(
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
        rayon_iter_concurrent_limit::iter_concurrent_limit!(
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
        let shard_index_idx: usize =
            usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2).unwrap();
        let chunk_representation = partial_decoder.chunk_representation.clone();
        async move {
            let offset = shard_index[shard_index_idx];
            let size = shard_index[shard_index_idx + 1];

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
    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)
        .expect("matching dimensionality")
        .expect("chunks_in_array_subset is determinate for a regular chunk grid");
    let chunk_bytes_and_subsets =
        futures::future::try_join_all(chunks.indices().into_iter().map(|chunk_indices| {
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("subset is determinate for a regular chunk grid");
            decode_inner_chunk_subset(chunk_indices, chunk_subset)
        }))
        .await?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?;
    Ok(out_array_subset)
}

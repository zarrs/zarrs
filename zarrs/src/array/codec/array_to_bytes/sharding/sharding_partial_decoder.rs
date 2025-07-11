use std::{num::NonZeroU64, sync::Arc};

use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_storage::byte_range::ByteRange;

use crate::array::{
    array_bytes::merge_chunks_vlen,
    codec::{
        ArrayCodecTraits, ArrayPartialDecoderTraits, ArraySubset, ArrayToBytesCodecTraits,
        ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecChain, CodecError,
        CodecOptions,
    },
    concurrency::{calc_concurrency_outer_inner, RecommendedConcurrency},
    ravel_indices, ArrayBytes, ArrayBytesFixedDisjointView, ArraySize, ChunkRepresentation,
    ChunkShape, DataType, DataTypeSize, RawBytes,
};

#[cfg(feature = "async")]
use crate::array::codec::{
    byte_interval_partial_decoder::AsyncByteIntervalPartialDecoder, AsyncArrayPartialDecoderTraits,
    AsyncBytesPartialDecoderTraits,
};

use super::{calculate_chunks_per_shard, ShardingIndexLocation};

/// Partial decoder for the sharding codec.
pub(crate) struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    chunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
}

fn inner_chunk_byte_range(
    shard_index: Option<&[u64]>,
    shard_shape: &[NonZeroU64],
    chunk_shape: &[NonZeroU64],
    chunk_indices: &[u64],
) -> Result<Option<ByteRange>, CodecError> {
    if let Some(shard_index) = shard_index {
        let chunks_per_shard = calculate_chunks_per_shard(shard_shape, chunk_shape)?;
        let chunks_per_shard = chunks_per_shard.to_array_shape();

        let shard_index_idx: usize =
            usize::try_from(ravel_indices(chunk_indices, &chunks_per_shard) * 2).unwrap();
        let offset = shard_index[shard_index_idx];
        let size = shard_index[shard_index_idx + 1];
        Ok(Some(ByteRange::new(offset..offset + size)))
    } else {
        Ok(None)
    }
}

impl ShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        chunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let shard_index = super::decode_shard_index_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            chunk_shape.as_slice(),
            &decoded_representation,
            options,
        )?;
        Ok(Self {
            input_handle,
            decoded_representation,
            chunk_shape,
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
        inner_chunk_byte_range(
            self.shard_index.as_deref(),
            self.decoded_representation.shape(),
            &self.chunk_shape,
            chunk_indices,
        )
    }

    /// Retrieve the encoded bytes of an inner chunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub(crate) fn retrieve_inner_chunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<RawBytes<'_>>, CodecError> {
        let byte_range = self.inner_chunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode_concat(&[byte_range], &CodecOptions::default())
        } else {
            Ok(None)
        }
    }
}

impl ArrayPartialDecoderTraits for ShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_handle.size() + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    #[allow(clippy::too_many_lines)]
    fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if indexer.dimensionality() != self.decoded_representation.dimensionality() {
            return Err(CodecError::InvalidArraySubsetDimensionalityError(
                indexer.clone(),
                self.decoded_representation.dimensionality(),
            ));
        }

        let Some(shard_index) = &self.shard_index else {
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                indexer.num_elements(),
            );
            return Ok(ArrayBytes::new_fill_value(
                array_size,
                self.decoded_representation.fill_value(),
            ));
        };

        let chunk_representation = unsafe {
            ChunkRepresentation::new_unchecked(
                self.chunk_shape.to_vec(),
                self.decoded_representation.data_type().clone(),
                self.decoded_representation.fill_value().clone(),
            )
        };

        let chunks_per_shard = calculate_chunks_per_shard(
            self.decoded_representation.shape(),
            chunk_representation.shape(),
        )?;
        let chunks_per_shard = chunks_per_shard.to_array_shape();
        let num_chunks = usize::try_from(chunks_per_shard.iter().product::<u64>()).unwrap();

        // Calculate inner chunk/codec concurrency
        let (inner_chunk_concurrent_limit, concurrency_limit_codec) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &RecommendedConcurrency::new_maximum(std::cmp::min(
                options.concurrent_target(),
                num_chunks,
            )),
            &self
                .inner_codecs
                .recommended_concurrency(&chunk_representation)?,
        );
        let options = options
            .into_builder()
            .concurrent_target(concurrency_limit_codec)
            .build();

        // let Some(array_subset) = indexer.as_array_subset() else {
        //     todo!("Support generic indexers")
        // };
        let array_subset = indexer;

        let chunks = array_subset.chunks(chunk_representation.shape())?;

        match self.decoded_representation.element_size() {
            DataTypeSize::Variable => {
                let decode_inner_chunk_subset = |(chunk_indices, chunk_subset): (
                    Vec<u64>,
                    ArraySubset,
                )| {
                    let shard_index_idx: usize =
                        usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2)
                            .unwrap();
                    let offset = shard_index[shard_index_idx];
                    let size = shard_index[shard_index_idx + 1];

                    // Get the subset of bytes from the chunk which intersect the array
                    let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;

                    let chunk_subset_bytes = if offset == u64::MAX && size == u64::MAX {
                        let array_size = ArraySize::new(
                            chunk_representation.data_type().size(),
                            chunk_subset_overlap.num_elements(),
                        );
                        ArrayBytes::new_fill_value(array_size, chunk_representation.fill_value())
                    } else {
                        // Partially decode the inner chunk
                        let partial_decoder = self.inner_codecs.clone().partial_decoder(
                            Arc::new(ByteIntervalPartialDecoder::new(
                                self.input_handle.clone(),
                                offset,
                                size,
                            )),
                            &chunk_representation,
                            &options,
                        )
                        .map_err(|err| if let CodecError::InvalidByteRangeError(_) = err {
                            CodecError::Other(
                                "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                                    .to_string(),
                            )
                        } else {
                            err
                        })?;
                        partial_decoder
                            .partial_decode(
                                &chunk_subset_overlap
                                    .relative_to(chunk_subset.start())
                                    .unwrap(),
                                &options,
                            )?
                            .into_owned()
                    };
                    Ok::<_, CodecError>((
                        chunk_subset_bytes,
                        chunk_subset_overlap
                            .relative_to(array_subset.start())
                            .unwrap(),
                    ))
                };
                // Decode the inner chunk subsets
                let chunk_bytes_and_subsets = rayon_iter_concurrent_limit::iter_concurrent_limit!(
                    inner_chunk_concurrent_limit,
                    chunks,
                    map,
                    decode_inner_chunk_subset
                )
                .collect::<Result<Vec<_>, _>>()?;

                // Convert into an array
                let out_array_subset =
                    merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?;
                Ok(out_array_subset)
            }
            DataTypeSize::Fixed(data_type_size) => {
                let array_subset_size = array_subset.num_elements_usize() * data_type_size;
                let mut out_array_subset = vec![0; array_subset_size];
                let out_array_subset_slice = UnsafeCellSlice::new(out_array_subset.as_mut_slice());

                let decode_inner_chunk_subset_into_slice = |(chunk_indices, chunk_subset): (
                    Vec<u64>,
                    ArraySubset,
                )| {
                    let shard_index_idx: usize =
                        usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2)
                            .unwrap();
                    let offset = shard_index[shard_index_idx];
                    let size = shard_index[shard_index_idx + 1];

                    // Get the subset of bytes from the chunk which intersect the array
                    let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;

                    let decoded_bytes = if offset == u64::MAX && size == u64::MAX {
                        let array_size = ArraySize::new(
                            chunk_representation.data_type().size(),
                            chunk_subset_overlap.num_elements(),
                        );
                        ArrayBytes::new_fill_value(array_size, chunk_representation.fill_value())
                    } else {
                        // Partially decode the inner chunk
                        let partial_decoder = self.inner_codecs.clone().partial_decoder(
                            Arc::new(ByteIntervalPartialDecoder::new(
                                self.input_handle.clone(),
                                offset,
                                size,
                            )),
                            &chunk_representation,
                            &options,
                        )
                        .map_err(|err| if let CodecError::InvalidByteRangeError(_) = err {
                            CodecError::Other(
                                "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                                    .to_string(),
                            )
                        } else {
                            err
                        })?;
                        partial_decoder
                            .partial_decode(
                                &chunk_subset_overlap
                                    .relative_to(chunk_subset.start())
                                    .unwrap(),
                                &options,
                            )?
                            .into_owned()
                    };
                    let decoded_bytes = decoded_bytes.into_fixed()?;
                    let mut output_view = unsafe {
                        // SAFETY: chunks represent disjoint array subsets
                        ArrayBytesFixedDisjointView::new(
                            out_array_subset_slice,
                            data_type_size,
                            array_subset.shape(),
                            chunk_subset_overlap
                                .relative_to(array_subset.start())
                                .unwrap(),
                        )?
                    };
                    output_view
                        .copy_from_slice(&decoded_bytes)
                        .map_err(CodecError::from)
                };

                rayon_iter_concurrent_limit::iter_concurrent_limit!(
                    inner_chunk_concurrent_limit,
                    chunks,
                    try_for_each,
                    decode_inner_chunk_subset_into_slice
                )?;
                Ok(ArrayBytes::from(out_array_subset))
            }
        }
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the sharding codec.
pub(crate) struct AsyncShardingPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    chunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
}

#[cfg(feature = "async")]
impl AsyncShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    pub(crate) async fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        chunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
    ) -> Result<AsyncShardingPartialDecoder, CodecError> {
        let shard_index = super::decode_shard_index_async_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            chunk_shape.as_slice(),
            &decoded_representation,
            options,
        )
        .await?;
        Ok(Self {
            input_handle,
            decoded_representation,
            chunk_shape,
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
        inner_chunk_byte_range(
            self.shard_index.as_deref(),
            self.decoded_representation.shape(),
            &self.chunk_shape,
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

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    #[allow(clippy::too_many_lines)]
    async fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if indexer.dimensionality() != self.decoded_representation.dimensionality() {
            return Err(CodecError::InvalidArraySubsetDimensionalityError(
                indexer.clone(),
                self.decoded_representation.dimensionality(),
            ));
        }

        let Some(shard_index) = &self.shard_index else {
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                indexer.num_elements(),
            );
            return Ok(ArrayBytes::new_fill_value(
                array_size,
                self.decoded_representation.fill_value(),
            ));
        };

        let chunks_per_shard =
            calculate_chunks_per_shard(self.decoded_representation.shape(), &self.chunk_shape)?;
        let chunks_per_shard = chunks_per_shard.to_array_shape();

        let chunk_representation = unsafe {
            ChunkRepresentation::new_unchecked(
                self.chunk_shape.to_vec(),
                self.decoded_representation.data_type().clone(),
                self.decoded_representation.fill_value().clone(),
            )
        };

        // let Some(array_subset) = indexer.as_array_subset() else {
        //     todo!("Support generic indexers")
        // };
        let array_subset = indexer;

        match self.decoded_representation.element_size() {
            DataTypeSize::Variable => {
                let chunks = array_subset.chunks(chunk_representation.shape())?;

                let decode_inner_chunk_subset = |(chunk_indices, chunk_subset): (Vec<u64>, _)| {
                    let shard_index_idx: usize =
                        usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2)
                            .unwrap();
                    let chunk_representation = chunk_representation.clone();
                    async move {
                        let offset = shard_index[shard_index_idx];
                        let size = shard_index[shard_index_idx + 1];

                        // Get the subset of bytes from the chunk which intersect the array
                        let chunk_subset_overlap = array_subset.overlap(&chunk_subset).unwrap(); // FIXME: unwrap

                        let chunk_subset_bytes = if offset == u64::MAX && size == u64::MAX {
                            let array_size = ArraySize::new(
                                self.data_type().size(),
                                chunk_subset_overlap.num_elements(),
                            );
                            ArrayBytes::new_fill_value(
                                array_size,
                                chunk_representation.fill_value(),
                            )
                        } else {
                            // Partially decode the inner chunk
                            let partial_decoder = self.inner_codecs.clone().async_partial_decoder(
                                Arc::new(AsyncByteIntervalPartialDecoder::new(
                                    self.input_handle.clone(),
                                    offset,
                                    size,
                                )),
                                &chunk_representation,
                                options,
                            ).await
                            .map_err(|err| if let CodecError::InvalidByteRangeError(_) = err {
                                CodecError::Other(
                                    "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                                        .to_string(),
                                )
                            } else {
                                err
                            })?;
                            partial_decoder
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
                let futures = chunks.iter().map(decode_inner_chunk_subset);
                let chunk_bytes_and_subsets = futures::future::try_join_all(futures).await?;

                // Convert into an array
                let out_array_subset =
                    merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?;
                Ok(out_array_subset)
            }
            DataTypeSize::Fixed(data_type_size) => {
                // Find filled / non filled chunks
                let chunk_info = array_subset
                    .chunks(chunk_representation.shape())?
                    .into_iter()
                    .map(|(chunk_indices, chunk_subset)| {
                        let chunk_index = ravel_indices(&chunk_indices, &chunks_per_shard);
                        let chunk_index = usize::try_from(chunk_index).unwrap();

                        // Read the offset/size
                        let offset = shard_index[chunk_index * 2];
                        let size = shard_index[chunk_index * 2 + 1];
                        if offset == u64::MAX && size == u64::MAX {
                            (chunk_subset, None)
                        } else {
                            let offset: usize = offset.try_into().unwrap();
                            let size: usize = size.try_into().unwrap();
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
                            let chunk_representation = chunk_representation.clone();
                            async move {
                            let partial_decoder = self
                                .inner_codecs
                                .clone()
                                .async_partial_decoder(
                                    Arc::new(AsyncByteIntervalPartialDecoder::new(
                                        self.input_handle.clone(),
                                        u64::try_from(*offset).unwrap(),
                                        u64::try_from(*size).unwrap(),
                                    )),
                                    &chunk_representation,
                                    options, // TODO: Adjust options for partial decoding?
                                )
                                .await
                                .map_err(|err| if let CodecError::InvalidByteRangeError(_) = err {
                                    CodecError::Other(
                                        "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                                            .to_string(),
                                    )
                                } else {
                                    err
                                })?;
                            let chunk_subset_overlap = array_subset.overlap(chunk_subset).unwrap(); // FIXME: unwrap
                            // Partial decoding is actually really slow with the blosc codec! Assume sharded chunks are small, and just decode the whole thing and extract bytes
                            // TODO: Investigate further
                            // let decoded_chunk = partial_decoder
                            //     .partial_decode(&[chunk_subset_overlap.relative_to(chunk_subset.start())?])
                            //     .await?
                            //     .remove(0);
                            let decoded_chunk = partial_decoder
                                .partial_decode(
                                    &ArraySubset::new_with_shape(chunk_subset.shape().to_vec()),
                                    options,
                                ) // TODO: Adjust options for partial decoding
                                .await?.into_owned();
                            let decoded_chunk = decoded_chunk
                                .extract_array_subset(
                                    &chunk_subset_overlap.relative_to(chunk_subset.start()).unwrap(),
                                    chunk_subset.shape(),
                                    self.decoded_representation.data_type()
                                )?
                                .into_fixed()?
                                .into_owned();
                            Ok::<_, CodecError>((decoded_chunk, chunk_subset_overlap))
                        }}),
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
                                .fill(self.decoded_representation.fill_value().as_ne_bytes())
                                .map_err(CodecError::from)
                        }
                    )?;
                }
                unsafe { shard.set_len(shard_size) };
                Ok(ArrayBytes::from(shard))
            }
        }
    }
}

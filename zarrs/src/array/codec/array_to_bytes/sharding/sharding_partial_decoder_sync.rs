use std::sync::Arc;

use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

use crate::array::{
    array_bytes::merge_chunks_vlen,
    codec::{
        ArrayPartialDecoderTraits, ArraySubset, ArrayToBytesCodecTraits,
        ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecChain, CodecError,
        CodecOptions,
    },
    ravel_indices, ArrayBytes, ArrayBytesFixedDisjointView, ArraySize, ChunkRepresentation,
    ChunkShape, DataType, DataTypeSize, RawBytes,
};

use super::{calculate_chunks_per_shard, ShardingIndexLocation};

/// Partial decoder for the sharding codec.
pub(crate) struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shard_representation: ChunkRepresentation,
    chunk_representation: ChunkRepresentation,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
}

impl ShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shard_representation: ChunkRepresentation,
        chunk_shape: &ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let shard_index = super::decode_shard_index_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            chunk_shape,
            &shard_representation,
            options,
        )?;
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
        self.shard_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_handle.size() + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    fn partial_decode(
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
                partial_decode_fixed_array_subset(self, indexer, options, data_type_size)
            }
            DataTypeSize::Variable => partial_decode_variable_array_subset(self, indexer, options),
        }
    }
}

fn get_inner_chunk_partial_decoder(
    partial_decoder: &ShardingPartialDecoder,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
    partial_decoder
        .inner_codecs
        .clone()
        .partial_decoder(
            Arc::new(ByteIntervalPartialDecoder::new(
                partial_decoder.input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            &partial_decoder.chunk_representation,
            options,
        )
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

fn partial_decode_fixed_array_subset(
    partial_decoder: &ShardingPartialDecoder,
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
    let (inner_chunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        &partial_decoder.inner_codecs,
        &partial_decoder.chunk_representation,
        &chunks_per_shard,
        options,
    )?;

    let array_subset_size = array_subset.num_elements_usize() * data_type_size;
    let mut out_array_subset = vec![0; array_subset_size];
    let out_array_subset_slice = UnsafeCellSlice::new(out_array_subset.as_mut_slice());

    let decode_inner_chunk_subset_into_slice =
        |(chunk_indices, chunk_subset): (Vec<u64>, ArraySubset)| {
            let shard_index_idx: usize =
                usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2).unwrap();
            let chunk_representation = &partial_decoder.chunk_representation;
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
                let inner_partial_decoder =
                    get_inner_chunk_partial_decoder(partial_decoder, &options, offset, size)?;
                inner_partial_decoder
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

    let chunks = array_subset.chunks(partial_decoder.chunk_representation.shape())?;
    rayon_iter_concurrent_limit::iter_concurrent_limit!(
        inner_chunk_concurrent_limit,
        chunks,
        try_for_each,
        decode_inner_chunk_subset_into_slice
    )?;
    Ok(ArrayBytes::from(out_array_subset))
}

fn partial_decode_variable_array_subset(
    partial_decoder: &ShardingPartialDecoder,
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
    let (inner_chunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        &partial_decoder.inner_codecs,
        &partial_decoder.chunk_representation,
        &chunks_per_shard,
        options,
    )?;
    let options = &options;

    let decode_inner_chunk_subset = |(chunk_indices, chunk_subset): (Vec<u64>, ArraySubset)| {
        let shard_index_idx: usize =
            usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2).unwrap();
        let chunk_representation = &partial_decoder.chunk_representation;
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
            let inner_partial_decoder =
                get_inner_chunk_partial_decoder(partial_decoder, options, offset, size)?;
            inner_partial_decoder
                .partial_decode(
                    &chunk_subset_overlap
                        .relative_to(chunk_subset.start())
                        .unwrap(),
                    options,
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
    let chunks = array_subset.chunks(partial_decoder.chunk_representation.shape())?;
    let chunk_bytes_and_subsets = rayon_iter_concurrent_limit::iter_concurrent_limit!(
        inner_chunk_concurrent_limit,
        chunks,
        map,
        decode_inner_chunk_subset
    )
    .collect::<Result<Vec<_>, _>>()?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?;
    Ok(out_array_subset)
}

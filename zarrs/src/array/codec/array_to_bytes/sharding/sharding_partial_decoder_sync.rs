use std::sync::Arc;

use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

use crate::{
    array::{
        array_bytes::merge_chunks_vlen,
        chunk_grid::RegularChunkGrid,
        codec::{
            ArrayPartialDecoderTraits, ArraySubset, ArrayToBytesCodecTraits,
            ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecChain, CodecError,
            CodecOptions,
        },
        ravel_indices, ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndices, ArraySize,
        ChunkRepresentation, ChunkShape, DataType, DataTypeSize, RawBytes, RawBytesOffsets,
    },
    array_subset::IncompatibleDimensionalityError,
    indexer::Indexer,
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
                .partial_decode_concat(&mut [byte_range].into_iter(), &CodecOptions::default())
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
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if indexer.dimensionality() != self.shard_representation.dimensionality() {
            return Err(CodecError::InvalidIndexerDimensionalityError(
                indexer.dimensionality(),
                self.shard_representation.dimensionality(),
            ));
        }

        match self.shard_representation.element_size() {
            DataTypeSize::Fixed(data_type_size) => {
                if let Some(subset) = indexer.as_array_subset() {
                    partial_decode_fixed_array_subset(self, subset, options, data_type_size)
                } else {
                    partial_decode_fixed_indexer(self, indexer, options, data_type_size)
                }
            }
            DataTypeSize::Variable => {
                if let Some(subset) = indexer.as_array_subset() {
                    partial_decode_variable_array_subset(self, subset, options)
                } else {
                    partial_decode_variable_indexer(self, indexer, options)
                }
            }
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

    let shard_chunk_grid = RegularChunkGrid::new(
        partial_decoder.shard_representation.shape_u64(),
        partial_decoder.chunk_representation.shape().into(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let decode_inner_chunk_subset_into_slice = |chunk_indices: Vec<u64>| {
        let shard_index_idx: usize =
            usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2).unwrap();
        let chunk_representation = &partial_decoder.chunk_representation;
        let offset = shard_index[shard_index_idx];
        let size = shard_index[shard_index_idx + 1];

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality");
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

    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    rayon_iter_concurrent_limit::iter_concurrent_limit!(
        inner_chunk_concurrent_limit,
        chunks.indices(),
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

    let shard_chunk_grid = RegularChunkGrid::new(
        partial_decoder.shard_representation.shape_u64(),
        partial_decoder.chunk_representation.shape().into(),
    )
    .expect("matching dimensionality");

    let decode_inner_chunk_subset = |chunk_indices: Vec<u64>| {
        let shard_index_idx: usize =
            usize::try_from(ravel_indices(&chunk_indices, &chunks_per_shard) * 2).unwrap();
        let chunk_representation = &partial_decoder.chunk_representation;
        let offset = shard_index[shard_index_idx];
        let size = shard_index[shard_index_idx + 1];

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality");
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
    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    let chunk_bytes_and_subsets = rayon_iter_concurrent_limit::iter_concurrent_limit!(
        inner_chunk_concurrent_limit,
        chunks.indices(),
        map,
        decode_inner_chunk_subset
    )
    .collect::<Result<Vec<_>, _>>()?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, array_subset.shape())?;
    Ok(out_array_subset)
}

fn partial_decode_fixed_indexer(
    partial_decoder: &ShardingPartialDecoder,
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
    let inner_chunk_partial_decoders = moka::sync::Cache::new(chunks_per_shard.iter().product());
    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != partial_decoder.chunk_representation.dimensionality() {
            return Err(CodecError::InvalidIndexerDimensionalityError(
                indices.len(),
                partial_decoder.chunk_representation.dimensionality(),
            ));
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i / cs)
            .collect();
        // FIXME: Check chunk is within the expected number of chunks
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard);

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];
        let inner_partial_decoder_entry = inner_chunk_partial_decoders
            .entry(chunk_index_1d)
            .or_try_insert_with(|| {
                get_inner_chunk_partial_decoder(partial_decoder, options, offset, size)
            })
            .map_err(Arc::unwrap_or_clone)?;
        let inner_partial_decoder = inner_partial_decoder_entry.value();

        // Get the element index
        let indices_in_inner_chunk: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let element_bytes = inner_partial_decoder
            .partial_decode(&[indices_in_inner_chunk], options)?
            .into_fixed()
            .expect("fixed data");
        output.extend_from_slice(&element_bytes);
    }

    debug_assert_eq!(output.len(), output_len);

    Ok(output.into())
}

fn partial_decode_variable_indexer(
    partial_decoder: &ShardingPartialDecoder,
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
    let inner_chunk_partial_decoders = moka::sync::Cache::new(chunks_per_shard.iter().product());
    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != partial_decoder.chunk_representation.dimensionality() {
            return Err(CodecError::InvalidIndexerDimensionalityError(
                indices.len(),
                partial_decoder.chunk_representation.dimensionality(),
            ));
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i / cs)
            .collect();
        // FIXME: Check chunk is within the expected number of chunks
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard);

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];
        let inner_partial_decoder_entry = inner_chunk_partial_decoders
            .entry(chunk_index_1d)
            .or_try_insert_with(|| {
                get_inner_chunk_partial_decoder(partial_decoder, options, offset, size)
            })
            .map_err(Arc::unwrap_or_clone)?;
        let inner_partial_decoder = inner_partial_decoder_entry.value();

        // Get the element index
        let indices_in_inner_chunk: ArrayIndices = indices
            .iter()
            .zip(partial_decoder.chunk_representation.shape())
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let (element_bytes, element_offsets) = inner_partial_decoder
            .partial_decode(&[indices_in_inner_chunk], options)?
            .into_variable()
            .expect("fixed data");
        debug_assert_eq!(element_offsets.len(), 2);
        bytes.extend_from_slice(&element_bytes);
        offsets.push(bytes.len());
    }

    Ok(ArrayBytes::new_vlen(bytes, RawBytesOffsets::new(offsets)?)?)
}

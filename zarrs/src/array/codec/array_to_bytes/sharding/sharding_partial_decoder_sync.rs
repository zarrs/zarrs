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
    indexer::{IncompatibleIndexerError, Indexer},
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

pub(crate) fn partial_decode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_representation: &ChunkRepresentation,
    chunk_representation: &ChunkRepresentation,
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::indexer::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    if indexer.dimensionality() != shard_representation.dimensionality() {
        return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
            indexer.dimensionality(),
            shard_representation.dimensionality(),
        )
        .into());
    }

    match shard_representation.element_size() {
        DataTypeSize::Fixed(_data_type_size) => {
            if let Some(subset) = indexer.as_array_subset() {
                partial_decode_fixed_array_subset(
                    input_handle,
                    shard_representation,
                    chunk_representation,
                    inner_codecs,
                    shard_index,
                    subset,
                    options,
                )
            } else {
                partial_decode_fixed_indexer(
                    input_handle,
                    shard_representation,
                    chunk_representation,
                    inner_codecs,
                    shard_index,
                    indexer,
                    options,
                )
            }
        }
        DataTypeSize::Variable => {
            if let Some(subset) = indexer.as_array_subset() {
                partial_decode_variable_array_subset(
                    input_handle,
                    shard_representation,
                    chunk_representation,
                    inner_codecs,
                    shard_index,
                    subset,
                    options,
                )
            } else {
                partial_decode_variable_indexer(
                    input_handle,
                    shard_representation,
                    chunk_representation,
                    inner_codecs,
                    shard_index,
                    indexer,
                    options,
                )
            }
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
        partial_decode(
            &self.input_handle,
            &self.shard_representation,
            &self.chunk_representation,
            &self.inner_codecs,
            self.shard_index.as_deref(),
            indexer,
            options,
        )
    }
}

fn get_inner_chunk_partial_decoder(
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    inner_codecs: Arc<CodecChain>,
    chunk_representation: &ChunkRepresentation,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
    inner_codecs
        .partial_decoder(
            Arc::new(ByteIntervalPartialDecoder::new(
                input_handle,
                byte_offset,
                byte_length,
            )),
            chunk_representation,
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

pub(crate) fn partial_decode_fixed_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_representation: &ChunkRepresentation,
    chunk_representation: &ChunkRepresentation,
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &ArraySubset,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type_size = shard_representation
        .data_type()
        .fixed_size()
        .expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return Ok(super::partial_decode_empty_shard(
            shard_representation,
            array_subset,
        ));
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_representation.shape(), chunk_representation.shape())?
            .to_array_shape();
    let (inner_chunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        chunk_representation,
        &chunks_per_shard,
        options,
    )?;

    let array_subset_size = array_subset.num_elements_usize() * data_type_size;
    let mut out_array_subset = vec![0; array_subset_size];
    let out_array_subset_slice = UnsafeCellSlice::new(out_array_subset.as_mut_slice());

    let shard_chunk_grid = RegularChunkGrid::new(
        shard_representation.shape_u64(),
        chunk_representation.shape().into(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let decode_inner_chunk_subset_into_slice = |chunk_indices: Vec<u64>| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

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
            let inner_partial_decoder = get_inner_chunk_partial_decoder(
                input_handle.clone(),
                inner_codecs.clone(),
                chunk_representation,
                &options,
                offset,
                size,
            )?;
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
    crate::iter_concurrent_limit!(
        inner_chunk_concurrent_limit,
        chunks.indices(),
        try_for_each,
        decode_inner_chunk_subset_into_slice
    )?;
    Ok(ArrayBytes::from(out_array_subset))
}

pub(crate) fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_representation: &ChunkRepresentation,
    chunk_representation: &ChunkRepresentation,
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &ArraySubset,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return Ok(super::partial_decode_empty_shard(
            shard_representation,
            array_subset,
        ));
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_representation.shape(), chunk_representation.shape())?
            .to_array_shape();
    let (inner_chunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        chunk_representation,
        &chunks_per_shard,
        options,
    )?;
    let options = &options;

    let shard_chunk_grid = RegularChunkGrid::new(
        shard_representation.shape_u64(),
        chunk_representation.shape().into(),
    )
    .expect("matching dimensionality");

    let decode_inner_chunk_subset = |chunk_indices: Vec<u64>| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

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
            let inner_partial_decoder = get_inner_chunk_partial_decoder(
                input_handle.clone(),
                inner_codecs.clone(),
                chunk_representation,
                options,
                offset,
                size,
            )?;
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
    let chunk_bytes_and_subsets = crate::iter_concurrent_limit!(
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

pub(crate) fn partial_decode_fixed_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_representation: &ChunkRepresentation,
    chunk_representation: &ChunkRepresentation,
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type_size = shard_representation
        .data_type()
        .fixed_size()
        .expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return Ok(super::partial_decode_empty_shard(
            shard_representation,
            indexer,
        ));
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_representation.shape(), chunk_representation.shape())?
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
        if indices.len() != chunk_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indices.len(),
                chunk_representation.dimensionality(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(chunk_representation.shape())
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
                .or_try_insert_with(|| {
                    get_inner_chunk_partial_decoder(
                        input_handle.clone(),
                        inner_codecs.clone(),
                        chunk_representation,
                        options,
                        offset,
                        size,
                    )
                })
                .map_err(Arc::unwrap_or_clone)?;
            inner_partial_decoder_entry.value()
        };
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = inner_chunk_partial_decoders
            .get_or_insert_with(&chunk_index_1d, || {
                get_inner_chunk_partial_decoder(
                        input_handle.clone(),
                        inner_codecs.clone(),
                        chunk_representation,
                        options,
                        offset,
                        size,
                )
            })?;


        // Get the element index
        let indices_in_inner_chunk: ArrayIndices = indices
            .iter()
            .zip(chunk_representation.shape())
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

pub(crate) fn partial_decode_variable_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_representation: &ChunkRepresentation,
    chunk_representation: &ChunkRepresentation,
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = shard_index else {
        return Ok(super::partial_decode_empty_shard(
            shard_representation,
            indexer,
        ));
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_representation.shape(), chunk_representation.shape())?
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
        if indices.len() != chunk_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indices.len(),
                chunk_representation.dimensionality(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(chunk_representation.shape())
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
                .or_try_insert_with(|| {
                    get_inner_chunk_partial_decoder(
                        input_handle.clone(),
                        inner_codecs.clone(),
                        chunk_representation,
                        options,
                        offset,
                        size,
                    )
                })
                .map_err(Arc::unwrap_or_clone)?;
            inner_partial_decoder_entry.value()
        };
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = inner_chunk_partial_decoders
            .get_or_insert_with(&chunk_index_1d, || {
                get_inner_chunk_partial_decoder(
                        input_handle.clone(),
                        inner_codecs.clone(),
                        chunk_representation,
                        options,
                        offset,
                        size,
                    )
            })?;

        // Get the element index
        let indices_in_inner_chunk: ArrayIndices = indices
            .iter()
            .zip(chunk_representation.shape())
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

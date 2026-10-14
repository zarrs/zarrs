use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use zarrs_data_type::FillValue;

use super::{ShardingCodecOptions, ShardingIndexLocation, calculate_chunks_per_shard};
use crate::array::array_bytes_internal::{
    FixedDecodeBuffers, build_nested_optional_target, extract_target_views, fill_target,
    merge_chunks_vlen, optional_innermost,
};
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::codec::CodecChain;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesRaw, ArrayIndicesTinyVec, ArraySubsetTraits,
    ChunkShape, ChunkShapeTraits, DataType, IncompatibleDimensionalityError, Indexer,
    ravel_indices,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
    ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecError, CodecOptions,
    InvalidNumberOfElementsError, decode_into_array_bytes_target,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Partial decoder for the sharding codec.
pub(crate) struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    data_type: DataType,
    fill_value: FillValue,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChain>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl ShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        data_type: DataType,
        fill_value: FillValue,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: &CodecChain,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
        let shard_index = super::decode_shard_index_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )?;

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
    pub(crate) fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
        } else {
            Ok(None)
        }
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn partial_decode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
            )?;
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
        );
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
}

impl ArrayPartialDecoderTraits for ShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
            + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    fn partial_decode(
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
    }

    fn partial_decode_into(
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
        } else {
            let decoded_value = self.partial_decode(indexer, options)?;
            decode_into_array_bytes_target(&decoded_value, output_target)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[expect(clippy::too_many_arguments)]
fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
    inner_codecs
        .clone()
        .partial_decoder(
            Arc::new(ByteIntervalPartialDecoder::new(
                input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            subchunk_shape,
            data_type,
            fill_value,
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

#[expect(clippy::too_many_arguments)]
fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
    let decode_subchunk_subset_into_slice = |chunk_indices: ArrayIndicesTinyVec| {
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
        // Calculate the chunk's position in the output view coordinate space
        let chunk_relative = chunk_subset_overlap.relative_to(&array_subset_start)?;
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
                &options,
                offset,
                size,
            )?;
            inner_partial_decoder.partial_decode_into(
                &chunk_subset_overlap
                    .relative_to(chunk_subset.start())
                    .unwrap(),
                target,
                &options,
            )
        }
    };

    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        chunks.indices(),
        try_for_each,
        decode_subchunk_subset_into_slice
    )?;
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    data_type: &DataType,
    fill_value: &FillValue,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChain>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let Some(shard_index) = &shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
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
    let options = &options;

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .expect("matching dimensionality");

    let array_subset_start = array_subset.start();
    let decode_subchunk_subset = |chunk_indices: ArrayIndicesTinyVec| {
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
            ArrayBytes::new_fill_value(data_type, chunk_subset_overlap.num_elements(), fill_value)?
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
            )?;
            inner_partial_decoder
                .partial_decode(
                    &chunk_subset_overlap
                        .relative_to(chunk_subset.start())
                        .unwrap(),
                    options,
                )?
                .into_owned()
                .into_variable()?
        };
        Ok::<_, CodecError>((
            chunk_subset_bytes,
            chunk_subset_overlap
                .relative_to(&array_subset_start)
                .unwrap(),
        ))
    };
    // Decode the subchunk subsets
    let chunks = shard_chunk_grid.chunks_in_array_subset(array_subset)?;
    let chunk_bytes_and_subsets = crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        chunks.indices(),
        map,
        decode_subchunk_subset
    )
    .collect::<Result<Vec<_>, _>>()?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, &array_subset.shape());
    Ok(ArrayBytes::Variable(out_array_subset))
}

#[expect(clippy::too_many_arguments)]
fn partial_decode_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
    let decode = |(subchunk_index, group): (u64, super::SubchunkIndices)| {
        let bytes = if let Some((offset, size)) =
            super::subchunk_offset_size(shard_index, usize::try_from(subchunk_index).unwrap())
        {
            let decoder = get_subchunk_partial_decoder(
                input_handle,
                data_type,
                fill_value,
                subchunk_shape,
                inner_codecs,
                &options,
                offset,
                size,
            )?;
            decoder.partial_decode(&group, &options)?.into_owned()
        } else {
            ArrayBytes::new_fill_value(data_type, group.len(), fill_value)?
        };
        Ok::<_, CodecError>((group.positions, bytes))
    };
    let decoded =
        crate::iter_concurrent_limit!(limit, groups.into_iter().collect::<Vec<_>>(), map, decode)
            .collect::<Result<Vec<_>, _>>()?;
    super::merge_indexer_subchunks(decoded, usize::try_from(indexer.len()).unwrap(), data_type)
}

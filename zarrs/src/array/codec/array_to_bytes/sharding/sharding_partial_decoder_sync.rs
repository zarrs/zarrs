use std::num::NonZeroU64;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use zarrs_chunk_grid::ChunkGridTraits;

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
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec, ArraySubsetTraits, ChunkGrid,
    ChunkShape, CodecChainBound, CowBytes, DataType, Indexer, ravel_indices,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderSubchunkingTraits,
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, ByteIntervalPartialDecoder,
    BytesPartialDecoderTraits, CodecError, CodecOptions, InvalidNumberOfElementsError,
    decode_into_array_bytes_target,
};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Partial decoder for the sharding codec.
pub struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: RegularBoundedChunkGrid,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

impl ShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: &CodecChainBound,
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
    pub fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<CowBytes<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
        } else {
            Ok(None)
        }
    }
}

pub(crate) fn partial_decode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    indexer.validate(subchunk_grid.array_shape())?;

    let Some(subset) = indexer.as_array_subset() else {
        return partial_decode_indexer(
            input_handle,
            subchunk_grid,
            subchunk_shape,
            inner_codecs,
            shard_index,
            indexer,
            options,
        );
    };

    // Fixed length data (including optional data with fixed length inner data): decode each subchunk directly into the output
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
            )?;
        }
        // SAFETY: every element of the output (and masks) is written by `partial_decode_fixed_array_subset_into`
        Ok(unsafe { buffers.into_array_bytes() })
    } else {
        partial_decode_merged_array_subset(
            input_handle,
            subchunk_grid,
            subchunk_shape,
            inner_codecs,
            shard_index,
            subset,
            options,
        )
    }
}

impl ArrayPartialDecoderSubchunkingTraits for ShardingPartialDecoder {
    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        nested_local_subchunk_grids(
            ChunkGrid::new(self.subchunk_grid.clone()),
            &self.inner_codecs,
        )
    }
}

impl ArrayPartialDecoderTraits for ShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
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
            &self.subchunk_grid,
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
            (true, Some(subset), output_target) => partial_decode_fixed_array_subset_into(
                &self.input_handle,
                &self.subchunk_grid,
                &self.subchunk_shape,
                &self.inner_codecs,
                self.shard_index.as_deref(),
                subset,
                options,
                output_target,
            ),
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
            }
            (_, _, output_target) => {
                let decoded_value = self.partial_decode(indexer, options)?;
                decode_into_array_bytes_target(&decoded_value, output_target)
            }
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
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
    let decode_subchunk_subset_into_slice = |chunk_indices: ArrayIndicesTinyVec| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, subchunk_grid.grid_shape()).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset_size = super::subchunk_offset_size(shard_index, shard_index_idx);

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = subchunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;
        // Calculate the chunk's position in the output view coordinate space
        let chunk_relative = chunk_subset_overlap.relative_to(&array_subset_start)?;
        let chunk_output_overlap_subset = chunk_relative.offset(output_view.subset().start())?;
        let mut subchunk_view = unsafe {
            // SAFETY: chunks represent disjoint array subsets
            output_view.subdivide(chunk_output_overlap_subset.clone())?
        };
        let mut subchunk_mask_views = mask_views
            .iter()
            .map(|mask_view| unsafe {
                // SAFETY: chunks represent disjoint array subsets
                mask_view.subdivide(chunk_output_overlap_subset.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let subchunk_target =
            build_nested_optional_target(&mut subchunk_view, &mut subchunk_mask_views);
        if let Some((offset, size)) = offset_size {
            // Partially decode the subchunk
            let inner_partial_decoder = get_subchunk_partial_decoder(
                input_handle,
                &chunk_subset.chunk_shape().expect("nonempty subchunk"),
                inner_codecs,
                &options,
                offset,
                size,
            )?;
            inner_partial_decoder.partial_decode_into(
                &chunk_subset_overlap
                    .relative_to(chunk_subset.start())
                    .unwrap(),
                subchunk_target,
                &options,
            )
        } else {
            fill_target(subchunk_target, data_type, fill_value)
        }
    };

    let chunks = subchunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    chunks
        .indices()
        .concurrent_limit(subchunk_concurrent_limit)
        .try_for_each(decode_subchunk_subset_into_slice)?;
    Ok(())
}

/// Partially decode an array subset by decoding the overlapping region of each subchunk and merging them.
///
/// This supports any data type, including variable length and optional data types.
fn partial_decode_merged_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = &shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        super::num_subchunks(subchunk_grid),
        options,
    )?;
    let options = &options;

    let array_subset_start = array_subset.start();
    let decode_subchunk_subset = |chunk_indices: ArrayIndicesTinyVec| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, subchunk_grid.grid_shape()).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset_size = super::subchunk_offset_size(shard_index, shard_index_idx);

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = subchunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;

        let chunk_subset_bytes = if let Some((offset, size)) = offset_size {
            // Partially decode the subchunk
            let inner_partial_decoder = get_subchunk_partial_decoder(
                input_handle,
                &chunk_subset.chunk_shape().expect("nonempty subchunk"),
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
        } else {
            ArrayBytes::new_fill_value(data_type, chunk_subset_overlap.num_elements(), fill_value)?
        };
        Ok::<_, CodecError>((
            chunk_subset_bytes,
            chunk_subset_overlap
                .relative_to(&array_subset_start)
                .unwrap(),
        ))
    };
    // Decode the subchunk subsets
    let chunks = subchunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let chunk_bytes_and_subsets = chunks
        .indices()
        .concurrent_limit(subchunk_concurrent_limit)
        .map(decode_subchunk_subset)
        .collect::<Result<Vec<_>, _>>()?;

    // Convert into an array
    merge_chunks(chunk_bytes_and_subsets, &array_subset.shape(), data_type)
}

fn partial_decode_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
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
    let decoded = groups
        .into_iter()
        .collect::<Vec<_>>()
        .concurrent_limit(subchunk_concurrent_limit)
        .map(|(subchunk_index, group)| {
            let shard_index_idx = usize::try_from(subchunk_index).unwrap();
            let bytes = if let Some((offset, size)) =
                super::subchunk_offset_size(shard_index, shard_index_idx)
            {
                let decoder = get_subchunk_partial_decoder(
                    input_handle,
                    &group.subchunk_shape,
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
        })
        .collect::<Result<Vec<_>, _>>()?;
    super::merge_indexer_subchunks(decoded, usize::try_from(indexer.len()).unwrap(), data_type)
}

/// Decode the elements of `indexer` into `output_view` by scattering each subchunk group directly.
///
/// The `indexer` must be validated against the shard shape.
#[expect(clippy::too_many_arguments)]
fn partial_decode_fixed_indexer_into<'a>(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
    output_view: &'a mut ArrayBytesFixedDisjointView<'a>,
) -> Result<(), CodecError> {
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
    let subchunk_decoder = |subchunk_index: u64, subchunk_shape: &[NonZeroU64]| {
        let shard_index_idx = usize::try_from(subchunk_index).unwrap();
        super::subchunk_offset_size(shard_index, shard_index_idx)
            .map(|(offset, size)| {
                get_subchunk_partial_decoder(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    &options,
                    offset,
                    size,
                )
            })
            .transpose()
    };

    if groups.len() == 1 {
        // Positions within a group are ascending and cover every element, so decode in place
        let (subchunk_index, group) = groups.into_iter().next().expect("one group");
        return if let Some(decoder) = subchunk_decoder(subchunk_index, &group.subchunk_shape)? {
            decoder.partial_decode_into(
                &group,
                ArrayBytesDecodeIntoTarget::Fixed(output_view),
                &options,
            )
        } else {
            output_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from)
        };
    }

    let decoded = groups
        .into_iter()
        .collect::<Vec<_>>()
        .concurrent_limit(subchunk_concurrent_limit)
        .map(|(subchunk_index, group)| {
            let bytes = subchunk_decoder(subchunk_index, &group.subchunk_shape)?
                .map(|decoder| {
                    Ok::<_, CodecError>(
                        decoder
                            .partial_decode(&group, &options)?
                            .into_owned()
                            .into_fixed()?,
                    )
                })
                .transpose()?;
            Ok::<_, CodecError>((group.positions, bytes))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (positions, bytes) in decoded {
        if let Some(bytes) = bytes {
            output_view.copy_elements_from_slice(&positions, &bytes)?;
        } else {
            output_view.fill_elements(&positions, fill_value.as_ne_bytes())?;
        }
    }
    Ok(())
}

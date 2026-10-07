use std::num::NonZeroU64;
use std::ops::IndexMut;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use bytes::Bytes;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::{ChunkGridCreateError, ChunkGridTraits};

#[cfg(feature = "async")]
use super::sharding_partial_decoder_async::AsyncShardingPartialDecoder;
use super::sharding_partial_decoder_sync::ShardingPartialDecoder;
use super::{
    CodecChain, RequireDivisibleSubchunks, ShardingCodecConfiguration,
    ShardingCodecConfigurationV1, ShardingCodecOptions, ShardingIndexLocation, SubchunkWriteOrder,
    calculate_chunks_per_shard, compute_index_encoded_size, decode_shard_index,
    sharding_index_shape, sharding_partial_encoder, subchunk_grid,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::array_bytes_internal::{
    FixedDecodeBuffers, build_nested_optional_target, extract_target_views, fill_target,
    merge_chunks,
};
use crate::array::chunk_grid::repeat::RepeatChunkGrid;
use crate::array::chunk_grid::{
    ChunkEdgeLengths, RectilinearChunkGrid, RegularBoundedChunkGrid, RegularChunkGrid,
};
use crate::array::concurrency::calc_concurrency_outer_inner;
use crate::array::{
    ArrayBytes, ArraySubset, BytesRepresentation, ChunkGrid, ChunkShape, ChunkShapeTraits,
    CodecChainBound, CowBytes, DataType, FillValue, transmute_to_bytes_vec, unravel_index,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits,
    ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits,
    BytesPartialEncoderTraits, ChunkGridDecoded, ChunkGridDecodedRef, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    UnboundArrayToBytesCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use zarrs_metadata::Configuration;
use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

/// Return the subchunk grid of a sharded `chunk_grid`.
///
/// Subchunks straddling a shard boundary are clipped to the shard shape, or rejected if `require_divisible`.
fn regular_subchunk_grid(
    chunk_grid: &ChunkGrid,
    subchunk_shape: &ChunkShape,
    require_divisible: bool,
) -> Result<ChunkGrid, ChunkGridCreateError> {
    if chunk_grid.dimensionality() != subchunk_shape.len() {
        return Err(ChunkGridCreateError::new(format!(
            "sharding subchunk shape dimensionality {} does not match chunk grid dimensionality {}",
            subchunk_shape.len(),
            chunk_grid.dimensionality()
        )));
    }

    let check_divisible = |edge_length: NonZeroU64, subchunk: NonZeroU64| {
        if require_divisible && !edge_length.get().is_multiple_of(subchunk.get()) {
            return Err(ChunkGridCreateError::new(format!(
                "invalid subchunk shape {subchunk_shape:?}, it must evenly divide shard shape {edge_length:?}"
            )));
        }
        Ok(())
    };

    if chunk_grid
        .name_v3()
        .is_some_and(|name| RegularChunkGrid::matches_name_v3(name.as_ref()))
    {
        let chunk_shape = chunk_grid
            .chunk_shape(&vec![0; chunk_grid.dimensionality()])?
            .ok_or_else(|| {
                ChunkGridCreateError::new("chunk grid does not contain an origin chunk")
            })?;
        for (&edge_length, &subchunk) in std::iter::zip(&chunk_shape, subchunk_shape) {
            check_divisible(edge_length, subchunk)?;
        }
        return Ok(ChunkGrid::new(RepeatChunkGrid::new(
            chunk_grid.grid_shape().to_vec(),
            ChunkGrid::new(RegularBoundedChunkGrid::new(
                chunk_shape.to_array_shape(),
                subchunk_shape.clone(),
            )?),
        )?));
    }

    let mut subchunk_grid_shape = Vec::with_capacity(chunk_grid.dimensionality());
    let mut subchunk_edge_lengths = Vec::with_capacity(chunk_grid.dimensionality());
    let mut needs_rectilinear = false;

    for (dim, subchunk) in subchunk_shape.iter().enumerate() {
        let chunk_edge_lengths = chunk_grid.chunk_edge_lengths(dim)?;
        let mut global_edge_lengths = Vec::new();
        for edge_length in chunk_edge_lengths {
            check_divisible(edge_length, *subchunk)?;
            global_edge_lengths.extend(
                RegularBoundedChunkGrid::new(vec![edge_length.get()], vec![*subchunk])?
                    .chunk_edge_lengths(0)?,
            );
        }
        let dimension_shape = global_edge_lengths
            .iter()
            .try_fold(0u64, |sum, edge| sum.checked_add(edge.get()))
            .ok_or_else(|| ChunkGridCreateError::new("subchunk grid shape overflow"))?;
        let edge_lengths = ChunkEdgeLengths::encode(&global_edge_lengths);
        if !matches!(edge_lengths, ChunkEdgeLengths::Scalar(_)) {
            needs_rectilinear = true;
        }
        subchunk_grid_shape.push(dimension_shape);
        subchunk_edge_lengths.push(edge_lengths);
    }

    if needs_rectilinear {
        Ok(ChunkGrid::new(RectilinearChunkGrid::new(
            subchunk_grid_shape,
            &subchunk_edge_lengths,
        )?))
    } else {
        let subchunk_shape = subchunk_edge_lengths
            .into_iter()
            .map(|edge_lengths| match edge_lengths {
                ChunkEdgeLengths::Scalar(edge_length) => Some(edge_length),
                ChunkEdgeLengths::Varying(_) => None,
            })
            .collect::<Option<ChunkShape>>()
            .expect("all edge lengths are scalar");
        Ok(ChunkGrid::new(RegularChunkGrid::new(
            subchunk_grid_shape,
            subchunk_shape,
        )?))
    }
}

/// Return the subset of the subchunk with the raveled `subchunk_index`, clipped to the shard shape.
fn subchunk_subset(subchunk_grid: &RegularBoundedChunkGrid, subchunk_index: usize) -> ArraySubset {
    let subchunk_indices = unravel_index(subchunk_index as u64, subchunk_grid.grid_shape())
        .expect("inbounds subchunk");
    subchunk_grid
        .subset(&subchunk_indices)
        .expect("matching dimensionality")
        .expect("inbounds subchunk")
}

/// A `sharding` codec implementation.
#[derive(Clone, Debug)]
pub struct ShardingCodec {
    /// An array of integers specifying the shape of the subchunks in a shard along each dimension of the outer array.
    pub(crate) subchunk_shape: ChunkShape,
    /// The codecs used to encode and decode subchunks.
    pub(crate) inner_codecs: Arc<CodecChain>,
    /// The codecs used to encode and decode the shard index.
    pub(crate) index_codecs: Arc<CodecChain>,
    /// Specifies whether the shard index is located at the beginning or end of the file.
    pub(crate) index_location: ShardingIndexLocation,
    /// Runtime options applied at array creation/opening time.
    pub(crate) options: ShardingCodecOptions,
}

/// A `sharding` codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
pub struct ShardingCodecBound {
    pub(crate) subchunk_shape: ChunkShape,
    pub(crate) inner_codecs: Arc<CodecChainBound>,
    pub(crate) index_codecs: Arc<CodecChainBound>,
    pub(crate) index_location: ShardingIndexLocation,
    pub(crate) options: ShardingCodecOptions,
    /// Reject subchunk shapes that do not evenly divide the shard shape when creating subchunk grids.
    pub(crate) require_divisible_subchunks: bool,
}

impl ShardingCodec {
    /// Create a new `sharding` codec.
    #[must_use]
    pub fn new(
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: Arc<CodecChain>,
        index_location: ShardingIndexLocation,
    ) -> Self {
        Self {
            subchunk_shape,
            inner_codecs,
            index_codecs,
            index_location,
            options: ShardingCodecOptions::default(),
        }
    }

    /// Create a new `sharding` codec from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &ShardingCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ShardingCodecConfiguration::V1(configuration) => {
                let inner_codecs = Arc::new(
                    CodecChain::from_metadata(&configuration.codecs)
                        .map_err(|err| PluginCreateError::Other(err.to_string()))?,
                );
                let index_codecs = Arc::new(
                    CodecChain::from_metadata(&configuration.index_codecs)
                        .map_err(|err| PluginCreateError::Other(err.to_string()))?,
                );
                Ok(Self::new(
                    configuration.chunk_shape.clone(),
                    inner_codecs,
                    index_codecs,
                    configuration.index_location,
                ))
            }
            _ => Err(PluginCreateError::Other(
                "this sharding_indexed codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Return a version of this codec with the provided [`ShardingCodecOptions`].
    #[must_use]
    pub fn with_options(mut self, options: ShardingCodecOptions) -> Self {
        self.options = options;
        self
    }

    /// Return a version of this codec with the provided [`SubchunkWriteOrder`].
    #[must_use]
    pub fn with_subchunk_write_order(mut self, order: SubchunkWriteOrder) -> Self {
        self.options = self.options.with_subchunk_write_order(order);
        self
    }
}

impl CodecTraits for ShardingCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ShardingCodecConfiguration::V1(ShardingCodecConfigurationV1 {
            chunk_shape: self.subchunk_shape.clone(),
            codecs: self.inner_codecs.create_metadatas(options),
            index_codecs: self.index_codecs.create_metadatas(options),
            index_location: self.index_location,
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: true,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToBytesCodecTraits for ShardingCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        let inner_codecs =
            self.inner_codecs
                .with_context(data_type, fill_value, codec_specific_options)?;
        let index_codecs = self.index_codecs.with_context(
            crate::array::data_type::uint64(),
            FillValue::from(u64::MAX),
            codec_specific_options,
        )?;
        let options = codec_specific_options
            .get_option::<ShardingCodecOptions>()
            .unwrap_or(&self.options)
            .clone();
        Ok(Arc::new(ShardingCodecBound {
            subchunk_shape: self.subchunk_shape.clone(),
            inner_codecs,
            index_codecs,
            index_location: self.index_location,
            options,
            require_divisible_subchunks: codec_specific_options
                .get_option::<RequireDivisibleSubchunks>()
                .is_some(),
        }))
    }
}

impl ArrayCodecTraits for ShardingCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
    }

    fn fill_value(&self) -> &FillValue {
        self.inner_codecs.fill_value()
    }

    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(shape, self.subchunk_shape.as_slice())?;
        let num_elements = chunks_per_shard.num_elements_nonzero_usize();
        Ok(RecommendedConcurrency::new_maximum(num_elements.into()))
    }
}

impl zarrs_codec::ArrayToBytesCodecSubchunkingTraits for ShardingCodecBound {
    fn decoded_subchunk_grids(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<Vec<ChunkGridDecoded>, ChunkGridCreateError> {
        let subchunk_grid = match decoded_chunk_grid {
            ChunkGridDecodedRef::None => ChunkGridDecoded::None,
            ChunkGridDecodedRef::Array(decoded_chunk_grid)
                if decoded_chunk_grid.array_shape().contains(&0) =>
            {
                ChunkGridDecoded::None
            }
            ChunkGridDecodedRef::Array(decoded_chunk_grid) => {
                ChunkGridDecoded::Array(regular_subchunk_grid(
                    decoded_chunk_grid,
                    &self.subchunk_shape,
                    self.require_divisible_subchunks,
                )?)
            }
            ChunkGridDecodedRef::ChunkLocal => ChunkGridDecoded::ChunkLocal,
        };
        let mut subchunk_grids = vec![subchunk_grid.clone()];
        subchunk_grids.extend(
            self.inner_codecs
                .decoded_subchunk_grids((&subchunk_grid).into())?,
        );
        Ok(subchunk_grids)
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for ShardingCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        let data_type = self.data_type();
        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes.validate(num_elements, data_type)?;

        // Get chunk bytes representation, and choose implementation based on whether the size is unbounded or not
        let chunk_bytes_representation = self
            .inner_codecs
            .encoded_representation(&self.subchunk_shape)?;

        bytes.validate(shape.num_elements_u64(), data_type)?;
        let bytes = match chunk_bytes_representation {
            BytesRepresentation::BoundedSize(size) | BytesRepresentation::FixedSize(size) => {
                self.encode_bounded(&bytes, shape, &self.subchunk_shape, size, options)
            }
            BytesRepresentation::UnboundedSize => {
                self.encode_unbounded(&bytes, shape, &self.subchunk_shape, options)
            }
        }?;
        Ok(CowBytes::from(bytes))
    }

    fn decode<'a>(
        &self,
        encoded_shard: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let data_type = self.data_type();
        let fill_value = self.fill_value();
        let shard_shape_u64: &[u64] = bytemuck::must_cast_slice(shape);

        // Fixed length data (including optional data with fixed length inner data): decode each subchunk directly into the output
        if !data_type.is_optional() && data_type.fixed_size() == Some(0) {
            return Ok(ArrayBytes::new_flen(vec![]));
        }
        if let Some(mut buffers) = FixedDecodeBuffers::new(data_type, shard_shape_u64) {
            {
                let (mut data_view, mut mask_views) = buffers.views()?;
                self.decode_into(
                    encoded_shard,
                    shape,
                    build_nested_optional_target(&mut data_view, &mut mask_views),
                    options,
                )?;
            }
            // SAFETY: every element of the output (and masks) is written by `decode_into`
            return Ok(unsafe { buffers.into_array_bytes() });
        }

        // Variable length data (including optional data with variable length inner data): decode each subchunk and merge
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let num_chunks = chunks_per_shard
            .iter()
            .map(|i| usize::try_from(i.get()).unwrap())
            .product::<usize>();

        let shard_index =
            self.decode_index(&encoded_shard, chunks_per_shard.as_slice(), options)?;

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_subchunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shape)?,
            &self
                .inner_codecs
                .recommended_concurrency(&self.subchunk_shape)?,
        );
        let options = options.with_concurrent_target(concurrency_limit_subchunks);

        let subchunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        let decode_subchunk = |chunk_index: usize| {
            let chunk_subset = subchunk_subset(&subchunk_grid, chunk_index);

            // Read the offset/size
            let offset = shard_index[chunk_index * 2];
            let size = shard_index[chunk_index * 2 + 1];
            let chunk_bytes = if offset == u64::MAX && size == u64::MAX {
                ArrayBytes::new_fill_value(data_type, chunk_subset.num_elements(), fill_value)?
            } else if usize::try_from(offset + size).unwrap() > encoded_shard.len() {
                return Err(CodecError::Other(
                    "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                        .to_string(),
                ));
            } else {
                let offset: usize = offset.try_into().unwrap();
                let size: usize = size.try_into().unwrap();
                let encoded_chunk = encoded_shard.clone().slice(offset..offset + size);
                self.inner_codecs.decode(
                    encoded_chunk,
                    &chunk_subset.chunk_shape().expect("nonempty subchunk"),
                    &options,
                )?
            };
            Ok((chunk_bytes, chunk_subset))
        };

        // Decode the subchunks
        let chunk_bytes_and_subsets = (0..num_chunks)
            .concurrent_limit(shard_concurrent_limit)
            .map(decode_subchunk)
            .collect::<Result<Vec<_>, _>>()?;

        // Convert into an array
        merge_chunks(chunk_bytes_and_subsets, shard_shape_u64, data_type)
    }

    fn compact<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Option<CowBytes<'a>>, CodecError> {
        // Calculate chunks per shard
        let chunks_per_shard = calculate_chunks_per_shard(shape, self.subchunk_shape.as_slice())?;

        // Decode the shard index
        let shard_index = self.decode_index(&bytes, chunks_per_shard.as_slice(), options)?;

        // Get index metadata
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;

        // Check if compaction is needed (no-op optimization)
        let mut needs_compaction = false;
        let mut chunks_size = 0;
        for &[offset, size] in shard_index.as_chunks::<2>().0 {
            if offset != u64::MAX && size != u64::MAX {
                chunks_size += size;
            }
        }
        if chunks_size != bytes.len() as u64 - index_encoded_size {
            needs_compaction = true;
        }

        if !needs_compaction {
            return Ok(None); // No compaction needed
        }

        // Calculate compact size
        let data_size: usize = shard_index
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|chunk| chunk[0] != u64::MAX)
            .map(|chunk| usize::try_from(chunk[1]).unwrap())
            .sum();

        let compact_size = data_size + usize::try_from(index_encoded_size).unwrap();

        // Build compacted shard
        let mut compact_shard = vec![0u8; compact_size];
        let mut new_index = vec![u64::MAX; shard_index.len()];

        let mut write_offset = match self.index_location {
            ShardingIndexLocation::Start => usize::try_from(index_encoded_size).unwrap(),
            ShardingIndexLocation::End => 0,
        };

        for (i, &[old_offset, size]) in shard_index.as_chunks::<2>().0.iter().enumerate() {
            if old_offset != u64::MAX && size != u64::MAX {
                let old_offset_usize = usize::try_from(old_offset).unwrap();
                let size_usize = usize::try_from(size).unwrap();

                // Validate bounds
                if old_offset_usize + size_usize > bytes.len() {
                    return Err(CodecError::Other(
                        "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                            .to_string(),
                    ));
                }

                // Copy chunk data
                compact_shard[write_offset..write_offset + size_usize]
                    .copy_from_slice(&bytes[old_offset_usize..old_offset_usize + size_usize]);

                // Update index
                new_index[i * 2] = u64::try_from(write_offset).unwrap();
                new_index[i * 2 + 1] = size;

                write_offset += size_usize;
            }
        }

        // Encode and write index
        let index_bytes = transmute_to_bytes_vec(new_index);
        let encoded_index =
            self.index_codecs
                .encode(ArrayBytes::from(index_bytes), &index_shape, options)?;

        match self.index_location {
            ShardingIndexLocation::Start => {
                compact_shard[..encoded_index.len()].copy_from_slice(&encoded_index);
            }
            ShardingIndexLocation::End => {
                let index_start = compact_size - encoded_index.len();
                compact_shard[index_start..].copy_from_slice(&encoded_index);
            }
        }

        Ok(Some(CowBytes::from(compact_shard)))
    }

    fn decode_into(
        &self,
        encoded_shard: CowBytes<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let data_type = self.data_type();
        let fill_value = self.fill_value();

        // Optional data is decoded into views of its inner data and each validity mask
        let (output_view, mask_views) = extract_target_views(&output_target);
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let num_chunks = chunks_per_shard
            .iter()
            .map(|i| usize::try_from(i.get()).unwrap())
            .product::<usize>();

        let shard_index =
            self.decode_index(&encoded_shard, chunks_per_shard.as_slice(), options)?;

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_subchunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shape)?,
            &self
                .inner_codecs
                .recommended_concurrency(&self.subchunk_shape)?,
        );
        let options = options.with_concurrent_target(concurrency_limit_subchunks);

        let subchunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        let decode_chunk = |chunk_index: usize| {
            let chunk_subset = subchunk_subset(&subchunk_grid, chunk_index);

            let output_subset_chunk = ArraySubset::new_with_start_shape(
                std::iter::zip(
                    output_view.subset().start().iter(),
                    chunk_subset.start().iter(),
                )
                .map(|(o, s)| o + s)
                .collect(),
                chunk_subset.shape().to_vec(),
            )
            .unwrap();
            let mut output_view_subchunk = unsafe {
                // SAFETY: subchunks represent disjoint array subsets
                output_view.subdivide(output_subset_chunk.clone())?
            };
            let mut mask_views_subchunk = mask_views
                .iter()
                .map(|mask_view| unsafe {
                    // SAFETY: subchunks represent disjoint array subsets
                    mask_view.subdivide(output_subset_chunk.clone())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_target_subchunk =
                build_nested_optional_target(&mut output_view_subchunk, &mut mask_views_subchunk);

            // Read the offset/size
            let offset = shard_index[chunk_index * 2];
            let size = shard_index[chunk_index * 2 + 1];
            if offset == u64::MAX && size == u64::MAX {
                fill_target(output_target_subchunk, data_type, fill_value)?;
            } else if usize::try_from(offset + size).unwrap() > encoded_shard.len() {
                return Err(CodecError::Other(
                    "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                        .to_string(),
                ));
            } else {
                let offset: usize = offset.try_into().unwrap();
                let size: usize = size.try_into().unwrap();
                let encoded_chunk = encoded_shard.clone().slice(offset..offset + size);
                self.inner_codecs.decode_into(
                    encoded_chunk,
                    &chunk_subset.chunk_shape().expect("nonempty subchunk"),
                    output_target_subchunk,
                    &options,
                )?;
            }

            Ok::<_, CodecError>(())
        };

        (0..num_chunks)
            .concurrent_limit(shard_concurrent_limit)
            .try_for_each(decode_chunk)?;

        Ok(())
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ShardingPartialDecoder::new(
            input_handle,
            ChunkShape::from(shape.to_vec()),
            self.subchunk_shape.clone(),
            self.inner_codecs.clone(),
            &self.index_codecs,
            self.index_location,
            options,
            self.options.clone(),
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            AsyncShardingPartialDecoder::new(
                input_handle,
                ChunkShape::from(shape.to_vec()),
                self.subchunk_shape.clone(),
                self.inner_codecs.clone(),
                &self.index_codecs,
                self.index_location,
                options,
                self.options.clone(),
            )
            .await?,
        ))
    }
    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            sharding_partial_encoder::ShardingPartialEncoder::new(
                input_output_handle,
                ChunkShape::from(shape.to_vec()),
                self.subchunk_shape.clone(),
                self.inner_codecs.clone(),
                self.index_codecs.clone(),
                self.index_location,
                options,
                self.options.clone(),
            )?,
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            super::sharding_partial_encoder_async::AsyncShardingPartialEncoder::new(
                input_output_handle,
                ChunkShape::from(shape.to_vec()),
                self.subchunk_shape.clone(),
                self.inner_codecs.clone(),
                self.index_codecs.clone(),
                self.index_location,
                options,
                self.options.clone(),
            )
            .await?,
        ))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        // Get the maximum size of encoded chunks
        let chunk_bytes_representation = self.inner_codecs.encoded_representation(shape)?;

        match chunk_bytes_representation {
            BytesRepresentation::BoundedSize(size) | BytesRepresentation::FixedSize(size) => {
                let chunks_per_shard =
                    calculate_chunks_per_shard(shape, self.subchunk_shape.as_slice())?;
                let index_encoded_size = compute_index_encoded_size(
                    self.index_codecs.as_ref(),
                    &sharding_index_shape(chunks_per_shard.as_slice()),
                )?;
                let shard_size = Self::encoded_shard_bounded_size(
                    index_encoded_size,
                    size,
                    chunks_per_shard.as_slice(),
                );
                Ok(BytesRepresentation::BoundedSize(shard_size))
            }
            BytesRepresentation::UnboundedSize => Ok(BytesRepresentation::UnboundedSize),
        }
    }
}

impl ShardingCodecBound {
    /// Return the subchunk shape.
    #[must_use]
    pub fn subchunk_shape(&self) -> &ChunkShape {
        &self.subchunk_shape
    }

    /// Return the codecs used to encode and decode subchunks.
    #[must_use]
    pub fn inner_codecs(&self) -> &Arc<CodecChainBound> {
        &self.inner_codecs
    }

    /// Return the codecs used to encode and decode the shard index.
    #[must_use]
    pub fn index_codecs(&self) -> &Arc<CodecChainBound> {
        &self.index_codecs
    }

    /// Return the shard index location.
    #[must_use]
    pub fn index_location(&self) -> ShardingIndexLocation {
        self.index_location
    }

    /// Return the runtime sharding options.
    #[must_use]
    pub fn options(&self) -> &ShardingCodecOptions {
        &self.options
    }

    /// Computed the bounded size of an encoded shard from
    ///  - the chunk bytes representation, and
    ///  - the number of chunks per shard.
    ///
    /// Equal to `num chunks * max chunk size + index size`
    fn encoded_shard_bounded_size(
        index_encoded_size: u64,
        chunk_encoded_size: u64,
        chunks_per_shard: &[NonZeroU64],
    ) -> u64 {
        let num_chunks = chunks_per_shard.iter().map(|i| i.get()).product::<u64>();
        num_chunks * chunk_encoded_size + index_encoded_size
    }

    /// Encode an inner chunk from the `decoded_value` or return None.
    fn encode_inner_by_chunk_index(
        &self,
        chunk_index: usize,
        decoded_value: &ArrayBytes,
        subchunk_grid: &RegularBoundedChunkGrid,
        options_inner: &CodecOptions,
    ) -> Option<Result<(usize, Bytes), CodecError>> {
        let data_type = self.inner_codecs.data_type();
        let fill_value = self.inner_codecs.fill_value();
        let chunk_subset = subchunk_subset(subchunk_grid, chunk_index);

        let bytes = decoded_value.extract_array_subset(
            &chunk_subset,
            subchunk_grid.array_shape(),
            data_type,
        );
        let bytes = match bytes {
            Ok(bytes) => bytes,
            Err(err) => return Some(Err(err)),
        };

        let is_fill_value = bytes.is_fill_value(fill_value);
        if is_fill_value {
            None
        } else {
            let chunk_shape = chunk_subset.chunk_shape().expect("nonempty subchunk");
            let encoded_chunk = self.inner_codecs.encode(bytes, &chunk_shape, options_inner);
            match encoded_chunk {
                Ok(encoded_chunk) => Some(Ok((chunk_index, encoded_chunk.into_bytes()))),
                Err(err) => Some(Err(err)),
            }
        }
    }

    /// Preallocate shard, encode and write chunks (in parallel), then truncate shard
    #[allow(clippy::too_many_lines)]
    fn encode_bounded(
        &self,
        decoded_value: &ArrayBytes,
        shard_shape: &[NonZeroU64],
        subchunk_shape: &[NonZeroU64],
        chunk_size_bounded: u64,
        options: &CodecOptions,
    ) -> Result<Vec<u8>, CodecError> {
        // Calculate maximum possible shard size
        let chunks_per_shard = calculate_chunks_per_shard(shard_shape, subchunk_shape)?;
        let subchunk_grid = subchunk_grid(shard_shape, subchunk_shape)?;
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;
        let shard_size_bounded = Self::encoded_shard_bounded_size(
            index_encoded_size,
            chunk_size_bounded,
            chunks_per_shard.as_slice(),
        );

        let shard_size_bounded = usize::try_from(shard_size_bounded).unwrap();
        let index_encoded_size = usize::try_from(index_encoded_size).unwrap();

        // Allocate an array for the shard
        let mut shard = Vec::with_capacity(shard_size_bounded);

        // Allocate the decoded shard index
        let mut shard_index = vec![u64::MAX; index_shape.num_elements_usize()];
        let encoded_shard_offset: usize = match self.index_location {
            ShardingIndexLocation::Start => index_encoded_size,
            ShardingIndexLocation::End => 0,
        };

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_subchunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shard_shape)?,
            &self.inner_codecs.recommended_concurrency(subchunk_shape)?,
        );
        let options = options.with_concurrent_target(concurrency_limit_subchunks);

        let n_chunks = chunks_per_shard
            .iter()
            .map(|i| usize::try_from(i.get()).unwrap())
            .product::<usize>();
        let shard_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut shard);
        // Encode the shards and update the shard index
        let encoded_shard_offset = match self.options.subchunk_write_order() {
            SubchunkWriteOrder::Unordered | SubchunkWriteOrder::Random => {
                let encoded_shard_offset_atomic: AtomicUsize = encoded_shard_offset.into();
                let shard_index_slice = UnsafeCellSlice::new(&mut shard_index);
                (0..n_chunks)
                    .concurrent_limit(shard_concurrent_limit)
                    .try_for_each(|chunk_index: usize| {
                        let maybe_chunk_encoded_with_id = self.encode_inner_by_chunk_index(
                            chunk_index,
                            decoded_value,
                            &subchunk_grid,
                            &options,
                        );
                        if let Some(chunk_encoded_with_id) = maybe_chunk_encoded_with_id {
                            // We don't need to worry about the id because the order here is random.
                            let chunk_encoded = chunk_encoded_with_id?.1;
                            let chunk_offset = encoded_shard_offset_atomic.fetch_add(
                                chunk_encoded.len(),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            if chunk_offset + chunk_encoded.len() > shard_size_bounded {
                                // This is a dev error, indicates the codec bounded size is not correct
                                return Err(CodecError::from(
                                    "Sharding did not allocate a large enough buffer",
                                ));
                            }

                            unsafe {
                                let shard_index_unsafe = shard_index_slice
                                    .index_mut(chunk_index * 2..chunk_index * 2 + 2);
                                shard_index_unsafe[0] = u64::try_from(chunk_offset).unwrap();
                                shard_index_unsafe[1] = u64::try_from(chunk_encoded.len()).unwrap();

                                shard_slice
                                    .index_mut(chunk_offset..chunk_offset + chunk_encoded.len())
                                    .copy_from_slice(&chunk_encoded);
                            }
                        }
                        Ok(())
                    })?;
                Ok::<_, CodecError>(
                    encoded_shard_offset_atomic.load(std::sync::atomic::Ordering::Relaxed),
                )
            }
            SubchunkWriteOrder::C => {
                // TODO: Replace this chunk order with the desired order i.e., a `Vec` of the ids i.e., for morton order.
                let chunk_order = 0..n_chunks;
                let encoded_chunk_ids_and_chunks: Vec<(usize, Bytes)> = chunk_order
                    .concurrent_limit(shard_concurrent_limit)
                    .filter_map(|chunk_index: usize| {
                        self.encode_inner_by_chunk_index(
                            chunk_index,
                            decoded_value,
                            &subchunk_grid,
                            &options,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let total_offset = encoded_chunk_ids_and_chunks.iter().fold(
                    encoded_shard_offset,
                    |acc: usize, (i, chunk)| {
                        let chunk_len_usize = chunk.len();
                        let chunk_length = u64::try_from(chunk_len_usize).unwrap();
                        let chunk_offset = u64::try_from(acc).unwrap();
                        let shard_index_unsafe = shard_index.index_mut(i * 2..i * 2 + 2);
                        shard_index_unsafe[0] = chunk_offset;
                        shard_index_unsafe[1] = chunk_length;
                        acc + chunk_len_usize
                    },
                );
                if total_offset > shard_size_bounded {
                    // This is a dev error, indicates the codec bounded size is not correct
                    return Err(CodecError::from(
                        "Sharding did not allocate a large enough buffer",
                    ));
                }
                encoded_chunk_ids_and_chunks
                    .concurrent_limit(shard_concurrent_limit)
                    .for_each(|(chunk_index, chunk): (usize, Bytes)| unsafe {
                        let shard_index_loc = &shard_index[chunk_index * 2..chunk_index * 2 + 2];
                        let chunk_offset = usize::try_from(shard_index_loc[0]).unwrap();
                        let chunk_encoded_len = usize::try_from(shard_index_loc[1]).unwrap();

                        shard_slice
                            .index_mut(chunk_offset..chunk_offset + chunk_encoded_len)
                            .copy_from_slice(&chunk);
                    });
                Ok(total_offset)
            }
        }?;

        // Truncate shard
        let shard_length = encoded_shard_offset
            + match self.index_location {
                ShardingIndexLocation::Start => 0,
                ShardingIndexLocation::End => index_encoded_size,
            };
        if shard_length > shard_size_bounded {
            // This is a dev error, indicates the codec bounded size is not correct.
            // The chunks may individually fit within the bounded size while the
            // chunks plus the shard index do not.
            return Err(CodecError::from(
                "Sharding did not allocate a large enough buffer",
            ));
        }

        // Encode and write array index
        let shard_index_bytes: CowBytes = transmute_to_bytes_vec(shard_index).into();
        let encoded_array_index =
            self.index_codecs
                .encode(shard_index_bytes.into(), &index_shape, &options)?;
        {
            // SAFETY: `shard_slice` is not read from until it has been written
            let shard_slice = unsafe { crate::vec_spare_capacity_to_mut_slice(&mut shard) };
            match self.index_location {
                ShardingIndexLocation::Start => {
                    shard_slice[..encoded_array_index.len()].copy_from_slice(&encoded_array_index);
                }
                ShardingIndexLocation::End => {
                    shard_slice[shard_length - encoded_array_index.len()..shard_length]
                        .copy_from_slice(&encoded_array_index);
                }
            }
        }
        // SAFETY: all elements have been initialised
        unsafe { shard.set_len(shard_length) };
        Ok(shard)
    }

    /// Encode subchunks (in parallel), then allocate shard, then write to shard (in parallel)
    // TODO: Collecting chunks then allocating shard can use a lot of memory, have a low memory variant
    // TODO: Also benchmark performance with just performing an alloc like 1x decoded size and writing directly into it, growing if needed
    #[allow(clippy::too_many_lines)]
    fn encode_unbounded(
        &self,
        decoded_value: &ArrayBytes,
        shard_shape: &[NonZeroU64],
        subchunk_shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Vec<u8>, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(shard_shape, subchunk_shape)?;
        let subchunk_grid = subchunk_grid(shard_shape, subchunk_shape)?;
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;
        let index_encoded_size = usize::try_from(index_encoded_size).unwrap();

        // Find chunks that are not entirely the fill value and collect their decoded bytes
        let n_chunks = chunks_per_shard
            .iter()
            .map(|i| usize::try_from(i.get()).unwrap())
            .product::<usize>();

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_subchunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shard_shape)?,
            &self.inner_codecs.recommended_concurrency(subchunk_shape)?,
        );
        let options_inner = options.with_concurrent_target(concurrency_limit_subchunks);

        #[cfg(not(target_arch = "wasm32"))]
        let iterator = match self.options.subchunk_write_order() {
            SubchunkWriteOrder::Unordered | SubchunkWriteOrder::Random | SubchunkWriteOrder::C => {
                (0..n_chunks).into_par_iter()
            }
        };
        #[cfg(target_arch = "wasm32")]
        let iterator = match self.options.subchunk_write_order() {
            SubchunkWriteOrder::Unordered | SubchunkWriteOrder::Random | SubchunkWriteOrder::C => {
                0..n_chunks
            }
        };

        let encoded_chunks: Vec<(usize, Bytes)> = iterator
            .concurrent_limit(shard_concurrent_limit)
            .filter_map(|chunk_index| {
                self.encode_inner_by_chunk_index(
                    chunk_index,
                    decoded_value,
                    &subchunk_grid,
                    &options_inner,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Allocate the shard
        let encoded_chunk_length = encoded_chunks
            .iter()
            .map(|(_, bytes)| bytes.len())
            .sum::<usize>();
        let shard_length = encoded_chunk_length + index_encoded_size;
        let mut shard = Vec::with_capacity(shard_length);

        // Allocate the decoded shard index
        let mut shard_index = vec![u64::MAX; index_shape.num_elements_usize()];
        let encoded_shard_offset = match self.index_location {
            ShardingIndexLocation::Start => index_encoded_size,
            ShardingIndexLocation::End => 0,
        };
        let shard_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut shard);

        // Write shard and update shard index
        if !encoded_chunks.is_empty() {
            match self.options.subchunk_write_order() {
                SubchunkWriteOrder::Unordered | SubchunkWriteOrder::Random => {
                    let encoded_shard_offset_atomic: AtomicUsize = encoded_shard_offset.into();
                    let shard_index_slice = UnsafeCellSlice::new(&mut shard_index);
                    encoded_chunks
                        .concurrent_limit(options.concurrent_target())
                        .for_each(|(chunk_index, chunk_encoded): (usize, Bytes)| {
                            let chunk_offset = encoded_shard_offset_atomic.fetch_add(
                                chunk_encoded.len(),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            unsafe {
                                let shard_index_unsafe = shard_index_slice
                                    .index_mut(chunk_index * 2..chunk_index * 2 + 2);
                                shard_index_unsafe[0] = u64::try_from(chunk_offset).unwrap();
                                shard_index_unsafe[1] = u64::try_from(chunk_encoded.len()).unwrap();

                                shard_slice
                                    .index_mut(chunk_offset..chunk_offset + chunk_encoded.len())
                                    .copy_from_slice(&chunk_encoded);
                            }
                        });
                }
                SubchunkWriteOrder::C => {
                    let mut offset = encoded_shard_offset;
                    for (i, chunk) in &encoded_chunks {
                        let chunk_len_usize = chunk.len();
                        let chunk_length = u64::try_from(chunk_len_usize).unwrap();
                        let chunk_offset = u64::try_from(offset).unwrap();
                        let shard_index_unsafe = shard_index.index_mut(i * 2..i * 2 + 2);
                        shard_index_unsafe[0] = chunk_offset;
                        shard_index_unsafe[1] = chunk_length;
                        offset += chunk_len_usize;
                    }
                    encoded_chunks
                        .concurrent_limit(options.concurrent_target())
                        .for_each(|(chunk_index, chunk): (usize, Bytes)| unsafe {
                            let shard_index_loc =
                                &shard_index[chunk_index * 2..chunk_index * 2 + 2];
                            let chunk_offset = usize::try_from(shard_index_loc[0]).unwrap();
                            let chunk_encoded_len = usize::try_from(shard_index_loc[1]).unwrap();

                            shard_slice
                                .index_mut(chunk_offset..chunk_offset + chunk_encoded_len)
                                .copy_from_slice(&chunk);
                        });
                }
            }
        }

        // Write shard index
        let encoded_array_index = self.index_codecs.encode(
            ArrayBytes::from(transmute_to_bytes_vec(shard_index)),
            &index_shape,
            options,
        )?;
        {
            // SAFETY: `shard_slice` is not read from until it has been written
            let shard_slice = unsafe { crate::vec_spare_capacity_to_mut_slice(&mut shard) };
            match self.index_location {
                ShardingIndexLocation::Start => {
                    shard_slice[..encoded_array_index.len()].copy_from_slice(&encoded_array_index);
                }
                ShardingIndexLocation::End => {
                    shard_slice[shard_length - encoded_array_index.len()..]
                        .copy_from_slice(&encoded_array_index);
                }
            }
        }
        // SAFETY: all elements have been initialised
        unsafe { shard.set_len(shard_length) };
        Ok(shard)
    }

    /// Decode the shard index inside the given encoded shard.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the decoded shard index is not valid.
    ///
    /// # Panics
    /// Panics if the encoded index size or the encoded shard minux its index length exceeds [`usize::MAX`].
    pub(crate) fn decode_index(
        &self,
        encoded_shard: &[u8],
        chunks_per_shard: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError> {
        // Get index array representation and encoded size
        let index_shape = sharding_index_shape(chunks_per_shard);
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;

        // Get encoded shard index
        if (encoded_shard.len() as u64) < index_encoded_size {
            return Err(CodecError::Other(
                "The encoded shard is smaller than the expected size of its index.".to_string(),
            ));
        }

        let encoded_shard_index = match self.index_location {
            ShardingIndexLocation::Start => {
                &encoded_shard[..index_encoded_size.try_into().unwrap()]
            }
            ShardingIndexLocation::End => {
                let encoded_shard_offset =
                    usize::try_from(encoded_shard.len() as u64 - index_encoded_size).unwrap();
                &encoded_shard[encoded_shard_offset..]
            }
        };

        // Decode the shard index
        decode_shard_index(
            encoded_shard_index,
            &index_shape,
            self.index_codecs.as_ref(),
            options,
        )
    }
}

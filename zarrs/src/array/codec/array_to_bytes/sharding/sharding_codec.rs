use itertools::Itertools;
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};

use zarrs_chunk_grid::{ChunkGridCreateError, ChunkGridTraits};

use super::{
    CodecChain, RequireDivisibleSubchunks, ShardingCodecConfiguration,
    ShardingCodecConfigurationV1, ShardingCodecOptions, ShardingIndexLocation, SubchunkWriteOrder,
    calculate_chunks_per_shard, compute_index_encoded_size, decode_shard_index,
    nested_local_subchunk_grids, sharding_index_shape, subchunk_grid,
};
use crate::array::chunk_grid::repeat::RepeatChunkGrid;
use crate::array::chunk_grid::{
    ChunkEdgeLengths, RectilinearChunkGrid, RegularBoundedChunkGrid, RegularChunkGrid,
};
use crate::array::{
    ArrayBytes, ArraySubset, BytesRepresentation, ChunkGrid, ChunkShape, ChunkShapeTraits,
    CodecChainBound, CowBytes, DataType, FillValue, transmute_to_bytes_vec,
};
use smallvec::SmallVec;
use zarrs_codec::plan::{
    ByteSource, DecodeRequest, PlanIndexer, SharedOutput, TargetViews, Task, TaskTarget,
    drive_memory_one, encode_hint, encoded_value,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayToBytesCodecTraits, ChunkGridDecoded,
    ChunkGridDecodedRef, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, ComputeResources, MultipartWrite,
    UnboundArrayToBytesCodecTraits,
};
use zarrs_metadata::Configuration;
use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};
use zarrs_storage::byte_range::ByteRange;

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
pub(super) fn subchunk_subset(
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_index: usize,
) -> ArraySubset {
    let subchunk_indices =
        zarrs_chunk_grid::unravel_index(subchunk_index as u64, subchunk_grid.grid_shape())
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
    /// The codec, for planning from methods that borrow it (planners take an [`Arc`] of the codec).
    pub(crate) this: Weak<ShardingCodecBound>,
}

pub(super) fn subchunk_order_iterator(
    subchunks_write_order: SubchunkWriteOrder,
    n_chunks: usize,
    subchunk_grid: &RegularBoundedChunkGrid,
) -> Vec<usize> {
    match subchunks_write_order {
        SubchunkWriteOrder::C
        | SubchunkWriteOrder::Lexicographic
        | SubchunkWriteOrder::Unordered => (0..n_chunks).collect_vec(),
        SubchunkWriteOrder::Colexicographic => (0..n_chunks)
            .sorted_by_cached_key(|&i| {
                let mut indices =
                    zarrs_chunk_grid::unravel_index(i as u64, subchunk_grid.grid_shape())
                        .expect("inbounds subchunk");
                indices.reverse();
                indices
            })
            .collect_vec(),
        SubchunkWriteOrder::Morton => (0..n_chunks)
            .map(|i| {
                let indices = zarrs_chunk_grid::unravel_index(i as u64, subchunk_grid.grid_shape())
                    .expect("inbounds subchunk");
                (indices, i)
            })
            .sorted_unstable_by(|(a, _), (b, _)| morton_cmp(a, b))
            .map(|(_, i)| i)
            .collect_vec(),
    }
}

/// Compare two indices in Morton (Z) order, first axis most significant: <https://en.wikipedia.org/wiki/Z-order_curve#Efficiently_building_quadtrees_and_octrees>
fn morton_cmp(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    // `x < y && x < x ^ y` iff the highest set bit of `x` is below that of `y`.
    let less_msb = |x: u64, y: u64| x < y && x < (x ^ y);
    let mut dim = 0;
    let mut msb = 0;
    for (d, (&ai, &bi)) in a.iter().zip(b).enumerate() {
        let diff = ai ^ bi;
        if less_msb(msb, diff) {
            msb = diff;
            dim = d;
        }
    }
    a[dim].cmp(&b[dim])
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
}

impl UnboundArrayToBytesCodecTraits for ShardingCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        if fill_value.is_none() {
            return Err(CodecCreateError::Other(
                "sharding requires a defined fill value".into(),
            ));
        }
        let inner_codecs =
            self.inner_codecs
                .with_context(data_type, fill_value, codec_specific_options)?;
        let index_codecs = self.index_codecs.with_context(
            crate::array::data_type::uint64(),
            Some(FillValue::from(u64::MAX)),
            codec_specific_options,
        )?;
        let options = codec_specific_options
            .get_option::<ShardingCodecOptions>()
            .unwrap_or(&self.options)
            .clone();
        let require_divisible_subchunks = codec_specific_options
            .get_option::<RequireDivisibleSubchunks>()
            .is_some();
        Ok(Arc::new_cyclic(|this| ShardingCodecBound {
            subchunk_shape: self.subchunk_shape.clone(),
            inner_codecs,
            index_codecs,
            index_location: self.index_location,
            options,
            require_divisible_subchunks,
            this: this.clone(),
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

    fn fill_value(&self) -> Option<&FillValue> {
        self.inner_codecs.fill_value()
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

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        let mut codecs = vec![self.inner_codecs.clone().into_dyn()];
        codecs.extend(self.inner_codecs.subchunk_codecs());
        codecs
    }
}

impl ArrayToBytesCodecTraits for ShardingCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let codec = self.arc();
        let hint = encode_hint(&bytes);
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        let task = Task::compute(move |ctx| codec.encode_shard_step(bytes, &shape, &options, ctx))
            .with_hint(hint);
        encoded_value(drive_memory_one(task, resources)?)
    }

    fn decode<'a>(
        &self,
        encoded_shard: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let data_type = self.data_type();
        let shard_shape_u64: &[u64] = bytemuck::must_cast_slice(shape);

        if !data_type.is_optional() && data_type.fixed_size() == Some(0) {
            return Ok(ArrayBytes::new_flen(vec![]));
        }
        // Fixed length data (including optional data with fixed length inner data): decode each subchunk directly into the output
        if let Some((output, views)) = SharedOutput::for_data_type(data_type, shard_shape_u64) {
            views.with_target(|target| {
                self.decode_into(encoded_shard.into(), shape, target, options, resources)
            })?;
            // SAFETY: every element of the output (and masks) is written by `decode_into`
            return unsafe { output.into_array_bytes() }.map_err(|_| {
                CodecError::Other("a shard output view outlived its decode".to_string())
            });
        }

        // Variable length data (including optional data with variable length inner data): decode each subchunk and merge
        let task = self.arc().plan_decode_shard(
            ByteSource::memory(encoded_shard),
            shape,
            DecodeRequest {
                indexer: PlanIndexer::Subset(ArraySubset::new_with_shape(shard_shape_u64)),
                target: TaskTarget::Value,
            },
            options,
        )?;
        drive_memory_one(task, resources)?
            .into_array()
            .ok_or_else(|| {
                CodecError::Other("a shard decode did not produce array bytes".to_string())
            })
    }

    fn compact<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<Option<CowBytes<'a>>, CodecError> {
        // Calculate chunks per shard
        let chunks_per_shard = calculate_chunks_per_shard(shape, self.subchunk_shape.as_slice())?;

        // Decode the shard index
        let shard_index =
            self.decode_index(&bytes, chunks_per_shard.as_slice(), options, resources)?;

        // Get index metadata
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;

        // Validate the byte ranges of the encoded subchunks
        let subchunk_byte_ranges = shard_index
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[offset, size]| {
                if offset == u64::MAX && size == u64::MAX {
                    Ok(None)
                } else {
                    ByteRange::FromStart(offset, Some(size))
                        .checked_range_usize(bytes.len())
                        .map(Some)
                        .ok_or_else(shard_index_out_of_bounds)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let data_size = subchunk_byte_ranges
            .iter()
            .flatten()
            .try_fold(0usize, |sum, byte_range| sum.checked_add(byte_range.len()))
            .ok_or_else(shard_index_out_of_bounds)?;
        let index_encoded_size = usize::try_from(index_encoded_size).unwrap();

        // Check if compaction is needed (no-op optimization)
        if data_size == bytes.len() - index_encoded_size {
            return Ok(None); // No compaction needed
        }

        // Build compacted shard
        let compact_size = data_size
            .checked_add(index_encoded_size)
            .ok_or_else(shard_index_out_of_bounds)?;
        let mut compact_shard = vec![0u8; compact_size];
        let mut new_index = vec![u64::MAX; shard_index.len()];

        let mut write_offset = match self.index_location {
            ShardingIndexLocation::Start => index_encoded_size,
            ShardingIndexLocation::End => 0,
        };

        for (i, byte_range) in subchunk_byte_ranges.into_iter().enumerate() {
            if let Some(byte_range) = byte_range {
                // Copy chunk data
                let size = byte_range.len();
                compact_shard[write_offset..write_offset + size]
                    .copy_from_slice(&bytes[byte_range]);

                // Update index
                new_index[i * 2] = u64::try_from(write_offset).unwrap();
                new_index[i * 2 + 1] = u64::try_from(size).unwrap();

                write_offset += size;
            }
        }

        // Encode and write index
        let index_bytes = transmute_to_bytes_vec(new_index);
        let encoded_index = self.index_codecs.encode(
            ArrayBytes::from(index_bytes),
            &index_shape,
            options,
            resources,
        )?;

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
        input: zarrs_codec::ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        let encoded_shard = input.into_bytes(options, resources)?;
        // SAFETY: the views of the target are only used through the whole views while the drive runs
        let views = unsafe { TargetViews::from_target(&output_target) }?;
        let task = self.arc().plan_decode_shard(
            ByteSource::memory(encoded_shard),
            shape,
            DecodeRequest {
                indexer: PlanIndexer::Subset(ArraySubset::new_with_shape(
                    bytemuck::must_cast_slice(shape),
                )),
                target: TaskTarget::Views(views),
            },
            options,
        )?;
        drive_memory_one(task, resources)?;
        Ok(())
    }

    fn plan_decode<'a>(
        self: Arc<Self>,
        source: zarrs_codec::plan::ByteSource<'a>,
        shape: &[NonZeroU64],
        request: zarrs_codec::plan::DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'a>, CodecError> {
        self.plan_decode_shard(source, shape, request, options)
    }

    fn plan_local_subchunk_grids<'a>(
        self: Arc<Self>,
        _source: zarrs_codec::plan::ByteSource<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'a>, CodecError> {
        // The subchunk grid depends only on the shard shape
        let grids = nested_local_subchunk_grids(
            ChunkGrid::new(subchunk_grid(shape, &self.subchunk_shape)?),
            &self.inner_codecs,
        )?;
        Ok(zarrs_codec::plan::Task::compute(move |_| {
            Ok(zarrs_codec::plan::Outcome::Value(
                zarrs_codec::plan::local_subchunk_grids_value(grids),
            ))
        }))
    }

    fn encode_step<'a>(
        self: Arc<Self>,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        ctx: &zarrs_codec::plan::ComputeCtx,
    ) -> Result<zarrs_codec::plan::Outcome<'a>, CodecError> {
        if ctx.runs_inline() {
            // Tasks of the subchunks would run serially
            let encoded = self.encode(bytes, shape, options, ctx.resources())?;
            return Ok(zarrs_codec::plan::Outcome::Value(
                zarrs_codec::plan::TaskValue::Bytes(Some(encoded)),
            ));
        }
        self.encode_shard_step(bytes, shape, options, ctx)
    }

    fn encode_write_step<'a>(
        self: Arc<Self>,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        key: zarrs_codec::plan::SourceKey,
        options: &CodecOptions,
        ctx: &zarrs_codec::plan::ComputeCtx,
    ) -> Result<zarrs_codec::plan::Outcome<'a>, CodecError> {
        // A shard is written as it is encoded with a multipart write, unless its tasks would run serially or it would
        // be written in one part. An index at the start is written last, over a placeholder, so the storage must
        // support patching multipart writes.
        if ctx.runs_inline()
            || options.multipart_write() != MultipartWrite::Always
            || bytes.size() <= options.multipart_part_size()
            || (self.index_location == ShardingIndexLocation::Start && !options.multipart_patch())
        {
            return zarrs_codec::plan::encode_and_set(self, bytes, shape, key, options, ctx);
        }
        if !options.store_empty_chunks()
            && self
                .inner_codecs
                .fill_value()
                .is_some_and(|fill| bytes.is_fill_value(fill))
        {
            return Ok(zarrs_codec::plan::Outcome::Write {
                ops: smallvec::smallvec![zarrs_codec::plan::WriteOp::Erase(key)],
                then: zarrs_codec::plan::TaskValue::None,
            });
        }
        self.encode_shard_write_step(bytes, shape, key, options, ctx)
    }

    fn plan_partial_encode<'a>(
        self: Arc<Self>,
        key: zarrs_codec::plan::SourceKey,
        shape: &[NonZeroU64],
        indexer: zarrs_codec::plan::PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        state: Option<zarrs_codec::plan::PartialEncodeState>,
        set_partial: bool,
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'a>, CodecError> {
        if indexer.is_whole(shape) {
            return Ok(zarrs_codec::plan::plan_partial_encode_whole(
                self, key, shape, indexer, bytes, options,
            ));
        }

        if set_partial {
            self.plan_partial_encode_shard(key, shape, indexer, bytes, state, options)
        } else {
            // Updated subchunks are appended to a shard, so the shard is rewritten compactly if it is written in full
            Ok(zarrs_codec::plan::plan_partial_encode_whole(
                self, key, shape, indexer, bytes, options,
            ))
        }
    }

    fn plan_locate_subchunk(
        self: Arc<Self>,
        source: zarrs_codec::plan::ByteSource<'static>,
        shape: &[NonZeroU64],
        coordinates: &[u64],
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'static>, CodecError> {
        self.plan_locate_subchunk_in_shard(source, shape, coordinates, options)
    }

    fn plan_store_encoded_subchunk(
        self: Arc<Self>,
        key: zarrs_codec::plan::SourceKey,
        shape: &[NonZeroU64],
        level: usize,
        coordinates: &[u64],
        encoded: Option<zarrs_storage::Bytes>,
        set_partial: bool,
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'static>, CodecError> {
        self.plan_store_encoded_subchunk_in_shard(
            key,
            shape,
            level,
            coordinates,
            encoded,
            set_partial,
            options,
        )
    }

    fn prepare_decode(
        self: Arc<Self>,
        source: zarrs_codec::plan::ByteSource<'static>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::PrepareDecode, CodecError> {
        self.prepare_decode_shard(source, shape, options)
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
                let num_chunks = chunks_per_shard.iter().map(|i| i.get()).product::<u64>();
                Ok(BytesRepresentation::BoundedSize(
                    num_chunks * size + index_encoded_size,
                ))
            }
            BytesRepresentation::UnboundedSize => Ok(BytesRepresentation::UnboundedSize),
        }
    }
}

impl ShardingCodecBound {
    /// The codec in an [`Arc`], for planning.
    pub(crate) fn arc(&self) -> Arc<Self> {
        // A clone of a bound codec refers to the codec it was cloned from
        self.this
            .upgrade()
            .unwrap_or_else(|| Arc::new(self.clone()))
    }

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
        resources: &ComputeResources,
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
            resources,
        )
    }
}

/// A corrupt shard index references bytes outside of the shard, or a subchunk end that overflows.
fn shard_index_out_of_bounds() -> CodecError {
    CodecError::Other(
        "The shard index references out-of-bounds bytes. The chunk may be corrupted.".to_string(),
    )
}

//! Planned decoding of the sharding codec.
//!
//! A shard is decoded by a task that reads the shard index, then plans each intersecting
//! subchunk with the inner codecs on a window of the shard. Subchunks are decoded directly into
//! their region of the target where possible, and otherwise as values that are merged.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_codec::plan::{
    ByteSource, ComputeCtx, DecodeRequest, Generator, Join, MissingDecoder, Outcome, PlanIndexer,
    PrepareDecode, PreparedDecoder, SharedOutput, SubchunkLocation, TargetViews, Task, TaskTarget,
    TaskValue, decode_whole, drive_memory_one, place_decoded, place_fill_value,
    subchunk_location_value, whole_chunk_hint,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoInput, ArrayCodecTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions,
    CowBytes, copy_fill_value_into,
};
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::byte_range::ByteRange;

use super::{
    IndexRead, ShardingCodecBound, calculate_chunks_per_shard, decode_shard_index,
    group_indices_by_subchunk, index_read, merge_indexer_subchunks, subchunk_grid,
    subchunk_offset_size,
};
use crate::array::array_bytes_internal::{merge_chunks, optional_innermost};
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{
    ArrayBytes, ArrayIndices, ArraySubset, ArraySubsetTraits, CodecChainBound, Indexer,
    ravel_indices, subset_indices_at,
};
use smallvec::SmallVec;

impl ShardingCodecBound {
    /// Plan the decoding of `request` from the shard `source`.
    pub(super) fn plan_decode_shard<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        request.indexer.validate(shape)?;
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let IndexRead {
            shape: index_shape,
            range: index_byte_range,
            hint: index_hint,
        } = index_read(&self, &chunks_per_shard)?;
        let subchunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        let options = *options;

        if request.indexer.is_whole(shape) {
            // Every subchunk is requested, so read the whole shard in one request and decode its subchunks from memory
            let hint = whole_chunk_hint(&*self, &source, shape);
            return Ok(Task::read(
                source,
                [ByteRange::FromStart(0, None)],
                move |mut reads, ctx| {
                    let Some(shard) = reads.pop().flatten() else {
                        return place_fill_value(
                            self.inner_codecs.data_type(),
                            self.inner_codecs.fill_value(),
                            request,
                        );
                    };
                    let source = ByteSource::memory(shard);
                    let encoded_index = source
                        .read_memory(index_byte_range)
                        .expect("the source is in memory")?;
                    let shard_index =
                        decode_index(&self, &encoded_index, &index_shape, &options, ctx)?;
                    drop(encoded_index);
                    self.plan_subchunks(source, shard_index, subchunk_grid, request, &options, ctx)
                },
            )
            .with_hint(hint));
        }

        let index_source = source.clone();
        Ok(
            Task::read(index_source, [index_byte_range], move |mut reads, ctx| {
                let Some(encoded_index) = reads.pop().flatten() else {
                    return place_fill_value(
                        self.inner_codecs.data_type(),
                        self.inner_codecs.fill_value(),
                        request,
                    );
                };
                let shard_index = decode_index(&self, &encoded_index, &index_shape, &options, ctx)?;
                drop(encoded_index);
                self.plan_subchunks(source, shard_index, subchunk_grid, request, &options, ctx)
            })
            .with_hint(index_hint),
        )
    }

    /// Plan locating the subchunk containing the element at `coordinates` of the shard `source` of `shape`.
    ///
    /// The task reads the shard index. Its value is the location of the subchunk
    /// ([`subchunk_location_value`]), which is [`None`] if the shard or the subchunk is empty.
    pub(super) fn plan_locate_subchunk_in_shard(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        coordinates: &[u64],
        options: &CodecOptions,
    ) -> Result<Task<'static>, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let subchunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        let out_of_bounds = || {
            CodecError::Other(format!(
                "coordinates {coordinates:?} are out of bounds of a shard with shape {shape:?}"
            ))
        };
        let subchunk_indices = subchunk_grid
            .chunk_indices(coordinates)?
            .ok_or_else(out_of_bounds)?;
        let subchunk_subset = subchunk_grid
            .subset(&subchunk_indices)?
            .ok_or_else(out_of_bounds)?;
        let subchunk_index = ravel_indices(
            &subchunk_indices,
            bytemuck::must_cast_slice(&chunks_per_shard),
        )
        .ok_or_else(out_of_bounds)?;
        let subchunk_index = usize::try_from(subchunk_index).unwrap();
        let coordinates: ArrayIndices = std::iter::zip(coordinates, subchunk_subset.start())
            .map(|(coordinate, start)| coordinate - start)
            .collect();
        let IndexRead {
            shape: index_shape,
            range: index_byte_range,
            hint: index_hint,
        } = index_read(&self, &chunks_per_shard)?;
        let options = *options;
        Ok(
            Task::read(source.clone(), [index_byte_range], move |mut reads, ctx| {
                let location = if let Some(encoded_index) = reads.pop().flatten() {
                    let shard_index =
                        decode_index(&self, &encoded_index, &index_shape, &options, ctx)?;
                    subchunk_offset_size(&shard_index, subchunk_index)
                        .map(|(offset, size)| {
                            Ok::<_, CodecError>(SubchunkLocation {
                                source: source.interval(offset, size)?,
                                shape: subset_shape(&subchunk_subset).to_vec(),
                                coordinates: coordinates.clone(),
                            })
                        })
                        .transpose()?
                } else {
                    None
                };
                Ok(Outcome::Value(subchunk_location_value(location)))
            })
            .with_hint(index_hint),
        )
    }

    /// Prepare a decoder of the shard `source`, which reads and decodes the shard index once.
    pub(super) fn prepare_decode_shard(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<PrepareDecode, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let IndexRead {
            shape: index_shape,
            range: index_byte_range,
            hint: index_hint,
        } = index_read(&self, &chunks_per_shard)?;
        let subchunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        Ok(PrepareDecode::Plan(
            Task::read(source.clone(), [index_byte_range], move |mut reads, ctx| {
                let decoder: Arc<dyn PreparedDecoder> =
                    if let Some(encoded_index) = reads.pop().flatten() {
                        let shard_index =
                            decode_index(&self, &encoded_index, &index_shape, &options, ctx)?;
                        Arc::new(ShardDecoder {
                            codec: self,
                            source,
                            shape,
                            shard_index: shard_index.into(),
                            subchunk_grid,
                        })
                    } else {
                        Arc::new(MissingDecoder::new(self.into_dyn(), &shape))
                    };
                Ok(Outcome::Value(TaskValue::Prepared(decoder)))
            })
            .with_hint(index_hint),
        ))
    }

    /// Plan the decoding of the subchunks of a shard that intersect `request`.
    fn plan_subchunks<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shard_index: impl ShardIndex + 'a,
        subchunk_grid: RegularBoundedChunkGrid,
        request: DecodeRequest<'a>,
        options: &CodecOptions,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let indexer = request.indexer.as_indexer();
        if indexer.is_empty() {
            return place_fill_value(
                self.inner_codecs.data_type(),
                self.inner_codecs.fill_value(),
                request,
            );
        }
        let Some(subset) = indexer
            .as_array_subset()
            .map(ArraySubsetTraits::to_array_subset)
        else {
            return self.plan_subchunks_indexer(
                &source,
                &shard_index,
                &subchunk_grid,
                request,
                options,
            );
        };

        // Fixed-length data decodes directly into the region of each subchunk in the target
        let fixed = optional_innermost(self.inner_codecs.data_type()).is_fixed();
        let parallelism = match ctx.resources().compute_target() {
            0 => rayon::current_num_threads(),
            target => target,
        };
        let subchunk_indices = subchunk_grid
            .chunks_in_array_subset(&subset)?
            .expect("subchunks are within the shard");
        match request.target {
            TaskTarget::Views(views) if fixed && views.subset().shape() == subset.shape() => {
                if subchunk_indices.num_elements() == 1 {
                    // A single subchunk may run without a separate task
                    let children = SubchunksIntoViews::generator(
                        self,
                        source,
                        shard_index,
                        subchunk_grid,
                        subset,
                        subchunk_indices,
                        views,
                        *options,
                        parallelism,
                    );
                    return match children.plan_index(0)? {
                        Some(child) if !child.needs_io() => child.run_without_io(ctx),
                        Some(child) => Ok(Outcome::Spawn(vec![child])),
                        None => Ok(Outcome::Done),
                    };
                }
                Ok(Outcome::Generate {
                    children: SubchunksIntoViews::generator(
                        self,
                        source,
                        shard_index,
                        subchunk_grid,
                        subset,
                        subchunk_indices,
                        views,
                        *options,
                        parallelism,
                    ),
                    join: None,
                })
            }
            TaskTarget::Value if fixed => {
                // Decode directly into an output allocated once, returned as the value
                let (output, views) =
                    SharedOutput::for_data_type(self.inner_codecs.data_type(), subset.shape())
                        .expect("fixed-length data");
                let children = SubchunksIntoViews::generator(
                    self.clone(),
                    source,
                    shard_index,
                    subchunk_grid,
                    subset,
                    subchunk_indices,
                    views,
                    *options,
                    parallelism,
                );
                Ok(Outcome::Generate {
                    children,
                    join: Some(Join::after(move |_ctx| {
                        // SAFETY: the subchunks cover the subset, and each wrote all elements of its region
                        let bytes = unsafe { output.into_array_bytes() }.map_err(|_| {
                            CodecError::Other(
                                "a subchunk output view outlived its task".to_string(),
                            )
                        })?;
                        Ok(Outcome::Value(TaskValue::Array(bytes)))
                    })),
                })
            }
            target @ (TaskTarget::Views(_) | TaskTarget::Value) => {
                // Decode the overlap of each subchunk as a value, then merge them
                let mut children = Vec::new();
                let mut overlaps = Vec::new();
                for subchunk in &subchunk_indices.indices() {
                    let subchunk_subset = subchunk_grid
                        .subset(&subchunk)?
                        .expect("subchunk is within the shard");
                    let overlap = subset.overlap(&subchunk_subset)?;
                    let subchunk_request = DecodeRequest {
                        indexer: PlanIndexer::Subset(overlap.relative_to(subchunk_subset.start())?),
                        target: TaskTarget::Value,
                    };
                    let child = self
                        .plan_subchunk(
                            &source,
                            &shard_index,
                            &subchunk_grid,
                            &subchunk,
                            subset_shape(&subchunk_subset),
                            subchunk_request,
                            options,
                        )?
                        .expect("a value is always planned");
                    children.push(child);
                    overlaps.push(overlap.relative_to(subset.start())?);
                }
                let shape = subset.shape().to_vec();
                Ok(Outcome::Fork {
                    children,
                    join: Join::new(move |values, _ctx| {
                        let chunks = std::iter::zip(values, overlaps)
                            .map(|(value, overlap)| Ok((array_value(value)?, overlap)))
                            .collect::<Result<Vec<_>, CodecError>>()?;
                        let merged = merge_chunks(chunks, &shape, self.inner_codecs.data_type())?;
                        place_decoded(merged, target)
                    }),
                })
            }
        }
    }

    /// Plan the decoding of the elements of a non-subset indexer by grouping them by subchunk.
    fn plan_subchunks_indexer<'a>(
        self: Arc<Self>,
        source: &ByteSource<'a>,
        shard_index: &[u64],
        subchunk_grid: &RegularBoundedChunkGrid,
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Outcome<'a>, CodecError> {
        let indexer = request.indexer.as_indexer();
        let num_elements = usize::try_from(indexer.len()).unwrap();
        let groups = group_indices_by_subchunk(subchunk_grid, &self.subchunk_shape, indexer);
        let mut children = Vec::with_capacity(groups.len());
        let mut positions = Vec::with_capacity(groups.len());
        for (subchunk_index, group) in groups {
            let subchunk = crate::array::unravel_index(subchunk_index, subchunk_grid.grid_shape())
                .expect("the subchunk is within the shard");
            let subchunk_subset = subchunk_grid
                .subset(&subchunk)?
                .expect("subchunk is within the shard");
            positions.push(group.positions.clone());
            let child = self
                .plan_subchunk(
                    source,
                    shard_index,
                    subchunk_grid,
                    &subchunk,
                    subset_shape(&subchunk_subset),
                    DecodeRequest {
                        indexer: PlanIndexer::Shared(Arc::new(group)),
                        target: TaskTarget::Value,
                    },
                    options,
                )?
                .expect("a value is always planned");
            children.push(child);
        }
        let target = request.target;
        Ok(Outcome::Fork {
            children,
            join: Join::new(move |values, _ctx| {
                let groups = std::iter::zip(values, positions)
                    .map(|(value, positions)| Ok((array_value(value)?.into_owned(), positions)))
                    .collect::<Result<Vec<_>, CodecError>>()?;
                let merged =
                    merge_indexer_subchunks(groups, num_elements, self.inner_codecs.data_type())?;
                place_decoded(merged, target)
            }),
        })
    }

    /// Plan the decoding of `request` from a subchunk.
    ///
    /// Returns [`None`] if the subchunk is empty and its target has been filled with the fill value.
    #[allow(clippy::too_many_arguments)]
    fn plan_subchunk<'a>(
        &self,
        source: &ByteSource<'a>,
        shard_index: &[u64],
        subchunk_grid: &RegularBoundedChunkGrid,
        subchunk: &[u64],
        subchunk_shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Task<'a>>, CodecError> {
        let shard_index_idx =
            ravel_indices(subchunk, subchunk_grid.grid_shape()).expect("inbounds subchunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        if let Some((offset, size)) = subchunk_offset_size(shard_index, shard_index_idx) {
            let subchunk_source = source
                .interval(offset, size)
                .map_err(|_| out_of_bounds_subchunk())?;
            Ok(Some(self.inner_codecs.clone().plan_decode(
                subchunk_source,
                subchunk_shape,
                request,
                options,
            )?))
        } else {
            match place_fill_value(
                self.inner_codecs.data_type(),
                self.inner_codecs.fill_value(),
                request,
            )? {
                Outcome::Done => Ok(None),
                outcome @ (Outcome::Value(_)
                | Outcome::Spawn(_)
                | Outcome::Fork { .. }
                | Outcome::Generate { .. }
                | Outcome::Write { .. }) => Ok(Some(Task::compute(move |_| Ok(outcome)))),
            }
        }
    }
}

/// The decoding of `subset` of a shard into views (with the shape of the subset), planned per subchunk or per block of
/// subchunks adjacent in the last dimension.
///
/// Where the shard is in memory and the inner codecs do not plan tasks of their own
/// ([`ArrayToBytesCodecTraits::plans_subtasks`]), a task decodes a block of subchunks with the inner codecs, bytes,
/// and views borrowed by the task, so that decoding a subchunk does not touch shared reference counts. Atomic
/// operations are full barriers on common architectures, which wait for the writes of a decoded subchunk to the
/// output, so they are costly per subchunk.
/// Otherwise, each subchunk is planned as a task with the inner codecs, which may read it or plan tasks of its own.
///
/// Empty subchunks are filled with the fill value.
struct SubchunksIntoViews<'a, I> {
    inner_codecs: Arc<CodecChainBound>,
    source: ByteSource<'a>,
    shard_index: I,
    subchunk_grid: RegularBoundedChunkGrid,
    subset: ArraySubset,
    /// The indices of the subchunks intersecting `subset`.
    subchunk_indices: ArraySubset,
    output_start: Vec<u64>,
    /// The regular shape of a subchunk.
    subchunk_shape: SmallVec<[NonZeroU64; 4]>,
    /// The array subset of a whole subchunk of the regular shape.
    whole_subchunk: ArraySubset,
    options: CodecOptions,
    /// The number of subchunks in a block, which is one if subchunks are planned as tasks.
    block_len: usize,
    /// The number of blocks in a row of subchunks in the last dimension.
    blocks_per_row: usize,
    /// The number of subchunks in a row of subchunks in the last dimension.
    row_len: usize,
}

impl<'a, I: ShardIndex + 'a> SubchunksIntoViews<'a, I> {
    #[allow(clippy::too_many_arguments)]
    fn generator(
        codec: Arc<ShardingCodecBound>,
        source: ByteSource<'a>,
        shard_index: I,
        subchunk_grid: RegularBoundedChunkGrid,
        subset: ArraySubset,
        subchunk_indices: ArraySubset,
        views: TargetViews<'a>,
        options: CodecOptions,
        parallelism: usize,
    ) -> Generator<'a> {
        let blocks =
            !codec.inner_codecs.plans_subtasks() && matches!(source.memory_slice(), Some(Ok(_)));
        let num_subchunks = subchunk_indices.num_elements_usize();
        let row_len = usize::try_from(*subchunk_indices.shape().last().expect("nonempty")).unwrap();
        let block_len = if blocks {
            (num_subchunks / (2 * parallelism.max(1)))
                .clamp(1, MAX_BLOCK_LEN)
                .min(row_len)
        } else {
            1
        };
        let blocks_per_row = row_len.div_ceil(block_len);
        let num_blocks = num_subchunks / row_len * blocks_per_row;
        let regular_shape: &[u64] = bytemuck::must_cast_slice(&codec.subchunk_shape);
        let state = Arc::new(Self {
            // A reference count of the inner codecs local to the generator
            inner_codecs: Arc::new(CodecChainBound::clone(&codec.inner_codecs)),
            whole_subchunk: ArraySubset::new_with_shape(regular_shape),
            subchunk_shape: codec.subchunk_shape.as_slice().into(),
            source,
            shard_index,
            subchunk_grid,
            output_start: views.subset().start().to_vec(),
            subset,
            subchunk_indices,
            options,
            block_len,
            blocks_per_row,
            row_len,
        });
        let views = views.with_local_keepalive();
        Generator::indexed(num_blocks, move |block| {
            let (first, len) = state.block(block);
            if !blocks {
                return state.plan_subchunk(first, &views);
            }
            let first_geometry = state.subchunk(first);
            let last_geometry = state.subchunk(first + len - 1);
            let last = first_geometry.shape.len() - 1;
            let block_subset = ArraySubset::from(
                std::iter::zip(&first_geometry.output_start, &first_geometry.overlap_shape)
                    .enumerate()
                    .map(|(dim, (&start, &extent))| {
                        let end = if dim == last {
                            last_geometry.output_start[last] + last_geometry.overlap_shape[last]
                        } else {
                            start + extent
                        };
                        start..end
                    }),
            );
            // SAFETY: the blocks are disjoint, and `views` is only subdivided
            let block_views = unsafe { views.subdivide(block_subset) }?;
            let state = state.clone();
            Ok(Some(Task::compute(move |ctx| {
                for index in first..first + len {
                    state.decode_subchunk(index, &block_views, ctx)?;
                }
                Ok(Outcome::Done)
            })))
        })
    }

    /// The index of the first subchunk of `block` in `subchunk_indices`, and the number of subchunks in the block.
    fn block(&self, block: usize) -> (usize, usize) {
        let row = block / self.blocks_per_row;
        let start = (block % self.blocks_per_row) * self.block_len;
        (
            row * self.row_len + start,
            self.block_len.min(self.row_len - start),
        )
    }

    /// The geometry of the subchunk at `index` in `subchunk_indices`.
    fn subchunk(&self, index: usize) -> SubchunkGeometry {
        let subchunk = subset_indices_at(&self.subchunk_indices, index as u64);
        let shard_shape = self.subchunk_grid.array_shape();
        let mut geometry = SubchunkGeometry {
            shard_index_idx: usize::try_from(
                ravel_indices(&subchunk, self.subchunk_grid.grid_shape())
                    .expect("inbounds subchunk"),
            )
            .unwrap(),
            shape: SmallVec::new(),
            overlap_start: SmallVec::new(),
            overlap_shape: SmallVec::new(),
            output_start: SmallVec::new(),
            whole: true,
        };
        for (dim, &chunk_index) in subchunk.iter().enumerate() {
            let regular = self.subchunk_shape[dim].get();
            let start = chunk_index * regular;
            let end = (start + regular).min(shard_shape[dim]);
            let subset_start = self.subset.start()[dim];
            let subset_end = subset_start + self.subset.shape()[dim];
            let (overlap_start, overlap_end) = (start.max(subset_start), end.min(subset_end));
            geometry.whole &= overlap_start == start && overlap_end == end;
            geometry
                .shape
                .push(NonZeroU64::new(end - start).expect("subchunks are nonempty"));
            geometry.overlap_start.push(overlap_start - start);
            geometry.overlap_shape.push(overlap_end - overlap_start);
            geometry
                .output_start
                .push(overlap_start - subset_start + self.output_start[dim]);
        }
        geometry
    }

    /// The bytes of the subchunk at `offset` with `size` in the shard.
    fn subchunk_source(&self, offset: u64, size: u64) -> Result<ByteSource<'a>, CodecError> {
        self.source
            .interval(offset, size)
            .map_err(|_| out_of_bounds_subchunk())
    }

    /// Plan the decoding of the subchunk at `index` in `subchunk_indices` into its region of `views`.
    ///
    /// Returns [`None`] if the subchunk is empty and its region has been filled with the fill value.
    fn plan_subchunk(
        &self,
        index: usize,
        views: &TargetViews<'a>,
    ) -> Result<Option<Task<'a>>, CodecError> {
        let geometry = self.subchunk(index);
        // SAFETY: the overlaps of subchunks with the subset are disjoint, and `views` is only subdivided
        let views = unsafe { views.subdivide(geometry.output_subset()) }?;
        let Some((offset, size)) =
            subchunk_offset_size(&self.shard_index, geometry.shard_index_idx)
        else {
            views.with_target(|target| {
                copy_fill_value_into(
                    self.inner_codecs.data_type(),
                    self.inner_codecs.fill_value(),
                    target,
                )
            })?;
            return Ok(None);
        };
        Ok(Some(self.inner_codecs.clone().plan_decode(
            self.subchunk_source(offset, size)?,
            &geometry.shape,
            DecodeRequest {
                indexer: PlanIndexer::Subset(geometry.overlap()),
                target: TaskTarget::Views(views),
            },
            &self.options,
        )?))
    }

    /// Decode the subchunk at `index` in `subchunk_indices` into its region of the `block_views`.
    fn decode_subchunk(
        &self,
        index: usize,
        block_views: &TargetViews<'_>,
        ctx: &ComputeCtx,
    ) -> Result<(), CodecError> {
        let geometry = self.subchunk(index);
        let inner_codecs = &self.inner_codecs;
        // SAFETY: the subchunks of a block are disjoint, and `block_views` is only subdivided
        unsafe {
            block_views.with_subdivided(geometry.output_subset(), |views| {
                let Some((offset, size)) =
                    subchunk_offset_size(&self.shard_index, geometry.shard_index_idx)
                else {
                    return views.with_target(|target| {
                        copy_fill_value_into(
                            inner_codecs.data_type(),
                            inner_codecs.fill_value(),
                            target,
                        )
                    });
                };
                let outcome = if geometry.whole {
                    // The bytes are borrowed from the shard, without a reference count of a window of it
                    let shard = self
                        .source
                        .memory_slice()
                        .expect("the source is in memory")?;
                    let bytes = usize::try_from(offset)
                        .ok()
                        .zip(usize::try_from(size).ok())
                        .and_then(|(offset, size)| shard.get(offset..offset.checked_add(size)?))
                        .ok_or_else(out_of_bounds_subchunk)?;
                    let indexer = if geometry.shape.as_slice() == self.subchunk_shape.as_slice() {
                        PlanIndexer::Borrowed(&self.whole_subchunk)
                    } else {
                        PlanIndexer::Subset(geometry.overlap())
                    };
                    decode_whole(
                        &**inner_codecs,
                        Some(ArrayBytesDecodeIntoInput::Bytes(CowBytes::Borrowed(bytes))),
                        &geometry.shape,
                        DecodeRequest {
                            indexer,
                            target: TaskTarget::Views(views),
                        },
                        &self.options,
                        ctx.resources(),
                    )?
                } else {
                    // Part of the subchunk is requested, which its codecs may decode without decoding all of it
                    inner_codecs
                        .clone()
                        .plan_decode(
                            self.subchunk_source(offset, size)?,
                            &geometry.shape,
                            DecodeRequest {
                                indexer: PlanIndexer::Subset(geometry.overlap()),
                                target: TaskTarget::Views(views),
                            },
                            &self.options,
                        )?
                        .run_without_io(ctx)?
                };
                let unexpected_value = || {
                    CodecError::Other(
                        "a subchunk decoded into a target produced a value".to_string(),
                    )
                };
                match outcome {
                    Outcome::Done => Ok(()),
                    Outcome::Value(_) => Err(unexpected_value()),
                    outcome @ (Outcome::Spawn(_)
                    | Outcome::Fork { .. }
                    | Outcome::Generate { .. }
                    | Outcome::Write { .. }) => {
                        // A partial decode planned tasks (e.g. a transpose of the decoded region), which only read memory
                        match drive_memory_one(
                            Task::compute(move |_| Ok(outcome)),
                            ctx.resources(),
                        )? {
                            TaskValue::None => Ok(()),
                            TaskValue::Array(_)
                            | TaskValue::Bytes(_)
                            | TaskValue::Prepared(_)
                            | TaskValue::Any(_) => Err(unexpected_value()),
                        }
                    }
                }
            })?
        }
    }
}

/// The geometry of a subchunk of [`SubchunksIntoViews`].
struct SubchunkGeometry {
    /// The index of the subchunk in the shard index.
    shard_index_idx: usize,
    /// The shape of the subchunk.
    shape: SmallVec<[NonZeroU64; 4]>,
    /// The start of the overlap of the subchunk with the requested subset, relative to the subchunk.
    overlap_start: SmallVec<[u64; 4]>,
    /// The shape of the overlap of the subchunk with the requested subset.
    overlap_shape: SmallVec<[u64; 4]>,
    /// The start of the overlap in the output.
    output_start: SmallVec<[u64; 4]>,
    /// Whether the whole subchunk is requested.
    whole: bool,
}

impl SubchunkGeometry {
    /// The overlap of the subchunk with the requested subset, relative to the subchunk.
    fn overlap(&self) -> ArraySubset {
        start_shape_subset(&self.overlap_start, &self.overlap_shape)
    }

    /// The region of the output the overlap is decoded into.
    fn output_subset(&self) -> ArraySubset {
        start_shape_subset(&self.output_start, &self.overlap_shape)
    }
}

/// The error of a shard index that references bytes out of bounds of the shard.
fn out_of_bounds_subchunk() -> CodecError {
    CodecError::Other(
        "The shard index references out-of-bounds bytes. The chunk may be corrupted.".to_string(),
    )
}

/// The array subset with `start` and `shape`, which have the same length.
fn start_shape_subset(start: &[u64], shape: &[u64]) -> ArraySubset {
    ArraySubset::from(std::iter::zip(start, shape).map(|(&start, &len)| start..start + len))
}

/// The maximum number of subchunks in a block of [`SubchunksIntoViews`].
const MAX_BLOCK_LEN: usize = 8;

/// The decoded offsets and sizes of the subchunks of a shard.
trait ShardIndex: core::ops::Deref<Target = [u64]> + MaybeSend + MaybeSync {}

impl<T: core::ops::Deref<Target = [u64]> + MaybeSend + MaybeSync> ShardIndex for T {}

/// A prepared decoder of a shard with a decoded index.
#[derive(Debug)]
struct ShardDecoder {
    codec: Arc<ShardingCodecBound>,
    source: ByteSource<'static>,
    shape: SmallVec<[NonZeroU64; 4]>,
    shard_index: Arc<[u64]>,
    subchunk_grid: RegularBoundedChunkGrid,
}

impl ShardDecoder {
    /// The indices of the subchunk containing every element of `indexer`, if it is a nonempty array subset within one
    /// subchunk.
    fn single_subchunk(&self, indexer: &dyn Indexer) -> Option<Vec<u64>> {
        let subset = indexer.as_array_subset()?;
        if subset.is_empty() {
            return None;
        }
        let (start, shape) = (subset.start(), subset.shape());
        std::iter::zip(
            std::iter::zip(start.iter(), shape.iter()),
            &self.codec.subchunk_shape,
        )
        .map(|((&start, &len), subchunk_len)| {
            let first = start / subchunk_len.get();
            let last = (start + len - 1) / subchunk_len.get();
            (first == last).then_some(first)
        })
        .collect()
    }
}

impl PreparedDecoder for ShardDecoder {
    fn plan_decode<'a>(
        self: Arc<Self>,
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        request.indexer.validate(&self.shape)?;
        if let Some(subchunk) = self.single_subchunk(request.indexer.as_indexer()) {
            // The request is within one subchunk, so it is planned as a request of that subchunk
            let subchunk_subset = self
                .subchunk_grid
                .subset(&subchunk)?
                .expect("subchunk is within the shard");
            let subset = request
                .indexer
                .as_indexer()
                .as_array_subset()
                .expect("a single subchunk is only found for array subsets")
                .to_array_subset();
            let subchunk_request = DecodeRequest {
                indexer: PlanIndexer::Subset(subset.relative_to(subchunk_subset.start())?),
                target: request.target,
            };
            let task = self.codec.plan_subchunk(
                &self.source,
                &self.shard_index,
                &self.subchunk_grid,
                &subchunk,
                subset_shape(&subchunk_subset),
                subchunk_request,
                options,
            )?;
            // An empty subchunk has filled the target
            return Ok(task.unwrap_or_else(|| Task::compute(|_| Ok(Outcome::Done))));
        }
        let options = *options;
        if request.indexer.is_whole(&self.shape) {
            // Every subchunk is requested, so read the whole shard in one request and decode its subchunks from memory
            let hint = whole_chunk_hint(&*self.codec, &self.source, &self.shape);
            return Ok(Task::read(
                self.source.clone(),
                [ByteRange::FromStart(0, None)],
                move |mut reads, ctx| {
                    let Some(shard) = reads.pop().flatten() else {
                        return place_fill_value(
                            self.codec.inner_codecs.data_type(),
                            self.codec.inner_codecs.fill_value(),
                            request,
                        );
                    };
                    self.codec.clone().plan_subchunks(
                        ByteSource::memory(shard),
                        self.shard_index.clone(),
                        self.subchunk_grid.clone(),
                        request,
                        &options,
                        ctx,
                    )
                },
            )
            .with_hint(hint));
        }
        Ok(Task::compute(move |ctx| {
            self.codec.clone().plan_subchunks(
                self.source.clone(),
                self.shard_index.clone(),
                self.subchunk_grid.clone(),
                request,
                &options,
                ctx,
            )
        }))
    }

    fn exists(&self) -> Option<bool> {
        Some(true)
    }

    fn size_held(&self) -> usize {
        self.source.size_held() + self.shard_index.len() * size_of::<u64>()
    }
}

/// Decode the shard index from its encoded bytes.
fn decode_index(
    codec: &ShardingCodecBound,
    encoded_index: &[u8],
    index_shape: &[NonZeroU64],
    options: &CodecOptions,
    ctx: &ComputeCtx,
) -> Result<Vec<u64>, CodecError> {
    decode_shard_index(
        encoded_index,
        index_shape,
        &codec.index_codecs,
        options,
        ctx.resources(),
    )
    .map_err(|err| CodecError::Other(format!("the shard index could not be decoded: {err}")))
}

/// The shape of a nonempty subchunk with `subchunk_subset`.
fn subset_shape(subchunk_subset: &ArraySubset) -> &[NonZeroU64] {
    bytemuck::checked::cast_slice(subchunk_subset.shape())
}

fn array_value(value: TaskValue<'_>) -> Result<ArrayBytes<'_>, CodecError> {
    value
        .into_array()
        .ok_or_else(|| CodecError::Other("expected decoded subchunk array bytes".to_string()))
}

//! Planned partial encoding of the sharding codec.
//!
//! A partial encode of a shard is a task that reads the shard index (unless it is known from a previous partial
//! encode), then forks a task per intersected subchunk that reads the subchunk if it is not entirely replaced, then
//! decodes, updates and encodes it. The join appends the encoded subchunks to the shard and rewrites the index.

use std::num::NonZeroU64;
use std::sync::Arc;

use smallvec::smallvec;
use zarrs_codec::plan::{
    ByteSource, ComputeCtx, Join, Outcome, PartialEncodeState, PlanIndexer, SourceKey, Task,
    TaskHint, TaskValue, WriteOp, decode_and_update, partial_encode_value,
    store_encoded_subchunk_memory,
};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecSubchunkingTraits, ArrayToBytesCodecTraits, CodecError,
    CodecOptions, CowBytes,
};
use zarrs_storage::Bytes;
use zarrs_storage::byte_range::ByteRange;

use super::sharding_planner::ShardSubchunk;
use super::{
    EMPTY, IndexRead, ShardingCodecBound, ShardingIndexLocation, SubchunkUpdate,
    calculate_chunks_per_shard, compute_index_encoded_size, decode_shard_index, index_read,
    num_subchunks, subchunk_grid, subchunk_offset_size, subchunk_updates,
};
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{ArrayBytes, ChunkShape, transmute_to_bytes};

impl ShardingCodecBound {
    /// Plan a partial encode of the shard `key`. See [`ArrayToBytesCodecTraits::plan_partial_encode`].
    ///
    /// The state of a shard is its decoded index.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_partial_encode_shard<'a>(
        self: Arc<Self>,
        key: SourceKey,
        shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        state: Option<PartialEncodeState>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        indexer.validate(shape)?;
        bytes.validate(indexer.as_indexer().len(), self.inner_codecs.data_type())?;
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let IndexRead {
            shape: index_shape,
            range: index_byte_range,
            hint: index_hint,
        } = index_read(&self, &chunks_per_shard)?;
        let chunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        let index_len = num_subchunks(&chunk_grid) * 2;
        let update = ShardUpdate {
            codec: self,
            key,
            chunk_grid,
            index_shape,
            options: *options,
        };

        let known_index = state
            .and_then(|state| state.downcast::<Vec<u64>>().ok())
            .filter(|index| index.len() == index_len);
        if let Some(index) = known_index {
            return Ok(Task::compute(move |ctx| {
                let index = Arc::try_unwrap(index).unwrap_or_else(|index| index.as_ref().clone());
                update.plan_subchunks(index, &indexer, &bytes, ctx)
            }));
        }
        Ok(Task::read(
            ByteSource::external(key),
            [index_byte_range],
            move |mut reads, ctx| {
                let index = match reads.pop().flatten() {
                    Some(encoded_index) => decode_shard_index(
                        &encoded_index,
                        &update.index_shape,
                        &update.codec.index_codecs,
                        &update.options,
                        ctx.resources(),
                    )?,
                    None => vec![EMPTY; index_len],
                };
                update.plan_subchunks(index, &indexer, &bytes, ctx)
            },
        )
        .with_hint(index_hint))
    }
}

impl ShardingCodecBound {
    /// Plan storing an encoded subchunk in the shard `key`, or erasing it. See
    /// [`ArrayToBytesCodecTraits::plan_store_encoded_subchunk`].
    ///
    /// The shard index is read, and the subchunk is appended to the shard and the index rewritten, as in a partial
    /// encode. A subchunk below level zero is stored in its enclosing subchunk, which is read and updated in memory.
    /// Without partial writes, the shard is read, updated in memory and written whole.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_store_encoded_subchunk_in_shard(
        self: Arc<Self>,
        key: SourceKey,
        shape: &[NonZeroU64],
        level: usize,
        coordinates: &[u64],
        encoded: Option<Bytes>,
        set_partial: bool,
        options: &CodecOptions,
    ) -> Result<Task<'static>, CodecError> {
        if self.inner_codecs.subchunk_codecs().len() < level {
            return Err(CodecError::UnsupportedEncodedSubchunk);
        }
        let ShardSubchunk {
            index: subchunk_index,
            shape: subchunk_shape,
            coordinates: subchunk_coordinates,
        } = self.subchunk_at(shape, coordinates)?;
        if !set_partial {
            let shape = ChunkShape::from(shape.to_vec());
            let coordinates = coordinates.to_vec();
            let options = *options;
            return Ok(Task::read(
                ByteSource::external(key),
                [ByteRange::FromStart(0, None)],
                move |mut reads, ctx| {
                    let mut shard = reads.pop().flatten().map(|shard| shard.to_vec());
                    store_encoded_subchunk_memory(
                        self.into_dyn(),
                        &mut shard,
                        &shape,
                        level,
                        &coordinates,
                        encoded,
                        &options,
                        ctx.resources(),
                    )?;
                    let op = match shard {
                        Some(shard) => WriteOp::Set(key, CowBytes::from(shard)),
                        None => WriteOp::Erase(key),
                    };
                    Ok(Outcome::Write {
                        ops: smallvec![op],
                        then: TaskValue::None,
                    })
                },
            ));
        }

        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let IndexRead {
            shape: index_shape,
            range: index_byte_range,
            hint: index_hint,
        } = index_read(&self, &chunks_per_shard)?;
        let chunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        let index_len = num_subchunks(&chunk_grid) * 2;
        let update = ShardUpdate {
            codec: self,
            key,
            chunk_grid,
            index_shape,
            options: *options,
        };
        Ok(Task::read(
            ByteSource::external(key),
            [index_byte_range],
            move |mut reads, ctx| {
                let shard_index = match reads.pop().flatten() {
                    Some(encoded_index) => decode_shard_index(
                        &encoded_index,
                        &update.index_shape,
                        &update.codec.index_codecs,
                        &update.options,
                        ctx.resources(),
                    )?,
                    None => vec![EMPTY; index_len],
                };
                if level == 0 {
                    let encoded = TaskValue::Bytes(encoded.map(CowBytes::Shared));
                    return update.write(shard_index, &[subchunk_index], vec![encoded], ctx);
                }

                // The enclosing subchunk is read (if it is stored) and updated in memory
                let inner_codecs = update.codec.inner_codecs.clone();
                let options = update.options;
                let store = move |enclosing: Option<CowBytes<'_>>, ctx: &ComputeCtx| {
                    let mut enclosing = enclosing.map(|enclosing| enclosing.to_vec());
                    store_encoded_subchunk_memory(
                        inner_codecs.into_dyn(),
                        &mut enclosing,
                        &subchunk_shape,
                        level - 1,
                        &subchunk_coordinates,
                        encoded,
                        &options,
                        ctx.resources(),
                    )?;
                    Ok(Outcome::Value(TaskValue::Bytes(
                        enclosing.map(CowBytes::from),
                    )))
                };
                let task = match subchunk_offset_size(&shard_index, subchunk_index) {
                    Some((offset, size)) => Task::read(
                        ByteSource::external(update.key),
                        [ByteRange::FromStart(offset, Some(size))],
                        move |mut reads, ctx| store(reads.pop().flatten(), ctx),
                    ),
                    None => Task::compute(move |ctx| store(None, ctx)),
                };
                Ok(Outcome::Fork {
                    children: vec![task],
                    join: Join::new(move |values, ctx| {
                        update.write(shard_index, &[subchunk_index], values, ctx)
                    }),
                })
            },
        )
        .with_hint(index_hint))
    }
}

/// A partial encode of a shard.
struct ShardUpdate {
    codec: Arc<ShardingCodecBound>,
    key: SourceKey,
    chunk_grid: RegularBoundedChunkGrid,
    index_shape: ChunkShape,
    options: CodecOptions,
}

impl ShardUpdate {
    /// Plan the update of the subchunks intersecting `indexer` to `bytes`, given the shard index.
    fn plan_subchunks<'a>(
        self,
        shard_index: Vec<u64>,
        indexer: &PlanIndexer<'_>,
        bytes: &ArrayBytes<'_>,
        _ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let indexer = indexer.as_indexer();
        if indexer.is_empty() {
            return Ok(Outcome::Value(partial_encode_value(Arc::new(shard_index))));
        }
        let updates = subchunk_updates(
            &self.chunk_grid,
            &self.codec.subchunk_shape,
            indexer,
            bytes,
            self.codec.inner_codecs.data_type(),
        )?;
        let mut subchunks = Vec::with_capacity(updates.len());
        let mut children = Vec::with_capacity(updates.len());
        for update in updates {
            let subchunk_index = usize::try_from(update.subchunk_index).unwrap();
            // Subchunks that are entirely replaced are not read
            let existing = if update.fully_covered {
                None
            } else {
                subchunk_offset_size(&shard_index, subchunk_index)
            };
            subchunks.push(subchunk_index);
            children.push(self.plan_subchunk(update, existing));
        }
        Ok(Outcome::Fork {
            children,
            join: Join::new(move |values, ctx| self.write(shard_index, &subchunks, values, ctx)),
        })
    }

    /// Plan the update of a subchunk, whose encoded bytes are at `existing` in the shard (if any).
    ///
    /// The value of the task is the encoded subchunk, or [`None`] if it is the fill value.
    fn plan_subchunk<'a>(
        &self,
        update: SubchunkUpdate<'_>,
        existing: Option<(u64, u64)>,
    ) -> Task<'a> {
        let SubchunkUpdate {
            subchunk_shape,
            indexer,
            bytes,
            ..
        } = update;
        // The extracted elements are owned, so this does not copy
        let bytes = bytes.into_owned();
        let codec = self.codec.clone();
        let options = self.options;
        let hint = TaskHint {
            read_bytes: existing.map(|(_offset, size)| size),
            compute_bytes: bytes.size() as u64,
        };
        let update = move |encoded: Option<CowBytes<'_>>, ctx: &ComputeCtx| {
            let inner_codecs = &codec.inner_codecs;
            let updated = decode_and_update(
                &**inner_codecs,
                encoded,
                &subchunk_shape,
                indexer.as_ref(),
                &bytes,
                &options,
                ctx.resources(),
            )?;
            let encoded = if inner_codecs
                .fill_value()
                .is_some_and(|fill| updated.is_fill_value(fill))
            {
                None
            } else {
                Some(
                    inner_codecs
                        .encode(updated, &subchunk_shape, &options, ctx.resources())?
                        .into_static(),
                )
            };
            Ok(Outcome::Value(TaskValue::Bytes(encoded)))
        };
        match existing {
            Some((offset, size)) => Task::read(
                ByteSource::external(self.key),
                [ByteRange::FromStart(offset, Some(size))],
                move |mut reads, ctx| update(reads.pop().flatten(), ctx),
            ),
            None => Task::compute(move |ctx| update(None, ctx)),
        }
        .with_hint(hint)
    }

    /// Write the encoded subchunks `values` of `subchunks` and the updated shard index.
    ///
    /// The encoded subchunks are appended to the shard, unless every existing subchunk is replaced, in which case the
    /// shard is rewritten. A shard of only empty subchunks is erased.
    fn write<'a>(
        self,
        mut shard_index: Vec<u64>,
        subchunks: &[usize],
        values: Vec<TaskValue<'a>>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        // The end of the existing encoded subchunks (the index is read from storage, so it may be corrupt)
        let max_data_offset = shard_index
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[offset, size]| {
                if offset == EMPTY && size == EMPTY {
                    Ok(0)
                } else {
                    offset.checked_add(size).ok_or_else(invalid_index)
                }
            })
            .try_fold(0, |max, end| end.map(|end| end.max(max)))?;

        // Rewrite the shard if no other subchunks remain
        for &subchunk_index in subchunks {
            shard_index[subchunk_index * 2] = EMPTY;
            shard_index[subchunk_index * 2 + 1] = EMPTY;
        }
        let rewrite = shard_index.iter().all(|&x| x == EMPTY);
        let max_data_offset = if rewrite { 0 } else { max_data_offset };

        // The offset of the new encoded subchunks
        let index_encoded_size =
            compute_index_encoded_size(&self.codec.index_codecs, &self.index_shape)?;
        let offset_new = match self.codec.index_location {
            ShardingIndexLocation::Start => max_data_offset.max(index_encoded_size),
            ShardingIndexLocation::End => max_data_offset,
        };

        // Update the shard index
        let mut encoded_subchunks = Vec::with_capacity(values.len());
        let mut offset = offset_new;
        for (&subchunk_index, value) in std::iter::zip(subchunks, values) {
            let encoded = value.into_bytes().ok_or_else(|| {
                CodecError::Other("a subchunk update did not produce encoded bytes".to_string())
            })?;
            if let Some(encoded) = encoded {
                let len = encoded.len() as u64;
                shard_index[subchunk_index * 2] = offset;
                shard_index[subchunk_index * 2 + 1] = len;
                encoded_subchunks.push(encoded);
                offset = offset.checked_add(len).ok_or_else(invalid_index)?;
            }
        }
        let data_end = offset;

        if shard_index.iter().all(|&x| x == EMPTY) {
            // Erase the shard if all subchunks are empty
            return Ok(Outcome::Write {
                ops: smallvec![WriteOp::Erase(self.key)],
                then: partial_encode_value(Arc::new(shard_index)),
            });
        }

        let encoded_index = self
            .codec
            .index_codecs
            .encode(
                ArrayBytes::new_flen(transmute_to_bytes(shard_index.as_slice())),
                &self.index_shape,
                &self.options,
                ctx.resources(),
            )?
            .into_static();
        // The encoded subchunks are written in one range (copied, as one write is cheaper than many for most stores)
        let data_len = usize::try_from(data_end - offset_new).unwrap();
        let mut data = Vec::with_capacity(data_len + encoded_index.len());
        if rewrite && self.codec.index_location == ShardingIndexLocation::Start {
            data.extend_from_slice(&encoded_index);
        }
        for encoded in &encoded_subchunks {
            data.extend_from_slice(encoded);
        }
        drop(encoded_subchunks);
        if self.codec.index_location == ShardingIndexLocation::End {
            data.extend_from_slice(&encoded_index);
        }
        // The appended bytes (including an index at the end) must end within the maximum shard size
        offset_new
            .checked_add(data.len() as u64)
            .ok_or_else(invalid_index)?;
        let op = if rewrite {
            // The shard is written from the start
            WriteOp::Set(self.key, CowBytes::from(data))
        } else {
            // The encoded subchunks are appended to the shard, and the index rewritten
            let mut offset_values = Vec::with_capacity(2);
            if self.codec.index_location == ShardingIndexLocation::Start {
                offset_values.push((0, encoded_index));
            }
            offset_values.push((offset_new, CowBytes::from(data)));
            WriteOp::SetPartial(self.key, offset_values)
        };
        Ok(Outcome::Write {
            ops: smallvec![op],
            then: partial_encode_value(Arc::new(shard_index)),
        })
    }
}

/// The error of a shard index with an encoded subchunk beyond the maximum shard size.
fn invalid_index() -> CodecError {
    CodecError::Other(
        "the shard index has an encoded subchunk beyond the maximum shard size".to_string(),
    )
}

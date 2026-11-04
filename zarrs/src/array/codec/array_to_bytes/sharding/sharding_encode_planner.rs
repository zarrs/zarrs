//! Planned encoding of the sharding codec.
//!
//! A shard is encoded by tasks that each encode a block of consecutive subchunks (in C order). Subchunks that are
//! entirely the fill value are omitted from the shard.
//!
//! With the unordered subchunk write order (the default) and a bounded encoded subchunk size, each task writes its
//! encoded subchunks straight into a shard allocated for the bound, as they are encoded. Otherwise, the encoded
//! subchunks are the values of the tasks, and the join copies them into the shard in C order.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use zarrs_codec::plan::{
    ComputeCtx, Fold, Generator, Join, MultipartId, MultipartOp, Outcome, SourceKey, Task,
    TaskValue, WriteOp, encoded_value,
};
use zarrs_codec::{ArrayCodecTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions, CowBytes};
use zarrs_plugin::{MaybeSend, MaybeSync};

use super::sharding_codec::subchunk_subset;
use super::{
    EMPTY, ShardingCodecBound, ShardingIndexLocation, SubchunkWriteOrder,
    calculate_chunks_per_shard, compute_index_encoded_size, num_subchunks, sharding_index_shape,
    subchunk_grid,
};
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{ArrayBytes, BytesRepresentation, ChunkShape, transmute_to_bytes_vec};
use zarrs_chunk_grid::ChunkGridTraits;

/// The maximum number of subchunks encoded by a task.
const MAX_BLOCK_LEN: usize = 8;

impl ShardingCodecBound {
    /// Encode `bytes`, a shard of `shape`, in the compute step `ctx` with a task per block of subchunks (unless there
    /// is one block). See [`ArrayToBytesCodecTraits::encode_step`].
    pub(super) fn encode_shard_step<'a>(
        self: Arc<Self>,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let unordered = matches!(
            self.options.subchunk_write_order(),
            SubchunkWriteOrder::Unordered
        );
        let encode = ShardEncode::new(self, bytes, shape, options)?;
        // The bound of the encoded size of a subchunk, if the encoded subchunks are written straight into the shard
        let direct_bound = if unordered {
            encode.subchunk_bound()?
        } else {
            None
        };
        encode.plan(direct_bound, ctx)
    }

    /// Encode `bytes`, a shard of `shape`, and write it to `key` as its blocks of subchunks are encoded, with a
    /// multipart write, in the compute step `ctx`. See [`ArrayToBytesCodecTraits::encode_write_step`].
    ///
    /// The encoded subchunks are written in C order, or in the order they are encoded with an unordered subchunk write
    /// order, followed by the index. An index at the start is written last, over a placeholder (patching the write).
    /// A shard of one block of subchunks is encoded whole and set.
    pub(super) fn encode_shard_write_step<'a>(
        self: Arc<Self>,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        key: SourceKey,
        options: &CodecOptions,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        if matches!(
            self.options.subchunk_write_order(),
            SubchunkWriteOrder::Colexicographic | SubchunkWriteOrder::Morton
        ) {
            return zarrs_codec::plan::encode_and_set(
                self.into_dyn(),
                bytes,
                shape,
                key,
                options,
                ctx,
            );
        }
        ShardEncode::new(self, bytes, shape, options)?.plan_write(key, ctx)
    }
}

/// The encoding of a shard.
struct ShardEncode<'a> {
    codec: Arc<ShardingCodecBound>,
    bytes: ArrayBytes<'a>,
    subchunk_grid: RegularBoundedChunkGrid,
    num_subchunks: usize,
    index_shape: ChunkShape,
    index_encoded_size: usize,
    options: CodecOptions,
}

impl<'a> ShardEncode<'a> {
    /// The encoding of `bytes`, a shard of `shape`, with `codec`.
    fn new(
        codec: Arc<ShardingCodecBound>,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<Self>, CodecError> {
        bytes.validate(
            shape.iter().map(|d| d.get()).product(),
            codec.inner_codecs.data_type(),
        )?;
        let chunks_per_shard = calculate_chunks_per_shard(shape, &codec.subchunk_shape)?;
        let index_shape = sharding_index_shape(&chunks_per_shard);
        let index_encoded_size = usize::try_from(compute_index_encoded_size(
            &codec.index_codecs,
            &index_shape,
        )?)
        .unwrap();
        let subchunk_grid = subchunk_grid(shape, &codec.subchunk_shape)?;
        Ok(Arc::new(ShardEncode {
            num_subchunks: num_subchunks(&subchunk_grid),
            codec,
            bytes,
            subchunk_grid,
            index_shape,
            index_encoded_size,
            options: *options,
        }))
    }

    /// The bound of the encoded size of a subchunk, if the inner codecs have one.
    fn subchunk_bound(&self) -> Result<Option<usize>, CodecError> {
        Ok(
            match self
                .codec
                .inner_codecs
                .encoded_representation(&self.codec.subchunk_shape)?
            {
                BytesRepresentation::BoundedSize(size) | BytesRepresentation::FixedSize(size) => {
                    Some(usize::try_from(size).unwrap())
                }
                BytesRepresentation::UnboundedSize => None,
            },
        )
    }

    /// The number of subchunks of a block, and the number of blocks, encoded in the compute step `ctx`.
    fn blocks(&self, ctx: &ComputeCtx) -> (usize, usize) {
        let parallelism = match ctx.resources().compute_target() {
            0 => rayon::current_num_threads(),
            target => target,
        };
        let block_len = (self.num_subchunks / (2 * parallelism.max(1))).clamp(1, MAX_BLOCK_LEN);
        (block_len, self.num_subchunks.div_ceil(block_len))
    }

    /// The raveled indices of the subchunks of the block `block` of blocks of `block_len` subchunks.
    fn block(&self, block: usize, block_len: usize) -> std::ops::Range<usize> {
        let start = block * block_len;
        start..(start + block_len).min(self.num_subchunks)
    }

    /// The generator of the encoded blocks of `block_len` subchunks of the shard.
    fn encoded_blocks(self: &Arc<Self>, block_len: usize, num_blocks: usize) -> Generator<'a> {
        let encode = self.clone();
        Generator::indexed(num_blocks, move |index| {
            let encode = encode.clone();
            let subchunks = encode.block(index, block_len);
            Ok(Some(Task::compute(move |ctx| {
                let encoded = encode.encode_block(subchunks, ctx)?;
                Ok(Outcome::Value(TaskValue::Any(Box::new(encoded))))
            })))
        })
    }

    /// Plan the encoding and writing of the shard to `key`, writing the encoded subchunks of each block as it is
    /// encoded.
    fn plan_write(
        self: Arc<Self>,
        key: SourceKey,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let (block_len, num_blocks) = self.blocks(ctx);
        if num_blocks == 1 {
            // A single block is encoded whole and set
            let encoded = self.encode_block(0..self.num_subchunks, ctx)?;
            let Outcome::Value(shard) = self.assemble(&[encoded], ctx)? else {
                unreachable!("an assembled shard is a value")
            };
            let shard = encoded_value(shard)?;
            return Ok(Outcome::Write {
                ops: smallvec::smallvec![WriteOp::Set(key, shard)],
                then: TaskValue::None,
            });
        }
        // The shard is at most the bounds of its subchunks and its index
        let size_hint = self
            .subchunk_bound()?
            .and_then(|bound| bound.checked_mul(self.num_subchunks))
            .and_then(|size| size.checked_add(self.index_encoded_size))
            .map(|size| size as u64);
        let children = self.encoded_blocks(block_len, num_blocks);
        let order = self.codec.options.subchunk_write_order();
        let mut write = ShardWrite {
            key,
            size_hint,
            part_size: self.options.multipart_part_size(),
            id: None,
            parts: Vec::new(),
            parts_len: 0,
            offset: 0,
            data_start: 0,
            shard_index: vec![EMPTY; self.num_subchunks * 2],
        };
        if self.codec.index_location == ShardingIndexLocation::Start {
            // A placeholder for the index, which is written last
            write
                .parts
                .push(CowBytes::from(vec![0u8; self.index_encoded_size]));
            write.parts_len = self.index_encoded_size;
            write.offset = self.index_encoded_size as u64;
            write.data_start = write.offset;
        }
        let fold = Fold::new(
            write,
            |write: &mut ShardWrite<'a>, value, _ctx| {
                let block = value.downcast::<EncodedBlock>().ok_or_else(block_error)?;
                Ok(write.write_block(block))
            },
            move |write, ctx| self.finish_write(write, ctx),
        );
        let fold = match order {
            SubchunkWriteOrder::C
            | SubchunkWriteOrder::Lexicographic
            | SubchunkWriteOrder::Colexicographic
            | SubchunkWriteOrder::Morton => fold,
            // The index locates the subchunks, so they are written as they are encoded
            SubchunkWriteOrder::Unordered => fold.unordered(),
        };
        Ok(Outcome::Stream { children, fold })
    }

    /// Write the rest of the shard `write` and its index and finish it, or erase (or set the index of) a shard without
    /// subchunks.
    fn finish_write(
        &self,
        mut write: ShardWrite<'a>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        if write.offset == write.data_start {
            // No subchunks: abandon a write begun with the placeholder of the index
            let mut ops = smallvec::SmallVec::new();
            if let Some(id) = write.id {
                ops.push(WriteOp::Multipart(id, MultipartOp::Abort));
            }
            ops.push(if self.options.store_empty_chunks() {
                WriteOp::Set(write.key, self.encode_index(write.shard_index, ctx)?)
            } else {
                WriteOp::Erase(write.key)
            });
            return Ok(Outcome::Write {
                ops,
                then: TaskValue::None,
            });
        }
        let encoded_index = self.encode_index(std::mem::take(&mut write.shard_index), ctx)?;
        let (mut ops, id) = match self.codec.index_location {
            ShardingIndexLocation::End => {
                write.parts.push(encoded_index);
                write.take_parts()
            }
            ShardingIndexLocation::Start => {
                let (mut ops, id) = write.take_parts();
                ops.push(WriteOp::Multipart(id, MultipartOp::Patch(0, encoded_index)));
                (ops, id)
            }
        };
        ops.push(WriteOp::Multipart(id, MultipartOp::Finish));
        Ok(Outcome::Write {
            ops,
            then: TaskValue::None,
        })
    }

    /// Plan the encoding of the blocks of subchunks of the shard.
    fn plan(
        self: Arc<Self>,
        direct_bound: Option<usize>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let (block_len, num_blocks) = self.blocks(ctx);

        if num_blocks == 1 {
            // A single block is encoded without a separate task
            let encoded = self.encode_block(0..self.num_subchunks, ctx)?;
            return self.assemble(&[encoded], ctx);
        }

        let num_subchunks = self.num_subchunks;
        let block = move |block: usize| {
            let start = block * block_len;
            start..(start + block_len).min(num_subchunks)
        };
        if let Some(bound) = direct_bound {
            let capacity = bound
                .checked_mul(self.num_subchunks)
                .and_then(|size| size.checked_add(self.index_encoded_size))
                .ok_or_else(|| CodecError::Other("the shard size is too large".to_string()))?;
            let data_start = match self.codec.index_location {
                ShardingIndexLocation::Start => self.index_encoded_size,
                ShardingIndexLocation::End => 0,
            };
            let sink = Arc::new(ShardSink::new(capacity, data_start, self.num_subchunks));
            let children = {
                let (encode, sink) = (self.clone(), sink.clone());
                Generator::indexed(num_blocks, move |index| {
                    let (encode, sink) = (encode.clone(), sink.clone());
                    let subchunks = block(index);
                    Ok(Some(Task::compute(move |ctx| {
                        for subchunk in subchunks {
                            if let Some(encoded) = encode.encode_subchunk(subchunk, ctx)? {
                                sink.write(subchunk, &encoded)?;
                            }
                        }
                        Ok(Outcome::Value(TaskValue::None))
                    })))
                })
            };
            Ok(Outcome::Generate {
                children,
                join: Some(Join::after(move |ctx| {
                    let sink = Arc::try_unwrap(sink).map_err(|_| {
                        CodecError::Other("a subchunk encode outlived its task".to_string())
                    })?;
                    let (shard, data_end, shard_index) = sink.into_parts();
                    self.finish(shard, data_end, shard_index, ctx)
                })),
            })
        } else {
            let children = self.encoded_blocks(block_len, num_blocks);
            Ok(Outcome::Generate {
                children,
                join: Some(Join::new(move |values, ctx| {
                    let blocks = values
                        .into_iter()
                        .map(|value| value.downcast::<EncodedBlock>().ok_or_else(block_error))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.assemble(&blocks, ctx)
                })),
            })
        }
    }

    /// Encode the subchunk with the raveled index `subchunk`, or return [`None`] if it is entirely the fill value.
    fn encode_subchunk(
        &self,
        subchunk: usize,
        ctx: &ComputeCtx,
    ) -> Result<Option<Bytes>, CodecError> {
        let inner_codecs = &self.codec.inner_codecs;
        let subset = subchunk_subset(&self.subchunk_grid, subchunk);
        let bytes = self.bytes.extract_array_subset(
            &subset,
            self.subchunk_grid.array_shape(),
            inner_codecs.data_type(),
        )?;
        if inner_codecs
            .fill_value()
            .is_some_and(|fill| bytes.is_fill_value(fill))
        {
            return Ok(None);
        }
        let shape: smallvec::SmallVec<[NonZeroU64; 4]> = subset
            .shape()
            .iter()
            .map(|&size| NonZeroU64::new(size).expect("nonempty subchunk"))
            .collect();
        Ok(Some(
            inner_codecs
                .encode(bytes, &shape, &self.options, ctx.resources())?
                .into_bytes(),
        ))
    }

    /// Encode the subchunks with the raveled indices `subchunks`.
    fn encode_block(
        &self,
        subchunks: std::ops::Range<usize>,
        ctx: &ComputeCtx,
    ) -> Result<EncodedBlock, CodecError> {
        let first = subchunks.start;
        let encoded = subchunks
            .map(|subchunk| self.encode_subchunk(subchunk, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(EncodedBlock { first, encoded })
    }

    /// Assemble the shard from the encoded subchunks of `blocks`, in C order.
    fn assemble(
        &self,
        blocks: &[EncodedBlock],
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let data_len: usize = blocks
            .iter()
            .flat_map(|block| &block.encoded)
            .flatten()
            .map(Bytes::len)
            .sum();
        let mut shard = Vec::with_capacity(data_len + self.index_encoded_size);
        if self.codec.index_location == ShardingIndexLocation::Start {
            // A placeholder for the index
            shard.resize(self.index_encoded_size, 0);
        }
        let mut shard_index = vec![EMPTY; self.num_subchunks * 2];
        let mut encoded: Vec<Option<&Bytes>> = vec![None; self.num_subchunks];
        for block in blocks {
            for (i, bytes) in (block.first..).zip(&block.encoded) {
                encoded[i] = bytes.as_ref();
            }
        }
        for i in super::sharding_codec::subchunk_order_iterator(
            self.codec.options.subchunk_write_order(),
            self.num_subchunks,
            &self.subchunk_grid,
        ) {
            if let Some(bytes) = encoded[i] {
                shard_index[i * 2] = shard.len() as u64;
                shard_index[i * 2 + 1] = bytes.len() as u64;
                shard.extend_from_slice(bytes);
            }
        }
        let data_end = shard.len();
        self.finish(shard, data_end, shard_index, ctx)
    }

    /// Encode `shard_index` into the shard, whose encoded subchunks end at `data_end`, and return the shard.
    ///
    /// The capacity of `shard` must hold the index, and if the index is at the start of the shard, the subchunks must
    /// start after it (so the bytes before them are reserved for it, initialised or not).
    fn finish(
        &self,
        mut shard: Vec<u8>,
        data_end: usize,
        shard_index: Vec<u64>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let encoded_index = self.encode_index(shard_index, ctx)?;
        let shard_len = match self.codec.index_location {
            // The index is reserved before the subchunks
            ShardingIndexLocation::Start => data_end,
            ShardingIndexLocation::End => data_end + self.index_encoded_size,
        };
        if shard_len > shard.capacity() {
            // A dev error, indicating the bounded size of the encoded subchunks is not correct
            return Err(CodecError::from(
                "Sharding did not allocate a large enough buffer",
            ));
        }
        let index_start = match self.codec.index_location {
            ShardingIndexLocation::Start => 0,
            ShardingIndexLocation::End => data_end,
        };
        // SAFETY: the bytes up to `shard_len` are initialised: the encoded subchunks are written up to `data_end`, and
        // the index is written below, before the start of the subchunks or after their end
        unsafe {
            std::ptr::copy_nonoverlapping(
                encoded_index.as_ptr(),
                shard.as_mut_ptr().add(index_start),
                encoded_index.len(),
            );
            shard.set_len(shard_len);
        }
        Ok(Outcome::Value(TaskValue::Bytes(Some(CowBytes::from(
            shard,
        )))))
    }
}

impl ShardEncode<'_> {
    /// Encode the decoded `shard_index`.
    fn encode_index(
        &self,
        shard_index: Vec<u64>,
        ctx: &ComputeCtx,
    ) -> Result<CowBytes<'static>, CodecError> {
        let encoded_index = self.codec.index_codecs.encode(
            ArrayBytes::from(transmute_to_bytes_vec(shard_index)),
            &self.index_shape,
            &self.options,
            ctx.resources(),
        )?;
        if encoded_index.len() != self.index_encoded_size {
            return Err(CodecError::Other(
                "the encoded shard index has an unexpected size".to_string(),
            ));
        }
        Ok(encoded_index.into_static())
    }
}

/// A shard written as its blocks of subchunks are encoded, with a multipart write.
struct ShardWrite<'a> {
    key: SourceKey,
    /// An upper bound on the size of the shard, if known.
    size_hint: Option<u64>,
    /// The size of the encoded subchunks collected before they are written as a part.
    part_size: usize,
    /// The multipart write, once it has begun with the first part.
    id: Option<MultipartId>,
    /// The encoded subchunks not yet written.
    parts: Vec<CowBytes<'a>>,
    /// The size of `parts`.
    parts_len: usize,
    /// The offset of the next encoded subchunk.
    offset: u64,
    /// The offset of the first encoded subchunk (after an index at the start).
    data_start: u64,
    /// The decoded shard index.
    shard_index: Vec<u64>,
}

impl<'a> ShardWrite<'a> {
    /// Collect the encoded subchunks of `block` (in the order of the block), and write the collected subchunks once
    /// they reach the part size, beginning the write if they are the first.
    fn write_block(&mut self, block: EncodedBlock) -> Outcome<'a> {
        let EncodedBlock { first, encoded } = block;
        let (shard_index, offset) = (&mut self.shard_index, &mut self.offset);
        let mut len = 0;
        let encoded = encoded.into_iter().enumerate().filter_map(|(i, encoded)| {
            let encoded = encoded?;
            let subchunk = first + i;
            shard_index[subchunk * 2] = *offset;
            shard_index[subchunk * 2 + 1] = encoded.len() as u64;
            *offset += encoded.len() as u64;
            len += encoded.len();
            Some(CowBytes::Shared(encoded))
        });
        if self.parts.is_empty() {
            // The parts reuse the allocation of the encoded subchunks of the block
            self.parts = encoded.collect();
        } else {
            self.parts.extend(encoded);
        }
        self.parts_len += len;
        if self.parts_len < self.part_size || self.parts.is_empty() {
            return Outcome::Done;
        }
        let (ops, _) = self.take_parts();
        Outcome::Write {
            ops,
            then: TaskValue::None,
        }
    }

    /// The writes of the collected parts, beginning the write if they are the first, and the write.
    fn take_parts(&mut self) -> (smallvec::SmallVec<[WriteOp<'a>; 1]>, MultipartId) {
        let mut ops = smallvec::SmallVec::new();
        let id = *self.id.get_or_insert_with(|| {
            let id = MultipartId::unique();
            ops.push(WriteOp::Multipart(
                id,
                MultipartOp::Begin(self.key, self.size_hint),
            ));
            id
        });
        self.parts_len = 0;
        if !self.parts.is_empty() {
            ops.push(WriteOp::Multipart(
                id,
                MultipartOp::Parts(std::mem::take(&mut self.parts)),
            ));
        }
        (ops, id)
    }
}

/// The encoded subchunks of a block, starting at the subchunk with the raveled index `first`.
struct EncodedBlock {
    first: usize,
    encoded: Vec<Option<Bytes>>,
}

/// The error of a task of a block of subchunks without an encoded block.
fn block_error() -> CodecError {
    CodecError::Other("a subchunk encode did not produce encoded subchunks".to_string())
}

/// A shard allocated for the bounded size of its encoded subchunks, which tasks write their encoded subchunks into.
struct ShardSink {
    /// The shard, whose spare capacity is written by tasks.
    shard: Vec<u8>,
    /// The start of the spare capacity of `shard`.
    data: *mut u8,
    /// The capacity of `shard`.
    capacity: usize,
    /// The offset of the next encoded subchunk.
    offset: AtomicUsize,
    /// The decoded shard index, whose entries are written by tasks.
    shard_index: Vec<u64>,
    /// The start of `shard_index`.
    index: *mut u64,
}

// SAFETY: tasks write disjoint regions of the shard (reserved with an atomic offset) and disjoint entries of the
// index (by subchunk), and the shard and index are only read once the tasks complete.
unsafe impl Send for ShardSink {}
unsafe impl Sync for ShardSink {}

impl ShardSink {
    /// Create a sink with the capacity for a shard of `capacity` bytes, whose subchunks start at `data_start`.
    fn new(capacity: usize, data_start: usize, num_subchunks: usize) -> Self {
        let mut shard = Vec::with_capacity(capacity);
        let mut shard_index = vec![EMPTY; num_subchunks * 2];
        Self {
            data: shard.as_mut_ptr(),
            capacity,
            shard,
            offset: AtomicUsize::new(data_start),
            index: shard_index.as_mut_ptr(),
            shard_index,
        }
    }

    /// Write the encoded subchunk with the raveled index `subchunk`.
    ///
    /// Each subchunk must be written at most once.
    fn write(&self, subchunk: usize, encoded: &[u8]) -> Result<(), CodecError> {
        let offset = self.offset.fetch_add(encoded.len(), Ordering::Relaxed);
        if offset + encoded.len() > self.capacity {
            // A dev error, indicating the bounded size of the encoded subchunks is not correct
            return Err(CodecError::from(
                "Sharding did not allocate a large enough buffer",
            ));
        }
        assert!(
            subchunk * 2 + 1 < self.shard_index.len(),
            "the subchunk is in the shard"
        );
        // SAFETY: the region of the shard is reserved for this subchunk by the atomic offset, and is within its
        // capacity. The index entries of the subchunk are only written by this subchunk.
        unsafe {
            std::ptr::copy_nonoverlapping(encoded.as_ptr(), self.data.add(offset), encoded.len());
            *self.index.add(subchunk * 2) = offset as u64;
            *self.index.add(subchunk * 2 + 1) = encoded.len() as u64;
        }
        Ok(())
    }

    /// The shard (with its encoded subchunks in its spare capacity), the end of its encoded subchunks, and the
    /// decoded shard index.
    fn into_parts(self) -> (Vec<u8>, usize, Vec<u64>) {
        let data_end = self.offset.load(Ordering::Relaxed);
        (self.shard, data_end, self.shard_index)
    }
}

/// The encoding of a shard is planned on compute threads.
const _: () = {
    const fn assert_send_sync<T: MaybeSend + MaybeSync>() {}
    assert_send_sync::<ShardEncode<'static>>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::codec::{BytesCodec, ShardingCodecBuilder};
    use crate::array::{FillValue, data_type, transmute_to_bytes_vec};
    use zarrs_codec::{
        CodecSpecificOptions, ComputeResources, Resources, UnboundArrayToBytesCodecTraits,
    };

    /// Encode and decode shards with each write order, index location, and inner representation, with one and many
    /// blocks of subchunks.
    #[test]
    fn encode_shard() {
        let nz = |v: u64| NonZeroU64::new(v).unwrap();
        let shape = [nz(12), nz(10)];
        // The subchunks in the first row are entirely the fill value
        let data: Vec<u16> = (0..120u16).map(|i| if i < 30 { 7 } else { i }).collect();
        let bytes = ArrayBytes::new_flen(transmute_to_bytes_vec(data));
        let index_end = |shard: &[u8]| -> Vec<u64> {
            // The index of 4 x 3 subchunks (16 bytes each) and its crc32c checksum
            let index = &shard[shard.len() - 4 * 3 * 16 - 4..shard.len() - 4];
            bytemuck::pod_collect_to_vec(index)
        };
        for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
            for order in [SubchunkWriteOrder::C, SubchunkWriteOrder::Unordered] {
                for gzip in [false, true] {
                    for compute_target in [1, 8] {
                        let mut builder =
                            ShardingCodecBuilder::new(vec![nz(3), nz(4)], &data_type::uint16());
                        builder.index_location(location);
                        builder.array_to_bytes_codec(Arc::new(BytesCodec::little()));
                        #[cfg(feature = "gzip")]
                        if gzip {
                            builder.bytes_to_bytes_codecs(vec![Arc::new(
                                crate::array::codec::GzipCodec::new(5).unwrap(),
                            )]);
                        }
                        #[cfg(not(feature = "gzip"))]
                        let _ = gzip;
                        let codec = builder
                            .build_arc()
                            .with_context(
                                data_type::uint16(),
                                Some(FillValue::from(7u16)),
                                &CodecSpecificOptions::default().with_option(
                                    crate::array::codec::ShardingCodecOptions::default()
                                        .with_subchunk_write_order(order),
                                ),
                            )
                            .unwrap();
                        let options = CodecOptions::default();
                        let resources = Resources::default().with_compute_target(compute_target);
                        let shard = codec
                            .encode(bytes.clone(), &shape, &options, resources.compute())
                            .unwrap();
                        let decoded = codec
                            .decode(shard.clone(), &shape, &options, resources.compute())
                            .unwrap();
                        assert_eq!(decoded, bytes, "{location:?} {order:?} gzip: {gzip}");
                        if location == ShardingIndexLocation::End {
                            let index = index_end(&shard);
                            // The subchunks of the fill value are empty
                            assert!(index[..3 * 2].iter().all(|&x| x == EMPTY));
                            if matches!(order, SubchunkWriteOrder::C) {
                                // The subchunks are in C order, and the shard has no gaps
                                let offsets: Vec<u64> =
                                    index[3 * 2..].chunks(2).map(|entry| entry[0]).collect();
                                assert!(offsets.is_sorted());
                                assert_eq!(offsets[0], 0);
                            }
                        }
                    }
                }
            }
        }

        // A shard of only the fill value is only its index
        let codec = ShardingCodecBuilder::new(vec![nz(3), nz(4)], &data_type::uint16())
            .build_arc()
            .with_context(
                data_type::uint16(),
                Some(FillValue::from(7u16)),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        let fill = ArrayBytes::new_flen(transmute_to_bytes_vec(vec![7u16; 120]));
        let shard = codec
            .encode(
                fill,
                &shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        assert_eq!(shard.len(), 4 * 3 * 16 + 4);
    }
}

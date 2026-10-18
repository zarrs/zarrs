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
use zarrs_codec::plan::{ComputeCtx, Generator, Join, Outcome, Task, TaskValue};
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
        bytes.validate(
            shape.iter().map(|d| d.get()).product(),
            self.inner_codecs.data_type(),
        )?;
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let index_shape = sharding_index_shape(&chunks_per_shard);
        let index_encoded_size = usize::try_from(compute_index_encoded_size(
            &self.index_codecs,
            &index_shape,
        )?)
        .unwrap();
        let subchunk_grid = subchunk_grid(shape, &self.subchunk_shape)?;
        // The bound of the encoded size of a subchunk, if the encoded subchunks are written straight into the shard
        let direct_bound = match self.options.subchunk_write_order() {
            SubchunkWriteOrder::Unordered | SubchunkWriteOrder::Random => {
                match self
                    .inner_codecs
                    .encoded_representation(&self.subchunk_shape)?
                {
                    BytesRepresentation::BoundedSize(size)
                    | BytesRepresentation::FixedSize(size) => Some(usize::try_from(size).unwrap()),
                    BytesRepresentation::UnboundedSize => None,
                }
            }
            SubchunkWriteOrder::C => None,
        };
        let encode = Arc::new(ShardEncode {
            num_subchunks: num_subchunks(&subchunk_grid),
            codec: self,
            bytes,
            subchunk_grid,
            index_shape,
            index_encoded_size,
            options: *options,
        });
        encode.plan(direct_bound, ctx)
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
    /// Plan the encoding of the blocks of subchunks of the shard.
    fn plan(
        self: Arc<Self>,
        direct_bound: Option<usize>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let parallelism = match ctx.resources().compute_target() {
            0 => rayon::current_num_threads(),
            target => target,
        };
        let block_len = (self.num_subchunks / (2 * parallelism.max(1))).clamp(1, MAX_BLOCK_LEN);
        let num_blocks = self.num_subchunks.div_ceil(block_len);

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
            let children = {
                let encode = self.clone();
                Generator::indexed(num_blocks, move |index| {
                    let encode = encode.clone();
                    let subchunks = block(index);
                    Ok(Some(Task::compute(move |ctx| {
                        let encoded = encode.encode_block(subchunks, ctx)?;
                        Ok(Outcome::Value(TaskValue::Any(Box::new(encoded))))
                    })))
                })
            };
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
        if bytes.is_fill_value(inner_codecs.fill_value()) {
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
        for block in blocks {
            for (subchunk, encoded) in (block.first..).zip(&block.encoded) {
                if let Some(encoded) = encoded {
                    shard_index[subchunk * 2] = shard.len() as u64;
                    shard_index[subchunk * 2 + 1] = encoded.len() as u64;
                    shard.extend_from_slice(encoded);
                }
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
                                FillValue::from(7u16),
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
                FillValue::from(7u16),
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

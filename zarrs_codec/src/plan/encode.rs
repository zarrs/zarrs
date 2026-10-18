//! Planned encoding of chunks.

use std::any::Any;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::Indexer;

use smallvec::smallvec;
use zarrs_storage::byte_range::ByteRange;

use super::{
    ByteSource, ComputeCtx, Join, Outcome, PlanIndexer, SourceKey, Task, TaskHint, TaskValue,
    WriteOp, whole_chunk_hint,
};
use crate::{
    ArrayBytes, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions,
    ComputeResources, CowBytes, update_array_bytes,
};

/// Decode the `encoded` chunk of `shape` with `codec` (or the fill value if [`None`]), and update its elements `indexer` to `bytes`.
///
/// This is the read-modify-write of a partial encode of a codec that cannot partially encode.
///
/// # Errors
/// Returns a [`CodecError`] if decoding fails or `indexer` or `bytes` are invalid for the chunk.
pub fn decode_and_update<'a>(
    codec: &dyn ArrayToBytesCodecTraits,
    encoded: Option<CowBytes<'_>>,
    shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    bytes: &ArrayBytes<'_>,
    options: &CodecOptions,
    resources: &ComputeResources,
) -> Result<ArrayBytes<'a>, CodecError> {
    let chunk = match encoded {
        Some(encoded) => codec.decode(encoded, shape, options, resources)?,
        None => ArrayBytes::new_fill_value(
            codec.data_type(),
            shape.iter().map(|s| s.get()).product(),
            codec.fill_value(),
        )?,
    };
    update_array_bytes(
        chunk,
        bytemuck::must_cast_slice(shape),
        indexer,
        bytes,
        codec.data_type().size(),
    )
}

/// The scheduling hints of a task that encodes `bytes`.
#[must_use]
pub fn encode_hint(bytes: &ArrayBytes<'_>) -> TaskHint {
    TaskHint {
        read_bytes: None,
        compute_bytes: bytes.size() as u64,
    }
}

/// The encoded chunk that is the value of a task planned by [`ArrayToBytesCodecTraits::plan_encode`].
///
/// # Errors
/// Returns a [`CodecError`] if `value` is not encoded bytes.
pub fn encoded_value(value: TaskValue<'_>) -> Result<CowBytes<'_>, CodecError> {
    value.into_bytes().flatten().ok_or_else(|| {
        CodecError::Other("an encode task did not produce encoded bytes".to_string())
    })
}

/// Plan the encoding of a chunk of `shape` with one compute step that calls [`ArrayToBytesCodecTraits::encode`].
///
/// This is the default plan of an array-to-bytes codec.
/// The value of the task is the encoded chunk ([`TaskValue::Bytes`]).
///
/// # Errors
/// Returns a [`CodecError`] if the shape is invalid for the codec.
pub fn plan_encode_whole<'a>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    bytes: ArrayBytes<'a>,
    shape: &[NonZeroU64],
    options: &CodecOptions,
) -> Result<Task<'a>, CodecError> {
    let hint = encode_hint(&bytes);
    let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
    let options = *options;
    Ok(Task::compute(move |ctx| {
        let encoded = codec.encode(bytes, &shape, &options, ctx.resources())?;
        Ok(Outcome::Value(TaskValue::Bytes(Some(encoded))))
    })
    .with_hint(hint))
}

/// Encode the chunk `bytes` of `shape` with `codec` and write it to `key`, or erase `key` if the chunk is the fill value
/// and empty chunks are not stored.
///
/// The encode tasks planned by the codec are forked ([`ArrayToBytesCodecTraits::plans_encode_subtasks`]), unless the
/// compute step runs inline ([`ComputeCtx::runs_inline`]), where they would run serially.
///
/// # Errors
/// Returns a [`CodecError`] if the codec fails.
pub fn encode_and_write<'a>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    key: SourceKey,
    shape: &[NonZeroU64],
    bytes: ArrayBytes<'a>,
    options: &CodecOptions,
    ctx: &ComputeCtx,
) -> Result<Outcome<'a>, CodecError> {
    let write = |op| Outcome::Write {
        ops: smallvec![op],
        then: TaskValue::None,
    };
    if !options.store_empty_chunks() && bytes.is_fill_value(codec.fill_value()) {
        return Ok(write(WriteOp::Erase(key)));
    }
    if codec.plans_encode_subtasks() && !ctx.runs_inline() {
        let task = codec.plan_encode(bytes, shape, options)?;
        return Ok(Outcome::Fork {
            children: vec![task],
            join: Join::new(move |mut values, _ctx| {
                let encoded = encoded_value(values.pop().unwrap_or_default())?;
                Ok(Outcome::Write {
                    ops: smallvec![WriteOp::Set(key, encoded)],
                    then: TaskValue::None,
                })
            }),
        });
    }
    let encoded = codec.encode(bytes, shape, options, ctx.resources())?;
    Ok(write(WriteOp::Set(key, encoded)))
}

/// Plan a partial encode that reads, decodes, updates, encodes and writes the whole chunk `key` of `shape`.
///
/// This is the default of [`ArrayToBytesCodecTraits::plan_partial_encode`].
/// The chunk is erased if it becomes the fill value and empty chunks are not stored (see [`encode_and_write`]).
/// The value of the task is [`TaskValue::None`], as no state is kept between partial encodes.
#[must_use]
pub fn plan_partial_encode_whole<'a>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    key: SourceKey,
    shape: &[NonZeroU64],
    indexer: PlanIndexer<'a>,
    bytes: ArrayBytes<'a>,
    options: &CodecOptions,
) -> Task<'a> {
    let source = ByteSource::external(key);
    let hint = whole_chunk_hint(&*codec, &source, shape);
    let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
    let options = *options;
    Task::read(
        source,
        [ByteRange::FromStart(0, None)],
        move |mut reads, ctx| {
            let chunk = decode_and_update(
                &*codec,
                reads.pop().flatten(),
                &shape,
                indexer.as_indexer(),
                &bytes,
                &options,
                ctx.resources(),
            )?;
            encode_and_write(codec, key, &shape, chunk, &options, ctx)
        },
    )
    .with_hint(hint)
}

/// Map a partial encode of the elements `indexer` of a chunk of `shape` to `bytes` through an elementwise array-to-array `codec`.
///
/// The elements are encoded as a 1-D array, and update the same elements of the encoded chunk.
/// This implements [`ArrayToArrayCodecTraits::partial_encode_request`] for elementwise codecs.
///
/// # Errors
/// Returns a [`CodecError`] if the indexer is invalid for `shape`, `bytes` are invalid for the indexer, or the codec fails.
pub fn partial_encode_request_elementwise<'a>(
    codec: &dyn ArrayToArrayCodecTraits,
    shape: &[NonZeroU64],
    indexer: PlanIndexer<'a>,
    bytes: ArrayBytes<'a>,
    options: &CodecOptions,
    resources: &ComputeResources,
) -> Result<Option<(PlanIndexer<'a>, ArrayBytes<'a>)>, CodecError> {
    indexer.validate(shape)?;
    let bytes = match NonZeroU64::new(indexer.as_indexer().len()) {
        Some(len) => codec.encode(bytes, &[len], options, resources)?,
        None => {
            bytes.validate(0, codec.data_type())?;
            ArrayBytes::new_fill_value(codec.encoded_data_type(), 0, codec.encoded_fill_value())?
        }
    };
    Ok(Some((indexer, bytes)))
}

/// The state of an encoded chunk kept between partial encodes (e.g. a decoded shard index).
///
/// See [`ArrayToBytesCodecTraits::plan_partial_encode`].
pub type PartialEncodeState = Arc<dyn Any + Send + Sync>;

/// The value of a task planned by [`ArrayToBytesCodecTraits::plan_partial_encode`].
#[must_use]
pub fn partial_encode_value<'a>(state: PartialEncodeState) -> TaskValue<'a> {
    TaskValue::Any(Box::new(state))
}

/// The state of an encoded chunk after a partial encode, from the value of its task.
///
/// Returns [`None`] if `value` is not a partial encode state.
#[must_use]
pub fn partial_encode_state(value: TaskValue<'_>) -> Option<PartialEncodeState> {
    value.downcast()
}

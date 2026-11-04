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
            codec.fill_value().ok_or(CodecError::UndefinedFillValue)?,
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

/// The encoded chunk that is the value of an [`ArrayToBytesCodecTraits::encode_step`] (or a task planned by
/// [`plan_encode`]).
///
/// # Errors
/// Returns a [`CodecError`] if `value` is not encoded bytes.
pub fn encoded_value(value: TaskValue<'_>) -> Result<CowBytes<'_>, CodecError> {
    value.into_bytes().flatten().ok_or_else(|| {
        CodecError::Other("an encode task did not produce encoded bytes".to_string())
    })
}

/// Plan the encoding of `bytes`, a chunk of `shape`, with `codec`.
///
/// The task encodes in a compute step ([`ArrayToBytesCodecTraits::encode_step`]), and its value is the encoded chunk
/// ([`TaskValue::Bytes`]).
#[must_use]
pub fn plan_encode<'a>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    bytes: ArrayBytes<'a>,
    shape: &[NonZeroU64],
    options: &CodecOptions,
) -> Task<'a> {
    let hint = encode_hint(&bytes);
    let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
    let options = *options;
    Task::compute(move |ctx| codec.encode_step(bytes, &shape, &options, ctx)).with_hint(hint)
}

/// Continue `outcome`, whose value is the encoded chunk, with `then`.
///
/// `then` runs straight away if the outcome is the value, otherwise once the tasks of the outcome complete.
///
/// # Errors
/// Returns a [`CodecError`] if the outcome is a value that is not encoded bytes, or `then` fails.
pub fn then_encoded<'a>(
    outcome: Outcome<'a>,
    then: impl FnOnce(CowBytes<'a>, &ComputeCtx) -> Result<Outcome<'a>, CodecError>
    + zarrs_plugin::MaybeSend
    + 'a,
    ctx: &ComputeCtx,
) -> Result<Outcome<'a>, CodecError> {
    match outcome {
        Outcome::Value(value) => then(encoded_value(value)?, ctx),
        outcome => Ok(Outcome::Fork {
            children: vec![Task::compute(move |_| Ok(outcome))],
            join: Join::new(move |mut values, ctx| {
                then(encoded_value(values.pop().unwrap_or_default())?, ctx)
            }),
        }),
    }
}

/// Encode the chunk `bytes` of `shape` with `codec` and write it to `key`, or erase `key` if the chunk is the fill value
/// and empty chunks are not stored.
///
/// The chunk is encoded and written in the compute step `ctx` ([`ArrayToBytesCodecTraits::encode_write_step`]), which
/// may fork tasks of the codec.
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
    codec.encode_write_step(bytes, shape, key, options, ctx)
}

/// Encode the chunk `bytes` of `shape` with `codec` and set it at `key` with one write, or erase `key` if the chunk is
/// the fill value and empty chunks are not stored.
///
/// The chunk is encoded in the compute step `ctx` ([`ArrayToBytesCodecTraits::encode_step`]), which may fork tasks of
/// the codec. This is the default of [`ArrayToBytesCodecTraits::encode_write_step`].
///
/// # Errors
/// Returns a [`CodecError`] if the codec fails.
pub fn encode_and_set<'a>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    bytes: ArrayBytes<'a>,
    shape: &[NonZeroU64],
    key: SourceKey,
    options: &CodecOptions,
    ctx: &ComputeCtx,
) -> Result<Outcome<'a>, CodecError> {
    let write = |op| Outcome::Write {
        ops: smallvec![op],
        then: TaskValue::None,
    };
    if !options.store_empty_chunks()
        && codec
            .fill_value()
            .is_some_and(|fill| bytes.is_fill_value(fill))
    {
        return Ok(write(WriteOp::Erase(key)));
    }
    let outcome = codec.encode_step(bytes, shape, options, ctx)?;
    then_encoded(
        outcome,
        move |encoded, _| Ok(write(WriteOp::Set(key, encoded))),
        ctx,
    )
}

/// Plan a partial encode that reads, decodes, updates, encodes and writes the whole chunk `key` of `shape`.
/// An update covering the whole chunk replaces it without reading or decoding the existing bytes.
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
    if indexer.is_whole(shape) {
        let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        return Task::compute(move |ctx| {
            indexer.validate(&shape)?;
            bytes.validate(indexer.as_indexer().len(), codec.data_type())?;
            encode_and_write(codec, key, &shape, bytes, &options, ctx)
        });
    }
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
) -> Result<PartialEncodeRequest<'a>, CodecError> {
    indexer.validate(shape)?;
    let bytes = match NonZeroU64::new(indexer.as_indexer().len()) {
        Some(len) => codec.encode(bytes, &[len], options, resources)?,
        None => {
            bytes.validate(0, codec.data_type())?;
            ArrayBytes::new_fill_value(
                codec.encoded_data_type(),
                0,
                codec
                    .encoded_fill_value()
                    .ok_or(CodecError::UndefinedFillValue)?,
            )?
        }
    };
    Ok(PartialEncodeRequest::Mapped(indexer, bytes))
}

/// A partial encode of the elements `indexer` of a chunk to `bytes`, mapped through an array-to-array codec or not.
///
/// See [`ArrayToArrayCodecTraits::partial_encode_request`].
#[derive(Debug)]
pub enum PartialEncodeRequest<'a> {
    /// The indexer of the encoded elements to update in the encoded chunk and their encoded bytes.
    Mapped(PlanIndexer<'a>, ArrayBytes<'a>),
    /// The unchanged request, as the codec does not map partial encodes.
    ///
    /// The chunk is decoded, updated and encoded in full by this codec and the codecs after it instead.
    Unmapped(PlanIndexer<'a>, ArrayBytes<'a>),
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

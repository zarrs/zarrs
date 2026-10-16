//! Planned encoding of chunks.

use std::any::Any;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::Indexer;

use super::{Outcome, PlanIndexer, Task, TaskHint, TaskValue};
use crate::{
    ArrayBytes, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions,
    CowBytes, Resources, update_array_bytes,
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
    resources: &Resources,
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
    resources: &Resources,
) -> Result<Option<(PlanIndexer<'a>, ArrayBytes<'a>)>, CodecError> {
    indexer
        .as_indexer()
        .validate(bytemuck::must_cast_slice(shape))?;
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

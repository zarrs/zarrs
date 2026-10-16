//! Planned encoding of chunks.

use std::any::Any;
use std::num::NonZeroU64;
use std::sync::Arc;

use super::{Outcome, PlanIndexer, Task, TaskHint, TaskValue};
use crate::{
    ArrayBytes, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions,
    CowBytes, Resources,
};

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
    match value {
        TaskValue::Bytes(Some(bytes)) => Ok(bytes),
        TaskValue::None
        | TaskValue::Array(_)
        | TaskValue::Bytes(None)
        | TaskValue::Prepared(_)
        | TaskValue::Any(_) => Err(CodecError::Other(
            "an encode task did not produce encoded bytes".to_string(),
        )),
    }
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

/// Map a partial encode of the elements `indexer` to `bytes` through an elementwise array-to-array `codec`.
///
/// The elements are encoded as a 1-D array, and update the same elements of the encoded chunk.
/// See [`ArrayToArrayCodecTraits::partial_encode_request`].
///
/// # Errors
/// Returns a [`CodecError`] if `bytes` are invalid for the indexer or the codec fails.
pub fn partial_encode_request_elementwise<'a>(
    codec: &dyn ArrayToArrayCodecTraits,
    indexer: PlanIndexer<'a>,
    bytes: ArrayBytes<'a>,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<(PlanIndexer<'a>, ArrayBytes<'a>), CodecError> {
    let bytes = match NonZeroU64::new(indexer.as_indexer().len()) {
        Some(len) => codec.encode(bytes, &[len], options, resources)?,
        None => {
            bytes.validate(0, codec.data_type())?;
            ArrayBytes::new_fill_value(codec.encoded_data_type(), 0, codec.encoded_fill_value())?
        }
    };
    Ok((indexer, bytes))
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
    match value {
        TaskValue::Any(any) => any
            .downcast::<PartialEncodeState>()
            .ok()
            .map(|state| *state),
        TaskValue::None | TaskValue::Array(_) | TaskValue::Bytes(_) | TaskValue::Prepared(_) => {
            None
        }
    }
}

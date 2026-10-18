//! Planned decoding of whole chunks.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::Indexer;
use zarrs_storage::byte_range::ByteRange;

use super::{
    ByteSource, DecodeRequest, Outcome, PlanIndexer, Task, TaskHint, TaskTarget, TaskValue,
};
use crate::{
    ArrayBytes, ArrayBytesDecodeIntoInput, ArrayToBytesCodecTraits, CodecError, CodecOptions,
    Resources, copy_array_subset_into_target, copy_fill_value_into, decode_into_array_bytes_target,
};
use zarrs_data_type::{DataType, FillValue};

/// Returns true if `indexer` selects every element of a chunk of `shape` in C order.
#[must_use]
pub fn is_whole_chunk(indexer: &dyn Indexer, shape: &[NonZeroU64]) -> bool {
    indexer.as_array_subset().is_some_and(|subset| {
        subset.dimensionality() == shape.len()
            && subset.start().iter().all(|&start| start == 0)
            && std::iter::zip(subset.shape().iter(), shape).all(|(&a, b)| a == b.get())
    })
}

/// The scheduling hints of a task that reads all of `source` and decodes a chunk of `shape` elements of `codec`.
#[must_use]
pub fn whole_chunk_hint(
    codec: &dyn ArrayToBytesCodecTraits,
    source: &ByteSource<'_>,
    shape: &[NonZeroU64],
) -> TaskHint {
    let num_elements = shape.iter().map(|s| s.get()).product::<u64>();
    TaskHint {
        read_bytes: source.len(),
        compute_bytes: num_elements * codec.data_type().fixed_size().unwrap_or(1) as u64,
    }
}

/// Plan the decoding of `request` from a chunk by reading and decoding the entire chunk.
///
/// This is the default plan of an array-to-bytes codec: one task that reads all of `source`, then
/// - decodes directly into the target if the request is for the whole chunk ([`ArrayToBytesCodecTraits::decode_into`]), or
/// - decodes the chunk and extracts the requested elements.
///
/// If the source does not exist, the requested elements are the fill value.
///
/// # Errors
/// Returns a [`CodecError`] if the request is invalid.
pub fn plan_decode_whole<'a>(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    source: ByteSource<'a>,
    shape: &[NonZeroU64],
    request: DecodeRequest<'a>,
    options: &CodecOptions,
) -> Result<Task<'a>, CodecError> {
    request.indexer.validate(shape)?;
    let hint = whole_chunk_hint(&*codec, &source, shape);
    let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
    let options = *options;
    Ok(Task::read(
        source,
        [ByteRange::FromStart(0, None)],
        move |mut reads, ctx| {
            let encoded = reads.pop().flatten();
            decode_whole(
                &*codec,
                encoded.map(ArrayBytesDecodeIntoInput::Bytes),
                &shape,
                request,
                &options,
                ctx.resources(),
            )
        },
    )
    .with_hint(hint))
}

/// Decode `request` from the `encoded` bytes of a whole chunk, or the fill value if [`None`].
///
/// Whole-chunk requests with an output target decode directly into the target with [`ArrayToBytesCodecTraits::decode_into`].
///
/// # Errors
/// Returns a [`CodecError`] if decoding fails or the request is invalid.
pub fn decode_whole<'a>(
    codec: &dyn ArrayToBytesCodecTraits,
    encoded: Option<ArrayBytesDecodeIntoInput<'a>>,
    shape: &[NonZeroU64],
    request: DecodeRequest<'a>,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Outcome<'a>, CodecError> {
    let data_type = codec.data_type();
    let Some(encoded) = encoded else {
        return place_fill_value(data_type, codec.fill_value(), request);
    };
    let whole = request.indexer.is_whole(shape);
    let DecodeRequest { indexer, target } = request;
    let indexer = indexer.as_indexer();
    match target {
        TaskTarget::Views(views) if whole => {
            views.with_target(|target| {
                codec.decode_into(encoded, shape, target, options, resources)
            })?;
            Ok(Outcome::Done)
        }
        TaskTarget::Views(views) => {
            let decoded = codec.decode(
                encoded.into_bytes(options, resources)?,
                shape,
                options,
                resources,
            )?;
            views.with_target(|target| {
                copy_array_subset_into_target(
                    &decoded,
                    bytemuck::must_cast_slice(shape),
                    indexer,
                    data_type,
                    target,
                )
            })?;
            Ok(Outcome::Done)
        }
        TaskTarget::Value => {
            let decoded = codec.decode(
                encoded.into_bytes(options, resources)?,
                shape,
                options,
                resources,
            )?;
            let bytes = if whole {
                decoded
            } else {
                decoded
                    .extract_array_subset(indexer, bytemuck::must_cast_slice(shape), data_type)?
                    .into_owned()
            };
            bytes.validate(indexer.len(), data_type)?;
            Ok(Outcome::Value(TaskValue::Array(bytes)))
        }
    }
}

/// How a request for the decoded elements of an array-to-array codec is planned.
///
/// See [`ArrayToArrayCodecTraits::plan_decode_request`](crate::ArrayToArrayCodecTraits::plan_decode_request).
#[derive(Debug)]
pub enum ArrayToArrayDecodePlan<'a> {
    /// Decode a request for encoded elements, which are the requested decoded elements in the same order, into the same target.
    ///
    /// For example, a codec that only rearranges elements maps the indexer, and a codec with an identity decode forwards the request unchanged.
    Forward(DecodeRequest<'a>),
    /// Decode `indexer` (of the encoded elements) as a value, decode it with the codec as an array of `shape`, and place the result in `target`.
    Decode {
        /// The encoded elements to decode.
        indexer: PlanIndexer<'a>,
        /// The shape passed to the decode of the codec.
        shape: Vec<NonZeroU64>,
        /// The target of the decoded elements.
        target: TaskTarget<'a>,
    },
}

/// Plan a request for the decoded elements of an elementwise array-to-array codec with a chunk of `shape`.
///
/// The requested elements are decoded from the same encoded elements, and then decoded by the codec.
///
/// # Errors
/// Returns a [`CodecError`] if the indexer is invalid for `shape`.
pub fn plan_decode_request_elementwise<'a>(
    shape: &[NonZeroU64],
    request: DecodeRequest<'a>,
) -> Result<ArrayToArrayDecodePlan<'a>, CodecError> {
    let indexer = request.indexer.as_indexer();
    indexer.validate(bytemuck::must_cast_slice(shape))?;
    let output_shape: Option<Vec<NonZeroU64>> = indexer
        .output_shape()
        .into_iter()
        .map(NonZeroU64::new)
        .collect();
    Ok(match output_shape {
        Some(shape) => ArrayToArrayDecodePlan::Decode {
            indexer: request.indexer,
            shape,
            target: request.target,
        },
        // No elements are requested
        None => ArrayToArrayDecodePlan::Forward(request),
    })
}

/// Place decoded `bytes` in `target`.
///
/// # Errors
/// Returns a [`CodecError`] if `bytes` is incompatible with the target.
pub fn place_decoded<'a>(
    bytes: ArrayBytes<'a>,
    target: TaskTarget<'a>,
) -> Result<Outcome<'a>, CodecError> {
    match target {
        TaskTarget::Views(views) => {
            views.with_target(|target| decode_into_array_bytes_target(&bytes, target))?;
            Ok(Outcome::Done)
        }
        TaskTarget::Value => Ok(Outcome::Value(TaskValue::Array(bytes))),
    }
}

/// Place the fill value in the elements of `request`.
///
/// # Errors
/// Returns a [`CodecError`] if the fill value is incompatible with the data type or the target.
pub fn place_fill_value<'a>(
    data_type: &DataType,
    fill_value: &FillValue,
    request: DecodeRequest<'a>,
) -> Result<Outcome<'a>, CodecError> {
    match request.target {
        TaskTarget::Views(views) => {
            views.with_target(|target| copy_fill_value_into(data_type, fill_value, target))?;
            Ok(Outcome::Done)
        }
        TaskTarget::Value => Ok(Outcome::Value(TaskValue::Array(
            ArrayBytes::new_fill_value(data_type, request.indexer.as_indexer().len(), fill_value)?,
        ))),
    }
}

/// The value of a task planned by [`ArrayToBytesCodecTraits::plan_local_subchunk_grids`]: the chunk-local subchunk
/// grids of a chunk, outermost first.
#[must_use]
pub fn local_subchunk_grids_value<'a>(grids: Vec<Option<crate::ChunkGrid>>) -> TaskValue<'a> {
    TaskValue::Any(Box::new(grids))
}

/// The chunk-local subchunk grids of a chunk, from the value of a task planned by
/// [`ArrayToBytesCodecTraits::plan_local_subchunk_grids`].
///
/// # Errors
/// Returns a [`CodecError`] if `value` is not chunk-local subchunk grids.
pub fn local_subchunk_grids(
    value: TaskValue<'_>,
) -> Result<Vec<Option<crate::ChunkGrid>>, CodecError> {
    value.downcast().ok_or_else(|| {
        CodecError::Other("a task did not produce chunk-local subchunk grids".to_string())
    })
}

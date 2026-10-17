//! Planned retrieval of encoded subchunks.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ArrayIndices;
use zarrs_metadata::ChunkShape;
use zarrs_storage::Bytes;
use zarrs_storage::byte_range::ByteRange;

use super::{ByteSource, Join, Outcome, Task, TaskValue};
use crate::{ArrayToBytesCodecTraits, CodecError, CodecOptions, CowBytes};

/// The encoded bytes of a subchunk and the shape to decode them with.
///
/// Decode the [`bytes`](Self::bytes) with the subchunk codec at the level the subchunk was retrieved from
/// ([`subchunk_codecs_at_level`](crate::ArrayToBytesCodecSubchunkingTraits::subchunk_codecs_at_level)), passing
/// [`shape`](Self::shape). That codec is a property of the array rather than of an individual subchunk, whereas the
/// shape is resolved per subchunk.
///
/// The bytes are in the *encoded domain* of the codec that created the subchunk, which is not necessarily the domain
/// of the array they were retrieved from. If the array has array-to-array codecs, then decoding yields array bytes
/// that those codecs have yet to decode:
/// - the values are in the data type of the subchunk codec, which may differ from the data type of the array (e.g.
///   with a `cast_value` codec), and
/// - the elements are ordered by [`shape`](Self::shape), which may differ in extent and even in dimensionality from
///   the shape of the subchunk in the subchunk grid of the array (e.g. with a `transpose`, `reshape`, or `squeeze`
///   codec).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedSubchunk {
    bytes: Bytes,
    shape: ChunkShape,
}

impl EncodedSubchunk {
    /// Create a new [`EncodedSubchunk`].
    ///
    /// The `shape` must be the shape of the subchunk in the encoded domain of the codec that created it.
    #[must_use]
    pub fn new(bytes: Bytes, shape: ChunkShape) -> Self {
        Self { bytes, shape }
    }

    /// Return the encoded bytes of the subchunk.
    ///
    /// These bytes are suitable for passthrough without decoding, such as copying a subchunk into another array with
    /// matching subchunk codecs.
    #[must_use]
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// Return the shape to decode the [`bytes`](Self::bytes) with.
    ///
    /// This is the shape of the subchunk in the encoded domain of its subchunk codec, which may differ in extent and
    /// dimensionality from the shape of the subchunk in the subchunk grid it was addressed by.
    #[must_use]
    pub fn shape(&self) -> &[NonZeroU64] {
        &self.shape
    }

    /// Consume the [`EncodedSubchunk`] and return its bytes and shape.
    #[must_use]
    pub fn into_parts(self) -> (Bytes, ChunkShape) {
        (self.bytes, self.shape)
    }
}

/// The location of a subchunk in the encoded bytes of a chunk.
///
/// This is the value of a task planned by [`ArrayToBytesCodecTraits::plan_locate_subchunk`]
/// ([`subchunk_location_value`]).
#[derive(Debug)]
pub struct SubchunkLocation {
    /// The encoded bytes of the subchunk.
    pub source: ByteSource<'static>,
    /// The shape of the subchunk, in the decoded domain of the subchunk codec.
    pub shape: ChunkShape,
    /// The coordinates of the located element in the subchunk.
    pub coordinates: ArrayIndices,
}

/// The value of a task planned by [`ArrayToBytesCodecTraits::plan_locate_subchunk`]: the location of a subchunk, or
/// [`None`] if the subchunk (or its chunk) is not stored.
#[must_use]
pub fn subchunk_location_value<'a>(location: Option<SubchunkLocation>) -> TaskValue<'a> {
    TaskValue::Any(Box::new(location))
}

/// The location of a subchunk, from the value of a task planned by [`ArrayToBytesCodecTraits::plan_locate_subchunk`].
///
/// # Errors
/// Returns a [`CodecError`] if `value` is not a subchunk location.
pub fn subchunk_location(value: TaskValue<'_>) -> Result<Option<SubchunkLocation>, CodecError> {
    value
        .downcast()
        .ok_or_else(|| CodecError::Other("a task did not produce a subchunk location".to_string()))
}

/// The encoded subchunk of the value of a task planned by [`plan_retrieve_encoded_subchunk`], or [`None`] if the
/// subchunk is not stored.
///
/// # Errors
/// Returns a [`CodecError`] if `value` is not an encoded subchunk.
pub fn encoded_subchunk(value: TaskValue<'_>) -> Result<Option<EncodedSubchunk>, CodecError> {
    value
        .downcast()
        .ok_or_else(|| CodecError::Other("a task did not produce an encoded subchunk".to_string()))
}

/// Plan retrieving the encoded subchunk at `level` that contains the element at `coordinates` of the chunk `source`
/// of `shape` encoded with `codec`.
///
/// The level zero subchunk containing the element is located with
/// [`ArrayToBytesCodecTraits::plan_locate_subchunk`]. Deeper levels descend into it with the level zero subchunk
/// codec, so only the indexes on the way down and the bytes of the subchunk are read.
///
/// The value of the task is the encoded subchunk ([`encoded_subchunk`]), which is [`None`] if it is not stored.
///
/// # Errors
/// Returns [`CodecError::UnsupportedEncodedSubchunk`] if `codec` does not have subchunks at `level`, or a
/// [`CodecError`] if `coordinates` are out of bounds of the chunk.
pub fn plan_retrieve_encoded_subchunk(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    source: ByteSource<'static>,
    shape: &[NonZeroU64],
    level: usize,
    coordinates: &[u64],
    options: &CodecOptions,
) -> Result<Task<'static>, CodecError> {
    let mut subchunk_codecs = codec.subchunk_codecs();
    if subchunk_codecs.len() <= level {
        return Err(CodecError::UnsupportedEncodedSubchunk);
    }
    let subchunk_codec = subchunk_codecs.swap_remove(0);
    let locate = codec.plan_locate_subchunk(source, shape, coordinates, options)?;
    let options = *options;
    Ok(Task::compute(move |_| {
        Ok(Outcome::Fork {
            children: vec![locate],
            join: Join::new(move |mut values, _ctx| {
                let Some(SubchunkLocation {
                    source,
                    shape,
                    coordinates,
                }) = subchunk_location(values.pop().unwrap_or_default())?
                else {
                    return Ok(Outcome::Value(TaskValue::Any(Box::new(
                        None::<EncodedSubchunk>,
                    ))));
                };
                let task = if level == 0 {
                    Task::read(
                        source,
                        [ByteRange::FromStart(0, None)],
                        move |mut reads, _| {
                            let subchunk = reads.pop().flatten().map(|bytes| {
                                EncodedSubchunk::new(CowBytes::into_bytes(bytes), shape)
                            });
                            Ok(Outcome::Value(TaskValue::Any(Box::new(subchunk))))
                        },
                    )
                } else {
                    plan_retrieve_encoded_subchunk(
                        subchunk_codec,
                        source,
                        &shape,
                        level - 1,
                        &coordinates,
                        &options,
                    )?
                };
                Ok(Outcome::Fork {
                    children: vec![task],
                    join: Join::new(|mut values, _ctx| {
                        Ok(Outcome::Value(values.pop().unwrap_or_default()))
                    }),
                })
            }),
        })
    }))
}

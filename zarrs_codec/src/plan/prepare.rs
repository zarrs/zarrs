//! Decoders of a chunk prepared once for planning many requests.

use std::num::NonZeroU64;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs_plugin::{MaybeSend, MaybeSync};

use zarrs_data_type::DataType;
use zarrs_storage::byte_range::ByteRange;

use super::{
    ByteSource, DecodeRequest, Join, Outcome, Task, TaskTarget, TaskValue, decode_whole,
    whole_chunk_hint,
};
use crate::{
    ArrayBytes, ArrayToBytesCodecTraits, CodecError, CodecOptions, decode_into_array_bytes_target,
};

/// A decoder of a chunk that plans requests from state prepared once (e.g. a shard index, or bytes decoded by
/// bytes-to-bytes codecs that cannot decode part of their input).
///
/// See [`ArrayToBytesCodecTraits::prepare_decode`].
pub trait PreparedDecoder: core::fmt::Debug + MaybeSend + MaybeSync {
    /// Plan the decoding of `request` from the chunk.
    ///
    /// The task is equivalent to [`ArrayToBytesCodecTraits::plan_decode`] on the source the decoder was prepared from.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the request is invalid.
    fn plan_decode<'a>(
        self: Arc<Self>,
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError>;

    /// Whether the chunk exists, if known without I/O.
    fn exists(&self) -> Option<bool>;

    /// The size of the chunk bytes held by the decoder.
    fn size_held(&self) -> usize;
}

/// The preparation of a [`PreparedDecoder`].
#[derive(Debug)]
pub enum PrepareDecode {
    /// A decoder prepared without I/O.
    Ready(Arc<dyn PreparedDecoder>),
    /// A task with the value [`TaskValue::Prepared`].
    Plan(Task<'static>),
}

impl PrepareDecode {
    /// Map the prepared decoder with `f`.
    #[must_use]
    pub fn map(
        self,
        f: impl FnOnce(Arc<dyn PreparedDecoder>) -> Arc<dyn PreparedDecoder> + MaybeSend + 'static,
    ) -> Self {
        match self {
            Self::Ready(decoder) => Self::Ready(f(decoder)),
            Self::Plan(task) => Self::Plan(Task::compute(move |_| {
                Ok(Outcome::Fork {
                    children: vec![task],
                    join: Join::new(move |mut values, _| {
                        let decoder = prepared(values.pop().unwrap_or_default())?;
                        Ok(Outcome::Value(TaskValue::Prepared(f(decoder))))
                    }),
                })
            })),
        }
    }
}

/// The prepared decoder of a [`TaskValue::Prepared`].
///
/// # Errors
/// Returns a [`CodecError`] if `value` is not a prepared decoder.
pub fn prepared(value: TaskValue<'_>) -> Result<Arc<dyn PreparedDecoder>, CodecError> {
    value
        .into_prepared()
        .ok_or_else(|| CodecError::Other("expected a prepared decoder".to_string()))
}

/// Prepare a decoder of the chunk `source` of `shape` by reading and decoding the entire chunk once.
///
/// The prepared decoder is a [`DecodedDecoder`], or a [`MissingDecoder`] if the chunk does not exist.
/// This is the default of [`ArrayToBytesCodecTraits::prepare_decode`], for codecs that cannot decode part of a chunk.
#[must_use]
pub fn prepare_decode_whole(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    source: ByteSource<'static>,
    shape: &[NonZeroU64],
    options: &CodecOptions,
) -> PrepareDecode {
    let hint = whole_chunk_hint(&*codec, &source, shape);
    let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
    let options = *options;
    PrepareDecode::Plan(
        Task::read(
            source,
            [ByteRange::FromStart(0, None)],
            move |mut reads, ctx| {
                let decoder: Arc<dyn PreparedDecoder> = if let Some(encoded) = reads.pop().flatten()
                {
                    let bytes = codec
                        .decode(encoded, &shape, &options, ctx.resources())?
                        .into_owned();
                    Arc::new(DecodedDecoder::new(
                        bytes,
                        &shape,
                        codec.data_type().clone(),
                    ))
                } else {
                    Arc::new(MissingDecoder::new(codec, &shape))
                };
                Ok(Outcome::Value(TaskValue::Prepared(decoder)))
            },
        )
        .with_hint(hint),
    )
}

/// A decoder that plans each request from a source with [`ArrayToBytesCodecTraits::plan_decode`].
///
/// This is the prepared decoder of codecs without state to prepare.
#[derive(Debug)]
pub struct SourceDecoder {
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    source: ByteSource<'static>,
    shape: SmallVec<[NonZeroU64; 4]>,
}

impl SourceDecoder {
    /// Create a decoder of the chunk with `shape` at `source` encoded with `codec`.
    #[must_use]
    pub fn new(
        codec: Arc<dyn ArrayToBytesCodecTraits>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
    ) -> Self {
        Self {
            codec,
            source,
            shape: shape.into(),
        }
    }
}

impl PreparedDecoder for SourceDecoder {
    fn plan_decode<'a>(
        self: Arc<Self>,
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        self.codec
            .clone()
            .plan_decode(self.source.clone(), &self.shape, request, options)
    }

    fn exists(&self) -> Option<bool> {
        (!self.source.is_external()).then_some(true)
    }

    fn size_held(&self) -> usize {
        self.source.size_held()
    }
}

/// A decoder of a chunk that does not exist, which decodes every request as the fill value.
#[derive(Debug)]
pub struct MissingDecoder {
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    shape: SmallVec<[NonZeroU64; 4]>,
}

impl MissingDecoder {
    /// Create a decoder of a missing chunk with `shape` encoded with `codec`.
    #[must_use]
    pub fn new(codec: Arc<dyn ArrayToBytesCodecTraits>, shape: &[NonZeroU64]) -> Self {
        Self {
            codec,
            shape: shape.into(),
        }
    }
}

impl PreparedDecoder for MissingDecoder {
    fn plan_decode<'a>(
        self: Arc<Self>,
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        request
            .indexer
            .as_indexer()
            .validate(bytemuck::must_cast_slice(&self.shape))?;
        let options = *options;
        Ok(Task::compute(move |ctx| {
            decode_whole(
                &*self.codec,
                None,
                &self.shape,
                request,
                &options,
                ctx.resources(),
            )
        }))
    }

    fn exists(&self) -> Option<bool> {
        Some(false)
    }

    fn size_held(&self) -> usize {
        0
    }
}

/// A decoder of a chunk that holds the decoded chunk, from which it extracts each request.
///
/// This is the prepared decoder of codecs that cannot decode part of a chunk.
#[derive(Debug)]
pub struct DecodedDecoder {
    bytes: ArrayBytes<'static>,
    shape: SmallVec<[NonZeroU64; 4]>,
    data_type: DataType,
}

impl DecodedDecoder {
    /// Create a decoder of the decoded chunk `bytes` with `shape` and `data_type`.
    #[must_use]
    pub fn new(bytes: ArrayBytes<'static>, shape: &[NonZeroU64], data_type: DataType) -> Self {
        Self {
            bytes,
            shape: shape.into(),
            data_type,
        }
    }
}

impl PreparedDecoder for DecodedDecoder {
    fn plan_decode<'a>(
        self: Arc<Self>,
        request: DecodeRequest<'a>,
        _options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        let shape: &[u64] = bytemuck::must_cast_slice(&self.shape);
        request.indexer.as_indexer().validate(shape)?;
        Ok(Task::compute(move |_| {
            let DecodeRequest { indexer, target } = request;
            let shape: &[u64] = bytemuck::must_cast_slice(&self.shape);
            let extracted =
                self.bytes
                    .extract_array_subset(indexer.as_indexer(), shape, &self.data_type)?;
            match target {
                TaskTarget::Views(views) => {
                    views
                        .with_target(|target| decode_into_array_bytes_target(&extracted, target))?;
                    Ok(Outcome::Done)
                }
                TaskTarget::Value => Ok(Outcome::Value(TaskValue::Array(extracted.into_owned()))),
            }
        }))
    }

    fn exists(&self) -> Option<bool> {
        Some(true)
    }

    fn size_held(&self) -> usize {
        self.bytes.size()
    }
}

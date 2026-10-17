use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_data_type::{DataType, FillValue};

use crate::plan::{ByteSource, DecodeRequest, PartialEncodeState, PlanIndexer, SourceKey, Task};
use crate::{
    ArrayBytes, ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayCodecTraits,
    BytesRepresentation, ChunkGridDecoded, ChunkGridDecodedRef, CodecCreateError, CodecError,
    CodecOptions, CodecSpecificOptions, CodecTraits, CowBytes, Resources,
    decode_into_array_bytes_target,
};

/// Subchunking traits for an array-to-bytes codec bound to a data type and fill value.
pub trait ArrayToBytesCodecSubchunkingTraits: ArrayCodecTraits {
    /// Return the decoded subchunk grids created by this codec.
    ///
    /// Grids are ordered from outermost to innermost. An empty vector indicates
    /// that the codec does not expose subchunks. A [`ChunkGridDecoded::None`] or
    /// [`ChunkGridDecoded::ChunkLocal`] entry preserves the level in the hierarchy
    /// when it cannot be resolved globally.
    ///
    /// # Errors
    /// Returns a [`ChunkGridCreateError`] if the chunk grid is not supported by this codec.
    fn decoded_subchunk_grids(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<Vec<ChunkGridDecoded>, ChunkGridCreateError>;

    /// Return the outermost decoded subchunk grid created by this codec.
    ///
    /// This is a compatibility wrapper around [`decoded_subchunk_grids`](Self::decoded_subchunk_grids).
    /// An empty hierarchy is returned as [`ChunkGridDecoded::None`].
    ///
    /// # Errors
    /// Returns a [`ChunkGridCreateError`] if the chunk grid is not supported by this codec.
    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        Ok(self
            .decoded_subchunk_grids(decoded_chunk_grid)?
            .into_iter()
            .next()
            .unwrap_or(ChunkGridDecoded::None))
    }
}

/// Marker trait for array-to-bytes codecs that do not expose subchunk grids.
pub trait ArrayToBytesCodecNoSubchunkingTraits {}

impl<T> ArrayToBytesCodecSubchunkingTraits for T
where
    T: ArrayCodecTraits + ArrayToBytesCodecNoSubchunkingTraits + ?Sized,
{
    fn decoded_subchunk_grids(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<Vec<ChunkGridDecoded>, ChunkGridCreateError> {
        Ok(Vec::new())
    }
}

/// Traits for array to bytes codecs.
pub trait UnboundArrayToBytesCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits>;

    /// Bind this codec to a decoded data type, fill value, and codec-specific options.
    ///
    /// A codec may read its own options type from `codec_specific_options`, and must bind any codecs it contains with the same `codec_specific_options`.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the `data_type` or `fill_value` is not supported by this codec.
    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError>;
}

/// Runtime traits for an array-to-bytes codec bound to a data type and fill value.
pub trait ArrayToBytesCodecTraits: ArrayToBytesCodecSubchunkingTraits + core::fmt::Debug {
    /// Return a dynamic version of the bound codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits>;

    /// Returns the size of the encoded representation given a size of the decoded representation.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decoded representation is not supported by this codec.
    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError>;

    /// Encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with the decoded representation.
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with the decoded representation.
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Compact a chunk to remove any extraneous data.
    ///
    /// The default implementation returns the input `bytes` unchanged.
    ///
    /// Returns `Ok(None)` if no compaction was performed.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with the decoded representation.
    #[allow(unused_variables)]
    fn compact<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<CowBytes<'a>>, CodecError> {
        Ok(None)
    }

    /// Decode into a subset of a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It works for fixed length data types and optional data types.
    ///
    /// The input can contain encoded bytes or a deferred bytes-to-bytes decode.
    /// The default implementation resolves the input normally, preserving ownership, then decodes and copies into the target.
    /// An override may instead let an efficient producer write directly into a suitable target and transform those bytes there.
    ///
    /// The decoded representation shape and dimensionality does not need to match the output target, but the number of elements must match.
    /// Chunk elements are written to the subset of the output in C order.
    ///
    /// For optional data types, provide [`ArrayBytesDecodeIntoTarget::Optional`] with data and mask views.
    /// For non-optional data types, convert a fixed view to a target using `.into()`.
    ///
    /// On error, the output target may have been partially written.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in the decoded representation does not match the number of elements in the output target.
    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        let bytes = input.into_bytes(options, resources)?;
        let bytes = self.decode(bytes, shape, options, resources)?;
        decode_into_array_bytes_target(&bytes, output_target)
    }

    /// Plan the decoding of `request` from the encoded chunk `source`.
    ///
    /// Returns a task that decodes the requested elements of the chunk into the target of the request.
    /// If the target is [`TaskTarget::Value`](crate::plan::TaskTarget::Value), the value of the task is the decoded [`ArrayBytes`].
    /// If the source does not exist, the requested elements are the fill value.
    ///
    /// The default implementation reads and decodes the entire chunk ([`plan_decode_whole`](crate::plan::plan_decode_whole)).
    /// Codecs that can decode part of a chunk from part of its encoded bytes (e.g. `sharding_indexed`) override this.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the request is invalid.
    fn plan_decode<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        crate::plan::plan_decode_whole(self.into_dyn(), source, shape, request, options)
    }

    /// Plan the encoding of `bytes`, a chunk of `shape`.
    ///
    /// Returns a task whose value is the encoded chunk ([`TaskValue::Bytes`](crate::plan::TaskValue::Bytes)).
    ///
    /// The default implementation encodes in one compute step with [`encode`](Self::encode) ([`plan_encode_whole`](crate::plan::plan_encode_whole)).
    /// Codecs that can encode parts of a chunk independently (e.g. `sharding_indexed`) may override this.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the shape is invalid for the codec.
    fn plan_encode<'a>(
        self: Arc<Self>,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        crate::plan::plan_encode_whole(self.into_dyn(), bytes, shape, options)
    }

    /// Plan a partial encode: update the elements of `indexer` of the encoded chunk `key` of `shape` to `bytes`, reading
    /// and writing only the parts of the encoded chunk that change (e.g. the updated subchunks and the index of a shard).
    ///
    /// `state` is the state of the chunk after a previous partial encode (e.g. its decoded shard index), which saves
    /// reading it again. The value of the task is the state of the chunk after the update
    /// ([`partial_encode_value`](crate::plan::partial_encode_value)).
    ///
    /// Returns [`None`] if the codec does not partially encode chunks ([`plans_partial_encode`](Self::plans_partial_encode)
    /// is false), so a chunk is decoded, updated and encoded in full instead. This is the default.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `indexer` or `bytes` are invalid for the chunk.
    #[allow(clippy::too_many_arguments)]
    fn plan_partial_encode<'a>(
        self: Arc<Self>,
        key: SourceKey,
        shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        state: Option<PartialEncodeState>,
        options: &CodecOptions,
    ) -> Result<Option<Task<'a>>, CodecError> {
        let _ = (key, shape, indexer, bytes, state, options);
        Ok(None)
    }

    /// Returns true if [`plan_partial_encode`](Self::plan_partial_encode) plans partial encodes (rather than returning
    /// [`None`]).
    fn plans_partial_encode(&self) -> bool {
        false
    }

    /// Returns true if [`plan_encode`](Self::plan_encode) plans tasks of its own (e.g. a task per subchunk of a shard).
    ///
    /// Otherwise, planning the encoding of a chunk and running it is equivalent to [`encode`](Self::encode), which
    /// planners may call directly.
    fn plans_encode_subtasks(&self) -> bool {
        false
    }

    /// Returns true if [`plan_decode`](Self::plan_decode) of a whole chunk plans tasks of its own (e.g. a task per
    /// subchunk of a shard).
    ///
    /// Otherwise, planning the decoding of a whole chunk in memory into a target and running it is equivalent to
    /// [`decode_into`](Self::decode_into), which planners may call directly.
    fn plans_subtasks(&self) -> bool {
        false
    }

    /// Plan resolving the chunk-local subchunk grids of the encoded chunk `source` of `shape`, outermost first.
    ///
    /// The value of the task is the grids ([`local_subchunk_grids_value`](crate::plan::local_subchunk_grids_value)).
    /// A [`None`] entry is a level whose grid cannot be resolved for the chunk.
    ///
    /// Codecs whose [`decoded_subchunk_grids`](ArrayToBytesCodecSubchunkingTraits::decoded_subchunk_grids) include a
    /// [`ChunkGridDecoded::ChunkLocal`] level override this, reading the encoded chunk if their subchunk grid depends
    /// on it. The default implementation resolves no subchunk grids.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the chunk shape is invalid for the codec.
    fn plan_local_subchunk_grids<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        let _ = (source, shape, options);
        Ok(Task::compute(|_| {
            Ok(crate::plan::Outcome::Value(
                crate::plan::local_subchunk_grids_value(Vec::new()),
            ))
        }))
    }

    /// Prepare a decoder of the encoded chunk `source` that plans many requests.
    ///
    /// Preparation reads and decodes what every request needs once (e.g. a shard index).
    ///
    /// The default implementation reads and decodes the entire chunk once ([`prepare_decode_whole`](crate::plan::prepare_decode_whole)),
    /// consistent with the default [`plan_decode`](Self::plan_decode).
    /// Codecs that override [`plan_decode`](Self::plan_decode) to decode part of a chunk should override this too, e.g.
    /// with a [`SourceDecoder`](crate::plan::SourceDecoder) that plans each request with [`plan_decode`](Self::plan_decode).
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the chunk shape is invalid for the codec.
    fn prepare_decode(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<crate::plan::PrepareDecode, CodecError> {
        Ok(crate::plan::prepare_decode_whole(
            self.into_dyn(),
            source,
            shape,
            options,
        ))
    }
}

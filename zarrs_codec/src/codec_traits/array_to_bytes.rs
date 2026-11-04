use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_data_type::{DataType, FillValue};

use crate::codec_partial_default::ArrayToBytesCodecPartialDefault;
use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits,
    ArrayPartialEncoderTraits, BytesPartialDecoderTraits, BytesPartialEncoderTraits,
    BytesRepresentation, ChunkGridDecoded, ChunkGridDecodedRef, CodecCreateError, CodecError,
    CodecOptions, CodecSpecificOptions, CodecTraits, ComputeResources, CowBytes, Resources,
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

    /// Return the codecs that encode the subchunks created by this codec, outermost first.
    ///
    /// There is one codec for each level of the grids returned by
    /// [`decoded_subchunk_grids`](Self::decoded_subchunk_grids). The level zero codec decodes an
    /// encoded subchunk of this codec, the level one codec decodes an encoded subchunk of a
    /// subchunk, and so on. An empty vector indicates that the codec does not expose subchunks.
    ///
    /// Each codec decodes into the domain of the array-to-bytes codec that created its
    /// subchunks: this codec at level zero, and the array-to-bytes codec of the level `N - 1`
    /// codec at level `N`. Decoding an encoded subchunk at level `N` therefore yields array bytes
    /// that the array-to-array codecs of the codecs at levels `0..N` have yet to decode, as do
    /// any array-to-array codecs preceding this codec (e.g. in a codec chain). Its data type,
    /// shape, dimensionality, and element order can differ from those of the subchunk in the
    /// level `N` grid, which is mapped through all of those array-to-array codecs.
    ///
    /// The codecs retain their bound data type and fill value, but are not bound to a particular
    /// subchunk shape. A caller decoding an encoded subchunk must pass its concrete shape, which
    /// can vary between subchunks.
    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>>;

    /// Return the codecs that encode the subchunks at `level`, if any.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    ///
    /// This is a compatibility wrapper around [`subchunk_codecs`](Self::subchunk_codecs).
    fn subchunk_codecs_at_level(&self, level: usize) -> Option<Arc<dyn ArrayToBytesCodecTraits>> {
        self.subchunk_codecs().into_iter().nth(level)
    }

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

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        Vec::new()
    }
}
#[cfg(feature = "async")]
use crate::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};

/// Traits for array to bytes codecs.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait UnboundArrayToBytesCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits>;

    /// Bind this codec to a decoded data type, fill value, and codec-specific options.
    ///
    /// A codec may read its own options type from `codec_specific_options`, and must bind any codecs it contains with the same `codec_specific_options`.
    ///
    /// A `fill_value` of [`None`] means the fill value is undefined (e.g. within the `data_codecs` of an `optional` codec with a `null` fill value).
    /// Partial decoders of a codec bound with an undefined fill value must return an error for missing input, and partial encoders must not omit chunks.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the `data_type` or `fill_value` is not supported by this codec.
    /// Codecs whose encoded representation depends on the fill value (e.g. `sharding_indexed`) must return an error if it is undefined.
    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError>;
}

/// Runtime traits for an array-to-bytes codec bound to a data type and fill value.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
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

    /// Encode a chunk with the supplied compute resources.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with the decoded representation.
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError>;

    /// Decode a chunk with the supplied compute resources.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with the decoded representation.
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
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
        resources: &ComputeResources,
    ) -> Result<Option<CowBytes<'a>>, CodecError> {
        Ok(None)
    }

    /// Decode into a subset of a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It works for fixed length data types and optional data types.
    ///
    /// The decoded representation shape and dimensionality does not need to match the output target, but the number of elements must match.
    /// Chunk elements are written to the subset of the output in C order.
    ///
    /// For optional data types, provide an `ArrayBytesDecodeIntoTarget` with a `mask` set to `Some`.
    /// For non-optional data types, convert a fixed view to target using `.into()` or create with `mask: None`.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in the decoded representation does not match the number of elements in the output target.
    fn decode_into(
        &self,
        bytes: CowBytes<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        let bytes = self.decode(bytes, shape, options, resources)?;
        decode_into_array_bytes_target(&bytes, output_target)
    }

    /// Returns whether decoding valid encoded bytes produces identical fixed bytes in native
    /// in-memory order.
    ///
    /// The result may depend on the decoded representation. The default implementation is
    /// conservative and returns `false`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decoded representation is not supported by this codec.
    #[expect(unused_variables)]
    fn is_decode_passthrough(&self, shape: &[NonZeroU64]) -> Result<bool, CodecError> {
        Ok(false)
    }

    /// Initialise a partial decoder.
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }

    /// Initialise a partial encoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial decoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial encoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }
}

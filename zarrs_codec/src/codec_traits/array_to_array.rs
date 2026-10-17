use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::{ArrayIndices, ChunkGrid, ChunkGridCreateError};
use zarrs_data_type::{DataType, FillValue};
use zarrs_metadata::ChunkShape;

use crate::{
    ArrayBytes, ArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded,
    ChunkGridEncodedRef, CodecCreateError, CodecError, CodecOptions, CodecSpecificOptions,
    CodecTraits, Resources,
};

/// Subchunking traits for an array-to-array codec bound to a data type and fill value.
pub trait ArrayToArrayCodecSubchunkingTraits: ArrayCodecTraits {
    /// Map a chunk grid from the decoded representation to the encoded representation.
    ///
    /// [`ChunkGridEncoded::ChunkLocal`] indicates that the codec can
    /// propagate a concrete grid for each chunk at runtime, but no global
    /// encoded grid exists. A codec must not map `ChunkLocal` back to `Array`.
    ///
    /// # Errors
    /// Returns a [`ChunkGridCreateError`] if the decoded chunk grid is not supported by this codec.
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError>;

    /// Map a subchunk grid from the encoded representation to the decoded representation.
    ///
    /// Returns [`None`] if this codec cannot preserve or transform the subchunk grid.
    ///
    /// # Errors
    /// Returns a [`ChunkGridCreateError`] if the decoded chunk grid or encoded subchunk grid is not supported by this codec.
    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError>;

    /// Map the coordinates of an element of a decoded chunk of `decoded_shape` to its coordinates in the encoded chunk.
    ///
    /// This addresses encoded subchunks through the codec: a subchunk of the decoded chunk is located by an element it
    /// contains.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `decoded_coordinates` are out of bounds of the decoded chunk.
    fn encoded_coordinates(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_coordinates: &[u64],
    ) -> Result<ArrayIndices, CodecError>;
}

/// Marker trait for array-to-array codecs with identity subchunking mappings.
///
/// Implement this trait only when the codec preserves element ordering and chunk shape.
pub trait ArrayToArrayCodecSubchunkingIdentityTraits {}

impl<T> ArrayToArrayCodecSubchunkingTraits for T
where
    T: ArrayCodecTraits + ArrayToArrayCodecSubchunkingIdentityTraits + ?Sized,
{
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        Ok(decoded_chunk_grid.into())
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        Ok(encoded_subchunk_grid.into())
    }

    fn encoded_coordinates(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_coordinates: &[u64],
    ) -> Result<ArrayIndices, CodecError> {
        let _ = decoded_shape;
        Ok(decoded_coordinates.to_vec())
    }
}

/// Traits for array to array codecs.
pub trait UnboundArrayToArrayCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits>;

    /// Bind this codec to a decoded data type, fill value, and codec-specific options.
    ///
    /// Binding eagerly validates and derives the encoded context.
    /// A codec may read its own options type from `codec_specific_options`, and must bind any codecs it contains with the same `codec_specific_options`.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the `data_type` or `fill_value` is not supported by this codec.
    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError>;
}

/// Runtime traits for an array-to-array codec bound to a data type and fill value.
pub trait ArrayToArrayCodecTraits: ArrayToArrayCodecSubchunkingTraits + core::fmt::Debug {
    /// Return a dynamic version of the bound codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits>;

    /// Return the encoded data type.
    fn encoded_data_type(&self) -> &DataType;

    /// Return the encoded fill value.
    fn encoded_fill_value(&self) -> &FillValue;

    /// Returns the shape of the encoded chunk for a given decoded chunk shape.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the `decoded_shape` is not supported by this codec.
    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec())
    }

    /// Map a partial decode granularity from the encoded representation to the decoded representation.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decoded shape or encoded granularity is not supported by this codec.
    fn partial_decode_granularity(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_granularity: &[NonZeroU64],
    ) -> Result<ChunkShape, CodecError> {
        let encoded_shape = self.encoded_shape(decoded_shape)?;
        if encoded_shape == decoded_shape {
            Ok(encoded_granularity.to_vec())
        } else {
            Ok(decoded_shape.to_vec())
        }
    }

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
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with the decoded representation.
    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Plan a request for decoded elements of a chunk of `shape` through this codec.
    ///
    /// The default implementation treats the codec as elementwise ([`plan_decode_request_elementwise`](crate::plan::plan_decode_request_elementwise)).
    /// Codecs that rearrange elements (e.g. `transpose`) or have an identity decode override this.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the request is invalid.
    fn plan_decode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        request: crate::plan::DecodeRequest<'a>,
    ) -> Result<crate::plan::ArrayToArrayDecodePlan<'a>, CodecError> {
        crate::plan::plan_decode_request_elementwise(shape, request)
    }

    /// Map a partial encode of the elements `indexer` of a chunk of `shape` to `bytes` through this codec.
    ///
    /// Returns the indexer of the encoded elements to update in the encoded chunk and their encoded bytes, or [`None`]
    /// if the codec does not map partial encodes ([`plans_partial_encode`](Self::plans_partial_encode) is false), so a
    /// chunk is decoded, updated and encoded in full instead. This is the default.
    ///
    /// Elementwise codecs can implement this with [`partial_encode_request_elementwise`](crate::plan::partial_encode_request_elementwise).
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `indexer` or `bytes` are invalid for the chunk, or the codec fails.
    fn partial_encode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        indexer: crate::plan::PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<(crate::plan::PlanIndexer<'a>, ArrayBytes<'a>)>, CodecError> {
        let _ = (shape, indexer, bytes, options, resources);
        Ok(None)
    }

    /// Returns true if [`partial_encode_request`](Self::partial_encode_request) maps partial encodes (rather than
    /// returning [`None`]).
    fn plans_partial_encode(&self) -> bool {
        false
    }

    /// Map a chunk-local subchunk grid of the encoded chunk to the decoded chunk of `shape`.
    ///
    /// Returns [`None`] if the grid cannot be represented in the decoded chunk.
    /// The default implementation returns the grid unchanged, which is valid for codecs that preserve element ordering
    /// and chunk shape.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the grid is incompatible with the encoded chunk.
    fn decoded_local_subchunk_grid(
        &self,
        shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        let _ = shape;
        Ok(Some(encoded_subchunk_grid.clone()))
    }
}

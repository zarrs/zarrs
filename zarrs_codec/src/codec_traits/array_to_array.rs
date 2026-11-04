use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::{ArrayIndices, ChunkGrid, ChunkGridCreateError};
use zarrs_data_type::{DataType, FillValue};
use zarrs_metadata::ChunkShape;

use crate::codec_partial_default::ArrayToArrayCodecPartialDefault;
use crate::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded, ChunkGridEncodedRef, CodecCreateError,
    CodecError, CodecOptions, CodecSpecificOptions, CodecTraits,
};
#[cfg(feature = "async")]
use crate::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};

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

    /// Map a chunk-local subchunk grid of the encoded chunk to the decoded chunk of `decoded_shape`.
    ///
    /// This is the chunk-local counterpart of [`decoded_subchunk_grid`](Self::decoded_subchunk_grid), used to resolve
    /// the subchunk grids of an individual chunk (e.g. by a partial decoder).
    ///
    /// Returns [`None`] if the grid cannot be represented in the decoded chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the grid is incompatible with the encoded chunk.
    fn decoded_local_subchunk_grid(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError>;

    /// Map the coordinates of an element of a decoded chunk of `decoded_shape` to its coordinates in the encoded chunk.
    ///
    /// This addresses encoded subchunks through the codec: a subchunk of the decoded chunk is located by an element it
    /// contains.
    ///
    /// Returns [`None`] if the codec does not map elements to elements of the encoded chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `decoded_coordinates` are out of bounds of the decoded chunk.
    fn encoded_coordinates(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_coordinates: &[u64],
    ) -> Result<Option<ArrayIndices>, CodecError>;
}

/// Check that `coordinates` are in bounds of a chunk of `shape`.
///
/// # Errors
/// Returns a [`CodecError`] if `coordinates` have a different dimensionality to `shape` or are out of bounds.
pub fn validate_chunk_coordinates(
    shape: &[NonZeroU64],
    coordinates: &[u64],
) -> Result<(), CodecError> {
    if coordinates.len() == shape.len()
        && std::iter::zip(coordinates, shape).all(|(&coordinate, length)| coordinate < length.get())
    {
        Ok(())
    } else {
        Err(CodecError::Other(format!(
            "coordinates {coordinates:?} are out of bounds of a chunk with shape {shape:?}"
        )))
    }
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
    ) -> Result<Option<ArrayIndices>, CodecError> {
        validate_chunk_coordinates(decoded_shape, decoded_coordinates)?;
        Ok(Some(decoded_coordinates.into()))
    }

    fn decoded_local_subchunk_grid(
        &self,
        _decoded_shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        Ok(Some(encoded_subchunk_grid.clone()))
    }
}

/// Traits for array to array codecs.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait UnboundArrayToArrayCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits>;

    /// Bind this codec to a decoded data type, fill value, and codec-specific options.
    ///
    /// Binding eagerly validates and derives the encoded context.
    /// A codec may read its own options type from `codec_specific_options`, and must bind any codecs it contains with the same `codec_specific_options`.
    ///
    /// A `fill_value` of [`None`] means the fill value is undefined (e.g. within the `data_codecs` of an `optional` codec with a `null` fill value).
    /// A codec must then bind the codecs that follow it with an undefined fill value, rather than one derived from a placeholder.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the `data_type` or `fill_value` is not supported by this codec.
    /// Codecs whose encoded representation depends on the fill value must return an error if it is undefined.
    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError>;
}

/// Runtime traits for an array-to-array codec bound to a data type and fill value.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait ArrayToArrayCodecTraits: ArrayToArrayCodecSubchunkingTraits + core::fmt::Debug {
    /// Return a dynamic version of the bound codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits>;

    /// Return the encoded data type.
    fn encoded_data_type(&self) -> &DataType;

    /// Return the encoded fill value, or [`None`] if it is undefined.
    fn encoded_fill_value(&self) -> Option<&FillValue>;

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
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Initialise a partial decoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
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
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }
}

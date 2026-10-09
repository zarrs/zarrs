use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_data_type::{DataType, FillValue};
use zarrs_metadata::ChunkShape;

use crate::codec_partial_default::ArrayToArrayCodecPartialDefault;
use crate::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded, ChunkGridEncodedRef, CodecCreateError,
    CodecError, CodecOptions, CodecSpecificOptions, CodecTraits, Resources,
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

    /// Initialise a partial decoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().clone(),
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
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().clone(),
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
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().clone(),
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
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        _ = resources;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().clone(),
            self.into_dyn(),
        )))
    }
}

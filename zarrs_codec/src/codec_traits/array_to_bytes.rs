use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_data_type::{DataType, FillValue};
use zarrs_metadata::ChunkShape;

use crate::codec_partial_default::ArrayToBytesCodecPartialDefault;
use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesRaw, ArrayCodecTraits,
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, BytesPartialDecoderTraits,
    BytesPartialEncoderTraits, BytesRepresentation, CodecCreateError, CodecError, CodecOptions,
    CodecSpecificOptions, CodecTraits, decode_into_array_bytes_target,
};
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

    /// Return a version of this codec reconfigured with the provided codec-specific options.
    ///
    /// The default implementation returns the codec unchanged.
    /// Override this to read your codec's options type from [`CodecSpecificOptions`].
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the codec cannot be reconfigured.
    #[expect(unused_variables)]
    fn with_codec_specific_options(
        self: Arc<Self>,
        opts: &CodecSpecificOptions,
    ) -> Result<Arc<dyn UnboundArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(self.into_dyn())
    }

    /// Bind this codec to a decoded data type and fill value.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the `data_type` or `fill_value` is not supported by this codec.
    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError>;
}

/// Runtime traits for an array-to-bytes codec bound to a data type and fill value.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait ArrayToBytesCodecTraits: ArrayCodecTraits + core::fmt::Debug {
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

    /// Return the partial decode granularity.
    ///
    /// This represents the shape of the smallest subset of a chunk that can be efficiently decoded if the chunk were subdivided into a regular grid.
    /// For most codecs, this is just the shape of the chunk.
    /// It is the shape of the sub chunks (inner chunks) for the sharding codec.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decoded shape is not supported by this codec.
    fn partial_decode_granularity(
        &self,
        decoded_shape: &[NonZeroU64],
    ) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec())
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
    ) -> Result<ArrayBytesRaw<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with the decoded representation.
    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
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
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
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
        bytes: ArrayBytesRaw<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let bytes = self.decode(bytes, shape, options)?;
        decode_into_array_bytes_target(&bytes, output_target)
    }

    /// Initialise a partial decoder.
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
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
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().clone(),
            self.into_dyn(),
        )))
    }
}

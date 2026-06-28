use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{
    BitroundCodecConfiguration, BitroundCodecConfigurationV1, BitroundDataTypeExt,
    bitround_codec_partial, round_bytes,
};
use crate::array::{ChunkGrid, ChunkShape, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecTraits, PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    SubchunkGrid, UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::Configuration;

/// A `bitround` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct BitroundCodec {
    keepbits: u32,
}

/// A `bitround` codec implementation.
#[derive(Clone, Debug)]
struct BitroundCodecBound {
    keepbits: u32,
    data_type: DataType,
    fill_value: FillValue,
    encoded_fill_value: FillValue,
}

impl BitroundCodec {
    /// Create a new `bitround` codec.
    ///
    /// `keepbits` is the number of bits to round to in the floating point mantissa.
    #[must_use]
    pub const fn new(keepbits: u32) -> Self {
        Self { keepbits }
    }

    /// Create a new `bitround` codec from a configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &BitroundCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            BitroundCodecConfiguration::V1(configuration) => Ok(Self {
                keepbits: configuration.keepbits,
            }),
            _ => Err(PluginCreateError::Other(
                "this bitround codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for BitroundCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        if options.codec_store_metadata_if_encode_only() {
            let configuration = BitroundCodecConfiguration::V1(BitroundCodecConfigurationV1 {
                keepbits: self.keepbits,
            });
            Some(configuration.into())
        } else {
            None
        }
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: true,
        }
    }
}

fn encode_fill_value(
    fill_value: &FillValue,
    data_type: &DataType,
    keepbits: u32,
) -> Result<FillValue, CodecCreateError> {
    let mut fill_value_bytes = ArrayBytes::new_fill_value(data_type, 1, fill_value)?
        .into_fixed()
        .map_err(CodecCreateError::other)?
        .into_owned();
    round_bytes(&mut fill_value_bytes, data_type, keepbits).map_err(CodecCreateError::other)?;
    Ok(FillValue::new(fill_value_bytes))
}

impl UnboundArrayToArrayCodecTraits for BitroundCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        data_type.codec_bitround()?;
        let encoded_fill_value = encode_fill_value(&fill_value, &data_type, self.keepbits)?;
        Ok(Arc::new(BitroundCodecBound {
            keepbits: self.keepbits,
            data_type,
            fill_value,
            encoded_fill_value,
        }))
    }
}

impl ArrayCodecTraits for BitroundCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Return the decoded data type bound to this codec.
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Return the decoded fill value bound to this codec.
    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[u64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: bitround is well suited to multithread, when is it optimal to kick in?
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for BitroundCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.encoded_fill_value
    }

    fn encoded_shape(&self, decoded_shape: &[u64]) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec())
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: &ChunkGrid,
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<SubchunkGrid, ChunkGridCreateError> {
        Ok(SubchunkGrid::Array(encoded_subchunk_grid.clone()))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let mut bytes = bytes.into_fixed()?;
        round_bytes(bytes.to_mut(), &self.data_type, self.keepbits)?;
        Ok(ArrayBytes::from(bytes))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_handle,
            &self.data_type,
            self.keepbits,
        )?))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_output_handle,
            &self.data_type,
            self.keepbits,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_handle,
            &self.data_type,
            self.keepbits,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_output_handle,
            &self.data_type,
            self.keepbits,
        )?))
    }
}

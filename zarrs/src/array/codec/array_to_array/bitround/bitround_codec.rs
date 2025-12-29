use std::sync::Arc;

use zarrs_plugin::PluginCreateError;

use super::{
    BitroundCodecConfiguration, BitroundCodecConfigurationV1, bitround_codec_partial, round_bytes,
};
#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use crate::array::{
    DataType, FillValue,
    codec::{
        ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
        ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    },
};
use crate::metadata::Configuration;
use std::num::NonZeroU64;
use zarrs_plugin::ExtensionIdentifier;

/// A `bitround` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct BitroundCodec {
    keepbits: u32,
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
    fn identifier(&self) -> &'static str {
        Self::IDENTIFIER
    }

    fn configuration(&self, _name: &str, options: &CodecMetadataOptions) -> Option<Configuration> {
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

impl ArrayCodecTraits for BitroundCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
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
impl ArrayToArrayCodecTraits for BitroundCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let mut bytes = bytes.into_fixed()?;
        round_bytes(bytes.to_mut(), data_type, self.keepbits)?;
        Ok(ArrayBytes::from(bytes))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_handle,
            data_type,
            self.keepbits,
        )?))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_output_handle,
            data_type,
            self.keepbits,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_handle,
            data_type,
            self.keepbits,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bitround_codec_partial::BitroundCodecPartial::new(
            input_output_handle,
            data_type,
            self.keepbits,
        )?))
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        // Use codec_bitround() from DataTypeExtension trait to check support
        if decoded_data_type.codec_bitround().is_some() {
            Ok(decoded_data_type.clone())
        } else {
            Err(CodecError::UnsupportedDataType(
                decoded_data_type.clone(),
                Self::IDENTIFIER.to_string(),
            ))
        }
    }
}

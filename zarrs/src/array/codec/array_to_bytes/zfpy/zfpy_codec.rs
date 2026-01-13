use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::super::zfp::ZfpCodec;
use crate::array::{BytesRepresentation, DataType, FillValue};
use crate::metadata::Configuration;
use crate::metadata_ext::codec::zfp::ZfpMode;
use crate::metadata_ext::codec::zfpy::{
    ZfpyCodecConfiguration, ZfpyCodecConfigurationMode, ZfpyCodecConfigurationNumcodecs,
};
use zarrs_codec::{
    ArrayBytes, ArrayBytesRaw, ArrayCodecTraits, ArrayToBytesCodecTraits, CodecError,
    CodecMetadataOptions, CodecOptions, CodecTraits, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency,
};

use crate::array::codec::{ArrayPartialDecoderTraits, BytesPartialDecoderTraits};
#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

/// A `zfpy` codec implementation.
///
/// This codec wraps [`ZfpCodec`] with `numcodecs.zfpy` compatibility,
/// including the redundant ZFP header that `zarr-python` expects.
#[derive(Clone, Copy, Debug)]
pub struct ZfpyCodec {
    inner: ZfpCodec,
}

impl ZfpyCodec {
    /// Create a new `zfpy` codec in fixed rate mode.
    #[must_use]
    pub fn new_fixed_rate(rate: f64) -> Self {
        Self {
            inner: ZfpCodec::new_fixed_rate(rate).with_write_header(true),
        }
    }

    /// Create a new `zfpy` codec in fixed precision mode.
    #[must_use]
    pub fn new_fixed_precision(precision: u32) -> Self {
        Self {
            inner: ZfpCodec::new_fixed_precision(precision).with_write_header(true),
        }
    }

    /// Create a new `zfpy` codec in fixed accuracy mode.
    #[must_use]
    pub fn new_fixed_accuracy(tolerance: f64) -> Self {
        Self {
            inner: ZfpCodec::new_fixed_accuracy(tolerance).with_write_header(true),
        }
    }

    /// Create a new `zfpy` codec in reversible mode.
    #[must_use]
    pub fn new_reversible() -> Self {
        Self {
            inner: ZfpCodec::new_reversible().with_write_header(true),
        }
    }

    /// Create a new `zfpy` codec from a configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &ZfpyCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ZfpyCodecConfiguration::Numcodecs(configuration) => match configuration.mode {
                ZfpyCodecConfigurationMode::FixedRate { rate } => Ok(Self::new_fixed_rate(rate)),
                ZfpyCodecConfigurationMode::FixedPrecision { precision } => {
                    Ok(Self::new_fixed_precision(precision))
                }
                ZfpyCodecConfigurationMode::FixedAccuracy { tolerance } => {
                    Ok(Self::new_fixed_accuracy(tolerance))
                }
                ZfpyCodecConfigurationMode::Reversible => Ok(Self::new_reversible()),
            },
            _ => Err(PluginCreateError::Other(
                "this zfpy codec configuration variant is unsupported".to_string(),
            ))?,
        }
    }
}

impl CodecTraits for ZfpyCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let mode = match self.inner.mode() {
            ZfpMode::FixedRate { rate } => ZfpyCodecConfigurationMode::FixedRate { rate },
            ZfpMode::FixedPrecision { precision } => {
                ZfpyCodecConfigurationMode::FixedPrecision { precision }
            }
            ZfpMode::FixedAccuracy { tolerance } => {
                ZfpyCodecConfigurationMode::FixedAccuracy { tolerance }
            }
            ZfpMode::Reversible => ZfpyCodecConfigurationMode::Reversible,
            ZfpMode::Expert { .. } => return None, // Expert mode not supported in zfpy
        };
        Some(ZfpyCodecConfiguration::Numcodecs(ZfpyCodecConfigurationNumcodecs { mode }).into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        self.inner.partial_decoder_capability()
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        self.inner.partial_encoder_capability()
    }
}

impl ArrayCodecTraits for ZfpyCodec {
    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        self.inner.recommended_concurrency(shape, data_type)
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for ZfpyCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        self.inner
            .encode(bytes, shape, data_type, fill_value, options)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.inner
            .decode(bytes, shape, data_type, fill_value, options)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Arc::new(self.inner).partial_decoder(input_handle, shape, data_type, fill_value, options)
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Arc::new(self.inner)
            .async_partial_decoder(input_handle, shape, data_type, fill_value, options)
            .await
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<BytesRepresentation, CodecError> {
        self.inner
            .encoded_representation(shape, data_type, fill_value)
    }
}

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::super::zfp::ZfpCodec;
use crate::array::{BytesRepresentation, DataType, FillValue};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayCodecTraits,
    ArrayToBytesCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, CowBytes, Resources, UnboundArrayToBytesCodecTraits,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::zfp::ZfpMode;
use zarrs_metadata_ext::codec::zfpy::{
    ZfpyCodecConfiguration, ZfpyCodecConfigurationMode, ZfpyCodecConfigurationNumcodecs,
};

/// A `zfpy` codec implementation.
///
/// This codec wraps [`ZfpCodec`] with `numcodecs.zfpy` compatibility,
/// including the redundant ZFP header that `zarr-python` expects.
#[derive(Clone, Copy, Debug)]
pub struct ZfpyCodec {
    inner: ZfpCodec,
}

#[derive(Clone, Debug)]
pub(crate) struct ZfpyCodecBound {
    inner: Arc<dyn ArrayToBytesCodecTraits>,
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
}

impl UnboundArrayToBytesCodecTraits for ZfpyCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        let inner_bound =
            Arc::new(self.inner).with_context(data_type, fill_value, codec_specific_options)?;
        Ok(Arc::new(ZfpyCodecBound { inner: inner_bound }))
    }
}

impl ArrayCodecTraits for ZfpyCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        self.inner.data_type()
    }

    fn fill_value(&self) -> &FillValue {
        self.inner.fill_value()
    }
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for ZfpyCodecBound {}

impl ArrayToBytesCodecTraits for ZfpyCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.inner.encode(bytes, shape, options, resources)
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.inner.decode(bytes, shape, options, resources)
    }

    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        self.inner
            .decode_into(input, shape, output_target, options, resources)
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        self.inner.encoded_representation(shape)
    }
}

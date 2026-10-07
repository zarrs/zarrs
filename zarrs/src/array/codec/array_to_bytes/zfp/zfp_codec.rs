use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};
use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpHeaderMask, ZfpScalarType};

use super::{
    ZfpCodecConfiguration, ZfpCodecConfigurationV1, ZfpDataTypeExt, ZfpEncoding,
    promote_before_zfp_encoding, zfp_config, zfp_decode, zfp_decode_into, zfp_dims,
    zfp_native_type_to_scalar_type,
};
use crate::array::{BytesRepresentation, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayCodecTraits,
    ArrayToBytesCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, CowBytes, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency, UnboundArrayToBytesCodecTraits,
    decode_into_array_bytes_target,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::zfp::ZfpMode;

/// A `zfp` codec implementation.
#[derive(Clone, Copy, Debug)]
pub struct ZfpCodec {
    mode: ZfpMode,
    write_header: bool,
}

/// A `zfp` codec implementation.
#[derive(Clone, Debug)]
pub(crate) struct ZfpCodecBound {
    data_type: DataType,
    fill_value: FillValue,
    encoding: ZfpEncoding,
    scalar_type: ZfpScalarType,
    config: ZfpConfig,
    write_header: bool,
}

impl ZfpCodec {
    /// Create a new `zfp` codec in expert mode.
    #[must_use]
    pub const fn new_expert(minbits: u32, maxbits: u32, maxprec: u32, minexp: i32) -> Self {
        Self {
            mode: ZfpMode::Expert {
                minbits,
                maxbits,
                maxprec,
                minexp,
            },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in fixed rate mode.
    #[must_use]
    pub const fn new_fixed_rate(rate: f64) -> Self {
        Self {
            mode: ZfpMode::FixedRate { rate },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in fixed precision mode.
    #[must_use]
    pub const fn new_fixed_precision(precision: u32) -> Self {
        Self {
            mode: ZfpMode::FixedPrecision { precision },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in fixed accuracy mode.
    #[must_use]
    pub const fn new_fixed_accuracy(tolerance: f64) -> Self {
        Self {
            mode: ZfpMode::FixedAccuracy { tolerance },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in reversible mode.
    #[must_use]
    pub const fn new_reversible() -> Self {
        Self {
            mode: ZfpMode::Reversible,
            write_header: false,
        }
    }

    /// Returns the zfp mode.
    #[must_use]
    pub(crate) const fn mode(&self) -> ZfpMode {
        self.mode
    }

    /// Set whether to write the zfp header.
    #[must_use]
    pub(crate) const fn with_write_header(mut self, write_header: bool) -> Self {
        self.write_header = write_header;
        self
    }

    /// Create a new `zfp` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &ZfpCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        let configuration = match configuration {
            ZfpCodecConfiguration::V1(configuration) => configuration.clone(),
            _ => Err(PluginCreateError::Other(
                "this zfp codec configuration variant is unsupported".to_string(),
            ))?,
        };

        Ok(match configuration.mode {
            ZfpMode::Expert {
                minbits,
                maxbits,
                maxprec,
                minexp,
            } => Self::new_expert(minbits, maxbits, maxprec, minexp),
            ZfpMode::FixedRate { rate } => Self::new_fixed_rate(rate),
            ZfpMode::FixedPrecision { precision } => Self::new_fixed_precision(precision),
            ZfpMode::FixedAccuracy { tolerance } => Self::new_fixed_accuracy(tolerance),
            ZfpMode::Reversible => Self::new_reversible(),
        })
    }
}

impl CodecTraits for ZfpCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(ZfpCodecConfiguration::V1(ZfpCodecConfigurationV1 { mode: self.mode }).into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToBytesCodecTraits for ZfpCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        let encoding = data_type.codec_zfp()?.zfp_encoding();
        let scalar_type = zfp_native_type_to_scalar_type(encoding.native_type());
        let config = zfp_config(&self.mode, scalar_type)?;
        Ok(Arc::new(ZfpCodecBound {
            data_type,
            fill_value,
            encoding,
            scalar_type,
            config,
            write_header: self.write_header,
        }))
    }
}

impl ArrayCodecTraits for ZfpCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: zfp supports multi thread, when is it optimal to kick in?
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for ZfpCodecBound {}

impl ArrayToBytesCodecTraits for ZfpCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        let bytes = bytes.into_fixed()?;
        let bytes_promoted = promote_before_zfp_encoding(&bytes, self.encoding);
        let dims = zfp_dims(shape).ok_or_else(|| CodecError::from("failed to create zfp field"))?;
        let field = bytes_promoted
            .field(dims)
            .ok_or_else(|| CodecError::from("failed to create zfp field"))?;

        let bufsize = self
            .config
            .maximum_size(self.scalar_type, dims)
            .ok_or_else(|| CodecError::from("failed to calculate zfp maximum size"))?;
        let mut bitstream = ZfpBitStream::new(bufsize)
            .map_err(|err| CodecError::Other(format!("failed to allocate zfp bitstream: {err}")))?;
        if self.write_header {
            bitstream
                .write_header(&self.config, &field.metadata(), ZfpHeaderMask::FULL)
                .map_err(|err| CodecError::Other(format!("failed to write zfp header: {err}")))?;
        }
        bitstream
            .compress(&self.config, &field)
            .map_err(|err| CodecError::Other(format!("zfp compression failed: {err}")))?;
        let bytes = bitstream
            .into_bytes()
            .map_err(|err| CodecError::Other(format!("failed to copy zfp bitstream: {err}")))?;
        Ok(CowBytes::from(bytes))
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        zfp_decode(
            &self.config,
            self.write_header,
            &bytes,
            shape,
            self.encoding,
        )
        .map(ArrayBytes::from)
    }

    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        mut output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let bytes = input.into_bytes(options)?;
        // Decode directly into an output that is one contiguous region
        if let ArrayBytesDecodeIntoTarget::Fixed(output) = &mut output_target
            && let Some(output) = output.as_mut_slice()
            && zfp_decode_into(
                &self.config,
                self.write_header,
                &bytes,
                shape,
                self.encoding,
                output,
            )?
        {
            return Ok(());
        }

        let bytes = self.decode(bytes, shape, options)?;
        decode_into_array_bytes_target(&bytes, output_target)
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        let dims = zfp_dims(shape).ok_or_else(|| CodecError::from("unsupported zfp shape"))?;
        let bufsize = self
            .config
            .maximum_size(self.scalar_type, dims)
            .ok_or_else(|| CodecError::from("failed to calculate zfp maximum size"))?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(BytesRepresentation::BoundedSize(bufsize as u64))
    }
}

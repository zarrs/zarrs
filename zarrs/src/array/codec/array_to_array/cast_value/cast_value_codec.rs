use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecTraits, PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_data_type::codec_traits::cast_value::{
    CastValueDataTypeExt, CastValueDataTypeTraits, CastValueKernel,
    CastValueOutOfRangeMode as DataTypeCastValueOutOfRangeMode,
    CastValueRoundingMode as DataTypeCastValueRoundingMode, select_cast_kernel,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::cast_value::{
    CastValueCodecConfiguration, CastValueCodecConfigurationV1, CastValueOutOfRangeMode,
    CastValueRoundingMode,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};

use crate::array::{DataType, FillValue};

pub(super) type ScalarMap = HashMap<Vec<u8>, Vec<u8>>;

/// A `cast_value` codec implementation.
#[derive(Clone, Debug)]
pub struct CastValueUnbound {
    target_data_type: DataType,
    configuration: CastValueCodecConfigurationV1,
}

/// A `cast_value` codec implementation.
pub type CastValueCodec = CastValueUnbound;

/// A `cast_value` codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
struct CastValueCodecBound {
    data_type: DataType,
    fill_value: FillValue,
    encoded_data_type: DataType,
    encoded_fill_value: FillValue,
    encode_scalar_map: Option<ScalarMap>,
    decode_scalar_map: Option<ScalarMap>,
    rounding: DataTypeCastValueRoundingMode,
    out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
    encode_kernel: Option<CastValueKernel>,
    decode_kernel: Option<CastValueKernel>,
}

impl CastValueUnbound {
    /// Create a new `cast_value` codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the configuration is unsupported.
    pub fn new_with_configuration(
        configuration: &CastValueCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            CastValueCodecConfiguration::V1(configuration) => {
                let target_data_type = DataType::from_metadata(&configuration.data_type)?;
                let target = target_data_type
                    .codec_castvalue()
                    .map_err(|err| PluginCreateError::Other(err.to_string()))?;
                validate_wrap_mode(&target_data_type, target, configuration.out_of_range)
                    .map_err(|err| PluginCreateError::Other(err.to_string()))?;
                Ok(Self {
                    target_data_type,
                    configuration: configuration.clone(),
                })
            }
            _ => Err(PluginCreateError::Other(
                "this cast_value codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for CastValueUnbound {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(CastValueCodecConfiguration::V1(self.configuration.clone()).into())
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

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToArrayCodecTraits for CastValueUnbound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        let source = data_type.codec_castvalue()?;
        let target = self.target_data_type.codec_castvalue()?;
        validate_wrap_mode(
            &self.target_data_type,
            target,
            self.configuration.out_of_range,
        )
        .map_err(CodecCreateError::other)?;
        let encode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.encode.as_ref())
            .map(|map| build_scalar_map(map, &data_type, &self.target_data_type))
            .transpose()
            .map_err(CodecCreateError::other)?;
        let decode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.decode.as_ref())
            .map(|map| build_scalar_map(map, &self.target_data_type, &data_type))
            .transpose()
            .map_err(CodecCreateError::other)?;

        let rounding = map_rounding(self.configuration.rounding);
        let out_of_range = map_out_of_range(self.configuration.out_of_range);
        // Select monomorphised bulk kernels when both data types expose a
        // numeric representation; otherwise the generic scalar path is used.
        let source_repr = source.cast_value_repr();
        let target_repr = target.cast_value_repr();
        let encode_kernel = source_repr.zip(target_repr).and_then(|(source, target)| {
            select_cast_kernel(source, target, rounding, out_of_range)
        });
        let decode_kernel = source_repr.zip(target_repr).and_then(|(source, target)| {
            select_cast_kernel(target, source, rounding, out_of_range)
        });
        let encoded = cast_element(
            fill_value.as_ne_bytes(),
            source,
            target,
            encode_scalar_map.as_ref(),
            encode_kernel.as_ref(),
            rounding,
            out_of_range,
        )
        .map_err(CodecCreateError::other)?;
        let decoded = cast_element(
            &encoded,
            target,
            source,
            decode_scalar_map.as_ref(),
            decode_kernel.as_ref(),
            rounding,
            out_of_range,
        )
        .map_err(CodecCreateError::other)?;
        if decoded != fill_value.as_ne_bytes() {
            return Err(CodecCreateError::Other(
                "cast_value fill value does not survive a round-trip cast".to_string(),
            ));
        }
        Ok(Arc::new(CastValueCodecBound {
            data_type,
            fill_value,
            encoded_data_type: self.target_data_type.clone(),
            encoded_fill_value: FillValue::new(encoded),
            encode_scalar_map,
            decode_scalar_map,
            rounding,
            out_of_range,
            encode_kernel,
            decode_kernel,
        }))
    }
}

impl ArrayCodecTraits for CastValueCodecBound {
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
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for CastValueCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.encoded_data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.encoded_fill_value
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let source = self.data_type.codec_castvalue()?;
        let target = self.encoded_data_type.codec_castvalue()?;
        Ok(ArrayBytes::from(cast_bytes(
            &bytes.into_fixed()?,
            &CastBytesContext {
                source_data_type: &self.data_type,
                target_data_type: &self.encoded_data_type,
                source,
                target,
                scalar_map: self.encode_scalar_map.as_ref(),
                rounding: self.rounding,
                out_of_range: self.out_of_range,
                kernel: self.encode_kernel,
            },
        )?))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let source = self.encoded_data_type.codec_castvalue()?;
        let target = self.data_type.codec_castvalue()?;
        Ok(ArrayBytes::from(cast_bytes(
            &bytes.into_fixed()?,
            &CastBytesContext {
                source_data_type: &self.encoded_data_type,
                target_data_type: &self.data_type,
                source,
                target,
                scalar_map: self.decode_scalar_map.as_ref(),
                rounding: self.rounding,
                out_of_range: self.out_of_range,
                kernel: self.decode_kernel,
            },
        )?))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_handle.data_type())?;
        Ok(Arc::new(self.partial(input_handle)?))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_output_handle.data_type())?;
        Ok(Arc::new(self.partial(input_output_handle)?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_handle.data_type())?;
        Ok(Arc::new(self.partial(input_handle)?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_output_handle.data_type())?;
        Ok(Arc::new(self.partial(input_output_handle)?))
    }
}

impl CastValueCodecBound {
    fn validate_partial_handle_data_type(&self, data_type: &DataType) -> Result<(), CodecError> {
        if data_type != &self.encoded_data_type {
            return Err(CodecError::Other(format!(
                "cast_value partial handle data type {data_type} does not match encoded data type {}",
                self.encoded_data_type
            )));
        }
        Ok(())
    }

    fn partial<T: ?Sized>(
        &self,
        input_output_handle: Arc<T>,
    ) -> Result<super::cast_value_codec_partial::CastValueCodecPartial<T>, CodecError> {
        super::cast_value_codec_partial::CastValueCodecPartial::new(
            input_output_handle,
            &self.data_type,
            &self.encoded_data_type,
            self.encode_scalar_map.clone(),
            self.decode_scalar_map.clone(),
            self.rounding,
            self.out_of_range,
            self.encode_kernel,
            self.decode_kernel,
        )
    }
}

fn build_scalar_map(
    map: &[[zarrs_metadata::FillValueMetadata; 2]],
    source_data_type: &DataType,
    target_data_type: &DataType,
) -> Result<ScalarMap, CodecError> {
    let mut scalar_map = HashMap::with_capacity(map.len());
    for [key, value] in map {
        let key = source_data_type
            .fill_value_v3(key)
            .map_err(|err| CodecError::Other(err.to_string()))?
            .as_ne_bytes()
            .to_vec();
        let value = target_data_type
            .fill_value_v3(value)
            .map_err(|err| CodecError::Other(err.to_string()))?
            .as_ne_bytes()
            .to_vec();
        scalar_map.entry(key).or_insert(value);
    }
    Ok(scalar_map)
}

pub(super) struct CastBytesContext<'a> {
    pub(super) source_data_type: &'a DataType,
    pub(super) target_data_type: &'a DataType,
    pub(super) source: &'a dyn CastValueDataTypeTraits,
    pub(super) target: &'a dyn CastValueDataTypeTraits,
    pub(super) scalar_map: Option<&'a ScalarMap>,
    pub(super) rounding: DataTypeCastValueRoundingMode,
    pub(super) out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
    pub(super) kernel: Option<CastValueKernel>,
}

pub(super) fn cast_bytes(
    bytes: &[u8],
    context: &CastBytesContext<'_>,
) -> Result<Vec<u8>, CodecError> {
    let source_size = context
        .source_data_type
        .fixed_size()
        .ok_or(CodecError::Other(
            "cast_value requires fixed-size source data types".to_string(),
        ))?;
    let target_size = context
        .target_data_type
        .fixed_size()
        .ok_or(CodecError::Other(
            "cast_value requires fixed-size target data types".to_string(),
        ))?;
    if !bytes.len().is_multiple_of(source_size) {
        return Err(CodecError::Other(
            "cast_value source bytes are not element aligned".to_string(),
        ));
    }
    let num_elements = bytes.len() / source_size;
    let mut output = Vec::with_capacity(num_elements * target_size);
    if let Some(scalar_map) = context.scalar_map {
        for source_bytes in bytes.chunks_exact(source_size) {
            if let Some(mapped) = scalar_map.get(source_bytes) {
                output.extend_from_slice(mapped);
            } else {
                cast_element_into(
                    source_bytes,
                    context.source,
                    context.target,
                    context.kernel.as_ref(),
                    context.rounding,
                    context.out_of_range,
                    &mut output,
                )?;
            }
        }
    } else if let Some(kernel) = &context.kernel {
        kernel
            .cast(bytes, &mut output)
            .map_err(|err| CodecError::Other(err.to_string()))?;
    } else {
        context
            .source
            .cast_value_cast_slice(
                bytes,
                source_size,
                context.target,
                context.rounding,
                context.out_of_range,
                &mut output,
            )
            .map_err(|err| CodecError::Other(err.to_string()))?;
    }
    Ok(output)
}

/// Cast one element via the kernel if selected, else the generic scalar path.
fn cast_element_into(
    source_bytes: &[u8],
    source: &dyn CastValueDataTypeTraits,
    target: &dyn CastValueDataTypeTraits,
    kernel: Option<&CastValueKernel>,
    rounding: DataTypeCastValueRoundingMode,
    out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
    output: &mut Vec<u8>,
) -> Result<(), CodecError> {
    if let Some(kernel) = kernel {
        kernel.cast(source_bytes, output)
    } else {
        source.cast_value_cast(source_bytes, target, rounding, out_of_range, output)
    }
    .map_err(|err| CodecError::Other(err.to_string()))
}

fn cast_element(
    source_bytes: &[u8],
    source: &dyn CastValueDataTypeTraits,
    target: &dyn CastValueDataTypeTraits,
    scalar_map: Option<&ScalarMap>,
    kernel: Option<&CastValueKernel>,
    rounding: DataTypeCastValueRoundingMode,
    out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
) -> Result<Vec<u8>, CodecError> {
    if let Some(mapped) = scalar_map.and_then(|map| map.get(source_bytes)) {
        return Ok(mapped.clone());
    }
    let mut output = Vec::new();
    cast_element_into(
        source_bytes,
        source,
        target,
        kernel,
        rounding,
        out_of_range,
        &mut output,
    )?;
    Ok(output)
}

fn map_rounding(rounding: Option<CastValueRoundingMode>) -> DataTypeCastValueRoundingMode {
    match rounding.unwrap_or_default() {
        CastValueRoundingMode::NearestEven => DataTypeCastValueRoundingMode::NearestEven,
        CastValueRoundingMode::TowardsZero => DataTypeCastValueRoundingMode::TowardsZero,
        CastValueRoundingMode::TowardsPositive => DataTypeCastValueRoundingMode::TowardsPositive,
        CastValueRoundingMode::TowardsNegative => DataTypeCastValueRoundingMode::TowardsNegative,
        CastValueRoundingMode::NearestAway => DataTypeCastValueRoundingMode::NearestAway,
    }
}

fn map_out_of_range(
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Option<DataTypeCastValueOutOfRangeMode> {
    match out_of_range {
        Some(CastValueOutOfRangeMode::Clamp) => Some(DataTypeCastValueOutOfRangeMode::Clamp),
        Some(CastValueOutOfRangeMode::Wrap) => Some(DataTypeCastValueOutOfRangeMode::Wrap),
        None => None,
    }
}

fn validate_wrap_mode(
    data_type: &DataType,
    target: &dyn CastValueDataTypeTraits,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<(), CodecError> {
    if matches!(out_of_range, Some(CastValueOutOfRangeMode::Wrap))
        && !target.cast_value_is_integral()
    {
        return Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            "cast_value out_of_range=wrap".to_string(),
        ));
    }
    Ok(())
}

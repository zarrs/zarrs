use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_data_type::codec_traits::cast_value::{
    CastValueDataTypeExt, CastValueDataTypeTraits,
    CastValueOutOfRangeMode as DataTypeCastValueOutOfRangeMode,
    CastValueRoundingMode as DataTypeCastValueRoundingMode,
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
pub struct CastValueCodec {
    target_data_type: DataType,
    configuration: CastValueCodecConfigurationV1,
}

impl CastValueCodec {
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
                if matches!(
                    configuration.out_of_range,
                    Some(CastValueOutOfRangeMode::Wrap)
                ) && !target.cast_value_is_integral()
                {
                    return Err(PluginCreateError::Other(
                        "cast_value out_of_range=wrap is only valid for integral target data types"
                            .to_string(),
                    ));
                }
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

impl CodecTraits for CastValueCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

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

impl ArrayCodecTraits for CastValueCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for CastValueCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        decoded_data_type.codec_castvalue()?;
        self.target_data_type.codec_castvalue()?;
        Ok(self.target_data_type.clone())
    }

    fn encoded_fill_value(
        &self,
        decoded_data_type: &DataType,
        decoded_fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        let source = decoded_data_type.codec_castvalue()?;
        let target = self.target_data_type.codec_castvalue()?;
        validate_wrap_mode(
            &self.target_data_type,
            target,
            self.configuration.out_of_range,
        )?;
        let encode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.encode.as_ref())
            .map(|map| build_scalar_map(map, decoded_data_type, &self.target_data_type))
            .transpose()?;
        let decode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.decode.as_ref())
            .map(|map| build_scalar_map(map, &self.target_data_type, decoded_data_type))
            .transpose()?;

        let encoded = cast_element(
            decoded_fill_value.as_ne_bytes(),
            source,
            target,
            encode_scalar_map.as_ref(),
            self.configuration.rounding,
            self.configuration.out_of_range,
        )?;
        let decoded = cast_element(
            &encoded,
            target,
            source,
            decode_scalar_map.as_ref(),
            self.configuration.rounding,
            self.configuration.out_of_range,
        )?;
        if decoded != decoded_fill_value.as_ne_bytes() {
            return Err(CodecError::Other(
                "cast_value fill value does not survive a round-trip cast".to_string(),
            ));
        }
        Ok(FillValue::new(encoded))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let source = data_type.codec_castvalue()?;
        let target = self.target_data_type.codec_castvalue()?;
        validate_wrap_mode(
            &self.target_data_type,
            target,
            self.configuration.out_of_range,
        )?;
        let scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.encode.as_ref())
            .map(|map| build_scalar_map(map, data_type, &self.target_data_type))
            .transpose()?;
        Ok(ArrayBytes::from(cast_bytes(
            &bytes.into_fixed()?,
            &CastBytesContext {
                source_data_type: data_type,
                target_data_type: &self.target_data_type,
                source,
                target,
                scalar_map: scalar_map.as_ref(),
                rounding: self.configuration.rounding,
                out_of_range: self.configuration.out_of_range,
            },
        )?))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let source = self.target_data_type.codec_castvalue()?;
        let target = data_type.codec_castvalue()?;
        let scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.decode.as_ref())
            .map(|map| build_scalar_map(map, &self.target_data_type, data_type))
            .transpose()?;
        Ok(ArrayBytes::from(cast_bytes(
            &bytes.into_fixed()?,
            &CastBytesContext {
                source_data_type: &self.target_data_type,
                target_data_type: data_type,
                source,
                target,
                scalar_map: scalar_map.as_ref(),
                rounding: self.configuration.rounding,
                out_of_range: self.configuration.out_of_range,
            },
        )?))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_handle.data_type())?;
        Ok(Arc::new(self.partial(input_handle, data_type)?))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_output_handle.data_type())?;
        Ok(Arc::new(self.partial(input_output_handle, data_type)?))
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
        self.validate_partial_handle_data_type(input_handle.data_type())?;
        Ok(Arc::new(self.partial(input_handle, data_type)?))
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
        self.validate_partial_handle_data_type(input_output_handle.data_type())?;
        Ok(Arc::new(self.partial(input_output_handle, data_type)?))
    }
}

impl CastValueCodec {
    fn validate_partial_handle_data_type(&self, data_type: &DataType) -> Result<(), CodecError> {
        if data_type != &self.target_data_type {
            return Err(CodecError::Other(format!(
                "cast_value partial handle data type {data_type} does not match encoded data type {}",
                self.target_data_type
            )));
        }
        Ok(())
    }

    fn partial<T: ?Sized>(
        &self,
        input_output_handle: Arc<T>,
        decoded_data_type: &DataType,
    ) -> Result<super::cast_value_codec_partial::CastValueCodecPartial<T>, CodecError> {
        let target = self.target_data_type.codec_castvalue()?;
        validate_wrap_mode(
            &self.target_data_type,
            target,
            self.configuration.out_of_range,
        )?;
        let encode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.encode.as_ref())
            .map(|map| build_scalar_map(map, decoded_data_type, &self.target_data_type))
            .transpose()?;
        let decode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.decode.as_ref())
            .map(|map| build_scalar_map(map, &self.target_data_type, decoded_data_type))
            .transpose()?;
        super::cast_value_codec_partial::CastValueCodecPartial::new(
            input_output_handle,
            decoded_data_type,
            &self.target_data_type,
            encode_scalar_map,
            decode_scalar_map,
            self.configuration.rounding,
            self.configuration.out_of_range,
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
    pub(super) rounding: Option<CastValueRoundingMode>,
    pub(super) out_of_range: Option<CastValueOutOfRangeMode>,
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
    let mut out = Vec::with_capacity(num_elements * target_size);
    for source_bytes in bytes.chunks_exact(source_size) {
        out.extend(cast_element(
            source_bytes,
            context.source,
            context.target,
            context.scalar_map,
            context.rounding,
            context.out_of_range,
        )?);
    }
    Ok(out)
}

fn cast_element(
    source_bytes: &[u8],
    source: &dyn CastValueDataTypeTraits,
    target: &dyn CastValueDataTypeTraits,
    scalar_map: Option<&ScalarMap>,
    rounding: Option<CastValueRoundingMode>,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<Vec<u8>, CodecError> {
    if let Some(mapped) = scalar_map.and_then(|map| map.get(source_bytes)) {
        return Ok(mapped.clone());
    }
    source
        .cast_value_cast(
            source_bytes,
            target,
            map_rounding(rounding),
            map_out_of_range(out_of_range),
        )
        .map_err(|err| CodecError::Other(err.to_string()))
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

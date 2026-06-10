use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
};
use zarrs_data_type::codec_traits::scale_offset::ScaleOffsetDataTypeExt;
use zarrs_metadata::{Configuration, FillValueMetadata};
use zarrs_metadata_ext::codec::scale_offset::{
    ScaleOffsetCodecConfiguration, ScaleOffsetCodecConfigurationV1,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};

use crate::array::{DataType, FillValue};

/// Native-endian byte representations of the optional `offset` and `scale` quantities.
type Quantities = (Option<Vec<u8>>, Option<Vec<u8>>);

/// A `scale_offset` codec implementation.
#[derive(Clone, Debug)]
pub struct ScaleOffsetCodec {
    configuration: ScaleOffsetCodecConfigurationV1,
}

impl ScaleOffsetCodec {
    /// Create a new `scale_offset` codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the configuration is unsupported.
    pub fn new_with_configuration(
        configuration: &ScaleOffsetCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ScaleOffsetCodecConfiguration::V1(configuration) => Ok(Self {
                configuration: configuration.clone(),
            }),
            _ => Err(PluginCreateError::Other(
                "this scale_offset codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Parse the native-endian offset/scale bytes for the input `data_type`.
    fn quantities(&self, data_type: &DataType) -> Result<Quantities, CodecError> {
        let offset = parse_quantity(data_type, self.configuration.offset.as_ref())?;
        let scale = parse_quantity(data_type, self.configuration.scale.as_ref())?;
        Ok((offset, scale))
    }
}

fn parse_quantity(
    data_type: &DataType,
    metadata: Option<&FillValueMetadata>,
) -> Result<Option<Vec<u8>>, CodecError> {
    match metadata {
        Some(metadata) => {
            let fill_value = data_type
                .fill_value_v3(metadata)
                .map_err(|err| CodecError::Other(err.to_string()))?;
            Ok(Some(fill_value.as_ne_bytes().to_vec()))
        }
        None => Ok(None),
    }
}

impl CodecTraits for ScaleOffsetCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(ScaleOffsetCodecConfiguration::V1(self.configuration.clone()).into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        // NOTE: the default array-to-array partial decoder supports partial read/decode
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for ScaleOffsetCodec {
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
impl ArrayToArrayCodecTraits for ScaleOffsetCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        // Validate that the data type supports the codec. The data type is unchanged.
        decoded_data_type.codec_scaleoffset()?;
        Ok(decoded_data_type.clone())
    }

    fn encoded_fill_value(
        &self,
        decoded_data_type: &DataType,
        decoded_fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        let support = decoded_data_type.codec_scaleoffset()?;
        let (offset, scale) = self.quantities(decoded_data_type)?;
        let mut bytes = decoded_fill_value.as_ne_bytes().to_vec();
        support
            .scale_offset_encode(&mut bytes, offset.as_deref(), scale.as_deref())
            .map_err(|err| CodecError::Other(err.to_string()))?;
        Ok(FillValue::new(bytes))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let support = data_type.codec_scaleoffset()?;
        let (offset, scale) = self.quantities(data_type)?;
        let mut bytes = bytes.into_fixed()?.into_owned();
        support
            .scale_offset_encode(&mut bytes, offset.as_deref(), scale.as_deref())
            .map_err(|err| CodecError::Other(err.to_string()))?;
        Ok(bytes.into())
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let support = data_type.codec_scaleoffset()?;
        let (offset, scale) = self.quantities(data_type)?;
        let mut bytes = bytes.into_fixed()?.into_owned();
        support
            .scale_offset_decode(&mut bytes, offset.as_deref(), scale.as_deref())
            .map_err(|err| CodecError::Other(err.to_string()))?;
        Ok(bytes.into())
    }
}

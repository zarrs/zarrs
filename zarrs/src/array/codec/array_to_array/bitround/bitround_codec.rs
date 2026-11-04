use std::sync::Arc;

use zarrs_codec::plan::{
    ArrayToArrayDecodePlan, DecodeRequest, PartialEncodeRequest, PlanIndexer,
    partial_encode_request_elementwise,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{
    BitroundCodecConfiguration, BitroundCodecConfigurationV1, BitroundDataTypeExt, round_bytes,
};
use crate::array::{ChunkShape, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecSubchunkingIdentityTraits,
    ArrayToArrayCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, ComputeResources, UnboundArrayToArrayCodecTraits,
};
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
    fill_value: Option<FillValue>,
    encoded_fill_value: Option<FillValue>,
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
}

fn encode_fill_value(
    fill_value: &FillValue,
    data_type: &DataType,
    keepbits: u32,
) -> Result<FillValue, CodecCreateError> {
    let mut fill_value_bytes = ArrayBytes::new_fill_value(data_type, 1, fill_value)?
        .into_fixed()
        .map_err(CodecCreateError::other)?
        .to_vec();
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
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        data_type.codec_bitround()?;
        let encoded_fill_value = fill_value
            .as_ref()
            .map(|fill_value| encode_fill_value(fill_value, &data_type, self.keepbits))
            .transpose()?;
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
    fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }
}

impl ArrayToArrayCodecSubchunkingIdentityTraits for BitroundCodecBound {}

impl ArrayToArrayCodecTraits for BitroundCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn partial_encode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<PartialEncodeRequest<'a>, CodecError> {
        partial_encode_request_elementwise(self, shape, indexer, bytes, options, resources)
    }

    fn plan_decode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
    ) -> Result<ArrayToArrayDecodePlan<'a>, CodecError> {
        // Decoding is the identity
        request.indexer.validate(shape)?;
        Ok(ArrayToArrayDecodePlan::Forward(request))
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> Option<&FillValue> {
        self.encoded_fill_value.as_ref()
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec())
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let mut bytes = bytes.into_fixed()?;
        bytes.with_mut(|bytes| round_bytes(bytes, &self.data_type, self.keepbits))?;
        Ok(ArrayBytes::from(bytes))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }
}

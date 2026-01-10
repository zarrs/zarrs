use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_plugin::ExtensionAliasesV3;

use super::{
    TransposeCodecConfiguration, TransposeOrder, apply_permutation, inverse_permutation, permute,
};
use crate::array::codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use crate::array::data_type::DataTypeExt;
use crate::array::{ArrayBytes, ChunkShape, DataType, FillValue};
use crate::metadata::Configuration;
use crate::metadata_ext::codec::transpose::TransposeCodecConfigurationV1;
use crate::plugin::PluginCreateError;

/// A Transpose codec implementation.
#[derive(Clone, Debug)]
pub struct TransposeCodec {
    pub(crate) order: TransposeOrder,
}

impl TransposeCodec {
    /// Create a new transpose codec from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &TransposeCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            TransposeCodecConfiguration::V1(configuration) => {
                Ok(Self::new(configuration.order.clone()))
            }
            _ => Err(PluginCreateError::Other(
                "this transpose codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Create a new transpose codec.
    #[must_use]
    pub const fn new(order: TransposeOrder) -> Self {
        Self { order }
    }

    /// Validate the shape and data type for this codec.
    fn validate(&self, shape: &[NonZeroU64], data_type: &DataType) -> Result<(), CodecError> {
        if data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::aliases_v3().default_name.to_string(),
            ));
        }
        if self.order.0.len() != shape.len() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }
        Ok(())
    }
}

impl CodecTraits for TransposeCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(&self, _options: &CodecMetadataOptions) -> Option<Configuration> {
        let configuration = TransposeCodecConfiguration::V1(TransposeCodecConfigurationV1 {
            order: self.order.clone(),
        });
        Some(configuration.into())
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
impl ArrayToArrayCodecTraits for TransposeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        if decoded_data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                decoded_data_type.clone(),
                Self::aliases_v3().default_name.to_string(),
            ));
        }
        Ok(decoded_data_type.clone())
    }

    fn encoded_fill_value(
        &self,
        _decoded_data_type: &DataType,
        decoded_fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        Ok(decoded_fill_value.clone())
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        if self.order.0.len() != decoded_shape.len() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }
        Ok(permute(decoded_shape, &self.order.0).expect("matching dimensionality"))
    }

    fn decoded_shape(
        &self,
        encoded_shape: &[NonZeroU64],
    ) -> Result<Option<ChunkShape>, CodecError> {
        if self.order.0.len() != encoded_shape.len() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }
        let transposed_shape = permute(encoded_shape, &inverse_permutation(&self.order.0))
            .expect("matching dimensionality");
        Ok(Some(transposed_shape))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.validate(shape, data_type)?;

        // Encode: apply the transpose order to the decoded shape
        let shape_u64 = bytemuck::must_cast_slice(shape);
        apply_permutation(&bytes, shape_u64, &self.order.0, data_type)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.validate(shape, data_type)?;

        // Decode: apply the inverse permutation to the encoded (transposed) shape
        let shape_u64 = bytemuck::must_cast_slice(shape);
        let transposed_shape = permute(shape_u64, &self.order.0).expect("validated");
        apply_permutation(
            &bytes,
            &transposed_shape,
            &inverse_permutation(&self.order.0),
            data_type,
        )
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_handle,
                shape,
                data_type,
                fill_value,
                self.order.0.clone(),
            ),
        ))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_output_handle,
                shape,
                data_type,
                fill_value,
                self.order.0.clone(),
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_handle,
                shape,
                data_type,
                fill_value,
                self.order.0.clone(),
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_output_handle,
                shape,
                data_type,
                fill_value,
                self.order.0.clone(),
            ),
        ))
    }
}

impl ArrayCodecTraits for TransposeCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: This could be increased, need to implement `transpose_array` without ndarray
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

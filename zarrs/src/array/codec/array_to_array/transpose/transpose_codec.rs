use std::{num::NonZeroU64, sync::Arc};

use crate::array::{codec::ArrayPartialEncoderTraits, DataType, FillValue};
use zarrs_metadata::Configuration;
use zarrs_registry::codec::TRANSPOSE;

use crate::{
    array::{
        codec::{
            ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToArrayCodecTraits,
            CodecError, CodecMetadataOptions, CodecOptions, CodecTraits, RecommendedConcurrency,
        },
        ChunkRepresentation, ChunkShape,
    },
    plugin::PluginCreateError,
};
use zarrs_metadata_ext::codec::transpose::TransposeCodecConfigurationV1;

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

use super::{
    calculate_order_decode, calculate_order_encode, permute, transpose_array,
    TransposeCodecConfiguration, TransposeOrder,
};

/// A Transpose codec implementation.
#[derive(Clone, Debug)]
pub struct TransposeCodec {
    order: TransposeOrder,
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
}

impl CodecTraits for TransposeCodec {
    fn identifier(&self) -> &str {
        TRANSPOSE
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = TransposeCodecConfiguration::V1(TransposeCodecConfigurationV1 {
            order: self.order.clone(),
        });
        Some(configuration.into())
    }

    fn partial_decoder_should_cache_input(&self) -> bool {
        false
    }

    fn partial_decoder_decodes_all(&self) -> bool {
        false
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
        Ok(permute(decoded_shape, &self.order.0).into())
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
        let mut permutation_decode = vec![0; self.order.0.len()];
        for (i, val) in self.order.0.iter().enumerate() {
            permutation_decode[*val] = i;
        }
        let transposed_shape = permute(encoded_shape, &permutation_decode);
        Ok(Some(transposed_shape.into()))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        bytes.validate(
            decoded_representation.num_elements(),
            decoded_representation.data_type().size(),
        )?;
        if self.order.0.len() != decoded_representation.dimensionality() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }

        match bytes {
            ArrayBytes::Variable(bytes, offsets) => {
                let order_encode = self.order.0.clone();
                let shape = decoded_representation
                    .shape()
                    .iter()
                    .map(|s| usize::try_from(s.get()).unwrap())
                    .collect::<Vec<_>>();
                Ok(super::transpose_vlen(
                    &bytes,
                    &offsets,
                    &shape,
                    order_encode,
                ))
            }
            ArrayBytes::Fixed(bytes) => {
                let order_encode =
                    calculate_order_encode(&self.order, decoded_representation.shape().len());
                let data_type_size = decoded_representation.data_type().fixed_size().unwrap();
                let bytes = transpose_array(
                    &order_encode,
                    &decoded_representation.shape_u64(),
                    data_type_size,
                    &bytes,
                )
                .map_err(|_| CodecError::Other("transpose_array invalid arguments?".to_string()))?;
                Ok(ArrayBytes::from(bytes))
            }
        }
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        bytes.validate(
            decoded_representation.num_elements(),
            decoded_representation.data_type().size(),
        )?;
        if self.order.0.len() != decoded_representation.dimensionality() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }

        match bytes {
            ArrayBytes::Variable(bytes, offsets) => {
                let mut order_decode = vec![0; decoded_representation.shape().len()];
                for (i, val) in self.order.0.iter().enumerate() {
                    order_decode[*val] = i;
                }
                let shape = decoded_representation
                    .shape()
                    .iter()
                    .map(|s| usize::try_from(s.get()).unwrap())
                    .collect::<Vec<_>>();
                Ok(super::transpose_vlen(
                    &bytes,
                    &offsets,
                    &shape,
                    order_decode,
                ))
            }
            ArrayBytes::Fixed(bytes) => {
                let order_decode =
                    calculate_order_decode(&self.order, decoded_representation.shape().len());
                let transposed_shape = permute(&decoded_representation.shape_u64(), &self.order.0);
                let data_type_size = decoded_representation.data_type().fixed_size().unwrap();
                let bytes =
                    transpose_array(&order_decode, &transposed_shape, data_type_size, &bytes)
                        .map_err(|_| CodecError::Other("transpose_array error".to_string()))?;
                Ok(ArrayBytes::from(bytes))
            }
        }
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::transpose_partial_decoder::TransposePartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
                self.order.clone(),
            ),
        ))
    }

    fn partial_encoder(
        self: Arc<Self>,
        _input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        _output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        todo!("the transpose codec does not yet support partial encoding")
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::transpose_partial_decoder::AsyncTransposePartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
                self.order.clone(),
            ),
        ))
    }
}

impl ArrayCodecTraits for TransposeCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: This could be increased, need to implement `transpose_array` without ndarray
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

// TODO: reshape partial decoder

use std::{num::NonZeroU64, sync::Arc};

use crate::array::{
    codec::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits},
    DataType, FillValue,
};
use num::Integer;
use zarrs_metadata::Configuration;
use zarrs_registry::codec::RESHAPE;

use crate::{
    array::{
        codec::{
            ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, CodecError,
            CodecMetadataOptions, CodecOptions, CodecTraits, RecommendedConcurrency,
        },
        ChunkRepresentation, ChunkShape,
    },
    plugin::PluginCreateError,
};
use zarrs_metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeDim, ReshapeShape,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// A `reshape` codec implementation.
#[derive(Clone, Debug)]
pub struct ReshapeCodec {
    shape: ReshapeShape,
}

impl ReshapeCodec {
    /// Create a new reshape codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &ReshapeCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ReshapeCodecConfiguration::V1(configuration) => {
                Ok(Self::new(configuration.shape.clone()))
            }
            _ => Err(PluginCreateError::Other(
                "this reshape codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Create a new reshape codec.
    #[must_use]
    pub const fn new(shape: ReshapeShape) -> Self {
        Self { shape }
    }
}

impl CodecTraits for ReshapeCodec {
    fn identifier(&self) -> &str {
        RESHAPE
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ReshapeCodecConfiguration::V1(ReshapeCodecConfigurationV1 {
            shape: self.shape.clone(),
        });
        Some(configuration.into())
    }

    fn partial_decoder_should_cache_input(&self) -> bool {
        false
    }

    fn partial_decoder_decodes_all(&self) -> bool {
        true // TODO: implement partial decoder and change to false
    }
}

#[cfg_attr(feature = "async", async_trait::async_trait)]
impl ArrayToArrayCodecTraits for ReshapeCodec {
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
        let mut encoded_shape = Vec::with_capacity(self.shape.0.len());
        let mut fill_index = None;
        for output_dim in &self.shape.0 {
            match output_dim {
                ReshapeDim::Size(size) => encoded_shape.push(*size),
                ReshapeDim::InputDims(input_dims) => {
                    let mut product = NonZeroU64::new(1).unwrap();
                    for input_dim in input_dims {
                        let input_shape = *decoded_shape
                            .get(usize::try_from(*input_dim).unwrap())
                            .ok_or_else(|| {
                                CodecError::Other(
                                    format!("reshape codec shape references a dimension ({input_dim}) larger than the chunk dimensionality ({})", decoded_shape.len()),
                                )
                            })?;
                        product = product.checked_mul(input_shape).unwrap();
                    }
                    encoded_shape.push(product);
                }
                ReshapeDim::Auto(_) => {
                    fill_index = Some(encoded_shape.len());
                    encoded_shape.push(NonZeroU64::new(1).unwrap());
                }
            }
        }

        let num_elements_input = decoded_shape.iter().map(|u| u.get()).product::<u64>();
        let num_elements_output = encoded_shape.iter().map(|u| u.get()).product::<u64>();
        if let Some(fill_index) = fill_index {
            let (quot, rem) = num_elements_input.div_rem(&num_elements_output);
            if rem == 0 {
                encoded_shape[fill_index] = NonZeroU64::new(quot).unwrap();
            } else {
                return Err(CodecError::Other(
                    format!("reshape codec no substitution for dim {fill_index} can satisfy decoded_shape {decoded_shape:?} == encoded_shape {encoded_shape:?}."),
                ));
            }
        } else if num_elements_input != num_elements_output {
            return Err(CodecError::Other(
                    format!("reshape codec encoded/decoded number of elements differ: decoded_shape {decoded_shape:?} ({num_elements_input}) encoded_shape {encoded_shape:?} ({num_elements_output})."),
                ));
        }

        Ok(encoded_shape.into())
    }

    fn decoded_shape(
        &self,
        _encoded_shape: &[NonZeroU64],
    ) -> Result<Option<ChunkShape>, CodecError> {
        Ok(None)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        _input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        todo!("the reshape codec does not yet support partial decoding")
    }

    fn partial_encoder(
        self: Arc<Self>,
        _input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        todo!("the reshape codec does not yet support partial encoding")
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        _input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        todo!("the reshape codec does not yet support partial decoding")
    }
}

impl ArrayCodecTraits for ReshapeCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

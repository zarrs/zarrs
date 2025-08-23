use std::{borrow::Cow, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::SHUFFLE;

use crate::array::{
    codec::{
        BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        RecommendedConcurrency,
    },
    BytesRepresentation, RawBytes,
};

use super::{ShuffleCodecConfiguration, ShuffleCodecConfigurationV1};

/// A `shuffle` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct ShuffleCodec {
    elementsize: usize,
}

impl ShuffleCodec {
    /// Create a new `shuffle` codec.
    #[must_use]
    pub fn new(elementsize: usize) -> Self {
        Self { elementsize }
    }

    /// Create a new `shuffle` codec.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &ShuffleCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ShuffleCodecConfiguration::V1(configuration) => Ok(Self {
                elementsize: configuration.elementsize,
            }),
            _ => Err(PluginCreateError::Other(
                "this shuffle codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for ShuffleCodec {
    fn identifier(&self) -> &str {
        SHUFFLE
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ShuffleCodecConfiguration::V1(ShuffleCodecConfigurationV1 {
            elementsize: self.elementsize,
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

fn is_multiple_of(lhs: usize, rhs: usize) -> bool {
    match rhs {
        // prevent division by zero
        0 => lhs == 0,
        _ => lhs % rhs == 0,
    }
}

#[cfg_attr(all(feature = "async", not(target_arch = "wasm32")), async_trait::async_trait)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for ShuffleCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: RawBytes<'a>,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        if !is_multiple_of(decoded_value.len(), self.elementsize) {
            return Err(CodecError::Other("the shuffle codec expects the input byte length to be an integer multiple of the elementsize".to_string()));
        }

        let mut encoded_value = decoded_value.to_vec();
        let count = encoded_value.len().div_ceil(self.elementsize);
        for i in 0..count {
            let offset = i * self.elementsize;
            for byte_index in 0..self.elementsize {
                let j = byte_index * count + i;
                encoded_value[j] = decoded_value[offset + byte_index];
            }
        }
        Ok(Cow::Owned(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: RawBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        if !is_multiple_of(encoded_value.len(), self.elementsize) {
            return Err(CodecError::Other("the shuffle codec expects the input byte length to be an integer multiple of the elementsize".to_string()));
        }

        let mut decoded_value = encoded_value.to_vec();
        let count = decoded_value.len().div_ceil(self.elementsize);
        for i in 0..self.elementsize {
            let offset = i * count;
            for byte_index in 0..count {
                let j = byte_index * self.elementsize + i;
                decoded_value[j] = encoded_value[offset + byte_index];
            }
        }
        Ok(Cow::Owned(decoded_value))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        *decoded_representation
    }
}

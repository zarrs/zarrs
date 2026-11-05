use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{ShuffleCodecConfiguration, ShuffleCodecConfigurationV1};
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    ComputeResources, InvalidBytesLengthError, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
};
use zarrs_metadata::Configuration;

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
            ShuffleCodecConfiguration::V1(configuration) if configuration.elementsize == 0 => {
                Err(PluginCreateError::Other(
                    "the shuffle codec elementsize must be non-zero".to_string(),
                ))
            }
            ShuffleCodecConfiguration::V1(configuration) => Ok(Self {
                elementsize: configuration.elementsize,
            }),
            _ => Err(PluginCreateError::Other(
                "this shuffle codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Return the number of elements in `len` bytes.
    ///
    /// # Errors
    /// Returns an error if the elementsize is zero or `len` is not an integer multiple of it.
    fn element_count(&self, len: usize) -> Result<usize, CodecError> {
        if self.elementsize == 0 {
            return Err(CodecError::Other(
                "the shuffle codec elementsize must be non-zero".to_string(),
            ));
        }
        if !len.is_multiple_of(self.elementsize) {
            return Err(CodecError::Other("the shuffle codec expects the input byte length to be an integer multiple of the elementsize".to_string()));
        }
        Ok(len / self.elementsize)
    }

    /// Unshuffle `encoded_value` into `decoded_value`, which must be the same length.
    fn unshuffle(&self, encoded_value: &[u8], decoded_value: &mut [u8]) -> Result<(), CodecError> {
        if encoded_value.len() != decoded_value.len() {
            return Err(
                InvalidBytesLengthError::new(decoded_value.len(), encoded_value.len()).into(),
            );
        }

        let count = self.element_count(encoded_value.len())?;
        for i in 0..self.elementsize {
            let offset = i * count;
            for byte_index in 0..count {
                let j = byte_index * self.elementsize + i;
                decoded_value[j] = encoded_value[offset + byte_index];
            }
        }
        Ok(())
    }
}

impl CodecTraits for ShuffleCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ShuffleCodecConfiguration::V1(ShuffleCodecConfigurationV1 {
            elementsize: self.elementsize,
        });
        Some(configuration.into())
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
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let count = self.element_count(decoded_value.len())?;
        let mut encoded_value = vec![0u8; decoded_value.len()];
        for i in 0..count {
            let offset = i * self.elementsize;
            for byte_index in 0..self.elementsize {
                let j = byte_index * count + i;
                encoded_value[j] = decoded_value[offset + byte_index];
            }
        }
        Ok(CowBytes::from(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let mut decoded_value = vec![0u8; encoded_value.len()];
        self.unshuffle(&encoded_value, &mut decoded_value)?;
        Ok(CowBytes::from(decoded_value))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        *decoded_representation
    }
}

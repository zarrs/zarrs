// Note: No validation that this codec is created *without* a specified endianness for multi-byte data types.

use std::sync::Arc;

use crate::array::{
    codec::{ArrayPartialEncoderTraits, BytesPartialEncoderTraits, PartialEncoderCapability},
    DataType,
};
use zarrs_data_type::DataTypeExtensionError;
use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::BYTES;

use crate::array::{
    codec::{
        ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
        BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        InvalidBytesLengthError, PartialDecoderCapability, RecommendedConcurrency,
    },
    ArrayBytes, BytesRepresentation, ChunkRepresentation, DataTypeSize, RawBytes,
};

#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};

use super::{
    bytes_codec_partial, reverse_endianness, BytesCodecConfiguration, BytesCodecConfigurationV1,
    Endianness,
};

/// A `bytes` codec implementation.
#[derive(Debug, Clone)]
pub struct BytesCodec {
    endian: Option<Endianness>,
}

impl Default for BytesCodec {
    fn default() -> Self {
        Self::new(Some(Endianness::native()))
    }
}

impl BytesCodec {
    /// Create a new `bytes` codec.
    ///
    /// `endian` is optional because an 8-bit type has no endianness.
    #[must_use]
    pub const fn new(endian: Option<Endianness>) -> Self {
        Self { endian }
    }

    /// Create a new `bytes` codec for little endian data.
    #[must_use]
    pub const fn little() -> Self {
        Self::new(Some(Endianness::Little))
    }

    /// Create a new `bytes` codec for big endian data.
    #[must_use]
    pub const fn big() -> Self {
        Self::new(Some(Endianness::Big))
    }

    /// Create a new `bytes` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &BytesCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            BytesCodecConfiguration::V1(configuration) => Ok(Self::new(configuration.endian)),
            _ => Err(PluginCreateError::Other(
                "this bytes codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    fn do_encode_or_decode<'a>(
        &self,
        mut value: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<RawBytes<'a>, CodecError> {
        let Some(data_type_size) = decoded_representation.data_type().fixed_size() else {
            return Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                BYTES.to_string(),
            ));
        };

        let array_size =
            usize::try_from(decoded_representation.num_elements() * data_type_size as u64).unwrap();
        if value.len() != array_size {
            return Err(InvalidBytesLengthError::new(value.len(), array_size).into());
        } else if data_type_size > 1 && self.endian.is_none() {
            return Err(CodecError::Other(format!(
                "tried to encode an array with element size {data_type_size} with endianness None"
            )));
        }

        if let Some(endian) = &self.endian {
            if !endian.is_native() {
                reverse_endianness(value.to_mut(), decoded_representation.data_type());
            }
        }
        Ok(value)
    }
}

impl CodecTraits for BytesCodec {
    fn identifier(&self) -> &str {
        BYTES
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = BytesCodecConfiguration::V1(BytesCodecConfigurationV1 {
            endian: self.endian,
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

impl ArrayCodecTraits for BytesCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: Recomment > 1 if endianness needs changing and input is sufficiently large
        // if let Some(endian) = &self.endian {
        //     if !endian.is_native() {
        //         FIXME: Support parallel
        //         let min_elements_per_thread = 32768; // 32^3
        //         unsafe {
        //             NonZeroU64::new_unchecked(
        //                 decoded_representation.num_elements().div_ceil(min_elements_per_thread),
        //             )
        //         }
        //     }
        // }
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for BytesCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        // Reject optional data types explicitly
        if decoded_representation.data_type().is_optional() {
            return Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                BYTES.to_string(),
            ));
        }

        bytes.validate(
            decoded_representation.num_elements(),
            decoded_representation.data_type(),
        )?;
        let bytes = bytes.into_fixed()?;
        let bytes_encoded = if let DataType::Extension(ext) = decoded_representation.data_type() {
            ext.codec_bytes()?
                .encode(bytes, self.endian)
                .map_err(DataTypeExtensionError::from)?
        } else {
            self.do_encode_or_decode(bytes, decoded_representation)?
        };
        Ok(bytes_encoded)
    }

    fn decode<'a>(
        &self,
        bytes: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        // Reject optional data types explicitly
        if decoded_representation.data_type().is_optional() {
            return Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                BYTES.to_string(),
            ));
        }

        let bytes_decoded = if let DataType::Extension(ext) = decoded_representation.data_type() {
            ext.codec_bytes()?
                .decode(bytes, self.endian)
                .map_err(DataTypeExtensionError::from)?
        } else {
            self.do_encode_or_decode(bytes, decoded_representation)?
        };
        Ok(ArrayBytes::from(bytes_decoded))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            decoded_representation.clone(),
            self.endian,
        )))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            decoded_representation.clone(),
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            decoded_representation.clone(),
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            decoded_representation.clone(),
            self.endian,
        )))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        // Reject optional data types explicitly
        if decoded_representation.data_type().is_optional() {
            return Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                BYTES.to_string(),
            ));
        }

        match decoded_representation.data_type().size() {
            DataTypeSize::Variable => Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                BYTES.to_string(),
            )),
            DataTypeSize::Fixed(data_type_size) => Ok(BytesRepresentation::FixedSize(
                decoded_representation.num_elements() * data_type_size as u64,
            )),
        }
    }
}

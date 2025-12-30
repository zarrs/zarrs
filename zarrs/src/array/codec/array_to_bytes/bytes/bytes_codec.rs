// Note: No validation that this codec is created *without* a specified endianness for multi-byte data types.

use std::sync::Arc;

use zarrs_data_type::DataTypeExtensionError;
use zarrs_plugin::PluginCreateError;

use super::{BytesCodecConfiguration, BytesCodecConfigurationV1, Endianness, bytes_codec_partial};
#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use crate::array::{
    ArrayBytes, ArrayBytesRaw, BytesRepresentation, ChunkShapeTraits, DataTypeSize, FillValue,
    codec::{
        ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
        BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        PartialDecoderCapability, RecommendedConcurrency,
    },
};
use crate::array::{
    DataType,
    codec::{ArrayPartialEncoderTraits, BytesPartialEncoderTraits, PartialEncoderCapability},
    data_type::DataTypeExt,
};
use crate::metadata::Configuration;
use std::num::NonZeroU64;
use zarrs_plugin::ExtensionIdentifier;

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
}

impl CodecTraits for BytesCodec {
    fn identifier(&self) -> &'static str {
        Self::IDENTIFIER
    }

    fn configuration(&self, _name: &str, _options: &CodecMetadataOptions) -> Option<Configuration> {
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
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: Recomment > 1 if endianness needs changing and input is sufficiently large
        // if let Some(endian) = &self.endian {
        //     if !endian.is_native() {
        //         FIXME: Support parallel
        //         let min_elements_per_thread = 32768; // 32^3
        //         let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        //         unsafe {
        //             NonZeroU64::new_unchecked(
        //                 num_elements.div_ceil(min_elements_per_thread),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        // Reject optional data types explicitly
        if data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::IDENTIFIER.to_string(),
            ));
        }

        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes.validate(num_elements, data_type)?;
        let bytes = bytes.into_fixed()?;

        // Use get_bytes_support() for all types
        let bytes_encoded = zarrs_data_type::get_bytes_support(&**data_type)
            .ok_or_else(|| {
                CodecError::UnsupportedDataType(data_type.clone(), Self::IDENTIFIER.to_string())
            })?
            .encode(bytes, self.endian)
            .map_err(DataTypeExtensionError::from)?;
        Ok(bytes_encoded)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        // Reject optional data types explicitly
        if data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::IDENTIFIER.to_string(),
            ));
        }

        // Use get_bytes_support() for all types
        let bytes_decoded: ArrayBytes = zarrs_data_type::get_bytes_support(&**data_type)
            .ok_or_else(|| {
                CodecError::UnsupportedDataType(data_type.clone(), Self::IDENTIFIER.to_string())
            })?
            .decode(bytes, self.endian)
            .map_err(DataTypeExtensionError::from)?
            .into();

        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes_decoded.validate(num_elements, data_type)?;

        Ok(bytes_decoded)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            shape,
            data_type,
            fill_value,
            self.endian,
        )))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            shape,
            data_type,
            fill_value,
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            shape,
            data_type,
            fill_value,
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            shape,
            data_type,
            fill_value,
            self.endian,
        )))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
    ) -> Result<BytesRepresentation, CodecError> {
        // Reject optional data types explicitly
        if data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::IDENTIFIER.to_string(),
            ));
        }

        match data_type.size() {
            DataTypeSize::Variable => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::IDENTIFIER.to_string(),
            )),
            DataTypeSize::Fixed(data_type_size) => Ok(BytesRepresentation::FixedSize(
                shape.num_elements_u64() * data_type_size as u64,
            )),
        }
    }
}

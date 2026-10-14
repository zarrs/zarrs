// Note: No validation that this codec is created *without* a specified endianness for multi-byte data types.

use std::sync::Arc;

use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

use super::{
    BytesCodecConfiguration, BytesCodecConfigurationV1, BytesDataTypeExt, Endianness,
    bytes_codec_partial,
};
use crate::array::{ArrayBytes, ArrayBytesRaw, BytesRepresentation, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToBytesCodecTraits, BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecError,
    CodecMetadataOptions, CodecOptions, CodecTraits, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use zarrs_metadata::Configuration;

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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
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

impl BytesCodec {
    fn decoded_size(
        &self,
        data_type: &DataType,
        shape: &[NonZeroU64],
    ) -> Result<(u64, u64), CodecError> {
        let data_type_size = data_type.fixed_size().ok_or_else(|| {
            CodecError::UnsupportedDataType(
                data_type.clone(),
                BytesCodec::aliases_v3().default_name.to_string(),
            )
        })?;
        let num_elements = shape
            .iter()
            .try_fold(1u64, |count, d| count.checked_mul(d.get()))
            .ok_or("the decoded chunk element count overflows u64")?;
        let num_bytes = num_elements
            .checked_mul(data_type_size as u64)
            .ok_or("the decoded chunk size in bytes overflows u64")?;
        Ok((num_elements, num_bytes))
    }

    fn decoded_len(
        &self,
        data_type: &DataType,
        shape: &[NonZeroU64],
    ) -> Result<(u64, usize), CodecError> {
        let (num_elements, num_bytes) = self.decoded_size(data_type, shape)?;
        let len = usize::try_from(num_bytes)
            .map_err(|_| CodecError::from("the decoded chunk size in bytes overflows usize"))?;
        Ok((num_elements, len))
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
        //         let (num_elements, _) = self.decoded_len(data_type, shape)?;
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

    fn is_decode_passthrough(
        &self,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
    ) -> Result<bool, CodecError> {
        if !data_type.is_fixed() || data_type.is_optional() {
            return Ok(false);
        }
        Ok(data_type.codec_bytes()?.is_decode_passthrough(self.endian))
    }

    fn decode_into(
        &self,
        bytes: ArrayBytesRaw<'_>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        output_target: zarrs_codec::ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if self.is_decode_passthrough(shape, data_type, fill_value)? {
            let (_, len) = self.decoded_len(data_type, shape)?;
            if bytes.len() != len {
                return Err(zarrs_codec::InvalidBytesLengthError::new(bytes.len(), len).into());
            }
            return zarrs_codec::decode_into_array_bytes_target(
                &ArrayBytes::new_flen(bytes),
                output_target,
            );
        }
        let bytes = self.decode(bytes, shape, data_type, fill_value, options)?;
        zarrs_codec::decode_into_array_bytes_target(&bytes, output_target)
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
                Self::aliases_v3().default_name.to_string(),
            ));
        }

        let (num_elements, _) = self.decoded_len(data_type, shape)?;
        bytes.validate(num_elements, data_type)?;
        let bytes = bytes.into_fixed()?;

        let bytes_encoded = data_type.codec_bytes()?.encode(bytes, self.endian)?;
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
                Self::aliases_v3().default_name.to_string(),
            ));
        }

        let bytes_decoded: ArrayBytes = data_type.codec_bytes()?.decode(bytes, self.endian)?.into();

        let (num_elements, _) = self.decoded_len(data_type, shape)?;
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
                Self::aliases_v3().default_name.to_string(),
            ));
        }

        let (_, size) = self.decoded_size(data_type, shape)?;
        Ok(BytesRepresentation::FixedSize(size))
    }
}

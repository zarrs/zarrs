#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]

use std::{mem::size_of, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::OPTIONAL;

use crate::array::{
    codec::{
        ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecError,
        CodecMetadataOptions, CodecOptions, CodecTraits, InvalidBytesLengthError,
        PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    },
    ArrayBytes, BytesRepresentation, ChunkRepresentation, CodecChain, DataType, RawBytes,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

use super::{OptionalCodecConfiguration, OptionalCodecConfigurationV1};

/// An `optional` codec implementation.
#[derive(Debug, Clone)]
pub struct OptionalCodec {
    mask_codecs: Arc<CodecChain>,
    data_codecs: Arc<CodecChain>,
}

impl OptionalCodec {
    /// Create a new `optional` codec.
    #[must_use]
    pub fn new(mask_codecs: Arc<CodecChain>, data_codecs: Arc<CodecChain>) -> Self {
        Self {
            mask_codecs,
            data_codecs,
        }
    }

    /// Create a new `optional` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &OptionalCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            OptionalCodecConfiguration::V1(configuration) => {
                let mask_codecs = Arc::new(CodecChain::from_metadata(&configuration.mask_codecs)?);
                let data_codecs = Arc::new(CodecChain::from_metadata(&configuration.data_codecs)?);
                Ok(Self::new(mask_codecs, data_codecs))
            }
            _ => Err(PluginCreateError::Other(
                "this optional codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    fn extract_optional_array_bytes<'a>(
        bytes: &'a ArrayBytes<'a>,
        data_type: &DataType,
        num_elements: usize,
    ) -> Result<(Vec<u8>, ArrayBytes<'a>), CodecError> {
        if let ArrayBytes::Optional { mask, data } = bytes {
            let DataType::Optional(_inner_type) = data_type else {
                return Err(CodecError::Other(
                    "optional codec requires an optional data type".to_string(),
                ));
            };

            // Mask is already byte-based (one byte per element)
            if mask.len() != num_elements {
                return Err(CodecError::Other(
                    "mask length does not match number of elements".to_string(),
                ));
            }

            Ok((mask.to_vec(), (**data).clone()))
        } else {
            // Fallback for legacy fixed format
            let DataType::Optional(inner_type) = data_type else {
                return Err(CodecError::Other(
                    "optional codec requires an optional data type".to_string(),
                ));
            };

            let inner_size = inner_type.fixed_size().ok_or_else(|| {
                CodecError::Other("inner data type must have a fixed size".to_string())
            })?;

            let bytes = bytes.clone().into_fixed()?;
            // The input should be num_elements * (inner_size bytes data + 1 byte mask)
            let expected_len = num_elements * (inner_size + 1);
            if bytes.len() != expected_len {
                return Err(InvalidBytesLengthError::new(bytes.len(), expected_len).into());
            }

            // Separate mask and data
            let mut mask = Vec::with_capacity(num_elements);
            let mut data = Vec::with_capacity(num_elements * inner_size);
            for i in 0..num_elements {
                let offset = i * (inner_size + 1);
                // Data comes first, then mask byte
                let is_valid = bytes[offset + inner_size] != 0;
                mask.push(u8::from(is_valid));
                if is_valid {
                    // Only include valid elements in the data
                    data.extend_from_slice(&bytes[offset..offset + inner_size]);
                }
            }
            Ok((mask, ArrayBytes::from(data)))
        }
    }

    fn reconstruct_optional_data_legacy(
        mask: &[u8],
        data: &[u8],
        data_type: &DataType,
    ) -> Result<Vec<u8>, CodecError> {
        let DataType::Optional(inner_type) = data_type else {
            return Err(CodecError::Other(
                "optional codec requires an optional data type".to_string(),
            ));
        };

        let inner_size = inner_type.fixed_size().ok_or_else(|| {
            CodecError::Other("inner data type must have a fixed size".to_string())
        })?;

        let num_elements = mask.len();
        let valid_count = mask.iter().filter(|&&v| v != 0).count();

        // Check that data has the right size
        let expected_data_len = valid_count * inner_size;
        if data.len() != expected_data_len {
            return Err(CodecError::Other(format!(
                "data length mismatch: expected {}, got {}",
                expected_data_len,
                data.len()
            )));
        }

        // Reconstruct the optional data with data first, then mask byte
        let mut result = Vec::with_capacity(num_elements * (inner_size + 1));
        let mut data_offset = 0;

        for &is_valid in mask {
            if is_valid != 0 {
                result.extend_from_slice(&data[data_offset..data_offset + inner_size]);
                data_offset += inner_size;
            } else {
                // Fill with zeros for missing elements
                result.extend_from_slice(&vec![0u8; inner_size]);
            }
            result.push(is_valid);
        }

        Ok(result)
    }
}

impl CodecTraits for OptionalCodec {
    fn identifier(&self) -> &str {
        OPTIONAL
    }

    fn configuration_opt(
        &self,
        _name: &str,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = OptionalCodecConfiguration::V1(OptionalCodecConfigurationV1 {
            mask_codecs: self.mask_codecs.create_metadatas_opt(options),
            data_codecs: self.data_codecs.create_metadatas_opt(options),
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        // For now, disable partial decoding
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

impl ArrayCodecTraits for OptionalCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // Sequential processing for now
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for OptionalCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let num_elements = decoded_representation.num_elements_usize();

        // Extract mask and data
        let (mask, data) = Self::extract_optional_array_bytes(
            &bytes,
            decoded_representation.data_type(),
            num_elements,
        )?;

        // Create representations for mask and data
        let mask_representation =
            ChunkRepresentation::new(decoded_representation.shape().to_vec(), DataType::Bool, 0u8)?;

        let DataType::Optional(inner_type) = decoded_representation.data_type() else {
            return Err(CodecError::Other(
                "optional codec requires an optional data type".to_string(),
            ));
        };

        // Encode mask
        let encoded_mask =
            self.mask_codecs
                .encode(ArrayBytes::from(mask), &mask_representation, options)?;

        // Encode data (only valid elements)
        #[allow(clippy::if_not_else)]
        let encoded_data = if data.size() > 0 {
            // Calculate number of valid elements from the data
            let inner_size = inner_type.fixed_size().unwrap();
            let num_valid = match &data {
                ArrayBytes::Fixed(bytes) => bytes.len() / inner_size,
                ArrayBytes::Variable(_, offsets) => offsets.len().saturating_sub(1),
                ArrayBytes::Optional { .. } => {
                    return Err(CodecError::Other(
                        "nested optional data types are not supported".to_string(),
                    ));
                }
            };

            let data_shape = if num_valid > 0 {
                vec![std::num::NonZeroU64::try_from(num_valid as u64).unwrap()]
            } else {
                vec![std::num::NonZeroU64::new(1).unwrap()]
            };
            // Create a zero-filled fill value of the correct size for the inner type
            let fill_value = crate::array::FillValue::new(vec![0u8; inner_size]);
            self.data_codecs.encode(
                data,
                &ChunkRepresentation::new(data_shape, (**inner_type).clone(), fill_value)?,
                options,
            )?
        } else {
            RawBytes::from(vec![])
        };

        // Concatenate: [mask_len (u64) | data_len (u64) | mask | data]
        let mut result = Vec::new();
        result.extend_from_slice(&(encoded_mask.len() as u64).to_le_bytes());
        result.extend_from_slice(&(encoded_data.len() as u64).to_le_bytes());
        result.extend_from_slice(&encoded_mask);
        result.extend_from_slice(&encoded_data);

        Ok(RawBytes::from(result))
    }

    fn decode<'a>(
        &self,
        bytes: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        // Extract mask length and data length from header
        if bytes.len() < 2 * size_of::<u64>() {
            return Err(InvalidBytesLengthError::new(bytes.len(), 2 * size_of::<u64>()).into());
        }

        let mask_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;

        if bytes.len() != 16 + mask_len + data_len {
            return Err(InvalidBytesLengthError::new(bytes.len(), 16 + mask_len + data_len).into());
        }

        let encoded_mask = &bytes[16..16 + mask_len];
        let encoded_data = &bytes[16 + mask_len..];

        // Decode mask
        let mask_representation =
            ChunkRepresentation::new(decoded_representation.shape().to_vec(), DataType::Bool, 0u8)?;

        let decoded_mask =
            self.mask_codecs
                .decode(encoded_mask.into(), &mask_representation, options)?;
        let mask = decoded_mask.into_fixed()?;

        // Decode data
        let DataType::Optional(inner_type) = decoded_representation.data_type() else {
            return Err(CodecError::Other(
                "optional codec requires an optional data type".to_string(),
            ));
        };

        #[allow(clippy::if_not_else)]
        let decoded_data = if !encoded_data.is_empty() {
            // We need to decode the data to get its actual length
            let valid_count = mask.iter().filter(|&&v| v != 0).count();
            let data_shape = if valid_count > 0 {
                vec![std::num::NonZeroU64::try_from(valid_count as u64).unwrap()]
            } else {
                vec![std::num::NonZeroU64::new(1).unwrap()]
            };
            let inner_size = inner_type.fixed_size().unwrap();
            // Create a zero-filled fill value of the correct size for the inner type
            let fill_value = crate::array::FillValue::new(vec![0u8; inner_size]);
            self.data_codecs.decode(
                encoded_data.into(),
                &ChunkRepresentation::new(data_shape, (**inner_type).clone(), fill_value)?,
                options,
            )?
        } else {
            ArrayBytes::from(Vec::<u8>::new())
        };

        // Reconstruct optional data in legacy format for compatibility with array operations
        let result = Self::reconstruct_optional_data_legacy(
            &mask,
            &decoded_data.into_fixed()?,
            decoded_representation.data_type(),
        )?;

        Ok(ArrayBytes::Fixed(result.into()))
    }

    fn partial_decoder(
        self: Arc<Self>,
        _input_handle: Arc<dyn crate::array::codec::BytesPartialDecoderTraits>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Err(CodecError::Other(
            "partial decoding is not supported for the optional codec".to_string(),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        _input_handle: Arc<dyn crate::array::codec::AsyncBytesPartialDecoderTraits>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Err(CodecError::Other(
            "async partial decoding is not supported for the optional codec".to_string(),
        ))
    }

    fn encoded_representation(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        // Variable size representation
        Ok(BytesRepresentation::BoundedSize(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{
        codec::{ArrayToBytesCodecTraits, CodecOptions, CodecTraits},
        ArrayBytes, ChunkRepresentation, DataType,
    };

    use super::*;

    #[test]
    fn codec_optional_configuration() {
        let codec_configuration: OptionalCodecConfiguration = serde_json::from_str(
            r#"{
                    "mask_codecs": [{"name": "packbits", "configuration": {}}],
                    "data_codecs": [{"name": "bytes", "configuration": {}}]
                }"#,
        )
        .unwrap();
        let codec = OptionalCodec::new_with_configuration(&codec_configuration).unwrap();
        let configuration = codec.configuration_opt(OPTIONAL, &CodecMetadataOptions::default());
        assert!(configuration.is_some());
    }

    fn codec_optional_round_trip_impl(
        data_type: DataType,
        fill_value: impl Into<crate::array::FillValue>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::num::NonZeroU64;

        let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(4).unwrap()];
        let chunk_representation = unsafe {
            // SAFETY: We control the data type and fill value
            ChunkRepresentation::new_unchecked(chunk_shape, data_type, fill_value)
        };

        // Create test data with some missing (invalid) elements
        // Format: [data bytes][mask byte]...
        let inner_size = match &chunk_representation.data_type() {
            DataType::Optional(inner) => inner.fixed_size().unwrap(),
            _ => panic!("Expected Optional data type"),
        };
        let num_elements = chunk_representation.num_elements_usize();

        let mut bytes = Vec::new();
        for i in 0..num_elements {
            // Every third element is invalid
            let is_valid = i % 3 != 0;

            if is_valid {
                // Add some test data first
                for j in 0..inner_size {
                    bytes.push((i + j) as u8);
                }
            } else {
                // Add zeros for invalid elements first
                bytes.extend_from_slice(&vec![0u8; inner_size]);
            }
            // Then add the mask byte
            bytes.push(u8::from(is_valid));
        }

        // Determine if we need endianness in the bytes codec configuration
        let data_codecs_config = match &chunk_representation.data_type() {
            DataType::Optional(inner) if inner.fixed_size().unwrap() > 1 => {
                r#"[{"name": "bytes", "configuration": {"endian": "little"}}]"#
            }
            _ => r#"[{"name": "bytes", "configuration": {}}]"#,
        };

        let codec_configuration: OptionalCodecConfiguration = serde_json::from_str(&format!(
            r#"{{
                    "mask_codecs": [{{"name": "packbits", "configuration": {{}}}}],
                    "data_codecs": {}
                }}"#,
            data_codecs_config
        ))
        .unwrap();
        let codec = OptionalCodec::new_with_configuration(&codec_configuration)?;

        let encoded = codec.encode(
            ArrayBytes::from(bytes.clone()),
            &chunk_representation,
            &CodecOptions::default(),
        )?;
        let decoded = codec.decode(encoded, &chunk_representation, &CodecOptions::default())?;

        // The codec now returns Fixed array bytes in legacy format
        let decoded_bytes = decoded.into_fixed()?;
        assert_eq!(bytes, decoded_bytes.as_ref());
        Ok(())
    }

    #[test]
    fn codec_optional_round_trip_u8() {
        codec_optional_round_trip_impl(
            DataType::Optional(Box::new(DataType::UInt8)),
            crate::array::FillValue::new_null(), // null/missing value
        )
        .unwrap();
    }

    #[test]
    fn codec_optional_round_trip_i32() {
        codec_optional_round_trip_impl(
            DataType::Optional(Box::new(DataType::Int32)),
            crate::array::FillValue::new_null(), // null/missing value
        )
        .unwrap();
    }

    #[test]
    fn codec_optional_round_trip_f32() {
        codec_optional_round_trip_impl(
            DataType::Optional(Box::new(DataType::Float32)),
            crate::array::FillValue::new_null(), // null/missing value
        )
        .unwrap();
    }
}

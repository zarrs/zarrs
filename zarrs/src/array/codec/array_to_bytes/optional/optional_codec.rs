#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]

use std::{mem::size_of, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::OPTIONAL;

use crate::array::{
    codec::{
        ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecChain,
        CodecError, CodecMetadataOptions, CodecOptions, CodecTraits, InvalidBytesLengthError,
        PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    },
    ArrayBytes, BytesRepresentation, ChunkRepresentation, DataType, RawBytes, RawBytesOffsets,
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
        let DataType::Optional(_inner_type) = data_type else {
            return Err(CodecError::Other(
                "optional codec requires an optional data type".to_string(),
            ));
        };

        if let Some(mask) = &bytes.mask {
            if mask.len() != num_elements {
                return Err(CodecError::Other(
                    "mask length does not match number of elements".to_string(),
                ));
            }

            // Extract data without mask
            let data = ArrayBytes {
                data: bytes.data.clone(),
                offsets: bytes.offsets.clone(),
                mask: None,
            };

            Ok((mask.to_vec(), data))
        } else {
            Err(CodecError::Other(
                "expected optional array bytes for optional codec".to_string(),
            ))
        }
    }
}

impl CodecTraits for OptionalCodec {
    fn identifier(&self) -> &str {
        OPTIONAL
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = OptionalCodecConfiguration::V1(OptionalCodecConfigurationV1 {
            mask_codecs: self.mask_codecs.create_metadatas(),
            data_codecs: self.data_codecs.create_metadatas(),
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

        // Extract mask and dense data
        let (mask, dense_data) = Self::extract_optional_array_bytes(
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
        let encoded_mask = self.mask_codecs.encode(
            ArrayBytes::from(mask.clone()),
            &mask_representation,
            options,
        )?;

        // Convert dense data to sparse data (extract only valid elements)
        let inner_size = inner_type.fixed_size().unwrap();

        // Check for nested optional (not supported)
        if dense_data.mask.is_some() {
            return Err(CodecError::Other(
                "nested optional data types are not supported".to_string(),
            ));
        }

        let sparse_data = match &dense_data.offsets {
            None => {
                // Fixed-length: Extract only valid elements based on mask
                let mut sparse_bytes = Vec::new();
                for (i, &mask_byte) in mask.iter().enumerate() {
                    if mask_byte != 0 {
                        let start = i * inner_size;
                        let end = start + inner_size;
                        sparse_bytes.extend_from_slice(&dense_data.data[start..end]);
                    }
                }
                ArrayBytes::new_flen(sparse_bytes)
            }
            Some(offsets) => {
                // Variable-length: Extract only valid elements based on mask
                let mut sparse_bytes = Vec::new();
                let mut sparse_offsets = Vec::new();
                sparse_offsets.push(0);

                for (i, &mask_byte) in mask.iter().enumerate() {
                    if mask_byte != 0 {
                        let start = offsets[i];
                        let end = offsets[i + 1];
                        sparse_bytes.extend_from_slice(&dense_data.data[start..end]);
                        sparse_offsets.push(sparse_bytes.len());
                    }
                }

                let sparse_offsets = unsafe { RawBytesOffsets::new_unchecked(sparse_offsets) };
                unsafe { ArrayBytes::new_vlen_unchecked(sparse_bytes, sparse_offsets) }
            }
        };

        // Encode sparse data
        let num_valid = mask.iter().filter(|&&v| v != 0).count();
        let encoded_data = if num_valid > 0 {
            let data_shape = vec![std::num::NonZeroU64::try_from(num_valid as u64).unwrap()];
            // Create a zero-filled fill value of the correct size for the inner type
            let fill_value = crate::array::FillValue::new(vec![0u8; inner_size]);
            self.data_codecs.encode(
                sparse_data,
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
        let mask = decoded_mask.into_fixed()?.into_owned();

        // Decode data
        let DataType::Optional(inner_type) = decoded_representation.data_type() else {
            return Err(CodecError::Other(
                "optional codec requires an optional data type".to_string(),
            ));
        };

        let num_elements = decoded_representation.num_elements_usize();
        let inner_size = inner_type.fixed_size().unwrap();

        // Decode sparse data (only valid elements)
        let valid_count = mask.iter().filter(|&&v| v != 0).count();
        let sparse_data = {
            let data_shape = vec![std::num::NonZeroU64::try_from(valid_count as u64).unwrap()];
            let fill_value = {
                if decoded_representation.fill_value().is_null() {
                    // Create a zero-filled fill value of the correct size for the inner type
                    crate::array::FillValue::new(vec![0u8; inner_size])
                } else {
                    decoded_representation.fill_value().clone()
                }
            };
            self.data_codecs
                .decode(
                    encoded_data.into(),
                    &ChunkRepresentation::new(data_shape, (**inner_type).clone(), fill_value)?,
                    options,
                )?
                .into_owned()
        };

        // Check for nested optional (not supported)
        if sparse_data.mask.is_some() {
            return Err(CodecError::Other(
                "nested optional data types are not supported".to_string(),
            ));
        }

        // Expand sparse data to dense format
        let dense_data = match &sparse_data.offsets {
            None => {
                // Fixed-length: Create dense array with placeholders for invalid elements
                let mut dense_bytes = vec![0u8; num_elements * inner_size];
                let mut sparse_idx = 0;

                for (i, &mask_byte) in mask.iter().enumerate() {
                    if mask_byte != 0 {
                        // Copy valid element from sparse data
                        let dense_start = i * inner_size;
                        let dense_end = dense_start + inner_size;
                        let sparse_start = sparse_idx * inner_size;
                        let sparse_end = sparse_start + inner_size;
                        dense_bytes[dense_start..dense_end]
                            .copy_from_slice(&sparse_data.data[sparse_start..sparse_end]);
                        sparse_idx += 1;
                    }
                    // Invalid elements remain as zero placeholders
                }

                ArrayBytes::new_flen(dense_bytes)
            }
            Some(sparse_offsets) => {
                // Variable-length: Create dense array with placeholders for invalid elements
                let mut dense_bytes = Vec::new();
                let mut dense_offsets = Vec::with_capacity(num_elements + 1);
                dense_offsets.push(0);

                let mut sparse_idx = 0;

                for &mask_byte in &mask {
                    if mask_byte != 0 {
                        // Copy valid element from sparse data
                        let sparse_start = sparse_offsets[sparse_idx];
                        let sparse_end = sparse_offsets[sparse_idx + 1];
                        dense_bytes.extend_from_slice(&sparse_data.data[sparse_start..sparse_end]);
                        sparse_idx += 1;
                    }
                    // For invalid elements, just add current offset (empty element)
                    dense_offsets.push(dense_bytes.len());
                }

                let dense_offsets = unsafe { RawBytesOffsets::new_unchecked(dense_offsets) };
                unsafe { ArrayBytes::new_vlen_unchecked(dense_bytes, dense_offsets) }
            }
        };

        // Return ArrayBytes with mask and dense data
        Ok(ArrayBytes {
            data: dense_data.data,
            offsets: dense_data.offsets,
            mask: Some(mask.into()),
        })
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
        let inner_size = match &chunk_representation.data_type() {
            DataType::Optional(inner) => inner.fixed_size().unwrap(),
            _ => panic!("Expected Optional data type"),
        };
        let num_elements = chunk_representation.num_elements_usize();

        // Build mask and dense data (all elements, including placeholders for invalid)
        let mut mask = Vec::new();
        let mut data = Vec::new();

        for i in 0..num_elements {
            // Every third element is invalid
            let is_valid = i % 3 != 0;
            mask.push(u8::from(is_valid));

            // Add data for all elements (dense format)
            for j in 0..inner_size {
                if is_valid {
                    data.push((i + j) as u8);
                } else {
                    // Placeholder for invalid elements
                    data.push(0u8);
                }
            }
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

        // Create ArrayBytes with mask for input
        let input = ArrayBytes {
            data: data.clone().into(),
            offsets: None,
            mask: Some(mask.clone().into()),
        };

        let encoded = codec.encode(input, &chunk_representation, &CodecOptions::default())?;
        let decoded = codec.decode(encoded, &chunk_representation, &CodecOptions::default())?;

        // The codec now returns optional ArrayBytes
        assert!(decoded.mask.is_some());
        assert!(decoded.offsets.is_none());
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

#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]

use std::{mem::size_of, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::OPTIONAL;

use crate::array::{
    codec::{
        ArrayCodecTraits, ArrayToBytesCodecTraits, CodecChain, CodecError, CodecMetadataOptions,
        CodecOptions, CodecTraits, InvalidBytesLengthError, PartialDecoderCapability,
        PartialEncoderCapability, RecommendedConcurrency,
    },
    ArrayBytes, BytesRepresentation, ChunkRepresentation, DataType, RawBytes, RawBytesOffsets,
};

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
    ) -> Result<(&'a ArrayBytes<'a>, &'a [u8]), CodecError> {
        let DataType::Optional(_inner_type) = data_type else {
            return Err(CodecError::Other(
                "optional codec requires an optional data type".to_string(),
            ));
        };

        // Match on the ArrayBytes variant
        if let ArrayBytes::Optional(data, mask) = bytes {
            if mask.len() != num_elements {
                return Err(CodecError::Other(
                    "mask length does not match number of elements".to_string(),
                ));
            }

            // Return references to the mask and data
            Ok((data.as_ref(), mask.as_ref()))
        } else {
            Err(CodecError::Other(
                "expected optional array bytes for optional codec".to_string(),
            ))
        }
    }

    /// Recursively extract sparse data from dense data based on a validity mask.
    /// This supports arbitrarily nested optional types.
    fn extract_sparse_data<'a>(
        dense_data: &'a ArrayBytes<'a>,
        mask: &[u8],
        inner_type: &DataType,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        match dense_data {
            ArrayBytes::Fixed(data) => {
                // Fixed-length: Extract only valid elements based on mask
                let inner_size = inner_type.fixed_size().unwrap();
                let mut sparse_bytes = Vec::new();
                for (i, &mask_byte) in mask.iter().enumerate() {
                    if mask_byte != 0 {
                        let start = i * inner_size;
                        let end = start + inner_size;
                        sparse_bytes.extend_from_slice(&data[start..end]);
                    }
                }
                Ok(ArrayBytes::new_flen(sparse_bytes))
            }
            ArrayBytes::Variable(data, offsets) => {
                // Variable-length: Extract only valid elements based on mask
                let mut sparse_bytes = Vec::new();
                let mut sparse_offsets = Vec::new();
                sparse_offsets.push(0);

                for (i, &mask_byte) in mask.iter().enumerate() {
                    if mask_byte != 0 {
                        let start = offsets[i];
                        let end = offsets[i + 1];
                        sparse_bytes.extend_from_slice(&data[start..end]);
                        sparse_offsets.push(sparse_bytes.len());
                    }
                }

                let sparse_offsets = unsafe { RawBytesOffsets::new_unchecked(sparse_offsets) };
                Ok(unsafe { ArrayBytes::new_vlen_unchecked(sparse_bytes, sparse_offsets) })
            }
            ArrayBytes::Optional(inner_data, inner_mask) => {
                // Nested optional: Extract valid elements from both outer and inner masks
                // Only elements where outer mask is valid should be included
                // For those elements, preserve their inner mask values

                // Extract the sparse inner mask (only for valid outer elements)
                let mut sparse_inner_mask = Vec::new();
                for (i, &mask_byte) in mask.iter().enumerate() {
                    if mask_byte != 0 {
                        sparse_inner_mask.push(inner_mask[i]);
                    }
                }

                // Get the inner-inner type (unwrap the Optional)
                let inner_inner_type = if let DataType::Optional(t) = inner_type {
                    t.as_ref()
                } else {
                    return Err(CodecError::Other(
                        "nested optional ArrayBytes requires nested optional DataType".to_string(),
                    ));
                };

                // Recursively extract sparse data from the inner data
                let sparse_inner_data =
                    Self::extract_sparse_data(inner_data, mask, inner_inner_type)?;

                Ok(sparse_inner_data.with_optional_mask(sparse_inner_mask))
            }
        }
    }

    /// Recursively expand sparse data to dense format based on a validity mask.
    /// This supports arbitrarily nested optional types and is the inverse of `extract_sparse_data`.
    fn expand_to_dense(
        sparse_data: ArrayBytes<'_>,
        mask: &[u8],
        inner_type: &DataType,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        match sparse_data {
            ArrayBytes::Fixed(sparse_bytes) => {
                // Fixed-length: Create dense array with placeholders for invalid elements
                let inner_size = inner_type.fixed_size().unwrap();
                let num_elements = mask.len();
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
                            .copy_from_slice(&sparse_bytes[sparse_start..sparse_end]);
                        sparse_idx += 1;
                    }
                    // Invalid elements remain as zero placeholders
                }

                Ok(ArrayBytes::new_flen(dense_bytes))
            }
            ArrayBytes::Variable(sparse_bytes, sparse_offsets) => {
                // Variable-length: Create dense array with placeholders for invalid elements
                let num_elements = mask.len();
                let mut dense_bytes = Vec::new();
                let mut dense_offsets = Vec::with_capacity(num_elements + 1);
                dense_offsets.push(0);

                let mut sparse_idx = 0;

                for &mask_byte in mask {
                    if mask_byte != 0 {
                        // Copy valid element from sparse data
                        let sparse_start = sparse_offsets[sparse_idx];
                        let sparse_end = sparse_offsets[sparse_idx + 1];
                        dense_bytes.extend_from_slice(&sparse_bytes[sparse_start..sparse_end]);
                        sparse_idx += 1;
                    }
                    // For invalid elements, just add current offset (empty element)
                    dense_offsets.push(dense_bytes.len());
                }

                let dense_offsets = unsafe { RawBytesOffsets::new_unchecked(dense_offsets) };
                Ok(unsafe { ArrayBytes::new_vlen_unchecked(dense_bytes, dense_offsets) })
            }
            ArrayBytes::Optional(sparse_inner_data, sparse_inner_mask) => {
                // Nested optional: Expand the inner data and then expand the inner mask

                // Get the inner-inner type (unwrap the Optional)
                let inner_inner_type = if let DataType::Optional(t) = inner_type {
                    t.as_ref()
                } else {
                    return Err(CodecError::Other(
                        "nested optional ArrayBytes requires nested optional DataType".to_string(),
                    ));
                };

                // Recursively expand the inner data
                let dense_inner_data =
                    Self::expand_to_dense(*sparse_inner_data, mask, inner_inner_type)?;

                // Expand the sparse inner mask to dense format
                let num_elements = mask.len();
                let mut dense_inner_mask = vec![0u8; num_elements];
                let mut sparse_idx = 0;

                for (i, &mask_byte) in mask.iter().enumerate() {
                    if mask_byte != 0 {
                        dense_inner_mask[i] = sparse_inner_mask[sparse_idx];
                        sparse_idx += 1;
                    }
                    // Invalid elements have mask value 0
                }

                Ok(dense_inner_data.with_optional_mask(dense_inner_mask))
            }
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
        let (dense_data, mask) = Self::extract_optional_array_bytes(
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

        // Convert dense data to sparse data (extract only valid elements)
        // This supports arbitrarily nested optional types
        let sparse_data = Self::extract_sparse_data(dense_data, mask, inner_type)?;

        // Encode sparse data
        let num_valid = mask.iter().filter(|&&v| v != 0).count();
        let encoded_data = if num_valid > 0 {
            let data_shape = vec![std::num::NonZeroU64::try_from(num_valid as u64).unwrap()];
            // Create a zero-filled fill value of the correct size for the inner type
            // For nested optional types, the fill value is null
            let fill_value = if inner_type.is_optional() {
                crate::array::FillValue::new_null()
            } else {
                let inner_size = inner_type.fixed_size().unwrap();
                crate::array::FillValue::new(vec![0u8; inner_size])
            };
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

        // Decode sparse data (only valid elements)
        let valid_count = mask.iter().filter(|&&v| v != 0).count();
        let sparse_data = {
            let data_shape = vec![std::num::NonZeroU64::try_from(valid_count as u64).unwrap()];
            let fill_value = {
                if inner_type.is_optional() {
                    // For nested optional types, use null
                    crate::array::FillValue::new_null()
                } else {
                    // Create a zero-filled fill value of the correct size for the inner type
                    let inner_size = inner_type.fixed_size().unwrap();
                    crate::array::FillValue::new(vec![0u8; inner_size])
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

        // Expand sparse data to dense format (supports nested optional types)
        let dense_data = Self::expand_to_dense(sparse_data, &mask, inner_type)?;

        // Return ArrayBytes with mask and dense data
        Ok(dense_data.with_optional_mask(mask))
    }

    // fn partial_decoder(
    //     self: Arc<Self>,
    //     _input_handle: Arc<dyn crate::array::codec::BytesPartialDecoderTraits>,
    //     _decoded_representation: &ChunkRepresentation,
    //     _options: &CodecOptions,
    // ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
    //     Err(CodecError::Other(
    //         "partial decoding is not supported for the optional codec".to_string(),
    //     ))
    // }

    // #[cfg(feature = "async")]
    // async fn async_partial_decoder(
    //     self: Arc<Self>,
    //     _input_handle: Arc<dyn crate::array::codec::AsyncBytesPartialDecoderTraits>,
    //     _decoded_representation: &ChunkRepresentation,
    //     _options: &CodecOptions,
    // ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
    //     Err(CodecError::Other(
    //         "async partial decoding is not supported for the optional codec".to_string(),
    //     ))
    // }

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

    /// Helper to build codec config recursively for nested optional types
    fn build_codec_config_for_type(data_type: &DataType) -> String {
        match data_type {
            DataType::Optional(inner) if inner.is_optional() => {
                // Nested optional - need another optional codec
                let inner_config = build_codec_config_for_type(inner.as_ref());
                format!(
                    r#"[{{"name": "optional", "configuration": {{
                        "mask_codecs": [{{"name": "packbits", "configuration": {{}}}}],
                        "data_codecs": {}
                    }}}}]"#,
                    inner_config
                )
            }
            DataType::Optional(inner) => {
                // Non-nested optional inner type - use bytes codec
                if inner.fixed_size().unwrap() > 1 {
                    r#"[{"name": "bytes", "configuration": {"endian": "little"}}]"#.to_string()
                } else {
                    r#"[{"name": "bytes", "configuration": {}}]"#.to_string()
                }
            }
            _ => {
                // Non-optional type
                if data_type.fixed_size().unwrap() > 1 {
                    r#"[{"name": "bytes", "configuration": {"endian": "little"}}]"#.to_string()
                } else {
                    r#"[{"name": "bytes", "configuration": {}}]"#.to_string()
                }
            }
        }
    }

    /// Helper to build nested ArrayBytes for testing
    fn build_nested_array_bytes(data_type: &DataType, num_elements: usize) -> ArrayBytes<'_> {
        match data_type {
            DataType::Optional(inner) if inner.is_optional() => {
                // Build nested optional structure
                let inner_array_bytes = build_nested_array_bytes(inner.as_ref(), num_elements);

                // Create outer mask (every third element is invalid at this level)
                let outer_mask: Vec<u8> = (0..num_elements).map(|i| u8::from(i % 3 != 0)).collect();

                ArrayBytes::Optional(Box::new(inner_array_bytes), outer_mask.into())
            }
            DataType::Optional(inner) => {
                // Innermost optional level - create data and mask
                let inner_size = inner.fixed_size().unwrap();
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

                ArrayBytes::Optional(Box::new(ArrayBytes::new_flen(data)), mask.into())
            }
            _ => {
                // Non-optional type - shouldn't reach here in these tests
                panic!("Expected Optional data type");
            }
        }
    }

    fn codec_optional_round_trip_impl(
        data_type: DataType,
        fill_value: impl Into<crate::array::FillValue>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::num::NonZeroU64;

        let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(4).unwrap()];
        let chunk_representation = unsafe {
            // SAFETY: We control the data type and fill value
            ChunkRepresentation::new_unchecked(chunk_shape, data_type.clone(), fill_value)
        };

        let num_elements = chunk_representation.num_elements_usize();

        // Build codec configuration recursively for nested optional types
        let data_codecs_config = build_codec_config_for_type(&data_type);

        let codec_configuration: OptionalCodecConfiguration = serde_json::from_str(&format!(
            r#"{{
                    "mask_codecs": [{{"name": "packbits", "configuration": {{}}}}],
                    "data_codecs": {}
                }}"#,
            data_codecs_config
        ))
        .unwrap();
        let codec = OptionalCodec::new_with_configuration(&codec_configuration)?;

        // Build nested ArrayBytes structure for input
        let input = build_nested_array_bytes(&data_type, num_elements);

        let encoded = codec.encode(input, &chunk_representation, &CodecOptions::default())?;
        let decoded = codec.decode(encoded, &chunk_representation, &CodecOptions::default())?;

        // The codec now returns optional ArrayBytes
        assert!(matches!(decoded, ArrayBytes::Optional(_, _)));
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

    #[test]
    fn codec_optional_round_trip_nested_2_level() {
        // Test Option<Option<u8>> with null fill value
        codec_optional_round_trip_impl(
            DataType::Optional(Box::new(DataType::Optional(Box::new(DataType::UInt8)))),
            crate::array::FillValue::new_null(), // null/missing value for outer optional
        )
        .unwrap();
    }

    #[test]
    fn codec_optional_round_trip_nested_2_level_i32() {
        // Test Option<Option<i32>> with null fill value
        codec_optional_round_trip_impl(
            DataType::Optional(Box::new(DataType::Optional(Box::new(DataType::Int32)))),
            crate::array::FillValue::new_null(), // null/missing value for outer optional
        )
        .unwrap();
    }

    #[test]
    fn codec_optional_round_trip_nested_3_level() {
        // Test Option<Option<Option<u8>>> with null fill value
        codec_optional_round_trip_impl(
            DataType::Optional(Box::new(DataType::Optional(Box::new(DataType::Optional(
                Box::new(DataType::UInt8),
            ))))),
            crate::array::FillValue::new_null(), // null/missing value for outer optional
        )
        .unwrap();
    }

    #[test]
    fn codec_optional_round_trip_nested_3_level_f64() {
        // Test Option<Option<Option<f64>>> with null fill value
        codec_optional_round_trip_impl(
            DataType::Optional(Box::new(DataType::Optional(Box::new(DataType::Optional(
                Box::new(DataType::Float64),
            ))))),
            crate::array::FillValue::new_null(), // null/missing value for outer optional
        )
        .unwrap();
    }

    #[test]
    fn codec_optional_nested_2_level_detailed() {
        use std::num::NonZeroU64;

        // Test Option<Option<u8>> with explicit mask construction
        let data_type = DataType::Optional(Box::new(DataType::Optional(Box::new(DataType::UInt8))));
        let chunk_shape = vec![NonZeroU64::new(8).unwrap()];
        let chunk_representation = unsafe {
            ChunkRepresentation::new_unchecked(
                chunk_shape,
                data_type,
                crate::array::FillValue::new_null(),
            )
        };

        // Create test data:
        // Element 0: Some(Some(10))  - outer valid=1, inner valid=1, data=10
        // Element 1: Some(None)      - outer valid=1, inner valid=0, data=0 (placeholder)
        // Element 2: None            - outer valid=0, inner valid=0 (placeholder), data=0
        // Element 3: Some(Some(30))  - outer valid=1, inner valid=1, data=30
        // Element 4: None            - outer valid=0, inner valid=0, data=0
        // Element 5: Some(Some(50))  - outer valid=1, inner valid=1, data=50
        // Element 6: Some(None)      - outer valid=1, inner valid=0, data=0
        // Element 7: Some(Some(70))  - outer valid=1, inner valid=1, data=70

        let outer_mask = vec![1u8, 1, 0, 1, 0, 1, 1, 1];
        let inner_mask = vec![1u8, 0, 0, 1, 0, 1, 0, 1]; // Dense: one for each element
        let data = vec![10u8, 0, 0, 30, 0, 50, 0, 70]; // Dense: one for each element

        // Construct nested optional ArrayBytes
        let inner_optional = ArrayBytes::new_flen(data).with_optional_mask(inner_mask);
        let input = inner_optional.with_optional_mask(outer_mask);

        // For nested optional types, we need nested optional codecs
        let codec_configuration: OptionalCodecConfiguration = serde_json::from_str(
            r#"{
                "mask_codecs": [{"name": "packbits", "configuration": {}}],
                "data_codecs": [{
                    "name": "optional",
                    "configuration": {
                        "mask_codecs": [{"name": "packbits", "configuration": {}}],
                        "data_codecs": [{"name": "bytes", "configuration": {}}]
                    }
                }]
            }"#,
        )
        .unwrap();
        let codec = OptionalCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                input.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();

        // Verify the decoded structure
        if let ArrayBytes::Optional(decoded_inner_box, decoded_outer_mask) = decoded {
            // Check outer mask
            assert_eq!(decoded_outer_mask.as_ref(), &[1u8, 1, 0, 1, 0, 1, 1, 1]);

            // Check inner optional structure
            if let ArrayBytes::Optional(decoded_data_box, decoded_inner_mask) = *decoded_inner_box {
                assert_eq!(decoded_inner_mask.as_ref(), &[1u8, 0, 0, 1, 0, 1, 0, 1]);

                // Check data
                if let ArrayBytes::Fixed(decoded_data) = *decoded_data_box {
                    assert_eq!(decoded_data.as_ref(), &[10u8, 0, 0, 30, 0, 50, 0, 70]);
                } else {
                    panic!("Expected Fixed ArrayBytes for innermost data");
                }
            } else {
                panic!("Expected Optional ArrayBytes for inner level");
            }
        } else {
            panic!("Expected Optional ArrayBytes for outer level");
        }
    }

    #[test]
    fn codec_optional_nested_2_level_with_inner_fill_value() {
        use std::num::NonZeroU64;

        // Test Option<u8> where the u8 has a non-zero fill value
        // This represents the outer optional wrapping a non-optional type
        let data_type = DataType::Optional(Box::new(DataType::UInt8));
        let chunk_shape = vec![NonZeroU64::new(6).unwrap()];
        let chunk_representation = unsafe {
            // Use a non-null fill value of 255 for missing elements
            ChunkRepresentation::new_unchecked(
                chunk_shape,
                data_type,
                crate::array::FillValue::new(vec![255u8]),
            )
        };

        // Create test data:
        // Element 0: Some(10)  - valid=1, data=10
        // Element 1: None      - valid=0, data=255 (fill value)
        // Element 2: Some(20)  - valid=1, data=20
        // Element 3: None      - valid=0, data=255
        // Element 4: Some(30)  - valid=1, data=30
        // Element 5: None      - valid=0, data=255

        let mask = vec![1u8, 0, 1, 0, 1, 0];
        let data = vec![10u8, 255, 20, 255, 30, 255];

        let input = ArrayBytes::new_flen(data).with_optional_mask(mask);

        let codec_configuration: OptionalCodecConfiguration = serde_json::from_str(
            r#"{
                "mask_codecs": [{"name": "packbits", "configuration": {}}],
                "data_codecs": [{"name": "bytes", "configuration": {}}]
            }"#,
        )
        .unwrap();
        let codec = OptionalCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                input.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();

        // Verify the decoded structure
        if let ArrayBytes::Optional(decoded_data_box, decoded_mask) = decoded {
            assert_eq!(decoded_mask.as_ref(), &[1u8, 0, 1, 0, 1, 0]);
            if let ArrayBytes::Fixed(decoded_data) = *decoded_data_box {
                // Note: The fill value is used as placeholders during encoding/decoding
                // but the exact values for invalid elements might be 0 from the sparse encoding
                // Let's verify valid elements are preserved
                let data_slice = decoded_data.as_ref();
                let mask_slice = decoded_mask.as_ref();
                assert_eq!(data_slice[0], 10u8); // valid element
                assert_eq!(data_slice[2], 20u8); // valid element
                assert_eq!(data_slice[4], 30u8); // valid element
                                                 // Invalid elements (indices 1, 3, 5) can be 0 or fill value depending on implementation
                assert_eq!(mask_slice[1], 0u8); // invalid
                assert_eq!(mask_slice[3], 0u8); // invalid
                assert_eq!(mask_slice[5], 0u8); // invalid
            } else {
                panic!("Expected Fixed ArrayBytes");
            }
        } else {
            panic!("Expected Optional ArrayBytes");
        }
    }

    #[test]
    fn codec_optional_nested_3_level_detailed() {
        use std::num::NonZeroU64;

        // Test Option<Option<Option<u16>>> with explicit mask construction
        let data_type = DataType::Optional(Box::new(DataType::Optional(Box::new(
            DataType::Optional(Box::new(DataType::UInt16)),
        ))));
        let chunk_shape = vec![NonZeroU64::new(6).unwrap()];
        let chunk_representation = unsafe {
            ChunkRepresentation::new_unchecked(
                chunk_shape,
                data_type,
                crate::array::FillValue::new_null(),
            )
        };

        // Create test data with 3 levels:
        // Element 0: Some(Some(Some(100)))  - outer=1, middle=1, inner=1, data=100
        // Element 1: Some(Some(None))       - outer=1, middle=1, inner=0, data=0
        // Element 2: Some(None)             - outer=1, middle=0, inner=0, data=0
        // Element 3: None                   - outer=0, middle=0, inner=0, data=0
        // Element 4: Some(Some(Some(400)))  - outer=1, middle=1, inner=1, data=400
        // Element 5: Some(Some(Some(500)))  - outer=1, middle=1, inner=1, data=500

        let outer_mask = vec![1u8, 1, 1, 0, 1, 1];
        let middle_mask = vec![1u8, 1, 0, 0, 1, 1];
        let inner_mask = vec![1u8, 0, 0, 0, 1, 1];
        // Data is u16, so 2 bytes per element, little-endian
        let data = vec![
            100u8, 0, // 100
            0, 0, // placeholder
            0, 0, // placeholder
            0, 0, // placeholder
            144, 1, // 400
            244, 1, // 500
        ];

        // Construct 3-level nested optional ArrayBytes
        let innermost = ArrayBytes::new_flen(data).with_optional_mask(inner_mask);
        let middle = innermost.with_optional_mask(middle_mask);
        let input = middle.with_optional_mask(outer_mask);

        // For 3-level nested optional types, we need 3 nested optional codecs
        let codec_configuration: OptionalCodecConfiguration = serde_json::from_str(
            r#"{
                "mask_codecs": [{"name": "packbits", "configuration": {}}],
                "data_codecs": [{
                    "name": "optional",
                    "configuration": {
                        "mask_codecs": [{"name": "packbits", "configuration": {}}],
                        "data_codecs": [{
                            "name": "optional",
                            "configuration": {
                                "mask_codecs": [{"name": "packbits", "configuration": {}}],
                                "data_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
                            }
                        }]
                    }
                }]
            }"#,
        )
        .unwrap();
        let codec = OptionalCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                input.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();

        // Verify the 3-level nested structure
        if let ArrayBytes::Optional(level2_box, outer_mask_decoded) = decoded {
            assert_eq!(outer_mask_decoded.as_ref(), &[1u8, 1, 1, 0, 1, 1]);

            if let ArrayBytes::Optional(level1_box, middle_mask_decoded) = *level2_box {
                assert_eq!(middle_mask_decoded.as_ref(), &[1u8, 1, 0, 0, 1, 1]);

                if let ArrayBytes::Optional(data_box, inner_mask_decoded) = *level1_box {
                    assert_eq!(inner_mask_decoded.as_ref(), &[1u8, 0, 0, 0, 1, 1]);

                    if let ArrayBytes::Fixed(data_decoded) = *data_box {
                        assert_eq!(
                            data_decoded.as_ref(),
                            &[100u8, 0, 0, 0, 0, 0, 0, 0, 144, 1, 244, 1]
                        );
                    } else {
                        panic!("Expected Fixed ArrayBytes for innermost data");
                    }
                } else {
                    panic!("Expected Optional ArrayBytes for level 1");
                }
            } else {
                panic!("Expected Optional ArrayBytes for level 2");
            }
        } else {
            panic!("Expected Optional ArrayBytes for outer level");
        }
    }

    #[test]
    fn codec_optional_nested_with_varying_fill_values() {
        use std::num::NonZeroU64;

        // Test Option<f32> with a specific fill value (e.g., NaN)
        let data_type = DataType::Optional(Box::new(DataType::Float32));
        let chunk_shape = vec![NonZeroU64::new(5).unwrap()];
        let nan_bytes = f32::NAN.to_le_bytes().to_vec();
        let chunk_representation = unsafe {
            ChunkRepresentation::new_unchecked(
                chunk_shape,
                data_type,
                crate::array::FillValue::new(nan_bytes),
            )
        };

        // Create test data with some valid and some invalid f32 values
        let mask = vec![1u8, 0, 1, 1, 0];
        let data = vec![
            1.5f32.to_le_bytes(),
            [0u8; 4], // placeholder
            2.5f32.to_le_bytes(),
            3.5f32.to_le_bytes(),
            [0u8; 4], // placeholder
        ]
        .concat();

        let input = ArrayBytes::new_flen(data).with_optional_mask(mask);

        let codec_configuration: OptionalCodecConfiguration = serde_json::from_str(
            r#"{
                "mask_codecs": [{"name": "packbits", "configuration": {}}],
                "data_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
            }"#,
        )
        .unwrap();
        let codec = OptionalCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                input.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();

        // Verify the decoded structure
        if let ArrayBytes::Optional(decoded_data_box, decoded_mask) = decoded {
            assert_eq!(decoded_mask.as_ref(), &[1u8, 0, 1, 1, 0]);
            if let ArrayBytes::Fixed(decoded_data) = *decoded_data_box {
                let data_slice = decoded_data.as_ref();
                // Check valid elements
                assert_eq!(
                    f32::from_le_bytes([
                        data_slice[0],
                        data_slice[1],
                        data_slice[2],
                        data_slice[3]
                    ]),
                    1.5f32
                );
                assert_eq!(
                    f32::from_le_bytes([
                        data_slice[8],
                        data_slice[9],
                        data_slice[10],
                        data_slice[11]
                    ]),
                    2.5f32
                );
                assert_eq!(
                    f32::from_le_bytes([
                        data_slice[12],
                        data_slice[13],
                        data_slice[14],
                        data_slice[15]
                    ]),
                    3.5f32
                );
            } else {
                panic!("Expected Fixed ArrayBytes");
            }
        } else {
            panic!("Expected Optional ArrayBytes");
        }
    }
}

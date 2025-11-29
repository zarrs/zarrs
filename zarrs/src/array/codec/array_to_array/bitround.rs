//! The `bitround` array to array codec.
//!
//! Round the mantissa of floating point data types to the specified number of bits.
//! Rounds integers from the most significant set bit.
//! Bit rounding leaves an array more amenable to compression.
//!
//! This codec requires the `bitround` feature, which is disabled by default.
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.bitround` codec in `zarr-python`.
//! However, it supports additional data types not supported by that implementation.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/blob/main/codecs/bitround/README.md>
//! - <https://codec.zarrs.dev/array_to_array/bitround>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `bitround`
//! - `numcodecs.bitround`
//! - `https://codec.zarrs.dev/array_to_array/bitround`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `bitround`
//!
//! ### Codec `configuration` Example - [`BitroundCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "keepbits": 10
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::bitround::BitroundCodecConfigurationV1;
//! # let configuration: BitroundCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
//! ```

mod bitround_codec;
mod bitround_codec_partial;

use std::sync::Arc;

pub use bitround_codec::BitroundCodec;

pub use crate::metadata_ext::codec::bitround::{
    BitroundCodecConfiguration, BitroundCodecConfigurationV1,
};
use crate::registry::codec::BITROUND;
use crate::{
    array::{
        codec::{Codec, CodecError, CodecPlugin},
        DataType,
    },
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

macro_rules! supported_dtypes {
    () => {
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::BFloat16
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::ComplexBFloat16
            | DataType::ComplexFloat16
            | DataType::ComplexFloat32
            | DataType::ComplexFloat64
            | DataType::Complex64
            | DataType::Complex128
            | DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            }
    };
}
macro_rules! unsupported_dtypes {
    // TODO: Add support for all int/float types?
    // TODO: Add support for extensions?
    () => {
        DataType::Bool
            | DataType::Int2
            | DataType::Int4
            | DataType::UInt2
            | DataType::UInt4
            | DataType::Float4E2M1FN
            | DataType::Float6E2M3FN
            | DataType::Float6E3M2FN
            | DataType::Float8E3M4
            | DataType::Float8E4M3
            | DataType::Float8E4M3B11FNUZ
            | DataType::Float8E4M3FNUZ
            | DataType::Float8E5M2
            | DataType::Float8E5M2FNUZ
            | DataType::Float8E8M0FNU
            | DataType::ComplexFloat4E2M1FN
            | DataType::ComplexFloat6E2M3FN
            | DataType::ComplexFloat6E3M2FN
            | DataType::ComplexFloat8E3M4
            | DataType::ComplexFloat8E4M3
            | DataType::ComplexFloat8E4M3B11FNUZ
            | DataType::ComplexFloat8E4M3FNUZ
            | DataType::ComplexFloat8E5M2
            | DataType::ComplexFloat8E5M2FNUZ
            | DataType::ComplexFloat8E8M0FNU
            | DataType::RawBits(_)
            | DataType::String
            | DataType::Bytes
            | DataType::Optional(_)
            | DataType::Extension(_)
    };
}

use supported_dtypes;
use unsupported_dtypes;

// Register the codec.
inventory::submit! {
    CodecPlugin::new(BITROUND, is_identifier_bitround, create_codec_bitround)
}

fn is_identifier_bitround(identifier: &str) -> bool {
    identifier == BITROUND
}

pub(crate) fn create_codec_bitround(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: BitroundCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(BITROUND, "codec", metadata.to_string()))?;
    let codec = Arc::new(BitroundCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToArray(codec))
}

fn round_bits8(mut input: u8, keepbits: u32, maxbits: u32) -> u8 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u8::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

const fn round_bits16(mut input: u16, keepbits: u32, maxbits: u32) -> u16 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u16::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

const fn round_bits32(mut input: u32, keepbits: u32, maxbits: u32) -> u32 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u32::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

const fn round_bits64(mut input: u64, keepbits: u32, maxbits: u32) -> u64 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u64::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

fn round_bytes(bytes: &mut [u8], data_type: &DataType, keepbits: u32) -> Result<(), CodecError> {
    match data_type {
        DataType::UInt8 | DataType::Int8 => {
            let round = |element: &mut u8| {
                *element = round_bits8(*element, keepbits, 8 - element.leading_zeros());
            };
            bytes.iter_mut().for_each(round);
            Ok(())
        }
        DataType::Float16
        | DataType::BFloat16
        | DataType::ComplexBFloat16
        | DataType::ComplexFloat16 => {
            let round = |chunk: &mut [u8]| {
                let element = u16::from_ne_bytes(chunk.try_into().unwrap());
                let element = u16::to_ne_bytes(round_bits16(element, keepbits, 10));
                chunk.copy_from_slice(&element);
            };
            bytes.chunks_exact_mut(2).for_each(round);
            Ok(())
        }
        DataType::UInt16 | DataType::Int16 => {
            let round = |chunk: &mut [u8]| {
                let element = u16::from_ne_bytes(chunk.try_into().unwrap());
                let element = u16::to_ne_bytes(round_bits16(
                    element,
                    keepbits,
                    16 - element.leading_zeros(),
                ));
                chunk.copy_from_slice(&element);
            };
            bytes.chunks_exact_mut(2).for_each(round);
            Ok(())
        }
        DataType::Float32 | DataType::Complex64 | DataType::ComplexFloat32 => {
            let round = |chunk: &mut [u8]| {
                let element = u32::from_ne_bytes(chunk.try_into().unwrap());
                let element = u32::to_ne_bytes(round_bits32(element, keepbits, 23));
                chunk.copy_from_slice(&element);
            };
            bytes.chunks_exact_mut(4).for_each(round);
            Ok(())
        }
        DataType::UInt32 | DataType::Int32 => {
            let round = |chunk: &mut [u8]| {
                let element = u32::from_ne_bytes(chunk.try_into().unwrap());
                let element = u32::to_ne_bytes(round_bits32(
                    element,
                    keepbits,
                    32 - element.leading_zeros(),
                ));
                chunk.copy_from_slice(&element);
            };
            bytes.chunks_exact_mut(4).for_each(round);
            Ok(())
        }
        DataType::Float64 | DataType::Complex128 | DataType::ComplexFloat64 => {
            let round = |chunk: &mut [u8]| {
                let element = u64::from_ne_bytes(chunk.try_into().unwrap());
                let element = u64::to_ne_bytes(round_bits64(element, keepbits, 52));
                chunk.copy_from_slice(&element);
            };
            bytes.chunks_exact_mut(8).for_each(round);
            Ok(())
        }
        DataType::UInt64
        | DataType::Int64
        | DataType::NumpyDateTime64 {
            unit: _,
            scale_factor: _,
        }
        | DataType::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _,
        } => {
            let round = |chunk: &mut [u8]| {
                let element = u64::from_ne_bytes(chunk.try_into().unwrap());
                let element = u64::to_ne_bytes(round_bits64(
                    element,
                    keepbits,
                    64 - element.leading_zeros(),
                ));
                chunk.copy_from_slice(&element);
            };
            bytes.chunks_exact_mut(8).for_each(round);
            Ok(())
        }
        unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            BITROUND.to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroU64, sync::Arc};

    use array_representation::ChunkRepresentation;

    use super::*;
    use crate::{
        array::{
            array_representation,
            codec::{ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesCodec, CodecOptions},
            ArrayBytes,
        },
        array_subset::ArraySubset,
    };

    #[test]
    fn codec_bitround_float() {
        // 1 sign bit, 8 exponent, 3 mantissa
        const JSON: &'static str = r#"{ "keepbits": 3 }"#;
        let chunk_representation =
            ChunkRepresentation::new(vec![NonZeroU64::new(4).unwrap()], DataType::Float32, 0.0f32)
                .unwrap();
        let elements: Vec<f32> = vec![
            //                         |
            0.0,
            // 1.23456789 -> 001111111001|11100000011001010010
            // 1.25       -> 001111111010
            1.23456789,
            // -8.3587192 -> 110000010000|01011011110101010000
            // -8.0       -> 110000010000
            -8.3587192834,
            // 98765.43210-> 010001111100|00001110011010110111
            // 98304.0    -> 010001111100
            98765.43210,
        ];
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = BitroundCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<f32>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, &[0.0f32, 1.25f32, -8.0f32, 98304.0f32]);
    }

    #[test]
    fn codec_bitround_uint() {
        const JSON: &'static str = r#"{ "keepbits": 3 }"#;
        let chunk_representation =
            ChunkRepresentation::new(vec![NonZeroU64::new(4).unwrap()], DataType::UInt32, 0u32)
                .unwrap();
        let elements: Vec<u32> = vec![0, 1024, 1280, 1664, 1685, 123145182, 4294967295];
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = BitroundCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<u32>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        for element in &decoded_elements {
            println!("{element} -> {element:#b}");
        }
        assert_eq!(
            decoded_elements,
            &[0, 1024, 1280, 1536, 1792, 117440512, 3758096384]
        );
    }

    #[test]
    fn codec_bitround_uint8() {
        const JSON: &'static str = r#"{ "keepbits": 3 }"#;
        let chunk_representation =
            ChunkRepresentation::new(vec![NonZeroU64::new(4).unwrap()], DataType::UInt8, 0u8)
                .unwrap();
        let elements: Vec<u32> = vec![0, 3, 7, 15, 17, 54, 89, 128, 255];
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = BitroundCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<u32>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        for element in &decoded_elements {
            println!("{element} -> {element:#b}");
        }
        assert_eq!(decoded_elements, &[0, 3, 7, 16, 16, 56, 96, 128, 224]);
    }

    #[test]
    fn codec_bitround_partial_decode() {
        const JSON: &'static str = r#"{ "keepbits": 2 }"#;
        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = Arc::new(BitroundCodec::new_with_configuration(&codec_configuration).unwrap());

        let elements: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let chunk_representation = ChunkRepresentation::new(
            vec![(elements.len() as u64).try_into().unwrap()],
            DataType::Float32,
            0.0f32,
        )
        .unwrap();
        let bytes: ArrayBytes = crate::array::transmute_to_bytes_vec(elements).into();

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_owned();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // bitround partial decoder does not hold bytes
        let decoded_regions = [
            ArraySubset::new_with_ranges(&[3..5]),
            ArraySubset::new_with_ranges(&[17..21]),
        ];
        let answer: &[Vec<f32>] = &[vec![3.0, 4.0], vec![16.0, 16.0, 20.0, 20.0]];
        for (decoded_region, expected) in decoded_regions.into_iter().zip(answer.iter()) {
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .unwrap();
            let decoded_partial_chunk = crate::array::convert_from_bytes_slice::<f32>(
                &decoded_partial_chunk.into_fixed().unwrap(),
            );
            assert_eq!(expected, &decoded_partial_chunk);
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_bitround_async_partial_decode() {
        const JSON: &'static str = r#"{ "keepbits": 2 }"#;
        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = Arc::new(BitroundCodec::new_with_configuration(&codec_configuration).unwrap());

        let elements: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let chunk_representation = ChunkRepresentation::new(
            vec![(elements.len() as u64).try_into().unwrap()],
            DataType::Float32,
            0.0f32,
        )
        .unwrap();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .async_partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded_regions = [
            ArraySubset::new_with_ranges(&[3..5]),
            ArraySubset::new_with_ranges(&[17..21]),
        ];
        let answer: &[Vec<f32>] = &[vec![3.0, 4.0], vec![16.0, 16.0, 20.0, 20.0]];
        for (decoded_region, expected) in decoded_regions.into_iter().zip(answer.iter()) {
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .await
                .unwrap();
            let decoded_partial_chunk = crate::array::convert_from_bytes_slice::<f32>(
                &decoded_partial_chunk.into_fixed().unwrap(),
            );
            assert_eq!(expected, &decoded_partial_chunk);
        }
    }
}

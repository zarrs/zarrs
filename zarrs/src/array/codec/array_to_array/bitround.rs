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
use zarrs_metadata::v3::MetadataV3;

use crate::array::DataType;
use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::bitround::{
    BitroundCodecConfiguration, BitroundCodecConfigurationV1,
};
use zarrs_plugin::PluginCreateError;

zarrs_plugin::impl_extension_aliases!(BitroundCodec,
    v3: "bitround", ["numcodecs.bitround", "https://codec.zarrs.dev/array_to_bytes/bitround"]
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<BitroundCodec>()
}

impl CodecTraitsV3 for BitroundCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: BitroundCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(BitroundCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

// Re-export the trait, plugin, and macro from zarrs_data_type
pub use zarrs_data_type::codec_traits::bitround::{
    BitroundDataTypeExt, BitroundDataTypePlugin, BitroundDataTypeTraits, impl_bitround_codec,
    round_bytes_float16, round_bytes_float32, round_bytes_float64, round_bytes_int8,
    round_bytes_int16, round_bytes_int32, round_bytes_int64,
};

fn round_bytes(
    bytes: &mut [u8],
    data_type: &DataType,
    keepbits: u32,
) -> Result<(), zarrs_data_type::DataTypeCodecError> {
    let bitround = data_type.codec_bitround()?;
    bitround.round(bytes, keepbits);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use zarrs_data_type::FillValue;

    use super::*;
    use crate::array::codec::BytesCodec;
    use crate::array::{ArrayBytes, ArraySubset, data_type};
    use zarrs_codec::{ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, CodecOptions};

    #[test]
    fn codec_bitround_float() {
        // 1 sign bit, 8 exponent, 3 mantissa
        const JSON: &str = r#"{ "keepbits": 3 }"#;
        let shape = vec![NonZeroU64::new(4).unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let elements: Vec<f32> = vec![
            //                         |
            0.0,
            // 1.234_567_9 -> 001111111001|11100000011001010010
            // 1.25        -> 001111111010
            1.234_567_9,
            // -8.358_719  -> 110000010000|01011011110101010000
            // -8.0        -> 110000010000
            -8.358_719,
            // 98_765.43   -> 010001111100|00001110011010110111
            // 98_304.0    -> 010001111100
            98_765.43,
        ];
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = BitroundCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<f32>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, &[0.0f32, 1.25f32, -8.0f32, 98304.0f32]);
    }

    #[test]
    fn codec_bitround_uint() {
        const JSON: &str = r#"{ "keepbits": 3 }"#;
        let shape = vec![NonZeroU64::new(7).unwrap()];
        let data_type = data_type::uint32();
        let fill_value = FillValue::from(0u32);
        let elements: Vec<u32> = vec![0, 1024, 1280, 1664, 1685, 123145182, 4294967295];
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = BitroundCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
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
        const JSON: &str = r#"{ "keepbits": 3 }"#;
        let shape = vec![NonZeroU64::new(9).unwrap()];
        let data_type = data_type::uint8();
        let fill_value = FillValue::from(0u8);
        let elements: Vec<u32> = vec![0, 3, 7, 15, 17, 54, 89, 128, 255];
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = BitroundCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<u32>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        for element in &decoded_elements {
            println!("{element} -> {element:#b}");
        }
        assert_eq!(decoded_elements, &[0, 3, 7, 16, 16, 56, 96, 128, 224]);
    }

    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn codec_bitround_partial_decode() {
        const JSON: &str = r#"{ "keepbits": 2 }"#;
        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = Arc::new(BitroundCodec::new_with_configuration(&codec_configuration).unwrap());

        let elements: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let shape = vec![(elements.len() as u64).try_into().unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let bytes: ArrayBytes = crate::array::transmute_to_bytes_vec(elements).into();

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap()
            .into_owned();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .partial_decoder(
                input_handle,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &shape,
                &data_type,
                &fill_value,
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

    #[allow(clippy::single_range_in_vec_init)]
    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_bitround_async_partial_decode() {
        use zarrs_data_type::FillValue;

        const JSON: &str = r#"{ "keepbits": 2 }"#;
        let codec_configuration: BitroundCodecConfiguration = serde_json::from_str(JSON).unwrap();
        let codec = Arc::new(BitroundCodec::new_with_configuration(&codec_configuration).unwrap());

        let elements: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let shape = vec![(elements.len() as u64).try_into().unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .async_partial_decoder(
                input_handle,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &shape,
                &data_type,
                &fill_value,
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

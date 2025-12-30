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
use zarrs_plugin::ExtensionIdentifier;

pub use crate::metadata_ext::codec::bitround::{
    BitroundCodecConfiguration, BitroundCodecConfigurationV1,
};
use crate::{
    array::{
        DataType,
        codec::{Codec, CodecError, CodecPlugin},
    },
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(BitroundCodec::IDENTIFIER, BitroundCodec::matches_name, BitroundCodec::default_name, create_codec_bitround)
}
zarrs_plugin::impl_extension_aliases!(BitroundCodec, "bitround",
    v3: "bitround", ["numcodecs.bitround", "https://codec.zarrs.dev/array_to_bytes/bitround"]
);

pub(crate) fn create_codec_bitround(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: BitroundCodecConfiguration = metadata.to_configuration().map_err(|_| {
        PluginMetadataInvalidError::new(BitroundCodec::IDENTIFIER, "codec", metadata.to_string())
    })?;
    let codec = Arc::new(BitroundCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToArray(codec))
}

/// Traits for a data type supporting the `bitround` codec.
///
/// The bitround codec rounds the mantissa of floating point data types or
/// rounds integers from the most significant set bit to the specified number of bits.
pub trait BitroundCodecDataTypeTraits {
    /// The number of bits to round to for floating point types.
    ///
    /// Returns `None` for integer types where rounding is from the MSB.
    fn mantissa_bits(&self) -> Option<u32>;

    /// Apply bit rounding to the bytes in-place.
    ///
    /// # Arguments
    /// * `bytes` - The bytes to round in-place
    /// * `keepbits` - The number of bits to keep
    fn round(&self, bytes: &mut [u8], keepbits: u32);
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

/// Helper to round 8-bit integer values (from MSB).
pub fn round_bytes_int8(bytes: &mut [u8], keepbits: u32) {
    for element in bytes.iter_mut() {
        *element = round_bits8(*element, keepbits, 8 - element.leading_zeros());
    }
}

/// Helper to round 16-bit integer values (from MSB).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 2.
pub fn round_bytes_int16(bytes: &mut [u8], keepbits: u32) {
    for chunk in bytes.as_chunks_mut::<2>().0 {
        let element = u16::from_ne_bytes(*chunk);
        let rounded = round_bits16(element, keepbits, 16 - element.leading_zeros());
        chunk.copy_from_slice(&u16::to_ne_bytes(rounded));
    }
}

/// Helper to round 32-bit integer values (from MSB).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 4.
pub fn round_bytes_int32(bytes: &mut [u8], keepbits: u32) {
    for chunk in bytes.as_chunks_mut::<4>().0 {
        let element = u32::from_ne_bytes(*chunk);
        let rounded = round_bits32(element, keepbits, 32 - element.leading_zeros());
        chunk.copy_from_slice(&u32::to_ne_bytes(rounded));
    }
}

/// Helper to round 64-bit integer values (from MSB).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 8.
pub fn round_bytes_int64(bytes: &mut [u8], keepbits: u32) {
    for chunk in bytes.as_chunks_mut::<8>().0 {
        let element = u64::from_ne_bytes(*chunk);
        let rounded = round_bits64(element, keepbits, 64 - element.leading_zeros());
        chunk.copy_from_slice(&u64::to_ne_bytes(rounded));
    }
}

/// Helper to round 16-bit float values (fixed mantissa bits).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 2.
pub fn round_bytes_float16(bytes: &mut [u8], keepbits: u32, mantissa_bits: u32) {
    for chunk in bytes.as_chunks_mut::<2>().0 {
        let element = u16::from_ne_bytes(*chunk);
        let rounded = round_bits16(element, keepbits, mantissa_bits);
        chunk.copy_from_slice(&u16::to_ne_bytes(rounded));
    }
}

/// Helper to round 32-bit float values (fixed mantissa bits).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 4.
pub fn round_bytes_float32(bytes: &mut [u8], keepbits: u32, mantissa_bits: u32) {
    for chunk in bytes.as_chunks_mut::<4>().0 {
        let element = u32::from_ne_bytes(*chunk);
        let rounded = round_bits32(element, keepbits, mantissa_bits);
        chunk.copy_from_slice(&u32::to_ne_bytes(rounded));
    }
}

/// Helper to round 64-bit float values (fixed mantissa bits).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 8.
pub fn round_bytes_float64(bytes: &mut [u8], keepbits: u32, mantissa_bits: u32) {
    for chunk in bytes.as_chunks_mut::<8>().0 {
        let element = u64::from_ne_bytes(*chunk);
        let rounded = round_bits64(element, keepbits, mantissa_bits);
        chunk.copy_from_slice(&u64::to_ne_bytes(rounded));
    }
}

// Generate the codec support infrastructure using the generic macro
crate::array::codec::define_data_type_support!(Bitround, BitroundCodecDataTypeTraits);

/// Macro to implement `BitroundCodecDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// // Float types (have mantissa bits):
/// impl_bitround_codec!(Float32DataType, 4, float32, 23);
/// impl_bitround_codec!(Float64DataType, 8, float64, 52);
/// impl_bitround_codec!(Float16DataType, 2, float16, 10);
/// impl_bitround_codec!(BFloat16DataType, 2, float16, 7);
///
/// // Integer types (no mantissa bits):
/// impl_bitround_codec!(Int32DataType, 4, int32);
/// impl_bitround_codec!(Int64DataType, 8, int64);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_bitround_codec {
    // Float16/BFloat16 types (use round_bytes_float16 with specified mantissa bits)
    ($marker:ty, 2, float16, $mantissa_bits:expr) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_float16(bytes, keepbits, $mantissa_bits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // Float32 types
    ($marker:ty, 4, float32, $mantissa_bits:expr) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_float32(bytes, keepbits, $mantissa_bits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // Float64 types
    ($marker:ty, 8, float64, $mantissa_bits:expr) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_float64(bytes, keepbits, $mantissa_bits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // Int8 types (no mantissa, round from MSB)
    ($marker:ty, 1, int8) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int8(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // Int16 types
    ($marker:ty, 2, int16) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int16(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // Int32 types
    ($marker:ty, 4, int32) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int32(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // Int64 types
    ($marker:ty, 8, int64) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int64(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // UInt8 types (use int8 rounding function)
    ($marker:ty, 1, uint8) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int8(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // UInt16 types (use int16 rounding function)
    ($marker:ty, 2, uint16) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int16(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // UInt32 types (use int32 rounding function)
    ($marker:ty, 4, uint32) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int32(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
    // UInt64 types (use int64 rounding function)
    ($marker:ty, 8, uint64) => {
        impl $crate::array::codec::BitroundCodecDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::array::codec::round_bytes_int64(bytes, keepbits);
            }
        }
        $crate::array::codec::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::BitroundPlugin,
            $crate::array::codec::BitroundCodecDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_bitround_codec as impl_bitround_codec;

fn round_bytes(bytes: &mut [u8], data_type: &DataType, keepbits: u32) -> Result<(), CodecError> {
    // Use get_bitround_support() for all types
    let bitround = get_bitround_support(&**data_type).ok_or_else(|| {
        CodecError::UnsupportedDataType(data_type.clone(), BitroundCodec::IDENTIFIER.to_string())
    })?;
    bitround.round(bytes, keepbits);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroU64, sync::Arc};

    use zarrs_data_type::FillValue;

    use super::*;
    use crate::array::data_type;
    use crate::{
        array::{
            ArrayBytes,
            codec::{ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesCodec, CodecOptions},
        },
        array_subset::ArraySubset,
    };

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

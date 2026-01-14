//! The `packbits` array to bytes codec.
//!
//! Packs together values with non-byte-aligned sizes.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/blob/8a28c319023598d40b9a5b5a0dae0a446d497520/codecs/packbits/README.md>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `packbits`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `packbits`
//!
//! ### Codec `configuration` Example - [`PackBitsCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "padding_encoding": "first_byte",
//!     "first_bit": null,
//!     "last_bit": null
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::packbits::PackBitsCodecConfiguration;
//! # serde_json::from_str::<PackBitsCodecConfiguration>(JSON).unwrap();
//! ```

mod data_type_extension_packbits_codec;
mod packbits_codec;
mod packbits_partial_decoder;

use std::sync::Arc;

use num::Integer;
pub use packbits_codec::PackBitsCodec;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::ExtensionAliasesV3;

use crate::array::DataType;
use zarrs_codec::{Codec, CodecError, CodecPluginV3};
pub use zarrs_metadata_ext::codec::packbits::{
    PackBitsCodecConfiguration, PackBitsCodecConfigurationV1,
};
use zarrs_plugin::{PluginConfigurationInvalidError, PluginCreateError};

zarrs_plugin::impl_extension_aliases!(PackBitsCodec, v3: "packbits");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<PackBitsCodec>(create_codec_packbits_v3)
}

pub(crate) fn create_codec_packbits_v3(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: PackBitsCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let codec = Arc::new(PackBitsCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

/// Traits for a data type supporting the `packbits` codec.
pub trait PackBitsCodecDataTypeTraits {
    /// The component size in bits.
    fn component_size_bits(&self) -> u64;

    /// The number of components.
    fn num_components(&self) -> u64;

    /// True if the components need sign extension.
    ///
    /// This should be set to `true` for signed integer types.
    fn sign_extension(&self) -> bool;
}

// Generate the codec support infrastructure using the generic macro
zarrs_codec::define_data_type_support!(PackBits, PackBitsCodecDataTypeTraits);

/// Macro to implement `PackBitsCodecDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// // For single-component types:
/// crate::array::codec::array_to_bytes::packbits::impl_packbits_codec!(Int32DataType, 32, signed, 1);
/// crate::array::codec::array_to_bytes::packbits::impl_packbits_codec!(UInt32DataType, 32, unsigned, 1);
/// crate::array::codec::array_to_bytes::packbits::impl_packbits_codec!(Float32DataType, 32, float, 1);
///
/// // For complex types (2 components):
/// crate::array::codec::array_to_bytes::packbits::impl_packbits_codec!(Complex64DataType, 32, float, 2);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_packbits_codec {
    // Multi-component, signed integer
    ($marker:ty, $bits:expr, signed, $components:expr) => {
        impl $crate::array::codec::PackBitsCodecDataTypeTraits for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                true
            }
        }
        $crate::array::codec::api::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::PackBitsPlugin,
            $crate::array::codec::PackBitsCodecDataTypeTraits
        );
    };
    // Multi-component, unsigned integer
    ($marker:ty, $bits:expr, unsigned, $components:expr) => {
        impl $crate::array::codec::PackBitsCodecDataTypeTraits for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                false
            }
        }
        $crate::array::codec::api::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::PackBitsPlugin,
            $crate::array::codec::PackBitsCodecDataTypeTraits
        );
    };
    // Multi-component, float (no sign extension)
    ($marker:ty, $bits:expr, float, $components:expr) => {
        impl $crate::array::codec::PackBitsCodecDataTypeTraits for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                false
            }
        }
        $crate::array::codec::api::register_data_type_extension_codec!(
            $marker,
            $crate::array::codec::PackBitsPlugin,
            $crate::array::codec::PackBitsCodecDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_packbits_codec as impl_packbits_codec;

struct PackBitsCodecComponents {
    pub component_size_bits: u64,
    pub num_components: u64,
    pub sign_extension: bool,
}

fn pack_bits_components(data_type: &DataType) -> Result<PackBitsCodecComponents, CodecError> {
    let packbits = get_packbits_support(data_type).ok_or_else(|| {
        CodecError::UnsupportedDataType(
            data_type.clone(),
            PackBitsCodec::aliases_v3().default_name.to_string(),
        )
    })?;
    Ok(PackBitsCodecComponents {
        component_size_bits: packbits.component_size_bits(),
        num_components: packbits.num_components(),
        sign_extension: packbits.sign_extension(),
    })
}

fn div_rem_8bit(bit: u64, element_size_bits: u64) -> (u64, u8) {
    let (element, element_bit) = bit.div_rem(&element_size_bits);
    let element_size_bits_padded = 8 * element_size_bits.div_ceil(8);
    let byte = (element * element_size_bits_padded + element_bit) / 8;
    let byte_bit = (element_bit % 8) as u8;
    (byte, byte_bit)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use num::Integer;
    use zarrs_data_type::FillValue;

    use crate::array::codec::BytesCodec;
    use crate::array::element::{Element, ElementOwned};
    use crate::array::{ArrayBytes, ArraySubset, data_type};
    use zarrs_codec::{ArrayToBytesCodecTraits, BytesPartialDecoderTraits, CodecOptions};
    use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

    #[test]
    fn div_rem_8bit() {
        use super::div_rem_8bit;

        assert_eq!(div_rem_8bit(0, 1), (0, 0));
        assert_eq!(div_rem_8bit(1, 1), (1, 0));
        assert_eq!(div_rem_8bit(2, 1), (2, 0));

        assert_eq!(div_rem_8bit(0, 3), (0, 0));
        assert_eq!(div_rem_8bit(1, 3), (0, 1));
        assert_eq!(div_rem_8bit(2, 3), (0, 2));
        assert_eq!(div_rem_8bit(3, 3), (1, 0));
        assert_eq!(div_rem_8bit(4, 3), (1, 1));
        assert_eq!(div_rem_8bit(5, 3), (1, 2));

        assert_eq!(div_rem_8bit(0, 12), (0, 0));
        assert_eq!(div_rem_8bit(7, 12), (0, 7));
        assert_eq!(div_rem_8bit(8, 12), (1, 0));
        assert_eq!(div_rem_8bit(9, 12), (1, 1));
        assert_eq!(div_rem_8bit(10, 12), (1, 2));
        assert_eq!(div_rem_8bit(11, 12), (1, 3));
        assert_eq!(div_rem_8bit(12, 12), (2, 0));
        assert_eq!(div_rem_8bit(13, 12), (2, 1));
    }

    #[test]
    fn codec_packbits_bool() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
            let data_type = data_type::bool();
            let fill_value = FillValue::from(false);

            let elements: Vec<bool> = (0..40).map(|i| i % 3 == 0).collect();
            let bytes = bool::into_array_bytes(&data_type, elements)?.into_owned();
            // T F F T F
            // F T F F T
            // F F T F F
            // T F F T F
            // ...

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= 40.div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded);

            // Partial decoding
            let decoded_region = ArraySubset::new_with_ranges(&[1..4, 1..4]);
            let input_handle = Arc::new(encoded);
            let partial_decoder = codec
                .partial_decoder(
                    input_handle.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // packbits partial decoder does not hold bytes
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .unwrap();
            let decoded_partial_chunk =
                bool::from_array_bytes(&data_type, decoded_partial_chunk).unwrap();
            let answer: Vec<bool> =
                vec![true, false, false, false, true, false, false, false, true];
            assert_eq!(answer, decoded_partial_chunk);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float32() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
            let data_type = data_type::float32();
            let fill_value = FillValue::from(0.0f32);

            let elements: Vec<f32> = (0..40).map(|i| i as f32).collect();
            let bytes = f32::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (40 * 32).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded);

            // Check it matches little endian bytes
            let decoded = BytesCodec::little()
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_int16() -> Result<(), Box<dyn std::error::Error>> {
        for last_bit in 11..15 {
            for first_bit in 0..4 {
                for encoding in [
                    PackBitsPaddingEncoding::None,
                    PackBitsPaddingEncoding::FirstByte,
                    PackBitsPaddingEncoding::LastByte,
                ] {
                    let codec = Arc::new(
                        super::PackBitsCodec::new(encoding, Some(first_bit), Some(last_bit))
                            .unwrap(),
                    );
                    let chunk_shape =
                        vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
                    let data_type = data_type::int16();
                    let fill_value = FillValue::from(0i16);
                    let elements: Vec<i16> = (-20..20).map(|i| (i as i16) << first_bit).collect();
                    let bytes = i16::to_array_bytes(&data_type, &elements)?.into_owned();

                    // Encoding
                    let encoded = codec.encode(
                        bytes.clone(),
                        &chunk_shape,
                        &data_type,
                        &fill_value,
                        &CodecOptions::default(),
                    )?;
                    assert!(
                        (encoded.len() as u64) <= (40 * (last_bit - first_bit + 1)).div_ceil(8) + 1
                    );

                    // Decoding
                    let decoded = codec
                        .decode(
                            encoded.clone(),
                            &chunk_shape,
                            &data_type,
                            &fill_value,
                            &CodecOptions::default(),
                        )
                        .unwrap();
                    assert_eq!(elements, i16::from_array_bytes(&data_type, decoded)?);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_uint2() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::uint2();
            let fill_value = FillValue::from(0u8);

            let elements: Vec<u8> = (0..4).map(|i| i as u8).collect();
            let bytes = u8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (4 * 4).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(elements, u8::from_array_bytes(&data_type, decoded)?);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_uint4() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::uint4();
            let fill_value = FillValue::from(0u8);

            let elements: Vec<u8> = (0..16).map(|i| i as u8).collect();
            let bytes = u8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(elements, u8::from_array_bytes(&data_type, decoded)?);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_int2() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::int2();
            let fill_value = FillValue::from(0i8);

            let elements: Vec<i8> = (-2..2).map(|i| i as i8).collect();
            let bytes = i8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (4 * 4).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(elements, i8::from_array_bytes(&data_type, decoded)?);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_int4() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::int4();
            let fill_value = FillValue::from(0i8);

            let elements: Vec<i8> = (-8..8).map(|i| i as i8).collect();
            let bytes = i8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(elements, i8::from_array_bytes(&data_type, decoded)?);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float4_e2m1fn() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float4_e2m1fn();
            let fill_value = FillValue::from(0u8);

            let bytes = ArrayBytes::new_flen((0..16).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float6_e2m3fn() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(64).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float6_e2m3fn();
            let fill_value = FillValue::from(0u8);

            let bytes = ArrayBytes::new_flen((0..64).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (6 * 64).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float6_e3m2fn() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap());
            let chunk_shape = vec![NonZeroU64::new(64).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float6_e3m2fn();
            let fill_value = FillValue::from(0u8);

            let bytes = ArrayBytes::new_flen((0..64).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(
                bytes.clone(),
                &chunk_shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )?;
            assert!((encoded.len() as u64) <= (6 * 64).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(
                    encoded.clone(),
                    &chunk_shape,
                    &data_type,
                    &fill_value,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }
}

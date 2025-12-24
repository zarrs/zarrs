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

mod packbits_codec;
mod packbits_partial_decoder;

use std::sync::Arc;

use num::Integer;
pub use packbits_codec::PackBitsCodec;

use crate::array::DataType;
use crate::array::codec::CodecError;
pub use crate::metadata_ext::codec::packbits::{
    PackBitsCodecConfiguration, PackBitsCodecConfigurationV1,
};
use crate::registry::codec::PACKBITS;
use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(PACKBITS, is_identifier_packbits, create_codec_packbits)
}

fn is_identifier_packbits(identifier: &str) -> bool {
    identifier == PACKBITS
}

pub(crate) fn create_codec_packbits(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: PackBitsCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(PACKBITS, "codec", metadata.to_string()))?;
    let codec = Arc::new(PackBitsCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

struct DataTypeExtensionPackBitsCodecComponents {
    pub component_size_bits: u64,
    pub num_components: u64,
    pub sign_extension: bool,
}

#[allow(clippy::too_many_lines)]
fn pack_bits_components(
    data_type: &DataType,
) -> Result<DataTypeExtensionPackBitsCodecComponents, CodecError> {
    type DT = DataType;
    match data_type {
        DT::Bool => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 1,
            num_components: 1,
            sign_extension: false,
        }),
        DT::UInt2 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 2,
            num_components: 1,
            sign_extension: false,
        }),
        DT::Int2 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 2,
            num_components: 1,
            sign_extension: true,
        }),
        DT::UInt4 | DT::Float4E2M1FN => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 4,
            num_components: 1,
            sign_extension: false,
        }),
        DT::Int4 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 4,
            num_components: 1,
            sign_extension: true,
        }),
        DT::Float6E2M3FN | DT::Float6E3M2FN => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 6,
            num_components: 1,
            sign_extension: false,
        }),
        DT::UInt8
        | DT::Float8E3M4
        | DT::Float8E4M3
        | DT::Float8E4M3B11FNUZ
        | DT::Float8E4M3FNUZ
        | DT::Float8E5M2
        | DT::Float8E5M2FNUZ
        | DT::Float8E8M0FNU => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 8,
            num_components: 1,
            sign_extension: false,
        }),
        DataType::ComplexFloat4E2M1FN => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 4,
            num_components: 2,
            sign_extension: false,
        }),
        DataType::ComplexFloat6E2M3FN | DataType::ComplexFloat6E3M2FN => {
            Ok(DataTypeExtensionPackBitsCodecComponents {
                component_size_bits: 6,
                num_components: 2,
                sign_extension: false,
            })
        }
        DataType::ComplexFloat8E3M4
        | DataType::ComplexFloat8E4M3
        | DataType::ComplexFloat8E4M3B11FNUZ
        | DataType::ComplexFloat8E4M3FNUZ
        | DataType::ComplexFloat8E5M2
        | DataType::ComplexFloat8E5M2FNUZ
        | DataType::ComplexFloat8E8M0FNU => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 8,
            num_components: 2,
            sign_extension: false,
        }),
        DT::Int8 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 8,
            num_components: 1,
            sign_extension: true,
        }),
        DT::UInt16 | DT::Float16 | DT::BFloat16 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 16,
            num_components: 1,
            sign_extension: false,
        }),
        DT::Int16 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 16,
            num_components: 1,
            sign_extension: true,
        }),
        DT::ComplexBFloat16 | DT::ComplexFloat16 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 16,
            num_components: 2,
            sign_extension: false,
        }),
        DT::UInt32 | DT::Float32 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 32,
            num_components: 1,
            sign_extension: false,
        }),
        DT::Int32
        | DT::NumpyDateTime64 {
            unit: _,
            scale_factor: _,
        }
        | DT::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _,
        } => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 32,
            num_components: 1,
            sign_extension: true,
        }),
        DT::UInt64 | DT::Float64 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 64,
            num_components: 1,
            sign_extension: false,
        }),
        DT::Int64 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 64,
            num_components: 1,
            sign_extension: true,
        }),
        DT::Complex64 | DT::ComplexFloat32 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 32,
            num_components: 2,
            sign_extension: false,
        }),
        DT::Complex128 | DT::ComplexFloat64 => Ok(DataTypeExtensionPackBitsCodecComponents {
            component_size_bits: 64,
            num_components: 2,
            sign_extension: false,
        }),
        DT::Extension(ext) => {
            let packbits = ext.codec_packbits()?;
            Ok(DataTypeExtensionPackBitsCodecComponents {
                component_size_bits: packbits.component_size_bits(),
                num_components: packbits.num_components(),
                sign_extension: packbits.sign_extension(),
            })
        }
        DT::String | DT::Bytes | DT::RawBits(_) | DT::Optional(_) => Err(
            CodecError::UnsupportedDataType(data_type.clone(), PACKBITS.to_string()),
        ),
    }
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
    use std::{num::NonZeroU64, sync::Arc};

    use num::Integer;
    use zarrs_data_type::FillValue;

    use crate::metadata_ext::codec::packbits::PackBitsPaddingEncoding;
    use crate::{
        array::{
            ArrayBytes, DataType,
            codec::{ArrayToBytesCodecTraits, BytesCodec, BytesPartialDecoderTraits, CodecOptions},
            element::{Element, ElementOwned},
        },
        array_subset::ArraySubset,
    };

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
            let data_type = DataType::Bool;
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
            assert!((encoded.len() as u64) <= (40 * 1).div_ceil(&8) + 1);

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
            let data_type = DataType::Float32;
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
                    let data_type = DataType::Int16;
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
            let data_type = DataType::UInt2;
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
            let data_type = DataType::UInt4;
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
            let data_type = DataType::Int2;
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
            let data_type = DataType::Int4;
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
            let data_type = DataType::Float4E2M1FN;
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
            let data_type = DataType::Float6E2M3FN;
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
            let data_type = DataType::Float6E3M2FN;
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

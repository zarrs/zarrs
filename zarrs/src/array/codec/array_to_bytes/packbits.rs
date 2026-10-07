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
//! None
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

use crate::array::DataType;
use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::packbits::{
    PackBitsCodecConfiguration, PackBitsCodecConfigurationV1,
};

zarrs_plugin::impl_extension_aliases!(PackBitsCodec, v3: "packbits");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<PackBitsCodec>()
}

impl CodecTraitsV3 for PackBitsCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: PackBitsCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(PackBitsCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToBytes(codec))
    }
}

// Re-export extension trait from zarrs_data_type
pub use zarrs_data_type::codec_traits::packbits::{
    PackBitsDataTypeExt, PackBitsDataTypePlugin, PackBitsDataTypeTraits,
    impl_pack_bits_data_type_traits,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct PackBitsCodecComponents {
    pub component_size_bits: u64,
    pub num_components: u64,
    pub sign_extension: bool,
}

fn pack_bits_components(
    data_type: &DataType,
) -> Result<PackBitsCodecComponents, zarrs_data_type::DataTypeCodecError> {
    let packbits = data_type.codec_packbits()?;
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

    use super::super::decode_into_test_util::{decode_into_view, decode_into_view_array};
    use crate::array::codec::BytesCodec;
    use crate::array::element::{Element, ElementOwned};
    use crate::array::{ArrayBytes, ArraySubset, CowBytes, DataType, data_type};
    use zarrs_codec::{
        BytesPartialDecoderTraits, CodecOptions, CodecSpecificOptions,
        UnboundArrayToBytesCodecTraits,
    };
    use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

    /// A data type, fill value, first and last bit, and elements.
    type DecodeIntoCase = (
        DataType,
        FillValue,
        Option<u64>,
        Option<u64>,
        ArrayBytes<'static>,
    );

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

    /// `decode_into` unpacks into a contiguous output and a view with two contiguous regions
    /// identically to `decode`, including where the output holds other values beforehand.
    #[test]
    fn codec_packbits_decode_into() -> Result<(), Box<dyn std::error::Error>> {
        let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(4).unwrap()];
        let options = CodecOptions::default();
        let cases: Vec<DecodeIntoCase> = vec![
            (
                data_type::bool(),
                FillValue::from(false),
                None,
                None,
                bool::into_array_bytes(
                    &data_type::bool(),
                    (0..16).map(|i| i % 3 == 0).collect::<Vec<bool>>(),
                )?
                .into_owned(),
            ),
            (
                data_type::uint4(),
                FillValue::from(0u8),
                None,
                None,
                u8::to_array_bytes(&data_type::uint4(), &(0..16).collect::<Vec<u8>>())?
                    .into_owned(),
            ),
            // Sign extension
            (
                data_type::int4(),
                FillValue::from(0i8),
                None,
                None,
                i8::to_array_bytes(&data_type::int4(), &(-8..8).collect::<Vec<i8>>())?.into_owned(),
            ),
            // Some of the bits of a component
            (
                data_type::int16(),
                FillValue::from(0i16),
                Some(1),
                Some(11),
                i16::to_array_bytes(
                    &data_type::int16(),
                    &(0..16).map(|i| i * 123 - 1000).collect::<Vec<i16>>(),
                )?
                .into_owned(),
            ),
            // Equivalent to the bytes codec
            (
                data_type::int16(),
                FillValue::from(0i16),
                None,
                None,
                i16::to_array_bytes(
                    &data_type::int16(),
                    &(0..16).map(|i| i * 123 - 1000).collect::<Vec<i16>>(),
                )?
                .into_owned(),
            ),
        ];
        for (data_type, fill_value, first_bit, last_bit, bytes) in cases {
            let element_size = data_type.fixed_size().unwrap();
            for encoding in [
                PackBitsPaddingEncoding::None,
                PackBitsPaddingEncoding::FirstByte,
                PackBitsPaddingEncoding::LastByte,
            ] {
                let codec = Arc::new(super::PackBitsCodec::new(encoding, first_bit, last_bit)?)
                    .with_context(
                        data_type.clone(),
                        fill_value.clone(),
                        &CodecSpecificOptions::default(),
                    )?;
                let encoded = codec.encode(bytes.clone(), &chunk_shape, &options)?;
                let decoded = codec
                    .decode(encoded.clone(), &chunk_shape, &options)?
                    .into_fixed()?
                    .to_vec();

                // One contiguous region
                let output =
                    decode_into_view(&codec, &encoded, &chunk_shape, [4, 4], 4, 0, element_size);
                assert_eq!(output?, decoded);

                // Four contiguous regions, with the rest of the output unchanged
                let output =
                    decode_into_view(&codec, &encoded, &chunk_shape, [4, 4], 8, 0, element_size)?;
                let row = 4 * element_size;
                for (output_row, decoded_row) in
                    std::iter::zip(output.chunks_exact(2 * row), decoded.chunks_exact(row))
                {
                    assert_eq!(&output_row[..row], decoded_row);
                    assert!(output_row[row..].iter().all(|byte| *byte == 0xFF));
                }

                // The encoded bytes are not the length of the chunk, and the output is not written to
                let truncated = CowBytes::from(encoded[..encoded.len() - 1].to_vec());
                for array_columns in [4, 8] {
                    let (output, result) = decode_into_view_array(
                        &codec,
                        &truncated,
                        &chunk_shape,
                        [4, 4],
                        array_columns,
                        0,
                        element_size,
                    );
                    assert!(result.is_err());
                    assert!(output.iter().all(|byte| *byte == 0xFF));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_bool() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
            let data_type = data_type::bool();
            let fill_value = FillValue::from(false);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<bool> = (0..40).map(|i| i % 3 == 0).collect();
            let bytes = bool::into_array_bytes(&data_type, elements)?.into_owned();
            // T F F T F
            // F T F F T
            // F F T F F
            // T F F T F
            // ...

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= 40.div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);

            // Partial decoding
            let decoded_region = ArraySubset::new_with_ranges(&[1..4, 1..4]);
            let input_handle = Arc::new(encoded);
            let partial_decoder = codec
                .partial_decoder(input_handle.clone(), &chunk_shape, &CodecOptions::default())
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
    fn codec_packbits_rejects_out_of_range_last_bit_when_bound() {
        let data_type = data_type::uint2();
        let fill_value = FillValue::from(0u8);
        let codec = Arc::new(
            super::PackBitsCodec::new(PackBitsPaddingEncoding::None, None, Some(2)).unwrap(),
        );

        assert!(
            codec
                .with_context(data_type, fill_value, &CodecSpecificOptions::default())
                .is_err()
        );
    }

    /// Shapes with sizes that overflow are errors rather than panics.
    #[test]
    fn codec_packbits_rejects_overflowing_shape() -> Result<(), Box<dyn std::error::Error>> {
        let codec = Arc::new(super::PackBitsCodec::default()).with_context(
            data_type::uint4(),
            FillValue::from(0u8),
            &CodecSpecificOptions::default(),
        )?;
        let options = CodecOptions::default();
        let max = NonZeroU64::new(u64::MAX).unwrap();
        let shapes = [
            // The number of elements overflows
            vec![max, NonZeroU64::new(2).unwrap()],
            // The number of bits overflows
            vec![NonZeroU64::new(u64::MAX / 2).unwrap()],
        ];
        for shape in shapes {
            assert!(codec.encoded_representation(&shape).is_err());
            assert!(
                codec
                    .encode(ArrayBytes::new_flen(vec![0u8]), &shape, &options)
                    .is_err()
            );
            assert!(
                codec
                    .decode(CowBytes::from(vec![0u8]), &shape, &options)
                    .is_err()
            );
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
            let chunk_shape = vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
            let data_type = data_type::float32();
            let fill_value = FillValue::from(0.0f32);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<f32> = (0..40).map(|i| i as f32).collect();
            let bytes = f32::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (40 * 32).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);

            // Check it matches little endian bytes
            let decoded = Arc::new(BytesCodec::little())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
                    let chunk_shape =
                        vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
                    let data_type = data_type::int16();
                    let fill_value = FillValue::from(0i16);
                    let codec = Arc::new(
                        super::PackBitsCodec::new(encoding, Some(first_bit), Some(last_bit))
                            .unwrap(),
                    )
                    .with_context(
                        data_type.clone(),
                        fill_value.clone(),
                        &CodecSpecificOptions::default(),
                    )?;
                    let elements: Vec<i16> = (-20..20).map(|i| (i as i16) << first_bit).collect();
                    let bytes = i16::to_array_bytes(&data_type, &elements)?.into_owned();

                    // Encoding
                    let encoded =
                        codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
                    assert!(
                        (encoded.len() as u64) <= (40 * (last_bit - first_bit + 1)).div_ceil(8) + 1
                    );

                    // Decoding
                    let decoded = codec
                        .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
            let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::uint2();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<u8> = (0..4).map(|i| i as u8).collect();
            let bytes = u8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 4).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::uint4();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<u8> = (0..16).map(|i| i as u8).collect();
            let bytes = u8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
            let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::int2();
            let fill_value = FillValue::from(0i8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<i8> = (-2..2).map(|i| i as i8).collect();
            let bytes = i8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 4).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::int4();
            let fill_value = FillValue::from(0i8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<i8> = (-8..8).map(|i| i as i8).collect();
            let bytes = i8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float4_e2m1fn();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let bytes = ArrayBytes::new_flen((0..16).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
            let chunk_shape = vec![NonZeroU64::new(64).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float6_e2m3fn();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let bytes = ArrayBytes::new_flen((0..64).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (6 * 64).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
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
            let chunk_shape = vec![NonZeroU64::new(64).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float6_e3m2fn();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let bytes = ArrayBytes::new_flen((0..64).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (6 * 64).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }
}

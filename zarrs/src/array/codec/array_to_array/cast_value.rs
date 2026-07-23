//! The `cast_value` array to array codec.
//!
//! Converts array scalar values to a configured target data type.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/blob/main/codecs/cast_value/README.md>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `cast_value`

mod cast_value_codec;
mod cast_value_codec_partial;

use std::sync::Arc;

pub use cast_value_codec::CastValueCodec;
use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
use zarrs_metadata::v3::MetadataV3;
pub use zarrs_metadata_ext::codec::cast_value::{
    CastValueCodecConfiguration, CastValueCodecConfigurationV1, CastValueOutOfRangeMode,
    CastValueRoundingMode, CastValueScalarMap,
};

zarrs_plugin::impl_extension_aliases!(CastValueCodec, v3: "cast_value");

inventory::submit! {
    CodecPluginV3::new::<CastValueCodec>()
}

impl CodecTraitsV3 for CastValueCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: CastValueCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(CastValueCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

pub use zarrs_data_type::codec_traits::cast_value::{
    CastValueDataTypeExt, CastValueDataTypePlugin, CastValueDataTypeTraits, CastValueIntStored,
    CastValueKernel, CastValueRepr, impl_cast_value_data_type_traits_float,
    impl_cast_value_data_type_traits_signed_integer,
    impl_cast_value_data_type_traits_unsigned_integer, select_cast_kernel,
};

#[cfg(test)]
mod tests {
    use zarrs_codec::{CodecOptions, UnboundArrayToArrayCodecTraits};
    use zarrs_data_type::FillValue;

    use super::*;
    use crate::array::{ArrayBytes, data_type};

    #[test]
    fn codec_cast_value_exact_round_trip() {
        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "uint16"
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let shape = [4];
        let data_type = data_type::uint8();
        let fill_value = FillValue::from(0u8);
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let elements = vec![0u8, 1, 127, 255];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements.clone()));

        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<u16>(
            encoded.clone().into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![0u16, 1, 127, 255]);

        let decoded = codec
            .decode(encoded, &shape, &CodecOptions::default())
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<u8>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, elements);
    }

    #[test]
    fn codec_cast_value_wrap_int16_to_int8() {
        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "int8",
                "out_of_range": "wrap"
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let shape = [4];
        let data_type = data_type::int16();
        let fill_value = FillValue::from(0i16);
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![
            127i16, 128, 129, -129,
        ]));

        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<i8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![127i8, -128, -127, 127]);
    }

    #[test]
    fn codec_cast_value_wrap_u64_to_int8() {
        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "int8",
                "out_of_range": "wrap"
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let shape = [1];
        let data_type = data_type::uint64();
        let fill_value = FillValue::from(0u64);
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![u64::MAX]));

        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<i8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![-1i8]);
    }

    #[test]
    fn codec_cast_value_wrap_float_to_uint8_decodes() {
        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "uint8",
                "out_of_range": "wrap"
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let shape = [2];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![255.0f32, 256.0]));

        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &shape, &CodecOptions::default())
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<f32>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, vec![255.0f32, 0.0]);
    }

    #[test]
    fn codec_cast_value_clamp_and_scalar_map_precedence() {
        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "uint8",
                "out_of_range": "clamp",
                "scalar_map": {
                    "encode": [[1.5, 42], ["NaN", 7]]
                }
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let shape = [3];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![
            1.5f32,
            300.0,
            f32::NAN,
        ]));

        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<u8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![42u8, 255, 7]);
    }

    #[test]
    fn codec_cast_value_duplicate_scalar_map_first_wins() {
        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "uint8",
                "scalar_map": {
                    "encode": [[1.0, 2], [1.0, 3]]
                }
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let shape = [1];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![1.0f32]));

        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<u8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![2u8]);
    }

    #[test]
    fn codec_cast_value_integer_rounding_modes() {
        let cases = [
            ("nearest-even", vec![2i8, 2, -2, -2]),
            ("towards-zero", vec![1i8, 2, -1, -2]),
            ("towards-positive", vec![2i8, 3, -1, -2]),
            ("towards-negative", vec![1i8, 2, -2, -3]),
            ("nearest-away", vec![2i8, 3, -2, -3]),
        ];
        let shape = [4];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);

        for (rounding, expected) in cases {
            let configuration: CastValueCodecConfiguration = serde_json::from_str(&format!(
                r#"{{
                    "data_type": "int8",
                    "rounding": "{rounding}"
                }}"#
            ))
            .unwrap();
            let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
            let codec = codec
                .with_context(data_type.clone(), fill_value.clone())
                .unwrap();
            let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![
                1.5f32, 2.5, -1.5, -2.5,
            ]));

            let encoded = codec
                .encode(bytes, &shape, &CodecOptions::default())
                .unwrap();
            let encoded_elements = crate::array::transmute_from_bytes_vec::<i8>(
                encoded.into_fixed().unwrap().into_owned(),
            );
            assert_eq!(encoded_elements, expected, "rounding={rounding}");
        }
    }

    #[test]
    fn codec_cast_value_large_uint64_rounding_to_float64() {
        // u64::MAX and 2^53 + 1 are not exactly representable in float64;
        // directed rounding must be applied to the exact integer, not a
        // ties-to-even pre-rounded intermediate
        let cases = [
            (
                "nearest-even",
                [18_446_744_073_709_551_616.0f64, 9_007_199_254_740_992.0],
            ),
            (
                "towards-zero",
                [18_446_744_073_709_549_568.0, 9_007_199_254_740_992.0],
            ),
            (
                "towards-positive",
                [18_446_744_073_709_551_616.0, 9_007_199_254_740_994.0],
            ),
            (
                "towards-negative",
                [18_446_744_073_709_549_568.0, 9_007_199_254_740_992.0],
            ),
            (
                "nearest-away",
                [18_446_744_073_709_551_616.0, 9_007_199_254_740_994.0],
            ),
        ];
        let shape = [2];
        let data_type = data_type::uint64();
        let fill_value = FillValue::from(0u64);
        let elements = vec![u64::MAX, 9_007_199_254_740_993u64];

        for (rounding, expected) in cases {
            let configuration: CastValueCodecConfiguration = serde_json::from_str(&format!(
                r#"{{
                    "data_type": "float64",
                    "rounding": "{rounding}"
                }}"#
            ))
            .unwrap();
            let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
            let codec = codec
                .with_context(data_type.clone(), fill_value.clone())
                .unwrap();
            let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements.clone()));

            let encoded = codec
                .encode(bytes, &shape, &CodecOptions::default())
                .unwrap();
            let encoded_elements = crate::array::transmute_from_bytes_vec::<f64>(
                encoded.into_fixed().unwrap().into_owned(),
            );
            assert_eq!(encoded_elements, expected, "rounding={rounding}");
        }
    }

    #[test]
    fn codec_cast_value_huge_float_wrap() {
        // floats at or beyond the i128 range must still wrap modulo 2^N;
        // every such float is a multiple of 256 so the result is 0
        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "uint8",
                "out_of_range": "wrap"
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let shape = [3];
        let codec_f32 = codec
            .with_context(data_type::float32(), FillValue::from(0.0f32))
            .unwrap();
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![
            f32::MAX,
            2.0f32.powi(127),
            f32::MIN,
        ]));
        let encoded = codec_f32
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<u8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![0u8, 0, 0]);

        // exactly 2^127 previously saturated to i128::MAX and wrapped to 255
        let codec_f64 = codec
            .with_context(data_type::float64(), FillValue::from(0.0f64))
            .unwrap();
        let shape = [1];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![2.0f64.powi(127)]));
        let encoded = codec_f64
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<u8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![0u8]);
    }

    #[test]
    fn codec_cast_value_huge_float_clamp() {
        // floats beyond the i128 range must still clamp to the target range
        let shape = [2];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![
            f32::MAX,
            f32::MIN,
        ]));

        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "uint8",
                "out_of_range": "clamp"
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let codec = codec
            .with_context(data_type::float32(), FillValue::from(0.0f32))
            .unwrap();
        let encoded = codec
            .encode(bytes.clone(), &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<u8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![255u8, 0]);

        let configuration: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "int8",
                "out_of_range": "clamp"
            }"#,
        )
        .unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        let codec = codec
            .with_context(data_type::float32(), FillValue::from(0.0f32))
            .unwrap();
        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<i8>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, vec![127i8, -128]);
    }

    #[test]
    fn codec_cast_value_kernel_matches_scalar_path() {
        use zarrs_data_type::codec_traits::cast_value::{
            CastValueDataTypeExt, CastValueOutOfRangeMode as DtOutOfRange,
            CastValueRoundingMode as DtRounding,
        };

        let data_types = [
            ("int2", data_type::int2()),
            ("int4", data_type::int4()),
            ("int8", data_type::int8()),
            ("int16", data_type::int16()),
            ("int32", data_type::int32()),
            ("int64", data_type::int64()),
            ("uint2", data_type::uint2()),
            ("uint4", data_type::uint4()),
            ("uint8", data_type::uint8()),
            ("uint16", data_type::uint16()),
            ("uint32", data_type::uint32()),
            ("uint64", data_type::uint64()),
            ("float16", data_type::float16()),
            ("bfloat16", data_type::bfloat16()),
            ("float32", data_type::float32()),
            ("float64", data_type::float64()),
        ];
        let roundings = [
            ("nearest-even", DtRounding::NearestEven),
            ("towards-zero", DtRounding::TowardsZero),
            ("towards-positive", DtRounding::TowardsPositive),
            ("towards-negative", DtRounding::TowardsNegative),
            ("nearest-away", DtRounding::NearestAway),
        ];
        let out_of_ranges = [
            (None, None),
            (Some("clamp"), Some(DtOutOfRange::Clamp)),
            (Some("wrap"), Some(DtOutOfRange::Wrap)),
        ];

        // xorshift64 for deterministic pseudo-random source bytes
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next_byte = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        };

        let num_elements = 64_usize;
        let shape = [num_elements as u64];
        let options = CodecOptions::default();

        for (source_name, source_dt) in &data_types {
            let source_size = source_dt.fixed_size().unwrap();
            let source_trait = source_dt.codec_castvalue().unwrap();
            assert!(source_trait.cast_value_repr().is_some(), "{source_name}");
            for (target_name, target_dt) in &data_types {
                let target_size = target_dt.fixed_size().unwrap();
                let target_trait = target_dt.codec_castvalue().unwrap();
                let target_is_float = target_name.contains("float");
                for (rounding_name, rounding) in roundings {
                    for (out_of_range_name, out_of_range) in out_of_ranges {
                        if target_is_float && matches!(out_of_range, Some(DtOutOfRange::Wrap)) {
                            continue; // invalid configuration
                        }
                        let case = format!(
                            "{source_name}->{target_name} rounding={rounding_name} out_of_range={out_of_range_name:?}"
                        );
                        let out_of_range_json = out_of_range_name
                            .map(|mode| format!(r#", "out_of_range": "{mode}""#))
                            .unwrap_or_default();
                        let configuration: CastValueCodecConfiguration =
                            serde_json::from_str(&format!(
                                r#"{{
                                    "data_type": "{target_name}",
                                    "rounding": "{rounding_name}"{out_of_range_json}
                                }}"#
                            ))
                            .unwrap();
                        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
                        let codec = codec
                            .with_context(source_dt.clone(), FillValue::new(vec![0; source_size]))
                            .unwrap();

                        let source_bytes: Vec<u8> = (0..num_elements * source_size)
                            .map(|_| next_byte())
                            .collect();

                        // reference: the generic per-element scalar path
                        let mut expected = Vec::new();
                        let expected_err = source_bytes.chunks_exact(source_size).any(|element| {
                            source_trait
                                .cast_value_cast(
                                    element,
                                    target_trait,
                                    rounding,
                                    out_of_range,
                                    &mut expected,
                                )
                                .is_err()
                        });

                        let encoded =
                            codec.encode(ArrayBytes::from(source_bytes.clone()), &shape, &options);
                        if expected_err {
                            assert!(encoded.is_err(), "{case}: expected encode error");
                            continue;
                        }
                        let encoded = encoded
                            .unwrap_or_else(|err| panic!("{case}: {err}"))
                            .into_fixed()
                            .unwrap()
                            .into_owned();
                        assert_eq!(encoded, expected, "{case}: encode mismatch");

                        // decode direction (kernel with source and target swapped)
                        let mut expected_decoded = Vec::new();
                        let expected_decode_err =
                            encoded.chunks_exact(target_size).any(|element| {
                                target_trait
                                    .cast_value_cast(
                                        element,
                                        source_trait,
                                        rounding,
                                        out_of_range,
                                        &mut expected_decoded,
                                    )
                                    .is_err()
                            });
                        let decoded =
                            codec.decode(ArrayBytes::from(encoded.clone()), &shape, &options);
                        if expected_decode_err {
                            assert!(decoded.is_err(), "{case}: expected decode error");
                        } else {
                            let decoded = decoded
                                .unwrap_or_else(|err| panic!("{case} (decode): {err}"))
                                .into_fixed()
                                .unwrap()
                                .into_owned();
                            assert_eq!(decoded, expected_decoded, "{case}: decode mismatch");
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "microfloat")]
    #[test]
    fn codec_cast_value_repr_microfloat_fallback() {
        use zarrs_data_type::codec_traits::cast_value::CastValueDataTypeExt;
        // microfloats have no kernel representation and use the scalar path
        assert!(
            data_type::float8_e4m3()
                .codec_castvalue()
                .unwrap()
                .cast_value_repr()
                .is_none()
        );
        assert!(
            data_type::uint8()
                .codec_castvalue()
                .unwrap()
                .cast_value_repr()
                .is_some()
        );
    }

    #[test]
    fn codec_cast_value_from_metadata() {
        let metadata: zarrs_metadata::v3::MetadataV3 = serde_json::from_str(
            r#"{
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8"
                }
            }"#,
        )
        .unwrap();
        let codec = zarrs_codec::Codec::from_metadata(&metadata).unwrap();
        assert!(matches!(codec, zarrs_codec::Codec::ArrayToArray(_)));
    }
}

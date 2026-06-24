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

pub use cast_value_codec::{CastValueCodec, CastValueUnbound};
use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
use zarrs_metadata::v3::MetadataV3;
pub use zarrs_metadata_ext::codec::cast_value::{
    CastValueCodecConfiguration, CastValueCodecConfigurationV1, CastValueOutOfRangeMode,
    CastValueRoundingMode, CastValueScalarMap,
};

zarrs_plugin::impl_extension_aliases!(CastValueUnbound, v3: "cast_value");

inventory::submit! {
    CodecPluginV3::new::<CastValueUnbound>()
}

impl CodecTraitsV3 for CastValueUnbound {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: CastValueCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(CastValueUnbound::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

pub use zarrs_data_type::codec_traits::cast_value::{
    CastValueDataTypeExt, CastValueDataTypePlugin, CastValueDataTypeTraits,
    impl_cast_value_data_type_traits_float, impl_cast_value_data_type_traits_signed_integer,
    impl_cast_value_data_type_traits_unsigned_integer,
};

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

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
        let shape = [NonZeroU64::new(4).unwrap()];
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
        let shape = [NonZeroU64::new(4).unwrap()];
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
        let shape = [NonZeroU64::new(1).unwrap()];
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
        let shape = [NonZeroU64::new(2).unwrap()];
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
        let shape = [NonZeroU64::new(3).unwrap()];
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
        let shape = [NonZeroU64::new(1).unwrap()];
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
        let shape = [NonZeroU64::new(4).unwrap()];
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

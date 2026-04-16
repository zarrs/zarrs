//! The `cast_value` array to array codec.
//!
//! Cast array values to a new data type by converting their numerical values.
//!
//! ### Specification
//! - <https://github.com/zarrs/zarrs/blob/main/spec/cast_value.md>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `cast_value`

mod cast_value_codec;

use std::sync::Arc;

pub use cast_value_codec::CastValueCodec;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::cast_value::{
    CastValueCodecConfiguration, CastValueCodecConfigurationV1, CastValueOutOfRangeMode,
    CastValueRoundingMode, CastValueScalarMap,
};
use zarrs_plugin::PluginCreateError;

zarrs_plugin::impl_extension_aliases!(CastValueCodec, v3: "cast_value");

inventory::submit! {
    CodecPluginV3::new::<CastValueCodec>()
}

impl CodecTraitsV3 for CastValueCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: CastValueCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(CastValueCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
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

    const JSON_IDENTITY: &str = r#"{ "data_type": "float32" }"#;
    const JSON_UINT8_WRAP: &str = r#"{
        "data_type": "uint8",
        "rounding": "towards-zero",
        "out_of_range": "wrap"
    }"#;
    const JSON_UINT8_SCALAR_MAP: &str = r#"{
        "data_type": "uint8",
        "rounding": "towards-zero",
        "out_of_range": "wrap",
        "scalar_map": {
            "encode": [["NaN", 0]],
            "decode": [[0, 0]]
        }
    }"#;

    #[test]
    fn codec_cast_value_configuration_valid() {
        assert!(serde_json::from_str::<CastValueCodecConfiguration>(JSON_IDENTITY).is_ok());
        assert!(serde_json::from_str::<CastValueCodecConfiguration>(JSON_UINT8_WRAP).is_ok());
    }

    #[test]
    fn codec_cast_value_configuration_invalid_unknown_field() {
        assert!(serde_json::from_str::<CastValueCodecConfiguration>(
            r#"{ "data_type": "uint8", "unknown": true }"#
        )
        .is_err());
    }

    #[test]
    fn codec_cast_value_round_trip_identity() {
        let shape = vec![NonZeroU64::new(4).unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let elements = vec![0.0f32, 1.25, -3.5, 8.0];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements.clone()));

        let configuration: CastValueCodecConfiguration = serde_json::from_str(JSON_IDENTITY).unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();

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
        assert_eq!(decoded_elements, elements);
    }

    #[test]
    fn codec_cast_value_float_to_uint8_wrap() {
        let shape = vec![NonZeroU64::new(5).unwrap()];
        let data_type = data_type::float64();
        let fill_value = FillValue::from(0.0f64);
        let elements = vec![1.0f64, 127.9, 128.0, 255.0, 257.0];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements));

        let configuration: CastValueCodecConfiguration =
            serde_json::from_str(JSON_UINT8_WRAP).unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(
                bytes,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let encoded = encoded.into_fixed().unwrap().into_owned();
        assert_eq!(encoded, vec![1, 127, 128, 255, 1]);
    }

    #[test]
    fn codec_cast_value_scalar_map_nan() {
        let shape = vec![NonZeroU64::new(3).unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let elements = vec![f32::NAN, 2.0f32, 3.0];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements));

        let configuration: CastValueCodecConfiguration =
            serde_json::from_str(JSON_UINT8_SCALAR_MAP).unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(
                bytes,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let encoded = encoded.into_fixed().unwrap().into_owned();
        assert_eq!(encoded[0], 0);
    }

    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn codec_cast_value_partial_decode() {
        let configuration: CastValueCodecConfiguration =
            serde_json::from_str(JSON_IDENTITY).unwrap();
        let codec = Arc::new(CastValueCodec::new_with_configuration(&configuration).unwrap());

        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
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
                input_handle,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[3..6]);
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_region, &CodecOptions::default())
            .unwrap();
        let decoded_partial_chunk = crate::array::convert_from_bytes_slice::<f32>(
            &decoded_partial_chunk.into_fixed().unwrap(),
        );
        assert_eq!(decoded_partial_chunk, &[3.0, 4.0, 5.0]);
    }
}

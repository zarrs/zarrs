//! The `scale_offset` array to array codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! Subtracts an `offset` then multiplies by a `scale`, which is commonly used to map
//! floating-point values into the support range of an integer data type (a lossy form of
//! compression when followed by a data type narrowing codec such as `cast_value`).
//!
//! The data type is unchanged by this codec.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/scale_offset>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `scale_offset`
//!
//! ### Codec `configuration` Example - [`ScaleOffsetCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "offset": 5,
//!     "scale": 0.1
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::scale_offset::ScaleOffsetCodecConfiguration;
//! # let configuration: ScaleOffsetCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod scale_offset_codec;

use std::sync::Arc;

pub use scale_offset_codec::ScaleOffsetCodec;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::scale_offset::{
    ScaleOffsetCodecConfiguration, ScaleOffsetCodecConfigurationV1,
};
use zarrs_plugin::PluginCreateError;

zarrs_plugin::impl_extension_aliases!(ScaleOffsetCodec, v3: "scale_offset");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<ScaleOffsetCodec>()
}

impl CodecTraitsV3 for ScaleOffsetCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: ScaleOffsetCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(ScaleOffsetCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

// Re-export the trait and macro from zarrs_data_type
pub use zarrs_data_type::codec_traits::scale_offset::{
    ScaleOffsetDataTypeExt, ScaleOffsetDataTypePlugin, ScaleOffsetDataTypeTraits, ScaleOffsetError,
    impl_scale_offset_data_type_traits,
};

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use zarrs_codec::{ArrayToArrayCodecTraits, CodecOptions};
    use zarrs_data_type::FillValue;
    use zarrs_metadata_ext::codec::scale_offset::ScaleOffsetCodecConfiguration;

    use super::ScaleOffsetCodec;
    use crate::array::{ArrayBytes, data_type};

    fn codec(json: &str) -> ScaleOffsetCodec {
        let configuration: ScaleOffsetCodecConfiguration = serde_json::from_str(json).unwrap();
        ScaleOffsetCodec::new_with_configuration(&configuration).unwrap()
    }

    #[test]
    fn codec_scale_offset_float32_round_trip() {
        let shape = [NonZeroU64::new(5).unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let elements: Vec<f32> = vec![5.0, 5.1, 5.2, 5.3, 5.4];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements.clone()));

        let codec = codec(r#"{ "offset": 5, "scale": 0.1 }"#);
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
        for (decoded, original) in decoded_elements.iter().zip(&elements) {
            assert!((decoded - original).abs() < 1e-5);
        }
    }

    #[test]
    fn codec_scale_offset_uint16_offset_only() {
        // Spec example: uint16 values in [1000, 1255] shifted down by 1000.
        let shape = [NonZeroU64::new(4).unwrap()];
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(1000u16);
        let elements: Vec<u16> = vec![1000, 1100, 1200, 1255];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements));

        let codec = codec(r#"{ "offset": 1000 }"#);
        let encoded = codec
            .encode(
                bytes,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<u16>(
            encoded.clone().into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, &[0, 100, 200, 255]);

        let decoded = codec
            .decode(
                encoded,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<u16>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, &[1000, 1100, 1200, 1255]);
    }

    #[test]
    fn codec_scale_offset_empty_is_noop() {
        let shape = [NonZeroU64::new(3).unwrap()];
        let data_type = data_type::int32();
        let fill_value = FillValue::from(0i32);
        let elements: Vec<i32> = vec![-5, 0, 42];
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(elements.clone()));

        let codec = codec("{}");
        let encoded = codec
            .encode(
                bytes,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let encoded_elements = crate::array::transmute_from_bytes_vec::<i32>(
            encoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(encoded_elements, elements);
    }

    #[test]
    fn codec_scale_offset_integer_overflow_errors() {
        let shape = [NonZeroU64::new(1).unwrap()];
        let data_type = data_type::uint8();
        let fill_value = FillValue::from(0u8);
        // (200 - 0) * 2 = 400 > u8::MAX
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![200u8]));

        let codec = codec(r#"{ "scale": 2 }"#);
        let result = codec.encode(
            bytes,
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn codec_scale_offset_unsigned_negative_intermediate_errors() {
        let shape = [NonZeroU64::new(1).unwrap()];
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(0u16);
        // 500 - 1000 is negative, not representable in uint16
        let bytes = ArrayBytes::from(crate::array::transmute_to_bytes_vec(vec![500u16]));

        let codec = codec(r#"{ "offset": 1000 }"#);
        let result = codec.encode(
            bytes,
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn codec_scale_offset_encoded_fill_value() {
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(1000u16);
        let codec = codec(r#"{ "offset": 1000 }"#);
        let encoded = codec.encoded_fill_value(&data_type, &fill_value).unwrap();
        assert_eq!(encoded, FillValue::from(0u16));
    }

    #[test]
    fn codec_scale_offset_encoded_data_type_unchanged() {
        let data_type = data_type::float32();
        let codec = codec(r#"{ "offset": 5, "scale": 0.1 }"#);
        assert_eq!(codec.encoded_data_type(&data_type).unwrap(), data_type);
    }

    #[test]
    fn codec_scale_offset_unsupported_data_type() {
        let shape = [NonZeroU64::new(1).unwrap()];
        let data_type = data_type::bool();
        let fill_value = FillValue::from(false);
        let bytes = ArrayBytes::from(vec![0u8]);

        let codec = codec(r#"{ "offset": 0 }"#);
        let result = codec.encode(
            bytes,
            &shape,
            &data_type,
            &fill_value,
            &CodecOptions::default(),
        );
        assert!(result.is_err());
    }
}

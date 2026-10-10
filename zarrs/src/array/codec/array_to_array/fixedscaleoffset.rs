//! The `fixedscaleoffset` array to array codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! This codec is compatible with the `numcodecs.fixedscaleoffset` codec in `zarr-python`.
//! However, it supports additional data types not supported by that implementation.
//!
//! The transform is lossy, and decoded data may differ slightly from that of `numcodecs`:
//! - The transform is computed in `f32` for `float32` data (encoded data when decoding) and otherwise `f64`, and encoded values are rounded with ties to even, as in `numcodecs`.
//! - Decoded integers are rounded to the nearest integer, whereas `numcodecs` truncates them.
//! - `numcodecs` computes in the integer data type if `offset` and `scale` are integers in the metadata, which can overflow.
//! - Values out of the range of the encoded data type are saturated, which is unspecified in `numcodecs`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.fixedscaleoffset>
//! - <https://codec.zarrs.dev/array_to_array/fixedscaleoffset>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.fixedscaleoffset`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `fixedscaleoffset`
//!
//! ### Codec `configuration` Example - [`FixedScaleOffsetCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "offset": 1000,
//!     "scale": 10,
//!     "dtype": "f8",
//!     "astype": "u1"
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::fixedscaleoffset::FixedScaleOffsetCodecConfigurationNumcodecsF64;
//! # let configuration: FixedScaleOffsetCodecConfigurationNumcodecsF64 = serde_json::from_str(JSON).unwrap();
//! ```

mod fixedscaleoffset_codec;

use std::sync::Arc;

pub use fixedscaleoffset_codec::FixedScaleOffsetCodec;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
#[allow(deprecated)]
pub use zarrs_metadata_ext::codec::fixedscaleoffset::FixedScaleOffsetCodecConfigurationNumcodecs;
pub use zarrs_metadata_ext::codec::fixedscaleoffset::{
    FixedScaleOffsetCodecConfiguration, FixedScaleOffsetCodecConfigurationNumcodecsF64,
};

zarrs_plugin::impl_extension_aliases!(FixedScaleOffsetCodec,
    v3: "numcodecs.fixedscaleoffset", [],
    v2: "fixedscaleoffset", []
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<FixedScaleOffsetCodec>()
}
inventory::submit! {
    CodecPluginV2::new::<FixedScaleOffsetCodec>()
}

impl CodecTraitsV3 for FixedScaleOffsetCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: FixedScaleOffsetCodecConfiguration =
            metadata.to_typed_configuration()?;
        let codec = Arc::new(FixedScaleOffsetCodec::new_with_configuration(
            &configuration,
        )?);
        Ok(Codec::ArrayToArray(codec))
    }
}

impl CodecTraitsV2 for FixedScaleOffsetCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: FixedScaleOffsetCodecConfiguration =
            metadata.to_typed_configuration()?;
        let codec = Arc::new(FixedScaleOffsetCodec::new_with_configuration(
            &configuration,
        )?);
        Ok(Codec::ArrayToArray(codec))
    }
}

// Re-export the trait and macro from zarrs_data_type
pub use zarrs_data_type::codec_traits::fixedscaleoffset::{
    FixedScaleOffsetDataTypeExt, FixedScaleOffsetDataTypePlugin, FixedScaleOffsetDataTypeTraits,
    FixedScaleOffsetElementType, FixedScaleOffsetFloatType,
    impl_fixed_scale_offset_data_type_traits,
};

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use zarrs_data_type::FillValue;

    use crate::array::codec::array_to_array::fixedscaleoffset::FixedScaleOffsetCodec;
    use crate::array::{ArrayBytes, data_type};
    use zarrs_codec::{
        CodecOptions, CodecSpecificOptions, CodecTraits, UnboundArrayToArrayCodecTraits,
    };
    use zarrs_metadata_ext::codec::fixedscaleoffset::FixedScaleOffsetCodecConfiguration;

    #[test]
    fn codec_fixedscaleoffset() {
        // 1 sign bit, 8 exponent, 3 mantissa
        const JSON: &str = r#"{ "offset": 1000, "scale": 10, "dtype": "f8", "astype": "u1" }"#;
        let shape = [NonZeroU64::new(4).unwrap()];
        let data_type = data_type::float64();
        let fill_value = FillValue::from(0.0f64);
        let elements: Vec<f64> = vec![
            1000.,
            1000.11111111,
            1000.22222222,
            1000.33333333,
            1000.44444444,
            1000.55555556,
            1000.66666667,
            1000.77777778,
            1000.88888889,
            1001.,
        ];
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes = ArrayBytes::from(bytes);

        let codec_configuration: FixedScaleOffsetCodecConfiguration =
            serde_json::from_str(JSON).unwrap();
        let codec =
            Arc::new(FixedScaleOffsetCodec::new_with_configuration(&codec_configuration).unwrap())
                .with_context(data_type, fill_value, &CodecSpecificOptions::default())
                .unwrap();

        let encoded = codec
            .encode(bytes.clone(), &shape, &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &shape, &CodecOptions::default())
            .unwrap();
        let decoded_elements =
            crate::array::transmute_from_bytes_vec::<f64>(decoded.into_fixed().unwrap().into_vec());
        assert_eq!(
            decoded_elements,
            &[
                1000., 1000.1, 1000.2, 1000.3, 1000.4, 1000.6, 1000.7, 1000.8, 1000.9, 1001.
            ]
        );
    }

    /// Encode and decode `elements` with a codec with `configuration`, returning the encoded and decoded bytes.
    fn encode_decode(
        configuration: serde_json::Value,
        data_type: crate::array::DataType,
        elements: Vec<u8>,
        num_elements: u64,
    ) -> (Vec<u8>, Vec<u8>) {
        let shape = [NonZeroU64::new(num_elements).unwrap()];
        let codec_configuration: FixedScaleOffsetCodecConfiguration =
            serde_json::from_value(configuration).unwrap();
        let fill_value = FillValue::new(vec![0; data_type.fixed_size().unwrap()]);
        let codec =
            Arc::new(FixedScaleOffsetCodec::new_with_configuration(&codec_configuration).unwrap())
                .with_context(data_type, fill_value, &CodecSpecificOptions::default())
                .unwrap();
        let encoded = codec
            .encode(ArrayBytes::from(elements), &shape, &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded.clone(), &shape, &CodecOptions::default())
            .unwrap();
        (
            encoded.into_fixed().unwrap().into_vec(),
            decoded.into_fixed().unwrap().into_vec(),
        )
    }

    #[test]
    fn codec_fixedscaleoffset_numcodecs() {
        let int16 = |values: &[i16]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect()
        };
        // Rounded with ties to even, as in numcodecs 0.17.0
        let (encoded, _) = encode_decode(
            serde_json::json!({"offset": 0, "scale": 0.5, "dtype": "u1"}),
            data_type::uint8(),
            vec![1, 3, 5, 7],
            4,
        );
        assert_eq!(encoded, [0, 2, 2, 4]);

        // Computed in f64, as in numcodecs 0.17.0
        // numcodecs truncates decoded integers (to -8219 and 27509), whereas they are rounded
        let (encoded, decoded) = encode_decode(
            serde_json::json!({"offset": 0, "scale": 0.1, "dtype": "<i2"}),
            data_type::int16(),
            int16(&[-8220, 27507]),
            2,
        );
        assert_eq!(encoded, int16(&[-822, 2751]));
        assert_eq!(decoded, int16(&[-8220, 27510]));

        // Computed in f32 for float32 data, as in numcodecs 0.17.0 (376907 if computed in f64)
        let (encoded, _) = encode_decode(
            serde_json::json!({"offset": 0.1, "scale": 1000.3, "dtype": "<f4", "astype": "<i4"}),
            data_type::float32(),
            f32::from_bits(1_136_423_517).to_ne_bytes().to_vec(),
            1,
        );
        assert_eq!(encoded, 376_906i32.to_ne_bytes());

        // Encoded values out of the range of the data type, but in the range of `astype`
        // numcodecs 0.17.0 computes in `int8` with integer `offset` and `scale`, which overflows (to -48, 0, -10)
        let (encoded, decoded) = encode_decode(
            serde_json::json!({"offset": -100, "scale": 10, "dtype": "i1", "astype": "<i2"}),
            data_type::int8(),
            [100i8, -100, 27]
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            3,
        );
        assert_eq!(encoded, int16(&[2000, 0, 1270]));
        assert_eq!(decoded, [100i8.to_ne_bytes()[0], 156, 27]);
    }

    #[test]
    fn codec_fixedscaleoffset_configuration_f64() {
        // `offset` and `scale` are not rounded to f32 (e.g. to 0.10000000149011612)
        let configuration: FixedScaleOffsetCodecConfiguration = serde_json::from_value(
            serde_json::json!({"offset": 0.1, "scale": 0.3, "dtype": "<f8"}),
        )
        .unwrap();
        assert!(matches!(
            configuration,
            FixedScaleOffsetCodecConfiguration::NumcodecsF64(_)
        ));
        let codec = FixedScaleOffsetCodec::new_with_configuration(&configuration).unwrap();
        let configuration = codec
            .configuration_v3(&zarrs_codec::CodecMetadataOptions::default())
            .unwrap();
        assert_eq!(
            serde_json::to_string(&configuration).unwrap(),
            r#"{"offset":0.1,"scale":0.3,"dtype":"<f8","astype":null}"#
        );
    }

    #[test]
    fn codec_fixedscaleoffset_single_byte_dtypes() {
        // numcodecs writes single byte data types with the `|` byteorder (e.g. `|i1`), and it may be omitted
        for (dtype, data_type) in [
            ("|i1", data_type::int8()),
            ("i1", data_type::int8()),
            ("|u1", data_type::uint8()),
            ("u1", data_type::uint8()),
        ] {
            let codec_configuration: FixedScaleOffsetCodecConfiguration = serde_json::from_value(
                serde_json::json!({"offset": 0, "scale": 1, "dtype": dtype, "astype": dtype}),
            )
            .unwrap();
            Arc::new(FixedScaleOffsetCodec::new_with_configuration(&codec_configuration).unwrap())
                .with_context(
                    data_type,
                    FillValue::from(0u8),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
        }
    }

    #[test]
    fn codec_fixedscaleoffset_encoded_fill_value() {
        const JSON: &str = r#"{ "offset": 1000, "scale": 10, "dtype": "f8", "astype": "u1" }"#;
        let data_type = data_type::float64();
        let fill_value = FillValue::from(1000.3f64);

        let codec_configuration: FixedScaleOffsetCodecConfiguration =
            serde_json::from_str(JSON).unwrap();
        let codec =
            Arc::new(FixedScaleOffsetCodec::new_with_configuration(&codec_configuration).unwrap())
                .with_context(data_type, fill_value, &CodecSpecificOptions::default())
                .unwrap();

        assert_eq!(codec.encoded_fill_value(), &FillValue::from(3u8));
    }
}

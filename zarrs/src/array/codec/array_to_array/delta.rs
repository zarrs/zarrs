//! The `delta` array to array codec.
//!
//! Encodes data as the difference between adjacent values. Decoding reconstructs
//! the original values via cumulative sum.
//!
//! <div class="warning">
//! The `zarrs.delta` variant is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! The `numcodecs.delta` variant is fully compatible with the `numcodecs.delta` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/numcodecs/blob/main/src/numcodecs/delta.py>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.delta` — numcodecs compatible (requires `dtype` in configuration)
//! - `zarrs.delta` — zarrs experimental (no `dtype` required)
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `delta`
//!
//! ### Codec `configuration` Example - [`DeltaCodecConfigurationV1`] (`zarrs.delta`):
//! ```rust
//! # let JSON = r#"
//! {}
//! # "#;
//! # use zarrs::metadata_ext::codec::delta::DeltaCodecConfigurationV1;
//! # let configuration: DeltaCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! ### Codec `configuration` Example - [`DeltaCodecConfigurationNumcodecs`] (`numcodecs.delta` / V2 `delta`):
//! ```rust
//! # let JSON = r#"
//! {
//!     "dtype": "<i2",
//!     "astype": "<i4"
//! }
//! # "#;
//! # use zarrs::array::codec::DeltaCodecConfigurationNumcodecs;
//! # let configuration: DeltaCodecConfigurationNumcodecs = serde_json::from_str(JSON).unwrap();
//! ```

mod delta_codec;

use std::sync::Arc;

pub use delta_codec::{DeltaCodec, NumcodecsDeltaCodec};
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::delta::{
    DeltaCodecConfiguration, DeltaCodecConfigurationNumcodecs, DeltaCodecConfigurationV1,
};
use zarrs_plugin::PluginCreateError;

// zarrs.delta — V3 only
zarrs_plugin::impl_extension_aliases!(DeltaCodec, v3: "zarrs.delta", []);

// numcodecs.delta (V3) + delta (V2)
zarrs_plugin::impl_extension_aliases!(NumcodecsDeltaCodec,
    v3: "numcodecs.delta", [],
    v2: "delta"
);

// Register DeltaCodec as V3 only
inventory::submit! {
    CodecPluginV3::new::<DeltaCodec>()
}

// Register NumcodecsDeltaCodec as both V3 and V2
inventory::submit! {
    CodecPluginV3::new::<NumcodecsDeltaCodec>()
}
inventory::submit! {
    CodecPluginV2::new::<NumcodecsDeltaCodec>()
}

impl CodecTraitsV3 for DeltaCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        crate::warn_experimental_extension(metadata.name(), "codec");
        let configuration: DeltaCodecConfigurationV1 = metadata.to_typed_configuration()?;
        let codec = Arc::new(DeltaCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

impl CodecTraitsV3 for NumcodecsDeltaCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: DeltaCodecConfigurationNumcodecs = metadata.to_typed_configuration()?;
        let codec = Arc::new(NumcodecsDeltaCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

impl CodecTraitsV2 for NumcodecsDeltaCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, PluginCreateError> {
        let configuration: DeltaCodecConfigurationNumcodecs = metadata.to_typed_configuration()?;
        let codec = Arc::new(NumcodecsDeltaCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use zarrs_data_type::FillValue;

    use crate::array::{ArrayBytes, data_type, transmute_to_bytes_vec};
    use zarrs_codec::{ArrayToArrayCodecTraits, CodecOptions};

    use super::*;

    #[test]
    fn codec_delta_zarrs_configuration_valid() {
        let json = r#"{}"#;
        assert!(serde_json::from_str::<DeltaCodecConfigurationV1>(json).is_ok());
    }

    #[test]
    fn codec_delta_zarrs_configuration_invalid_unknown_field() {
        let json = r#"{"astype": "<i2"}"#;
        assert!(serde_json::from_str::<DeltaCodecConfigurationV1>(json).is_err());
    }

    #[test]
    fn codec_delta_numcodecs_configuration_valid() {
        let json = r#"{"dtype": "<i2"}"#;
        assert!(serde_json::from_str::<DeltaCodecConfigurationNumcodecs>(json).is_ok());
        let json = r#"{"dtype": "<i2", "astype": "<i4"}"#;
        assert!(serde_json::from_str::<DeltaCodecConfigurationNumcodecs>(json).is_ok());
    }

    #[test]
    fn codec_delta_numcodecs_configuration_invalid_missing_dtype() {
        let json = r#"{}"#;
        assert!(serde_json::from_str::<DeltaCodecConfigurationNumcodecs>(json).is_err());
    }

    #[test]
    fn codec_delta_zarrs_round_trip_i16() {
        let shape = [NonZeroU64::new(10).unwrap()];
        let data_type = data_type::int16();
        let fill_value = FillValue::from(0i16);
        let elements: Vec<i16> = (100..110).collect();
        let bytes = ArrayBytes::from(transmute_to_bytes_vec(elements.clone()));

        let codec = DeltaCodec::new();

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
        let decoded_elements = crate::array::transmute_from_bytes_vec::<i16>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, elements);
    }

    #[test]
    fn codec_delta_numcodecs_round_trip_i16() {
        let shape = [NonZeroU64::new(10).unwrap()];
        let data_type = data_type::int16();
        let fill_value = FillValue::from(0i16);
        let elements: Vec<i16> = (100..110).collect();
        let bytes = ArrayBytes::from(transmute_to_bytes_vec(elements.clone()));

        let configuration: DeltaCodecConfigurationNumcodecs =
            serde_json::from_str(r#"{"dtype": "<i2"}"#).unwrap();
        let codec = NumcodecsDeltaCodec::new_with_configuration(&configuration).unwrap();

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
        let decoded_elements = crate::array::transmute_from_bytes_vec::<i16>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, elements);
    }

    #[test]
    fn codec_delta_numcodecs_dtype_mismatch_error() {
        let shape = [NonZeroU64::new(4).unwrap()];
        let data_type = data_type::int32(); // wrong: codec expects i16
        let fill_value = FillValue::from(0i32);
        let elements: Vec<i32> = vec![1, 2, 3, 4];
        let bytes = ArrayBytes::from(transmute_to_bytes_vec(elements));

        let configuration: DeltaCodecConfigurationNumcodecs =
            serde_json::from_str(r#"{"dtype": "<i2"}"#).unwrap();
        let codec = NumcodecsDeltaCodec::new_with_configuration(&configuration).unwrap();

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
    fn codec_delta_zarrs_round_trip_f32() {
        let shape = [NonZeroU64::new(5).unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let elements: Vec<f32> = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let bytes = ArrayBytes::from(transmute_to_bytes_vec(elements.clone()));

        let codec = DeltaCodec::new();

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
    fn codec_delta_numcodecs_example() {
        // Example from the numcodecs docs:
        // x = [100, 102, 104, ..., 118] (dtype=i2)
        // codec = Delta(dtype='i2', astype='i1')
        // encoded = [100, 2, 2, 2, 2, 2, 2, 2, 2, 2] (dtype=i8)
        let shape = [NonZeroU64::new(10).unwrap()];
        let data_type = data_type::int16();
        let fill_value = FillValue::from(0i16);
        let elements: Vec<i16> = (0..10).map(|i| 100 + i * 2).collect();
        let bytes = ArrayBytes::from(transmute_to_bytes_vec(elements.clone()));

        let configuration: DeltaCodecConfigurationNumcodecs =
            serde_json::from_str(r#"{"dtype": "<i2", "astype": "|i1"}"#).unwrap();
        let codec = NumcodecsDeltaCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();

        // Verify encoded values: first=100, rest=2
        let encoded_bytes = encoded.clone().into_fixed().unwrap().into_owned();
        let encoded_i8 = crate::array::transmute_from_bytes_vec::<i8>(encoded_bytes);
        assert_eq!(encoded_i8[0], 100i8);
        for &v in &encoded_i8[1..] {
            assert_eq!(v, 2i8);
        }

        // Round-trip
        let decoded = codec
            .decode(
                encoded,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<i16>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(decoded_elements, elements);
    }
}

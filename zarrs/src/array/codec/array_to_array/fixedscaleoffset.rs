//! The `fixedscaleoffset` array to array codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.fixedscaleoffset` codec in `zarr-python`.
//! However, it supports additional data types not supported by that implementation.
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
//! # use zarrs::metadata_ext::codec::fixedscaleoffset::FixedScaleOffsetCodecConfigurationNumcodecs;
//! # let configuration: FixedScaleOffsetCodecConfigurationNumcodecs = serde_json::from_str(JSON).unwrap();
//! ```

mod fixedscaleoffset_codec;

use std::sync::Arc;

pub use fixedscaleoffset_codec::FixedScaleOffsetCodec;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use zarrs_codec::{Codec, CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::fixedscaleoffset::{
    FixedScaleOffsetCodecConfiguration, FixedScaleOffsetCodecConfigurationNumcodecs,
};
use zarrs_plugin::PluginCreateError;

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
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: FixedScaleOffsetCodecConfiguration =
            metadata.to_typed_configuration()?;
        let codec = Arc::new(FixedScaleOffsetCodec::new_with_configuration(
            &configuration,
        )?);
        Ok(Codec::ArrayToArray(codec))
    }
}

impl CodecTraitsV2 for FixedScaleOffsetCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, PluginCreateError> {
        let configuration: FixedScaleOffsetCodecConfiguration =
            metadata.to_typed_configuration()?;
        let codec = Arc::new(FixedScaleOffsetCodec::new_with_configuration(
            &configuration,
        )?);
        Ok(Codec::ArrayToArray(codec))
    }
}

// Re-export the trait and types from zarrs_data_type
pub use zarrs_data_type::codec_traits::{
    FixedScaleOffsetDataTypeExt, FixedScaleOffsetDataTypePlugin, FixedScaleOffsetDataTypeTraits,
    FixedScaleOffsetElementType, FixedScaleOffsetFloatType,
    impl_fixed_scale_offset_data_type_traits,
};

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use zarrs_data_type::FillValue;

    use crate::array::codec::array_to_array::fixedscaleoffset::FixedScaleOffsetCodec;
    use crate::array::{ArrayBytes, data_type};
    use zarrs_codec::{ArrayToArrayCodecTraits, CodecOptions};
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
        let codec = FixedScaleOffsetCodec::new_with_configuration(&codec_configuration).unwrap();

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
        let decoded_elements = crate::array::transmute_from_bytes_vec::<f64>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(
            decoded_elements,
            &[
                1000., 1000.1, 1000.2, 1000.3, 1000.4, 1000.6, 1000.7, 1000.8, 1000.9, 1001.
            ]
        );
    }
}

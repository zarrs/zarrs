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

pub use crate::metadata_ext::codec::fixedscaleoffset::{
    FixedScaleOffsetCodecConfiguration, FixedScaleOffsetCodecConfigurationNumcodecs,
};
use crate::registry::codec::FIXEDSCALEOFFSET;
use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(FIXEDSCALEOFFSET, is_identifier_fixedscaleoffset, create_codec_fixedscaleoffset)
}

fn is_identifier_fixedscaleoffset(identifier: &str) -> bool {
    identifier == FIXEDSCALEOFFSET
}

pub(crate) fn create_codec_fixedscaleoffset(
    metadata: &MetadataV3,
) -> Result<Codec, PluginCreateError> {
    let configuration: FixedScaleOffsetCodecConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(FIXEDSCALEOFFSET, "codec", metadata.to_string())
        })?;
    let codec = Arc::new(FixedScaleOffsetCodec::new_with_configuration(
        &configuration,
    )?);
    Ok(Codec::ArrayToArray(codec))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use crate::array::{
        codec::{
            array_to_array::fixedscaleoffset::FixedScaleOffsetCodec, ArrayToArrayCodecTraits,
            CodecOptions,
        },
        ArrayBytes, ChunkRepresentation, DataType,
    };
    use crate::metadata_ext::codec::fixedscaleoffset::FixedScaleOffsetCodecConfiguration;

    #[test]
    fn codec_fixedscaleoffset() {
        // 1 sign bit, 8 exponent, 3 mantissa
        const JSON: &'static str =
            r#"{ "offset": 1000, "scale": 10, "dtype": "f8", "astype": "u1" }"#;
        let chunk_representation =
            ChunkRepresentation::new(vec![NonZeroU64::new(4).unwrap()], DataType::Float64, 0.0f64)
                .unwrap();
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
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let decoded_elements = crate::array::transmute_from_bytes_vec::<f64>(
            decoded.into_fixed().unwrap().into_owned(),
        );
        assert_eq!(
            decoded_elements,
            &[1000., 1000.1, 1000.2, 1000.3, 1000.4, 1000.6, 1000.7, 1000.8, 1000.9, 1001.]
        );
    }
}

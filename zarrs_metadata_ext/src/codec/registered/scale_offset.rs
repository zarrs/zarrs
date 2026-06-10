//! `scale_offset` codec metadata.

use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::{ConfigurationSerialize, FillValueMetadata};

/// A wrapper to handle various versions of `scale_offset` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum ScaleOffsetCodecConfiguration {
    /// Version 1.0.
    V1(ScaleOffsetCodecConfigurationV1),
}

impl ConfigurationSerialize for ScaleOffsetCodecConfiguration {}

/// `scale_offset` codec configuration parameters (version 1.0).
///
/// The `offset` and `scale` values are JSON-encoded scalars using the Zarr V3 fill value encoding
/// for the input array's data type. A missing `offset` encodes the additive identity (`0`) and a
/// missing `scale` encodes the multiplicative identity (`1`).
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, Default)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ScaleOffsetCodecConfigurationV1 {
    /// The quantity subtracted from input values during encoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<FillValueMetadata>,
    /// The quantity multiplied with input values during encoding, after `offset` is subtracted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<FillValueMetadata>,
}

#[cfg(test)]
mod tests {
    use zarrs_metadata::v3::MetadataV3;

    use super::*;

    #[test]
    fn codec_scale_offset_metadata() {
        serde_json::from_str::<MetadataV3>(
            r#"{
                "name": "scale_offset",
                "configuration": {
                    "offset": 5,
                    "scale": 0.1
                }
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_scale_offset_config_offset_only() {
        let config: ScaleOffsetCodecConfiguration = serde_json::from_str(
            r#"{
                "offset": 1000
            }"#,
        )
        .unwrap();

        let ScaleOffsetCodecConfiguration::V1(config) = config;
        assert!(config.offset.is_some());
        assert!(config.scale.is_none());

        // Verify exact round-trip: absent optional fields stay absent
        let serialized = serde_json::to_value(&config).unwrap();
        assert!(serialized.get("offset").is_some());
        assert!(serialized.get("scale").is_none());
    }

    #[test]
    fn codec_scale_offset_config_empty() {
        let config: ScaleOffsetCodecConfiguration = serde_json::from_str("{}").unwrap();
        let ScaleOffsetCodecConfiguration::V1(config) = config;
        assert!(config.offset.is_none());
        assert!(config.scale.is_none());

        let serialized = serde_json::to_value(&config).unwrap();
        assert!(serialized.get("offset").is_none());
        assert!(serialized.get("scale").is_none());
    }

    #[test]
    fn codec_scale_offset_config_unknown_field() {
        assert!(serde_json::from_str::<ScaleOffsetCodecConfiguration>(
            r#"{ "offset": 5, "unknown": 1 }"#
        )
        .is_err());
    }
}

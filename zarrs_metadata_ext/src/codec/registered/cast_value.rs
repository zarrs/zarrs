//! `cast_value` codec metadata.

// NOTE: All optional fields use Option for exact metadata round tripping from other implementations.

use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{ConfigurationSerialize, FillValueMetadata};

/// A wrapper to handle various versions of `cast_value` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum CastValueCodecConfiguration {
    /// Version 1.0.
    V1(CastValueCodecConfigurationV1),
}

impl ConfigurationSerialize for CastValueCodecConfiguration {}

/// `cast_value` codec configuration parameters (version 1.0).
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct CastValueCodecConfigurationV1 {
    /// The encoded data type.
    pub data_type: MetadataV3,
    /// The rounding mode used when an exact cast is not possible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounding: Option<CastValueRoundingMode>,
    /// The handling of out-of-range values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_of_range: Option<CastValueOutOfRangeMode>,
    /// Optional exact scalar mappings applied before normal casting rules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_map: Option<CastValueScalarMap>,
}

/// The rounding mode for `cast_value`.
#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug, Display, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CastValueRoundingMode {
    /// Round to nearest with ties to even.
    #[default]
    NearestEven,
    /// Round towards zero.
    TowardsZero,
    /// Round towards positive infinity.
    TowardsPositive,
    /// Round towards negative infinity.
    TowardsNegative,
    /// Round to nearest with ties away from zero.
    NearestAway,
}

/// The out-of-range handling mode for `cast_value`.
#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug, Display)]
#[serde(rename_all = "kebab-case")]
pub enum CastValueOutOfRangeMode {
    /// Clamp the value to the target type range.
    Clamp,
    /// Wrap the value modulo `2^N`.
    Wrap,
}

/// Optional scalar maps for `cast_value`.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, Default)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct CastValueScalarMap {
    /// Exact mappings applied during encode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encode: Option<Vec<[FillValueMetadata; 2]>>,
    /// Exact mappings applied during decode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<Vec<[FillValueMetadata; 2]>>,
}

#[cfg(test)]
mod tests {
    use zarrs_metadata::v3::MetadataV3;

    use super::*;

    #[test]
    fn codec_cast_value_metadata() {
        serde_json::from_str::<MetadataV3>(
            r#"{
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8",
                    "rounding": "towards-zero",
                    "out_of_range": "wrap",
                    "scalar_map": {
                        "encode": [["NaN", 0]],
                        "decode": [[0, 0]]
                    }
                }
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_cast_value_config_scalar_map() {
        let config: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "uint8",
                "rounding": "towards-zero",
                "out_of_range": "wrap",
                "scalar_map": {
                    "encode": [
                        ["NaN", 0],
                        ["+Infinity", 0],
                        ["-Infinity", 0]
                    ]
                }
            }"#,
        )
        .unwrap();

        let CastValueCodecConfiguration::V1(config) = config;
        assert_eq!(config.rounding, Some(CastValueRoundingMode::TowardsZero));
        assert_eq!(config.out_of_range, Some(CastValueOutOfRangeMode::Wrap));
        assert!(config.scalar_map.is_some());

        let scalar_map = config.scalar_map.as_ref().unwrap();
        assert!(scalar_map.encode.is_some());
        assert!(scalar_map.decode.is_none());

        // Verify exact round-trip: present fields stay present, absent fields stay absent
        let serialized = serde_json::to_value(&config).unwrap();
        assert!(serialized.get("data_type").is_some());
        assert!(serialized.get("rounding").is_some());
        assert!(serialized.get("out_of_range").is_some());
        assert!(serialized.get("scalar_map").is_some());
        assert!(serialized.get("decode").is_none());
    }

    #[test]
    fn codec_cast_value_config_defaults() {
        let config: CastValueCodecConfiguration = serde_json::from_str(
            r#"{
                "data_type": "float32"
            }"#,
        )
        .unwrap();

        let CastValueCodecConfiguration::V1(config) = config;
        assert_eq!(config.rounding, None);
        assert_eq!(config.out_of_range, None);
        assert!(config.scalar_map.is_none());

        // Verify exact round-trip: absent optional fields stay absent
        let serialized = serde_json::to_value(&config).unwrap();
        assert!(serialized.get("rounding").is_none());
        assert!(serialized.get("out_of_range").is_none());
        assert!(serialized.get("scalar_map").is_none());

        let reserialized = serde_json::to_string(&config).unwrap();
        assert!(!reserialized.contains("rounding"));
        assert!(!reserialized.contains("out_of_range"));
        assert!(!reserialized.contains("scalar_map"));
    }
}

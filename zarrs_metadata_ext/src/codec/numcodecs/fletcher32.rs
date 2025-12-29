use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `fletcher32` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum Fletcher32CodecConfiguration {
    /// Version 1.0 draft.
    V1(Fletcher32CodecConfigurationV1),
}

impl ConfigurationSerialize for Fletcher32CodecConfiguration {}

/// `fletcher32` codec configuration parameters (version 1.0 draft).
///
/// ### Example (Zarr V3)
/// ```json
/// {
///     "name": "fletcher32",
///     "configuration": {}
/// }
/// ```
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct Fletcher32CodecConfigurationV1 {}

#[cfg(test)]
mod tests {
    use zarrs_metadata::v3::MetadataV3;

    use super::*;

    #[test]
    fn codec_fletcher32_config1() {
        serde_json::from_str::<Fletcher32CodecConfiguration>(r"{}").unwrap();
    }

    #[test]
    fn codec_fletcher32_config_outer1() {
        serde_json::from_str::<MetadataV3>(
            r#"{
            "name": "fletcher32",
            "configuration": {}
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_fletcher32_config_outer2() {
        serde_json::from_str::<MetadataV3>(
            r#"{
            "name": "fletcher32"
        }"#,
        )
        .unwrap();
    }
}

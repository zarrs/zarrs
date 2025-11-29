use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `bitround` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum BitroundCodecConfiguration {
    /// Version 1.0 draft.
    V1(BitroundCodecConfigurationV1),
}

impl ConfigurationSerialize for BitroundCodecConfiguration {}

/// `bitround` codec configuration parameters (version 1.0 draft).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
pub struct BitroundCodecConfigurationV1 {
    /// The number of mantissa bits to keep for a floating point data type.
    pub keepbits: u32,
}

#[cfg(test)]
mod tests {
    use zarrs_metadata::v3::MetadataV3;

    use super::*;

    #[test]
    fn codec_bitround_metadata() {
        serde_json::from_str::<MetadataV3>(
            r#"{ 
            "name": "bitround",
            "configuration": {
                "keepbits": 10
            }
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_bitround_config() {
        serde_json::from_str::<BitroundCodecConfiguration>(
            r#"{
                "keepbits": 10
            }"#,
        )
        .unwrap();
    }
}

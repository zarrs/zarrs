//! `optional` codec metadata (`zarrs` experimental).

use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `optional` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum OptionalCodecConfiguration {
    /// Version 1.0 draft.
    V1(OptionalCodecConfigurationV1),
}

impl ConfigurationSerialize for OptionalCodecConfiguration {}

/// `optional` codec configuration parameters (version 1.0 draft).
///
/// ### Example (Zarr V3)
/// ```json
/// {
///     "name": "optional",
///     "configuration": {
///         "mask_codecs": [
///             {
///                 "name": "packbits",
///                 "configuration": {}
///             },
///             {
///                 "name": "gzip",
///                 "configuration": {"level": 5}
///             }
///         ],
///         "data_codecs": [
///             {
///                 "name": "bytes",
///                 "configuration": {"endian": "little"}
///             },
///             {
///                 "name": "gzip",
///                 "configuration": {"level": 5}
///             }
///         ]
///     }
/// }
/// ```
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct OptionalCodecConfigurationV1 {
    /// The codec chain for encoding/decoding the mask (flattened bool array).
    pub mask_codecs: Vec<MetadataV3>,
    /// The codec chain for encoding/decoding the data (flattened bytes, excluding missing elements).
    pub data_codecs: Vec<MetadataV3>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_simple() {
        let configuration = serde_json::from_str::<OptionalCodecConfigurationV1>(
            r#"
        {
            "mask_codecs": [
                {
                    "name": "packbits",
                    "configuration": {}
                }
            ],
            "data_codecs": [
                {
                    "name": "bytes",
                    "configuration": {}
                }
            ]
        }
        "#,
        )
        .unwrap();
        assert_eq!(configuration.mask_codecs.len(), 1);
        assert_eq!(configuration.data_codecs.len(), 1);
    }

    #[test]
    fn optional_with_compression() {
        let configuration = serde_json::from_str::<OptionalCodecConfigurationV1>(
            r#"
        {
            "mask_codecs": [
                {
                    "name": "packbits",
                    "configuration": {}
                },
                {
                    "name": "gzip",
                    "configuration": {"level": 5}
                }
            ],
            "data_codecs": [
                {
                    "name": "bytes",
                    "configuration": {"endian": "little"}
                },
                {
                    "name": "gzip",
                    "configuration": {"level": 5}
                }
            ]
        }
        "#,
        )
        .unwrap();
        assert_eq!(configuration.mask_codecs.len(), 2);
        assert_eq!(configuration.data_codecs.len(), 2);
    }

    #[test]
    fn optional_missing_mask_codecs() {
        let config = r#"
        {
            "data_codecs": []
        }
        "#;
        assert!(serde_json::from_str::<OptionalCodecConfigurationV1>(config).is_err());
    }

    #[test]
    fn optional_missing_data_codecs() {
        let config = r#"
        {
            "mask_codecs": []
        }
        "#;
        assert!(serde_json::from_str::<OptionalCodecConfigurationV1>(config).is_err());
    }
}

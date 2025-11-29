use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::{ConfigurationSerialize, Endianness};

/// A wrapper to handle various versions of `bytes` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum BytesCodecConfiguration {
    /// Version 1.0.
    V1(BytesCodecConfigurationV1),
}

impl ConfigurationSerialize for BytesCodecConfiguration {}

/// `bytes` codec configuration parameters (version 1.0).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct BytesCodecConfigurationV1 {
    /// The target endianness. Required if the data type is larger than one byte.
    /// A string equal to either "big" or "little" in JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endian: Option<Endianness>,
}

impl BytesCodecConfigurationV1 {
    /// Create a new `bytes` codec configuration given an optional [`Endianness`].
    #[must_use]
    pub const fn new(endian: Option<Endianness>) -> Self {
        Self { endian }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_bytes_big() {
        serde_json::from_str::<BytesCodecConfiguration>(r#"{"endian":"big"}"#).unwrap();
    }

    #[test]
    fn codec_bytes_little() {
        serde_json::from_str::<BytesCodecConfiguration>(r#"{"endian":"little"}"#).unwrap();
    }

    #[test]
    fn codec_bytes_empty() {
        serde_json::from_str::<BytesCodecConfiguration>(r#"{}"#).unwrap();
    }

    #[test]
    fn codec_bytes_invalid() {
        assert!(serde_json::from_str::<BytesCodecConfiguration>(r#"{"endian":""}"#).is_err());
    }
}

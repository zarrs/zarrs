use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `vlen-array` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum VlenArrayCodecConfiguration {
    /// Version 1.0 draft.
    V1(VlenArrayCodecConfigurationV1),
}

impl ConfigurationSerialize for VlenArrayCodecConfiguration {}

/// `vlen-array` codec configuration parameters (version 1.0 draft).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct VlenArrayCodecConfigurationV1 {}

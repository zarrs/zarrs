use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `squeeze` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum SqueezeCodecConfiguration {
    /// Version 0.0 draft.
    V0(SqueezeCodecConfigurationV0),
}

impl ConfigurationSerialize for SqueezeCodecConfiguration {}

/// `squeeze` codec configuration parameters (version 0.0 draft).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct SqueezeCodecConfigurationV0 {}

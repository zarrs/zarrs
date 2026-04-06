use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `zarrs.delta` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum DeltaCodecConfiguration {
    /// Version 1.0.
    V1(DeltaCodecConfigurationV1),
}

impl ConfigurationSerialize for DeltaCodecConfiguration {}

/// `zarrs.delta` codec configuration parameters (version 1.0).
///
/// Unlike the `numcodecs.delta` variant, no `dtype` field is required.
/// The output has the same numeric type as the input.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct DeltaCodecConfigurationV1 {}

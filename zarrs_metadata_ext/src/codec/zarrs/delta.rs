use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::v3::MetadataV3;
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
/// Unlike the `numcodecs.delta` variant, no `dtype` field is required. The codec operates on
/// whatever numeric type the array uses.
///
/// # Notes
/// If `astype` is an integer type, it must be large enough to store both the absolute value
/// of the first element and all delta values. No overflow checks are performed.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct DeltaCodecConfigurationV1 {
    /// Zarr V3 data type for encoded (delta) data.
    ///
    /// Defaults to the array data type if absent.
    pub astype: Option<MetadataV3>,
}

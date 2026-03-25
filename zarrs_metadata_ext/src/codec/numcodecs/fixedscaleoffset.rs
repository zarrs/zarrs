use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `fixedscaleoffset` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum FixedScaleOffsetCodecConfiguration {
    /// `numcodecs` version 0.0.0.
    Numcodecs(FixedScaleOffsetCodecConfigurationNumcodecs),
}

impl ConfigurationSerialize for FixedScaleOffsetCodecConfiguration {}

/// `fixedscaleoffset` codec configuration parameters (numcodecs).
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct FixedScaleOffsetCodecConfigurationNumcodecs {
    /// Value to subtract from data.
    pub offset: f32,
    /// Value to multiply by data.
    pub scale: f32,
    /// Zarr V2 data type to use for decoded data.
    ///
    /// The byte order (|, <, >) can be omitted, but must be valid for the data type if present.
    pub dtype: String,
    /// Zarr V2 data type to use for encoded data.
    ///
    /// The byte order (|, <, >) can be omitted, but must be valid for the data type if present.
    pub astype: Option<String>,
}

#![allow(deprecated)]

use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `fixedscaleoffset` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum FixedScaleOffsetCodecConfiguration {
    /// `numcodecs` version 0.0.0.
    NumcodecsF64(FixedScaleOffsetCodecConfigurationNumcodecsF64),
    /// `numcodecs` version 0.0.0, with `offset` and `scale` rounded to `f32`.
    ///
    /// This variant is never deserialized, [`FixedScaleOffsetCodecConfiguration::NumcodecsF64`] is preferred.
    #[deprecated(
        since = "0.4.5",
        note = "rounds `offset` and `scale` to `f32`, use `NumcodecsF64` instead"
    )]
    Numcodecs(FixedScaleOffsetCodecConfigurationNumcodecs),
}

impl ConfigurationSerialize for FixedScaleOffsetCodecConfiguration {}

/// `fixedscaleoffset` codec configuration parameters (numcodecs).
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct FixedScaleOffsetCodecConfigurationNumcodecsF64 {
    /// Value to subtract from data.
    pub offset: f64,
    /// Value to multiply by data.
    pub scale: f64,
    /// Zarr V2 data type to use for decoded data.
    ///
    /// The byte order (|, <, >) can be omitted, but must be valid for the data type if present.
    pub dtype: String,
    /// Zarr V2 data type to use for encoded data.
    ///
    /// The byte order (|, <, >) can be omitted, but must be valid for the data type if present.
    pub astype: Option<String>,
}

/// `fixedscaleoffset` codec configuration parameters (numcodecs), with `offset` and `scale` rounded to `f32`.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
#[deprecated(
    since = "0.4.5",
    note = "rounds `offset` and `scale` to `f32`, use `FixedScaleOffsetCodecConfigurationNumcodecsF64` instead"
)]
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

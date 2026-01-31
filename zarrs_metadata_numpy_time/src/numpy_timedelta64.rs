//! `numpy.timedelta64` data type metadata.

use std::num::NonZeroU32;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

use crate::NumpyTimeUnit;

/// The `numpy.timedelta64` data type configuration.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct NumpyTimeDelta64DataTypeConfigurationV1 {
    /// The `NumPy` temporal unit.
    pub unit: NumpyTimeUnit,
    /// The `NumPy` temporal scale factor.
    pub scale_factor: NonZeroU32, // 31
}

impl ConfigurationSerialize for NumpyTimeDelta64DataTypeConfigurationV1 {}

// /// A wrapper to handle various versions of `numpy.timedelta64` data type configuration parameters.
// #[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
// #[non_exhaustive]
// #[serde(untagged)]
// pub enum NumpyTimeDelta64DataTypeConfiguration {
//     /// Version 1.0.
//     V1(NumpyTimeDelta64DataTypeConfigurationV1),
// }

//! `numpy.datetim64` data type metadata.

use std::num::NonZeroU32;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

use super::numpy_time_unit::NumpyTimeUnit;

/// The `numpy.datetime64` data type configuration.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct NumpyDateTime64DataTypeConfigurationV1 {
    /// The `NumPy` temporal unit.
    pub unit: NumpyTimeUnit,
    /// The `NumPy` temporal scale factor.
    pub scale_factor: NonZeroU32, // 31
}

impl ConfigurationSerialize for NumpyDateTime64DataTypeConfigurationV1 {}

// /// A wrapper to handle various versions of `numpy.datetime64` data type configuration parameters.
// #[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
// #[non_exhaustive]
// #[serde(untagged)]
// pub enum NumpyDateTime4DataTypeConfiguration {
//     /// Version 1.0.
//     V1(NumpyDateTime64DataTypeConfigurationV1),
// }

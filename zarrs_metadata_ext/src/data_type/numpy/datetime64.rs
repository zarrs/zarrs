use std::num::NonZeroU32;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

use super::time_unit::NumpyTimeUnit;

/// The `numpy.datetime64` data type configuration.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct NumpyDateTime64DataTypeConfiguration {
    /// The `NumPy` temporal unit.
    pub unit: NumpyTimeUnit,
    /// The `NumPy` temporal scale factor.
    pub scale_factor: NonZeroU32, // 31
}

impl ConfigurationSerialize for NumpyDateTime64DataTypeConfiguration {}

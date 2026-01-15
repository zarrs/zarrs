//! `fixed_length_utf32` data type metadata.
//!
//! See <https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/fixed-length-utf32>.

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// The `fixed_length_utf32` data type configuration.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct FixedLengthUtf32DataTypeConfigurationV1 {
    /// The length in bytes.
    ///
    /// Must be divisible by 4 and in range `[0, 2147483644]`.
    pub length_bytes: u32,
}

impl ConfigurationSerialize for FixedLengthUtf32DataTypeConfigurationV1 {}

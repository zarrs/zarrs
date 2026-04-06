use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::v2::DataTypeMetadataV2;
use zarrs_metadata::ConfigurationSerialize;

impl ConfigurationSerialize for DeltaCodecConfigurationNumcodecs {}

/// `numcodecs.delta` / V2 `delta` codec configuration parameters.
///
/// The `dtype` field specifies the data type of the decoded (original) array and is validated
/// at encode/decode time to ensure the input matches. The `astype` field optionally specifies
/// the data type for the encoded (delta) values; if absent, the same type as `dtype` is used.
///
/// # Notes
/// If `astype` is an integer type, it must be large enough to store both the absolute value
/// of the first element and all delta values. No overflow checks are performed.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct DeltaCodecConfigurationNumcodecs {
    /// Zarr V2 data type for decoded (original) data.
    pub dtype: DataTypeMetadataV2,
    /// Zarr V2 data type for encoded (delta) data.
    ///
    /// Defaults to `dtype` if absent.
    pub astype: Option<DataTypeMetadataV2>,
}

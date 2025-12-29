use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `vlen_v2` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum VlenV2CodecConfiguration {
    /// Version 0.0 draft.
    V0(VlenV2CodecConfigurationV0),
}

impl ConfigurationSerialize for VlenV2CodecConfiguration {}

/// `vlen_v2` codec configuration parameters (version 0.0 draft).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, Default)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct VlenV2CodecConfigurationV0 {}

impl VlenV2CodecConfigurationV0 {
    /// Create a new `vlen_v2` codec configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_vlen_v2() {
        serde_json::from_str::<VlenV2CodecConfiguration>(r"{}").unwrap();
    }
}

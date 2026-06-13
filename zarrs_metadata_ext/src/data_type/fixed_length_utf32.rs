//! `fixed_length_utf32` data type metadata.

use std::num::NonZeroU64;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// The `fixed_length_utf32` data type configuration (version 1.0).
#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct FixedLengthUTF32DataTypeConfigurationV1 {
    /// The length of each element in bytes (must be a multiple of 4, at least 4).
    pub length_bytes: NonZeroU64,
}

impl ConfigurationSerialize for FixedLengthUTF32DataTypeConfigurationV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_config() {
        let config: FixedLengthUTF32DataTypeConfigurationV1 =
            serde_json::from_str(r#"{"length_bytes":16}"#).unwrap();
        assert_eq!(config.length_bytes, NonZeroU64::new(16).unwrap());
    }

    #[test]
    fn missing_length_bytes() {
        assert!(serde_json::from_str::<FixedLengthUTF32DataTypeConfigurationV1>(r#"{}"#).is_err());
    }
}

//! `optional` data type metadata.

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::{Configuration, ConfigurationSerialize};

/// The `optional` data type configuration (version 1.0).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct OptionalDataTypeConfigurationV1 {
    /// The inner data type name.
    pub name: String,
    /// The configuration of the inner data type.
    pub configuration: Configuration,
}

impl ConfigurationSerialize for OptionalDataTypeConfigurationV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_int32() {
        let config = r#"{"name":"int32","configuration":{}}"#;
        serde_json::from_str::<OptionalDataTypeConfigurationV1>(config).unwrap();
    }

    #[test]
    fn optional_complex() {
        let config = r#"{"name":"numpy.datetime64","configuration":{"unit":"s","scale_factor":1}}"#;
        serde_json::from_str::<OptionalDataTypeConfigurationV1>(config).unwrap();
    }

    #[test]
    fn optional_missing_name() {
        let config = r#"{"configuration":{}}"#;
        assert!(serde_json::from_str::<OptionalDataTypeConfigurationV1>(config).is_err());
    }

    #[test]
    fn optional_missing_configuration() {
        let config = r#"{"name":"int32"}"#;
        assert!(serde_json::from_str::<OptionalDataTypeConfigurationV1>(config).is_err());
    }
}

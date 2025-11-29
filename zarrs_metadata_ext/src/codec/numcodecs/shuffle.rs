use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `shuffle` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum ShuffleCodecConfiguration {
    /// Version 1.0 draft.
    V1(ShuffleCodecConfigurationV1),
}

impl ConfigurationSerialize for ShuffleCodecConfiguration {}

/// `shuffle` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ShuffleCodecConfigurationV1 {
    /// The element size.
    pub elementsize: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle() {
        let configuration = serde_json::from_str::<ShuffleCodecConfigurationV1>(
            r#"
        {
            "elementsize": 4
        }
        "#,
        )
        .unwrap();
        assert_eq!(configuration.elementsize, 4);
    }
}

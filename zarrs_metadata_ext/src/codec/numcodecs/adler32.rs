use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `adler32` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum Adler32CodecConfiguration {
    /// Version 1.0 draft.
    V1(Adler32CodecConfigurationV1),
}

impl ConfigurationSerialize for Adler32CodecConfiguration {}

/// `adler32` codec configuration parameters (version 1.0 draft).
///
/// ### Example (Zarr V3)
/// ```json
/// {
///     "name": "adler32",
///     "configuration": {}
/// }
/// ```
///
/// ```json
/// {
///     "name": "adler32",
///     "configuration": {
///         "location": "end"
///     }
/// }
/// ```
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct Adler32CodecConfigurationV1 {
    /// The location of the checksum.
    #[serde(
        default = "checksum_location_default",
        skip_serializing_if = "checksum_location_is_default"
    )]
    pub location: Adler32CodecConfigurationChecksumLocation,
}

/// The location of the checksum.
#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug, Display, Default)]
#[serde(rename_all = "lowercase")]
pub enum Adler32CodecConfigurationChecksumLocation {
    #[default]
    /// The checksum is encoded at the start of the byte sequence.
    Start,
    /// The checksum is encoded at the end of the byte sequence.
    End,
}

fn checksum_location_default() -> Adler32CodecConfigurationChecksumLocation {
    Adler32CodecConfigurationChecksumLocation::default()
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn checksum_location_is_default(location: &Adler32CodecConfigurationChecksumLocation) -> bool {
    *location == Adler32CodecConfigurationChecksumLocation::default()
}

#[cfg(test)]
mod tests {
    use zarrs_metadata::v3::MetadataV3;

    use super::*;

    #[test]
    fn codec_adler32_config1() {
        serde_json::from_str::<Adler32CodecConfiguration>(r"{}").unwrap();
    }

    #[test]
    fn codec_adler32_config_outer1() {
        serde_json::from_str::<MetadataV3>(
            r#"{
            "name": "adler32",
            "configuration": {
                "location": "end"
            }
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_adler32_config_outer2() {
        serde_json::from_str::<MetadataV3>(
            r#"{
            "name": "adler32"
        }"#,
        )
        .unwrap();
    }
}

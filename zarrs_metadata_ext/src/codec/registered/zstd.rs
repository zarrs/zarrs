use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `zstd` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum ZstdCodecConfiguration {
    /// Version 1.0 / `numcodecs` version 0.13.
    V1(ZstdCodecConfigurationV1),
    /// `numcodecs` version 0.1.
    Numcodecs(ZstdCodecConfigurationNumcodecs),
}

impl ConfigurationSerialize for ZstdCodecConfiguration {}

/// `zstd` codec configuration parameters (version 1.0).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ZstdCodecConfigurationV1 {
    /// The compression level.
    pub level: ZstdCompressionLevel,
    /// A boolean that indicates whether to store a checksum when writing that will be verified when reading.
    pub checksum: bool,
}

impl ZstdCodecConfigurationV1 {
    /// Create a new `zstd` codec configuration given a [`ZstdCompressionLevel`].
    #[must_use]
    pub const fn new(level: ZstdCompressionLevel, checksum: bool) -> Self {
        Self { level, checksum }
    }
}

/// `zstd` codec configuration parameters (`numcodecs` version 0.1).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ZstdCodecConfigurationNumcodecs {
    /// The compression level.
    pub level: ZstdCompressionLevel,
}

/// Convert [`ZstdCodecConfigurationNumcodecs`] to [`ZstdCodecConfiguration`].
#[must_use]
pub fn codec_zstd_v2_numcodecs_to_v3(zstd: &ZstdCodecConfiguration) -> ZstdCodecConfiguration {
    match zstd {
        ZstdCodecConfiguration::V1(zstd) => ZstdCodecConfiguration::V1(zstd.clone()),
        ZstdCodecConfiguration::Numcodecs(ZstdCodecConfigurationNumcodecs { level }) => {
            ZstdCodecConfiguration::V1(ZstdCodecConfigurationV1::new(*level, false))
        }
    }
}

/// A `Zstd` compression level. An integer from -131072 to 22 which controls the speed and level of compression (has no impact on decoding).
///
/// A value of 0 indicates to use the default compression level.
/// Otherwise, a higher level is expected to achieve a higher compression ratio at the cost of lower speed.
#[derive(Serialize, Clone, Copy, Eq, PartialEq, Debug)]
pub struct ZstdCompressionLevel(i32);

impl<'de> serde::Deserialize<'de> for ZstdCompressionLevel {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(d)?;
        match value {
            Value::Number(number) => {
                if let Some(number) = number.as_i64() {
                    if (-131_072..=22).contains(&number) {
                        #[allow(clippy::cast_possible_truncation)]
                        return Ok(Self(number as i32));
                    }
                }
            }
            Value::String(string) => {
                // COMPATIBILITY: support data created with zarr-python that uses a string for the level
                // https://github.com/zarr-developers/zarr-python/blob/a52048ddb2d5d069c3404e7457439a9ecb5e40c3/tests/test_v2.py#L278-L280
                if let Ok(number) = string.parse::<i64>() {
                    if (-131_072..=22).contains(&number) {
                        #[allow(clippy::cast_possible_truncation)]
                        return Ok(Self(number as i32));
                    }
                }
            }
            _ => {}
        }
        Err(serde::de::Error::custom(
            "Zstd compression level must be an integer between -131072 and 22",
        ))
    }
}

impl ZstdCompressionLevel {
    /// Create a new `Zstd` compression level.
    #[must_use]
    pub const fn new(level: i32) -> Self {
        Self(level)
    }
}

impl From<i32> for ZstdCompressionLevel {
    fn from(value: i32) -> Self {
        Self(value)
    }
}

impl From<ZstdCompressionLevel> for i32 {
    fn from(value: ZstdCompressionLevel) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_zstd_configuration_valid() {
        const JSON_VALID: &str = r#"{
        "level": 22,
        "checksum": false
    }"#;
        serde_json::from_str::<ZstdCodecConfiguration>(JSON_VALID).unwrap();
    }

    // #[test]
    // fn codec_zstd_configuration_invalid1() {
    //     const JSON_INVALID1: &str = r#"{
    //     "level": 5
    // }"#;
    //     assert!(serde_json::from_str::<ZstdCodecConfiguration>(JSON_INVALID1).is_err());
    // }

    #[test]
    fn codec_zstd_configuration_invalid2() {
        const JSON_INVALID2: &str = r#"{
        "level": 23,
        "checksum": true
    }"#;
        assert!(serde_json::from_str::<ZstdCodecConfiguration>(JSON_INVALID2).is_err());
    }
}

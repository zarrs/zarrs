use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `gzip` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum GzipCodecConfiguration {
    /// Version 1.0.
    V1(GzipCodecConfigurationV1),
}

impl ConfigurationSerialize for GzipCodecConfiguration {}

/// `gzip` codec configuration parameters (version 1.0).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct GzipCodecConfigurationV1 {
    /// The compression level.
    pub level: GzipCompressionLevel,
}

impl GzipCodecConfigurationV1 {
    /// Create a new `gzip` codec configuration given a [`GzipCompressionLevel`].
    #[must_use]
    pub const fn new(level: GzipCompressionLevel) -> Self {
        Self { level }
    }
}

/// A compression level. Used by the `gzip` codec.
///
/// An integer from 0 to 9 which controls the speed and level of compression.
/// A level of 1 is the fastest compression method and produces the least compressions, while 9 is slowest and produces the most compression.
/// Compression is turned off completely when level is 0.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Display)]
pub struct GzipCompressionLevel(u32);

/// An invalid compression level.
#[derive(Debug, thiserror::Error)]
#[error("Invalid compression level {0}, must be 0-9")]
pub struct GzipCompressionLevelError(u32);

impl TryFrom<u32> for GzipCompressionLevel {
    type Error = GzipCompressionLevelError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value < 10 {
            Ok(Self(value))
        } else {
            Err(GzipCompressionLevelError(value))
        }
    }
}

impl serde::Serialize for GzipCompressionLevel {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u32(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for GzipCompressionLevel {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        if let serde_json::Value::Number(level) = value {
            if let Some(level) = level.as_u64().and_then(|level| u32::try_from(level).ok()) {
                if level < 10 {
                    return Ok(Self(level));
                }
            }
        }
        Err(serde::de::Error::custom(
            "compression level must be an integer between 0 and 9.",
        ))
    }
}

impl GzipCompressionLevel {
    /// Create a new compression level.
    ///
    /// # Errors
    /// Errors if `compression_level` is not between 0-9.
    pub fn new<N: num::Unsigned + std::cmp::PartialOrd<u32>>(
        compression_level: N,
    ) -> Result<Self, N>
    where
        u32: From<N>,
    {
        if compression_level < 10 {
            Ok(Self(u32::from(compression_level)))
        } else {
            Err(compression_level)
        }
    }

    /// The underlying integer compression level.
    #[must_use]
    pub const fn as_u32(&self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_gzip_configuration_valid() {
        const JSON_VALID: &str = r#"{
            "level": 1
        }"#;
        serde_json::from_str::<GzipCodecConfiguration>(JSON_VALID).unwrap();
    }

    #[test]
    fn codec_gzip_configuration_invalid1() {
        const JSON_INVALID1: &str = r#"{
            "level": -1
        }"#;
        assert!(serde_json::from_str::<GzipCodecConfiguration>(JSON_INVALID1).is_err());
    }

    #[test]
    fn codec_gzip_configuration_invalid2() {
        const JSON_INVALID2: &str = r#"{
            "level": 10
        }"#;
        assert!(serde_json::from_str::<GzipCodecConfiguration>(JSON_INVALID2).is_err());
    }
}

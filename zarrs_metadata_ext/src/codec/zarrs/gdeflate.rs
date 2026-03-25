use derive_more::{Display, From};
use serde::{Deserialize, Deserializer, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `gdeflate` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum GDeflateCodecConfiguration {
    /// Version 0.0 draft.
    V0(GDeflateCodecConfigurationV0),
}

impl ConfigurationSerialize for GDeflateCodecConfiguration {}

/// `gdeflate` codec configuration parameters (version 0.0 draft).
///
/// ### Example: encode with a compression level of 12
/// ```rust
/// # let JSON = r#"
/// {
///     "level": 12
/// }
/// # "#;
/// # use zarrs_metadata_ext::codec::gdeflate::GDeflateCodecConfigurationV0;
/// # let configuration: GDeflateCodecConfigurationV0 = serde_json::from_str(JSON).unwrap();
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct GDeflateCodecConfigurationV0 {
    /// The compression level.
    pub level: GDeflateCompressionLevel,
}

/// An integer from 0 to 12 controlling the compression level
///
/// A level of 1 is the fastest compression method and produces the least compression, while 12 is slowest and produces the most compression.
/// Compression is turned off when the compression level is 0.
#[derive(Serialize, Copy, Clone, Debug, Eq, PartialEq)]
pub struct GDeflateCompressionLevel(u32);

/// An invalid gdeflate compression level.
#[derive(Debug, thiserror::Error)]
#[error("Invalid gdeflate compression level {0}, must be 0-12")]
pub struct GDeflateCompressionLevelError(u32);

macro_rules! gdeflate_compression_level_try_from {
    ( $t:ty ) => {
        impl TryFrom<$t> for GDeflateCompressionLevel {
            type Error = GDeflateCompressionLevelError;
            fn try_from(level: $t) -> Result<Self, Self::Error> {
                let level = u32::from(level);
                if level <= 12 {
                    Ok(Self(level))
                } else {
                    Err(GDeflateCompressionLevelError(level))
                }
            }
        }
    };
}

gdeflate_compression_level_try_from!(u8);
gdeflate_compression_level_try_from!(u16);
gdeflate_compression_level_try_from!(u32);

impl<'de> Deserialize<'de> for GDeflateCompressionLevel {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let level = u32::deserialize(d)?;
        if level <= 12 {
            Ok(Self(level))
        } else {
            Err(serde::de::Error::custom(
                "gdeflate compression level must be between 0 and 9",
            ))
        }
    }
}

impl GDeflateCompressionLevel {
    /// Create a new compression level.
    ///
    /// # Errors
    /// Errors if `compression_level` is not between 0-12.
    pub fn new<N: num::Unsigned + std::cmp::PartialOrd<u32>>(
        compression_level: N,
    ) -> Result<Self, GDeflateCompressionLevelError>
    where
        u32: From<N>,
    {
        let compression_level = u32::from(compression_level);
        if compression_level <= 12 {
            Ok(Self(compression_level))
        } else {
            Err(GDeflateCompressionLevelError(compression_level))
        }
    }

    /// The underlying integer compression level.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)] // it won't wrap
    pub fn as_i32(&self) -> i32 {
        self.0 as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_gdeflate_valid1() {
        let json = r#"
        {
            "level": 12
        }"#;
        assert!(serde_json::from_str::<GDeflateCodecConfiguration>(json).is_ok());
    }

    #[test]
    fn codec_gdeflate_invalid_level() {
        let json = r#"
        {
            "level": 13
        }"#;
        let codec_configuration = serde_json::from_str::<GDeflateCodecConfiguration>(json);
        assert!(codec_configuration.is_err());
    }
}

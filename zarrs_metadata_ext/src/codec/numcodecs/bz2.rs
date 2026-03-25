use derive_more::{Display, From};
use serde::{Deserialize, Deserializer, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `bz2` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum Bz2CodecConfiguration {
    /// Version 1.0 draft.
    V1(Bz2CodecConfigurationV1),
}

impl ConfigurationSerialize for Bz2CodecConfiguration {}

/// `bz2` codec configuration parameters (version 1.0 draft).
///
/// ### Example: encode with a compression level of 9
/// ```rust
/// # let JSON = r#"
/// {
///     "level": 9
/// }
/// # "#;
/// # use zarrs_metadata_ext::codec::bz2::Bz2CodecConfigurationV1;
/// # let configuration: Bz2CodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct Bz2CodecConfigurationV1 {
    /// The compression level.
    pub level: Bz2CompressionLevel,
}

/// An integer from 0 to 9 controlling the compression level
///
/// A level of 1 is the fastest compression method and produces the least compression, while 9 is slowest and produces the most compression.
/// Compression is turned off when the compression level is 0.
#[derive(Serialize, Copy, Clone, Debug, Eq, PartialEq)]
pub struct Bz2CompressionLevel(u32);

macro_rules! bz2_compression_level_try_from {
    ( $t:ty ) => {
        impl TryFrom<$t> for Bz2CompressionLevel {
            type Error = $t;
            fn try_from(level: $t) -> Result<Self, Self::Error> {
                if level <= 9 {
                    Ok(Self(u32::from(level)))
                } else {
                    Err(level)
                }
            }
        }
    };
}

bz2_compression_level_try_from!(u8);
bz2_compression_level_try_from!(u16);
bz2_compression_level_try_from!(u32);

impl<'de> Deserialize<'de> for Bz2CompressionLevel {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let level = u32::deserialize(d)?;
        if level <= 9 {
            Ok(Self(level))
        } else {
            Err(serde::de::Error::custom(
                "bz2 compression level must be between 0 and 9",
            ))
        }
    }
}

impl Bz2CompressionLevel {
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
    fn codec_bz2_valid1() {
        let json = r#"
        {
            "level": 5
        }"#;
        assert!(serde_json::from_str::<Bz2CodecConfiguration>(json).is_ok());
    }

    #[test]
    fn codec_bz2_invalid_level() {
        let json = r#"
        {
            "level": 10
        }"#;
        let codec_configuration = serde_json::from_str::<Bz2CodecConfiguration>(json);
        assert!(codec_configuration.is_err());
    }
}

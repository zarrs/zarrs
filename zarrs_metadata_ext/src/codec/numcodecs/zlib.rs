use derive_more::{Display, From};
use serde::{Deserialize, Deserializer, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `zlib` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum ZlibCodecConfiguration {
    /// Version 1.0 draft.
    V1(ZlibCodecConfigurationV1),
}

impl ConfigurationSerialize for ZlibCodecConfiguration {}

/// `zlib` codec configuration parameters (version 1.0 draft).
///
/// ### Example: encode with a compression level of 9
/// ```rust
/// # let JSON = r#"
/// {
///     "level": 9
/// }
/// # "#;
/// # use zarrs_metadata_ext::codec::zlib::ZlibCodecConfigurationV1;
/// # let configuration: ZlibCodecConfigurationV1 = serde_json::from_str(JSON).unwrap();
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ZlibCodecConfigurationV1 {
    /// The compression level.
    ///
    /// Defaults to 1 (the `numcodecs` default) if absent.
    #[serde(default = "default_level")]
    pub level: ZlibCompressionLevel,
}

const fn default_level() -> ZlibCompressionLevel {
    ZlibCompressionLevel(1)
}

/// An integer from -1 to 9 controlling the compression level
///
/// A level of 1 is the fastest compression method and produces the least compression, while 9 is slowest and produces the most compression.
/// Compression is turned off when the compression level is 0.
/// A level of -1 is the zlib default (`Z_DEFAULT_COMPRESSION`), equivalent to 6.
#[derive(Serialize, Copy, Clone, Debug, Eq, PartialEq)]
pub struct ZlibCompressionLevel(i8);

macro_rules! zlib_compression_level_try_from {
    ( $t:ty ) => {
        impl TryFrom<$t> for ZlibCompressionLevel {
            type Error = $t;
            fn try_from(level: $t) -> Result<Self, Self::Error> {
                match i8::try_from(level) {
                    Ok(level_i8) if level_i8 <= 9 => Ok(Self(level_i8)),
                    _ => Err(level),
                }
            }
        }
    };
}

zlib_compression_level_try_from!(u8);
zlib_compression_level_try_from!(u16);
zlib_compression_level_try_from!(u32);

impl<'de> Deserialize<'de> for ZlibCompressionLevel {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match i8::try_from(i64::deserialize(d)?) {
            Ok(level) if (-1..=9).contains(&level) => Ok(Self(level)),
            _ => Err(serde::de::Error::custom(
                "zlib compression level must be between -1 and 9",
            )),
        }
    }
}

impl ZlibCompressionLevel {
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
            Ok(Self(u32::from(compression_level) as i8))
        } else {
            Err(compression_level)
        }
    }

    /// The effective integer compression level from 0 to 9.
    ///
    /// The zlib default level of -1 is returned as 6.
    #[must_use]
    pub const fn as_u32(&self) -> u32 {
        if self.0 < 0 {
            6
        } else {
            self.0 as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_zlib_valid1() {
        let json = r#"
        {
            "level": 5
        }"#;
        assert!(serde_json::from_str::<ZlibCodecConfiguration>(json).is_ok());
    }

    #[test]
    fn codec_zlib_invalid_level() {
        let json = r#"
        {
            "level": 10
        }"#;
        let codec_configuration = serde_json::from_str::<ZlibCodecConfiguration>(json);
        assert!(codec_configuration.is_err());
        assert!(serde_json::from_str::<ZlibCodecConfiguration>(r#"{"level": -2}"#).is_err());
    }

    #[test]
    fn codec_zlib_default_level() {
        let ZlibCodecConfiguration::V1(configuration) =
            serde_json::from_str::<ZlibCodecConfiguration>("{}").unwrap();
        assert_eq!(configuration.level.as_u32(), 1);
        assert_eq!(
            serde_json::to_string(&configuration).unwrap(),
            r#"{"level":1}"#
        );
    }

    #[test]
    fn codec_zlib_level_minus_one() {
        let json = r#"{"level":-1}"#;
        let ZlibCodecConfiguration::V1(configuration) =
            serde_json::from_str::<ZlibCodecConfiguration>(json).unwrap();
        assert_eq!(configuration.level.as_u32(), 6);
        assert_eq!(serde_json::to_string(&configuration).unwrap(), json);
    }
}

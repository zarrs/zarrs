use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

use zarrs_metadata::{ConfigurationSerialize, DataTypeSize};

/// A wrapper to handle various versions of `blosc` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum BloscCodecConfiguration {
    /// Version 1.0.
    V1(BloscCodecConfigurationV1),
    /// Numcodecs.
    Numcodecs(BloscCodecConfigurationNumcodecs),
}

impl ConfigurationSerialize for BloscCodecConfiguration {}

/// An integer from 0 to 9 controlling the compression level
///
/// A level of 1 is the fastest compression method and produces the least compressions, while 9 is slowest and produces the most compression.
/// Compression is turned off when the compression level is 0.
#[derive(Serialize, Copy, Clone, Debug, Eq, PartialEq)]
pub struct BloscCompressionLevel(u8);

impl From<BloscCompressionLevel> for u8 {
    fn from(val: BloscCompressionLevel) -> Self {
        val.0
    }
}

impl TryFrom<u8> for BloscCompressionLevel {
    type Error = u8;
    fn try_from(level: u8) -> Result<Self, Self::Error> {
        if level <= 9 {
            Ok(Self(level))
        } else {
            Err(level)
        }
    }
}

impl<'de> serde::Deserialize<'de> for BloscCompressionLevel {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let level = u8::deserialize(d)?;
        if level <= 9 {
            Ok(Self(level))
        } else {
            Err(serde::de::Error::custom("clevel must be between 0 and 9"))
        }
    }
}

/// The `blosc` shuffle mode.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, Eq, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
#[repr(u32)]
pub enum BloscShuffleMode {
    /// No shuffling.
    #[default]
    NoShuffle = 0, // blosc_src::BLOSC_NOSHUFFLE,
    /// Byte-wise shuffling.
    Shuffle = 1, // blosc_src::BLOSC_SHUFFLE,
    /// Bit-wise shuffling.
    BitShuffle = 2, // blosc_src::BLOSC_BITSHUFFLE,
}

/// The `blosc` compressor.
///
/// See <https://www.blosc.org/pages/>.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BloscCompressor {
    /// [BloscLZ](https://github.com/Blosc/c-blosc/blob/master/blosc/blosclz.h): blosc default compressor, heavily based on [FastLZ](http://fastlz.org/).
    BloscLZ,
    /// [LZ4](http://fastcompression.blogspot.com/p/lz4.html): a compact, very popular and fast compressor.
    LZ4,
    /// [LZ4HC](http://fastcompression.blogspot.com/p/lz4.html): a tweaked version of LZ4, produces better compression ratios at the expense of speed.
    LZ4HC,
    /// [Snappy](https://code.google.com/p/snappy): a popular compressor used in many places.
    Snappy,
    /// [Zlib](http://www.zlib.net/): a classic; somewhat slower than the previous ones, but achieving better compression ratios.
    Zlib,
    /// [Zstd](http://www.zstd.net/): an extremely well balanced codec; it provides the best compression ratios among the others above, and at reasonably fast speed.
    Zstd,
}

/// `blosc` codec configuration parameters (version 1.0).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct BloscCodecConfigurationV1 {
    /// The compressor.
    pub cname: BloscCompressor,
    /// The compression level.
    pub clevel: BloscCompressionLevel,
    /// The shuffle mode.
    ///
    /// Defaults to noshuffle if unspecified.
    #[serde(default)]
    pub shuffle: BloscShuffleMode,
    /// The type size in bytes.
    ///
    /// Required unless shuffle is "noshuffle", in which case the value is ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typesize: Option<usize>,
    /// The compression block size. Automatically determined if 0.
    pub blocksize: usize,
}

/// `blosc` codec configuration parameters (numcodecs).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(from = "BloscCodecConfigurationNumcodecs16_0")]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct BloscCodecConfigurationNumcodecs {
    /// The compressor.
    pub cname: BloscCompressor,
    /// The compression level.
    pub clevel: BloscCompressionLevel,
    /// The shuffle mode.
    pub shuffle: BloscShuffleModeNumcodecs,
    /// The compression block size. Automatically determined if 0.
    #[serde(default)]
    pub blocksize: usize,
}

/// `typesize` was temporarily added in numcodecs 0.16.0 and will likely be removed in 0.16.1
///
/// <https://github.com/zarr-developers/numcodecs/pull/713#issuecomment-2797357740>
/// <https://github.com/zarr-developers/numcodecs/pull/739>
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BloscCodecConfigurationNumcodecs16_0 {
    cname: BloscCompressor,
    clevel: BloscCompressionLevel,
    shuffle: BloscShuffleModeNumcodecs,
    #[serde(default)]
    blocksize: usize,
    #[allow(dead_code)]
    typesize: Option<usize>,
}

impl From<BloscCodecConfigurationNumcodecs16_0> for BloscCodecConfigurationNumcodecs {
    fn from(value: BloscCodecConfigurationNumcodecs16_0) -> Self {
        BloscCodecConfigurationNumcodecs {
            cname: value.cname,
            clevel: value.clevel,
            shuffle: value.shuffle,
            blocksize: value.blocksize,
        }
    }
}

/// Blosc shuffle modes (numcodecs).
#[derive(Serialize_repr, Deserialize_repr, Clone, Eq, PartialEq, Debug, Display)]
#[repr(i8)]
pub enum BloscShuffleModeNumcodecs {
    /// No shuffling.
    NoShuffle = 0,
    /// Byte-wise shuffling.
    Shuffle = 1,
    /// Bit-wise shuffling.
    BitShuffle = 2,
    /// Bit-wise shuffling will be used for buffers with itemsize 1, and byte-wise shuffling will be used otherwise.
    AutoShuffle = -1,
}

/// Convert [`BloscCodecConfigurationNumcodecs`] to [`BloscCodecConfiguration`].
#[must_use]
pub fn codec_blosc_v2_numcodecs_to_v3(
    blosc: &BloscCodecConfigurationNumcodecs,
    data_type_size: Option<DataTypeSize>,
) -> BloscCodecConfiguration {
    let (shuffle, typesize) = match (&blosc.shuffle, data_type_size) {
        (BloscShuffleModeNumcodecs::NoShuffle, _) | (_, None) => {
            (BloscShuffleMode::NoShuffle, None)
        }
        // Fixed
        (BloscShuffleModeNumcodecs::Shuffle, Some(DataTypeSize::Fixed(data_type_size))) => {
            (BloscShuffleMode::Shuffle, Some(data_type_size))
        }
        (BloscShuffleModeNumcodecs::BitShuffle, Some(DataTypeSize::Fixed(data_type_size))) => {
            (BloscShuffleMode::BitShuffle, Some(data_type_size))
        }
        (BloscShuffleModeNumcodecs::AutoShuffle, Some(DataTypeSize::Fixed(data_type_size))) => {
            if data_type_size == 1 {
                (BloscShuffleMode::BitShuffle, Some(data_type_size))
            } else {
                (BloscShuffleMode::Shuffle, Some(data_type_size))
            }
        }
        // Variable
        (
            BloscShuffleModeNumcodecs::Shuffle
            | BloscShuffleModeNumcodecs::BitShuffle
            | BloscShuffleModeNumcodecs::AutoShuffle,
            Some(DataTypeSize::Variable),
        ) => {
            // FIXME: Check blosc auto behaviour with variable sized data type
            //        Currently defaulting to "bitshuffle"
            //        What do other implementations do for a variable sized data type?
            //        May need to make this function fallible
            (BloscShuffleMode::NoShuffle, None)
        }
    };

    BloscCodecConfiguration::V1(BloscCodecConfigurationV1 {
        cname: blosc.cname,
        clevel: blosc.clevel,
        shuffle,
        typesize,
        blocksize: blosc.blocksize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_blosc_valid1() {
        serde_json::from_str::<BloscCodecConfiguration>(
            r#"
        {
            "cname": "lz4",
            "clevel": 5,
            "shuffle": "shuffle",
            "typesize": 4,
            "blocksize": 0
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_blosc_valid2() {
        serde_json::from_str::<BloscCodecConfiguration>(
            r#"
        {
            "cname": "lz4",
            "clevel": 4,
            "shuffle": "bitshuffle",
            "typesize": 4,
            "blocksize": 0
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_blosc_invalid_no_typesize() {
        serde_json::from_str::<BloscCodecConfiguration>(
            r#"
        {
            "cname": "lz4",
            "clevel": 4,
            "shuffle": "bitshuffle",
            "blocksize": 0
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_blosc_valid_no_shuffle() {
        serde_json::from_str::<BloscCodecConfiguration>(
            r#"
        {
            "cname": "lz4",
            "clevel": 4,
            "blocksize": 0
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_blosc_valid_no_typesize() {
        serde_json::from_str::<BloscCodecConfiguration>(
            r#"
        {
            "cname": "lz4",
            "clevel": 4,
            "shuffle": "shuffle",
            "blocksize": 0
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_blosc_invalid_clevel() {
        let json = r#"
    {
        "cname": "lz4",
        "clevel": 10,
        "shuffle": "shuffle",
        "typesize": 4,
        "blocksize": 0
    }"#;
        let codec_configuration = serde_json::from_str::<BloscCodecConfiguration>(json);
        assert!(codec_configuration.is_err());
    }

    #[test]
    fn codec_blosc_invalid_cname() {
        let json = r#"
    {
        "cname": "",
        "clevel": 1,
        "shuffle": "shuffle",
        "typesize": 4,
        "blocksize": 0
    }"#;
        let codec_configuration = serde_json::from_str::<BloscCodecConfiguration>(json);
        assert!(codec_configuration.is_err());
    }

    #[test]
    fn codec_blosc_invalid_shuffle() {
        let json = r#"
    {
        "cname": "lz4",
        "clevel": 1,
        "shuffle": "",
        "typesize": 4,
        "blocksize": 0
    }"#;
        let codec_configuration = serde_json::from_str::<BloscCodecConfiguration>(json);
        assert!(codec_configuration.is_err());
    }

    #[test]
    fn codec_blosc_v2_numcodecs() {
        serde_json::from_str::<BloscCodecConfigurationNumcodecs>(
            r#"
        {
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 2,
            "blocksize": 0
        }"#,
        )
        .unwrap();
    }
}

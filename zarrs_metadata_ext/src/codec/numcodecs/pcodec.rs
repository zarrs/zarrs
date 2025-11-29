use derive_more::{Display, From};
use serde::{Deserialize, Deserializer, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `pcodec` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum PcodecCodecConfiguration {
    /// Version 1.0 draft.
    V1(PcodecCodecConfigurationV1),
}

impl ConfigurationSerialize for PcodecCodecConfiguration {}

impl Default for PcodecCodecConfiguration {
    fn default() -> Self {
        Self::V1(PcodecCodecConfigurationV1::default())
    }
}

/// `pcodec` codec configuration parameters (version 1.0 draft).
///
/// This configuration matches the implementation in `numcodecs`.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
#[serde(default)] // for compatibility with zarrs < 0.19
#[serde(deny_unknown_fields)]
pub struct PcodecCodecConfigurationV1 {
    /// A compression level from 0-12, where 12 takes the longest and compresses the most.
    pub level: PcodecCompressionLevel,
    /// The pcodec mode spec.
    pub mode_spec: PcodecModeSpecConfiguration,
    /// The delta encoding strategy.
    pub delta_spec: PcodecDeltaSpecConfiguration,
    /// The paging spec.
    pub paging_spec: PcodecPagingSpecConfiguration,
    /// Either a delta encoding level from 0-7 or None.
    ///
    /// If set to None, pcodec will try to infer the optimal delta encoding order.
    /// The default is None.
    pub delta_encoding_order: Option<PcodecDeltaEncodingOrder>,
    /// The maximum number of values to encode per pcodec page.
    ///
    /// If set too high or too low, pcodec's compression ratio may drop.
    /// See <https://docs.rs/pco/latest/pco/enum.PagingSpec.html#variant.EqualPagesUpTo>.
    ///
    /// The default is `1 << 18`.
    pub equal_pages_up_to: usize,
}

impl Default for PcodecCodecConfigurationV1 {
    fn default() -> Self {
        Self {
            level: PcodecCompressionLevel::default(),
            mode_spec: PcodecModeSpecConfiguration::default(),
            delta_spec: PcodecDeltaSpecConfiguration::default(),
            paging_spec: PcodecPagingSpecConfiguration::default(),
            delta_encoding_order: None,
            equal_pages_up_to: default_equal_pages_up_to(),
        }
    }
}

/// The [`pco::ModeSpec`](https://docs.rs/pco/latest/pco/enum.ModeSpec.html).
///
/// `TryFloatMult`, `TryFloatQuant`, and `TryIntMult` are not currently supported.
#[derive(Serialize, Deserialize, Default, Clone, Copy, PartialEq, Debug, Display)]
#[serde(rename_all = "snake_case")]
pub enum PcodecModeSpecConfiguration {
    /// See <https://docs.rs/pco/latest/pco/enum.ModeSpec.html#variant.Auto>.
    #[default]
    Auto,
    /// See <https://docs.rs/pco/latest/pco/enum.ModeSpec.html#variant.Classic>.
    Classic,
}

/// The [`pco::DeltaSpec`](https://docs.rs/pco/latest/pco/enum.DeltaSpec.html).
///
/// The delta encoding order for `TryConsecutive` is serialised in a separate field.
#[derive(Serialize, Deserialize, Default, Debug, Clone, Copy, PartialEq, Display)]
#[serde(rename_all = "snake_case")]
pub enum PcodecDeltaSpecConfiguration {
    /// See <https://docs.rs/pco/latest/pco/enum.DeltaSpec.html#variant.Auto>.
    #[default]
    Auto,
    /// See <https://docs.rs/pco/latest/pco/enum.DeltaSpec.html#variant.None>.
    None,
    /// See <https://docs.rs/pco/latest/pco/enum.DeltaSpec.html#variant.TryConsecutive>.
    TryConsecutive,
    /// See <https://docs.rs/pco/latest/pco/enum.DeltaSpec.html#variant.TryLookback>.
    TryLookback,
}

/// The [`pco::PagingSpec`](https://docs.rs/pco/latest/pco/enum.PagingSpec.html).
///
/// The `Exact` paging spec is not supported.
#[derive(Serialize, Deserialize, Default, Debug, Clone, Copy, PartialEq, Display)]
#[serde(rename_all = "snake_case")]
pub enum PcodecPagingSpecConfiguration {
    /// See <https://docs.rs/pco/latest/pco/enum.PagingSpec.html#variant.EqualPagesUpTo>.
    #[default]
    EqualPagesUpTo,
}

/// An integer from 0 to 12 controlling the compression level.
///
/// The default is 8.
///
/// See <https://docs.rs/pco/latest/pco/struct.ChunkConfig.html#structfield.compression_level>.
///
/// - Level 0 achieves only a small amount of compression.
/// - Level 8 achieves very good compression and runs only slightly slower.
/// - Level 12 achieves marginally better compression than 8 and may run several times slower.
#[derive(Serialize, Copy, Clone, Debug, Eq, PartialEq)]
pub struct PcodecCompressionLevel(u8);

impl Default for PcodecCompressionLevel {
    fn default() -> Self {
        Self(8)
    }
}

macro_rules! pcodec_compression_level_try_from {
    ( $t:ty ) => {
        impl TryFrom<$t> for PcodecCompressionLevel {
            type Error = $t;
            fn try_from(level: $t) -> Result<Self, Self::Error> {
                if level <= 12 {
                    Ok(Self(unsafe { u8::try_from(level).unwrap_unchecked() }))
                } else {
                    Err(level)
                }
            }
        }
    };
}

pcodec_compression_level_try_from!(u8);
pcodec_compression_level_try_from!(u16);
pcodec_compression_level_try_from!(u32);
pcodec_compression_level_try_from!(u64);
pcodec_compression_level_try_from!(usize);

impl<'de> Deserialize<'de> for PcodecCompressionLevel {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let level = u8::deserialize(d)?;
        if level <= 12 {
            Ok(Self(level))
        } else {
            Err(serde::de::Error::custom(
                "pcodec compression level must be between 0 and 12",
            ))
        }
    }
}

impl PcodecCompressionLevel {
    /// Create a new compression level.
    ///
    /// # Errors
    /// Errors if `compression_level` is not between 0-12.
    pub fn new<N: num::Unsigned + std::cmp::PartialOrd<u8>>(compression_level: N) -> Result<Self, N>
    where
        u8: TryFrom<N>,
    {
        if compression_level <= 12 {
            Ok(Self(unsafe {
                u8::try_from(compression_level).unwrap_unchecked()
            }))
        } else {
            Err(compression_level)
        }
    }

    /// The underlying integer compression level.
    #[must_use]
    pub const fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

/// An integer from 0 to 7 controlling the delta encoding order.
///
/// It is the number of times to apply delta encoding before compressing.
/// See <https://docs.rs/pco/latest/pco/enum.DeltaSpec.html>.
///
/// - 0th order takes numbers as-is. This is perfect for columnar data were the order is essentially random.
/// - 1st order takes consecutive differences, leaving `[0, 2, 0, 2, 0, 2, 0]`. This is best for continuous but noisy time series data, like stock prices or most time series data.
/// - 2nd order takes consecutive differences again, leaving `[2, -2, 2, -2, 2, -2]`. This is best for piecewise-linear or somewhat quadratic data.
/// - Even higher-order is best for time series that are very smooth, like temperature or light sensor readings.
#[derive(Serialize, Copy, Clone, Debug, Eq, PartialEq)]
pub struct PcodecDeltaEncodingOrder(u8);

macro_rules! pcodec_delta_encoding_order_level_try_from {
    ( $t:ty ) => {
        impl TryFrom<$t> for PcodecDeltaEncodingOrder {
            type Error = $t;
            fn try_from(level: $t) -> Result<Self, Self::Error> {
                if level <= 7 {
                    Ok(Self(unsafe { u8::try_from(level).unwrap_unchecked() }))
                } else {
                    Err(level)
                }
            }
        }
    };
}

pcodec_delta_encoding_order_level_try_from!(u8);
pcodec_delta_encoding_order_level_try_from!(u16);
pcodec_delta_encoding_order_level_try_from!(u32);
pcodec_delta_encoding_order_level_try_from!(u64);
pcodec_delta_encoding_order_level_try_from!(usize);

impl<'de> Deserialize<'de> for PcodecDeltaEncodingOrder {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let level = u8::deserialize(d)?;
        if level <= 7 {
            Ok(Self(level))
        } else {
            Err(serde::de::Error::custom(
                "pcodec compression level must be between 0 and 7",
            ))
        }
    }
}

impl PcodecDeltaEncodingOrder {
    /// Create a new delta encoding order.
    ///
    /// # Errors
    /// Errors if `delta_encoding_order` is not between 0-7.
    pub fn new<N: num::Unsigned + std::cmp::PartialOrd<u8>>(
        delta_encoding_order: N,
    ) -> Result<Self, N>
    where
        u8: TryFrom<N>,
    {
        if delta_encoding_order <= 7 {
            Ok(Self(unsafe {
                u8::try_from(delta_encoding_order).unwrap_unchecked()
            }))
        } else {
            Err(delta_encoding_order)
        }
    }

    /// The underlying delta encoding order.
    #[must_use]
    pub const fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

const fn default_equal_pages_up_to() -> usize {
    // pco::constants::DEFAULT_MAX_PAGE_N
    1 << 18
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_pcodec_valid_empty() {
        serde_json::from_str::<PcodecCodecConfiguration>(
            r#"{
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_valid_auto() {
        serde_json::from_str::<PcodecCodecConfiguration>(
            r#"{
            "level": 8,
            "delta_encoding_order": 2,
            "mode_spec": "auto",
            "equal_pages_up_to": 262144
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_valid_classic() {
        serde_json::from_str::<PcodecCodecConfiguration>(
            r#"{
            "level": 8,
            "delta_encoding_order": 2,
            "mode_spec": "classic",
            "equal_pages_up_to": 262144
        }"#,
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_invalid_level() {
        assert!(serde_json::from_str::<PcodecCodecConfiguration>(
            r#"{
            "level": 13,
            "delta_encoding_order": 2,
            "mode_spec": "auto",
            "equal_pages_up_to": 262144
        }"#,
        )
        .is_err());
    }

    #[test]
    fn codec_pcodec_invalid_delta_encoding_order() {
        assert!(serde_json::from_str::<PcodecCodecConfiguration>(
            r#"{
            "level": 8,
            "delta_encoding_order": 8,
            "mode_spec": "auto",
            "equal_pages_up_to": 262144
        }"#,
        )
        .is_err());
    }
}

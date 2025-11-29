use derive_more::From;
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `zfp` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum ZfpCodecConfiguration {
    /// Version 1.0 draft.
    V1(ZfpCodecConfigurationV1),
}

impl ConfigurationSerialize for ZfpCodecConfiguration {}

/// `zfp` codec configuration parameters (version 1.0 draft).
///
/// Further information on the meaning of these parameters can be found in the [zfp documentation](https://zfp.readthedocs.io/en/latest/).
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
// #[serde(deny_unknown_fields)] // NOTE: Not supported with flatten
pub struct ZfpCodecConfigurationV1 {
    /// The zfp mode.
    #[serde(flatten)]
    pub mode: ZfpMode,
}

/// The zfp mode.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(tag = "mode", rename_all = "snake_case")]
// #[serde(deny_unknown_fields)]
pub enum ZfpMode {
    /// Expert mode.
    Expert {
        /// The minimum number of compressed bits used to represent a block.
        ///
        /// Usually this parameter equals one bit, unless each and every block is to be stored using a fixed number of bits to facilitate random access, in which case it should be set to the same value as `maxbits`.
        minbits: u32,
        /// The maximum number of bits used to represent a block.
        ///
        /// This parameter sets a hard upper bound on compressed block size and governs the rate in fixed-rate mode. It may also be used as an upper storage limit to guard against buffer overruns in combination with the accuracy constraints given by `zfp_stream.maxprec` and `zfp_stream.minexp`.
        /// `maxbits` must be large enough to allow the common block exponent and any control bits to be encoded. This implies `maxbits` ≥ 9 for single-precision data and `maxbits` ≥ 12 for double-precision data.
        maxbits: u32,
        /// The maximum number of bit planes encoded.
        ///
        /// This parameter governs the number of most significant uncompressed bits encoded per transform coefficient.
        /// It does not directly correspond to the number of uncompressed mantissa bits for the floating-point or integer values being compressed, but is closely related.
        /// This is the parameter that specifies the precision in fixed-precision mode, and it provides a mechanism for controlling the relative error.
        /// Note that this parameter selects how many bits planes to encode regardless of the magnitude of the common floating-point exponent within the block.
        maxprec: u32,
        /// The smallest absolute bit plane number encoded (applies to floating-point data only; this parameter is ignored for integer data).
        ///
        /// The place value of each transform coefficient bit depends on the common floating-point exponent, $e$, that scales the integer coefficients. If the most significant coefficient bit has place value $2^e$, then the number of bit planes encoded is (one plus) the difference between e and `zfp_stream.minexp`.
        /// This parameter governs the absolute error in fixed-accuracy mode.
        minexp: i32,
    },
    /// Fixed rate mode.
    FixedRate {
        /// The rate is the number of compressed bits per value.
        rate: f64,
    },
    /// Fixed precision mode.
    FixedPrecision {
        /// The precision specifies how many uncompressed bits per value to store, and indirectly governs the relative error.
        precision: u32,
    },
    /// Fixed accuracy mode.
    FixedAccuracy {
        /// The tolerance ensures that values in the decompressed array differ from the input array by no more than this tolerance.
        tolerance: f64,
    },
    /// Reversible mode.
    Reversible,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_zfp_configuration_expert() {
        const JSON: &'static str = r#"{
        "mode": "expert",
        "minbits": 1,
        "maxbits": 12,
        "maxprec": 10,
        "minexp": -2
    }"#;
        serde_json::from_str::<ZfpCodecConfiguration>(JSON).unwrap();
    }

    #[test]
    fn codec_zfp_configuration_fixed_rate() {
        const JSON: &'static str = r#"{
        "mode": "fixed_rate",
        "rate": 12
    }"#;
        serde_json::from_str::<ZfpCodecConfiguration>(JSON).unwrap();
    }

    #[test]
    fn codec_zfp_configuration_fixed_precision() {
        const JSON: &'static str = r#"{
        "mode": "fixed_precision",
        "precision": 12
    }"#;
        serde_json::from_str::<ZfpCodecConfiguration>(JSON).unwrap();
    }

    #[test]
    fn codec_zfp_configuration_fixed_accuracy() {
        const JSON: &'static str = r#"{
        "mode": "fixed_accuracy",
        "tolerance": 0.001
    }"#;
        serde_json::from_str::<ZfpCodecConfiguration>(JSON).unwrap();
    }

    #[test]
    fn codec_zfp_configuration_reversible() {
        const JSON: &'static str = r#"{
        "mode": "reversible"
    }"#;
        serde_json::from_str::<ZfpCodecConfiguration>(JSON).unwrap();
    }

    #[test]
    fn codec_zfp_configuration_invalid2() {
        const JSON_INVALID2: &'static str = r#"{
        "mode": "unknown"
    }"#;
        assert!(serde_json::from_str::<ZfpCodecConfiguration>(JSON_INVALID2).is_err());
    }
}

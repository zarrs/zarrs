use zfp_rs::{ZfpConfig, ZfpDimensionality, ZfpScalarType, ZfpStreamAlignment};

use super::ZfpMode;

/// Create a `zfp` compression configuration for a given mode and scalar type.
///
/// Returns [`None`] if the mode is not supported for the scalar type.
pub(super) fn zfp_config(mode: &ZfpMode, scalar_type: ZfpScalarType) -> Option<ZfpConfig> {
    match mode {
        ZfpMode::Expert {
            minbits,
            maxbits,
            maxprec,
            minexp,
        } => Some(ZfpConfig::expert(*minbits, *maxbits, *maxprec, *minexp)),
        ZfpMode::FixedRate { rate } => Some(ZfpConfig::fixed_rate(
            *rate,
            scalar_type,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::None,
        )),
        ZfpMode::FixedPrecision { precision } => Some(ZfpConfig::fixed_precision(*precision)),
        ZfpMode::FixedAccuracy { tolerance } => match scalar_type {
            ZfpScalarType::Float | ZfpScalarType::Double => {
                Some(ZfpConfig::fixed_accuracy(*tolerance))
            }
            ZfpScalarType::Int32 | ZfpScalarType::Int64 => None,
        },
        ZfpMode::Reversible => Some(ZfpConfig::reversible()),
    }
}

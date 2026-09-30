use zarrs_codec::CodecCreateError;
use zfp_rs::{ZfpConfig, ZfpDimensionality, ZfpScalarType, ZfpStreamAlignment};

use super::ZfpMode;

/// Create a `zfp` compression configuration for a given mode and scalar type.
///
/// # Errors
/// Returns an error if the mode is not supported for the scalar type or its parameters are invalid.
pub(super) fn zfp_config(
    mode: &ZfpMode,
    scalar_type: ZfpScalarType,
) -> Result<ZfpConfig, CodecCreateError> {
    let invalid = |err| CodecCreateError::Other(format!("invalid zfp {mode:?} mode: {err}"));
    match mode {
        ZfpMode::Expert {
            minbits,
            maxbits,
            maxprec,
            minexp,
        } => ZfpConfig::expert(*minbits, *maxbits, *maxprec, *minexp).map_err(invalid),
        ZfpMode::FixedRate { rate } => ZfpConfig::fixed_rate(
            *rate,
            scalar_type,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::Unaligned,
        )
        .map_err(invalid),
        ZfpMode::FixedPrecision { precision } => Ok(ZfpConfig::fixed_precision(*precision)),
        ZfpMode::FixedAccuracy { tolerance } => match scalar_type {
            ZfpScalarType::F32 | ZfpScalarType::F64 => Ok(ZfpConfig::fixed_accuracy(*tolerance)),
            ZfpScalarType::I32 | ZfpScalarType::I64 => Err(CodecCreateError::Other(format!(
                "zfp {mode:?} mode is unsupported for the {scalar_type:?} zfp scalar type"
            ))),
        },
        ZfpMode::Reversible => Ok(ZfpConfig::reversible()),
    }
}

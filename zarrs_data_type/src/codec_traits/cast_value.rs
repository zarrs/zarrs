//! The `cast_value` codec data type traits.

mod kernels;

use half::{bf16, f16};
use thiserror::Error;

pub use kernels::{
    CastValueIntStored, CastValueIntStoredPrimitive, CastValueKernel, CastValueRepr,
    select_cast_kernel,
};

/// Rounding mode used by the `cast_value` codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastValueRoundingMode {
    /// Round to nearest with ties to even.
    NearestEven,
    /// Round towards zero.
    TowardsZero,
    /// Round towards positive infinity.
    TowardsPositive,
    /// Round towards negative infinity.
    TowardsNegative,
    /// Round to nearest with ties away from zero.
    NearestAway,
}

/// Out-of-range handling used by the `cast_value` codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastValueOutOfRangeMode {
    /// Clamp to the target range.
    Clamp,
    /// Wrap modulo the integer target range.
    Wrap,
}

/// A scalar value decoded by a `CastValueDataTypeTraits` implementation.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum CastValueScalar {
    /// A signed integer value.
    Signed(i128),
    /// An unsigned integer value.
    Unsigned(u128),
    /// A floating point value.
    Float(CastValueFloatScalar),
}

/// A floating point scalar value.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum CastValueFloatScalar {
    /// `float16`.
    F16(f16),
    /// `bfloat16`.
    BF16(bf16),
    /// `float32` or a smaller float represented through `f32`.
    F32(f32),
    /// `float64`.
    F64(f64),
}

impl CastValueFloatScalar {
    /// Return the value as `f64`.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        match self {
            Self::F16(value) => value.to_f64(),
            Self::BF16(value) => value.to_f64(),
            Self::F32(value) => f64::from(value),
            Self::F64(value) => value,
        }
    }

    /// Return the value as `f32`.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        match self {
            Self::F16(value) => value.to_f32(),
            Self::BF16(value) => value.to_f32(),
            Self::F32(value) => value,
            #[expect(clippy::cast_possible_truncation)]
            Self::F64(value) => value as f32,
        }
    }

    /// Returns `true` if the value is NaN.
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.to_f64().is_nan()
    }

    /// Returns `true` if the value is infinite.
    #[must_use]
    pub fn is_infinite(self) -> bool {
        self.to_f64().is_infinite()
    }

    /// Returns `true` if the value has a negative sign.
    #[must_use]
    pub fn is_sign_negative(self) -> bool {
        self.to_f64().is_sign_negative()
    }
}

/// Errors raised while casting scalar values.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CastValueError {
    /// Invalid source element bytes.
    #[error("invalid source element bytes for cast_value")]
    InvalidElementBytes,
    /// The source scalar is not representable in the target data type.
    #[error("scalar is not representable in target data type")]
    NotRepresentable,
    /// Wrapping was requested for a non-integral target data type.
    #[error("cast_value wrap mode is only valid for integral target data types")]
    InvalidWrapTarget,
}

/// Traits for a data type supporting the `cast_value` codec.
pub trait CastValueDataTypeTraits {
    /// Returns `true` if the data type is integral.
    fn cast_value_is_integral(&self) -> bool;

    /// The numeric representation for kernel-based casting, if any.
    ///
    /// Data types returning `Some` opt in to the monomorphised bulk cast
    /// kernels selected by [`select_cast_kernel`]; those returning `None`
    /// (the default) use the generic scalar path.
    fn cast_value_repr(&self) -> Option<CastValueRepr> {
        None
    }

    /// Decode one element from native-endian codec bytes.
    ///
    /// # Errors
    /// Returns an error if `bytes` are incompatible with the data type.
    fn cast_value_read(&self, bytes: &[u8]) -> Result<CastValueScalar, CastValueError>;

    /// Encode one scalar to native-endian codec bytes, appending them to `output`.
    ///
    /// # Errors
    /// Returns an error if `value` cannot be represented with the requested modes.
    fn cast_value_write(
        &self,
        value: CastValueScalar,
        rounding: CastValueRoundingMode,
        out_of_range: Option<CastValueOutOfRangeMode>,
        output: &mut Vec<u8>,
    ) -> Result<(), CastValueError>;

    /// Cast one element to another `cast_value`-compatible data type, appending the encoded bytes to `output`.
    ///
    /// Implementors may override this for specialised direct conversions.
    ///
    /// # Errors
    /// Returns an error if the source cannot be converted to the target.
    fn cast_value_cast(
        &self,
        source: &[u8],
        target: &dyn CastValueDataTypeTraits,
        rounding: CastValueRoundingMode,
        out_of_range: Option<CastValueOutOfRangeMode>,
        output: &mut Vec<u8>,
    ) -> Result<(), CastValueError> {
        target.cast_value_write(
            self.cast_value_read(source)?,
            rounding,
            out_of_range,
            output,
        )
    }

    /// Cast contiguous elements of `element_size` bytes each, appending the encoded bytes to `output`.
    ///
    /// Implementors may override this for specialised bulk conversions.
    ///
    /// # Errors
    /// Returns an error if `source` is not a multiple of `element_size` or an element cannot be converted to the target.
    fn cast_value_cast_slice(
        &self,
        source: &[u8],
        element_size: usize,
        target: &dyn CastValueDataTypeTraits,
        rounding: CastValueRoundingMode,
        out_of_range: Option<CastValueOutOfRangeMode>,
        output: &mut Vec<u8>,
    ) -> Result<(), CastValueError> {
        if element_size == 0 || !source.len().is_multiple_of(element_size) {
            return Err(CastValueError::InvalidElementBytes);
        }
        for element in source.chunks_exact(element_size) {
            self.cast_value_cast(element, target, rounding, out_of_range, output)?;
        }
        Ok(())
    }
}

// Generate the codec support infrastructure using the generic macro.
crate::define_data_type_support!(CastValue);

// The integer helpers below only support bit widths up to 127; 128-bit integer
// data types would overflow the `i128`/`u128` arithmetic used here.

const fn signed_min(bits: u32) -> i128 {
    debug_assert!(bits >= 1 && bits < 128);
    -(1_i128 << (bits - 1))
}

const fn signed_max(bits: u32) -> i128 {
    debug_assert!(bits >= 1 && bits < 128);
    (1_i128 << (bits - 1)) - 1
}

const fn unsigned_max(bits: u32) -> u128 {
    debug_assert!(bits >= 1 && bits < 128);
    (1_u128 << bits) - 1
}

// Wrapping modulo `2^bits` uses masks rather than `rem_euclid`/`%`: the
// euclidean remainder modulo a power of two is the low `bits` of the two's
// complement representation, and a runtime modulus would otherwise cost a
// 128-bit division per out-of-range element.

fn wrap_signed(value: i128, bits: u32) -> i128 {
    #[expect(clippy::cast_sign_loss)]
    wrap_signed_unsigned(value as u128, bits)
}

fn wrap_unsigned(value: i128, bits: u32) -> u128 {
    #[expect(clippy::cast_sign_loss)]
    {
        (value as u128) & unsigned_max(bits)
    }
}

fn wrap_signed_unsigned(value: u128, bits: u32) -> i128 {
    let modulus = 1_u128 << bits;
    let wrapped = value & (modulus - 1);
    let sign_bit = 1_u128 << (bits - 1);
    #[expect(clippy::cast_possible_wrap)]
    if wrapped >= sign_bit {
        (wrapped as i128) - (modulus as i128)
    } else {
        wrapped as i128
    }
}

fn wrap_unsigned_unsigned(value: u128, bits: u32) -> u128 {
    value & unsigned_max(bits)
}

fn round_float_to_integer(value: f64, rounding: CastValueRoundingMode) -> f64 {
    match rounding {
        CastValueRoundingMode::NearestEven => value.round_ties_even(),
        CastValueRoundingMode::TowardsZero => value.trunc(),
        CastValueRoundingMode::TowardsPositive => value.ceil(),
        CastValueRoundingMode::TowardsNegative => value.floor(),
        CastValueRoundingMode::NearestAway => value.round(),
    }
}

/// A scalar rounded to an integer value.
enum RoundedScalar {
    /// An integer representable in `i128`.
    Int(i128),
    /// A finite integer-valued float with a magnitude beyond the `i128` range.
    Big(f64),
}

/// 2^127: the smallest integer magnitude not representable in `i128`.
const TWO_POW_127: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;

/// 2^128: the only `u128 as f64` result that cannot round-trip through `u128`.
const TWO_POW_128: f64 = 340_282_366_920_938_463_463_374_607_431_768_211_456.0;

fn scalar_to_rounded_integer(
    value: CastValueScalar,
    rounding: CastValueRoundingMode,
) -> Result<RoundedScalar, CastValueError> {
    match value {
        CastValueScalar::Signed(value) => Ok(RoundedScalar::Int(value)),
        CastValueScalar::Unsigned(value) => i128::try_from(value)
            .map(RoundedScalar::Int)
            .map_err(|_| CastValueError::NotRepresentable),
        CastValueScalar::Float(value) => {
            if value.is_nan() || value.is_infinite() {
                return Err(CastValueError::NotRepresentable);
            }
            let value = round_float_to_integer(value.to_f64(), rounding);
            if value >= TWO_POW_127 || value < -TWO_POW_127 {
                Ok(RoundedScalar::Big(value))
            } else {
                #[expect(clippy::cast_possible_truncation)]
                Ok(RoundedScalar::Int(value as i128))
            }
        }
    }
}

/// Reduce a finite integer-valued float modulo `2^bits`.
///
/// This is exact: any float with a magnitude of at least 2^53 has enough
/// trailing zero bits that `rem_euclid` by a power of two cannot round.
fn wrap_float(value: f64, bits: u32) -> u128 {
    #[expect(clippy::cast_precision_loss)]
    let modulus = (1_u128 << bits) as f64;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        value.rem_euclid(modulus) as u128
    }
}

fn scalar_to_f64(value: CastValueScalar, rounding: CastValueRoundingMode) -> f64 {
    match value {
        CastValueScalar::Signed(value) => {
            integer_magnitude_to_f64(value.unsigned_abs(), value < 0, rounding)
        }
        CastValueScalar::Unsigned(value) => integer_magnitude_to_f64(value, false, rounding),
        CastValueScalar::Float(value) => value.to_f64(),
    }
}

/// Convert an integer magnitude to `f64`, exactly rounded per `rounding`.
///
/// A plain `as f64` cast always rounds ties-to-even; magnitudes above 2^53
/// need correction for the other rounding modes.
fn integer_magnitude_to_f64(
    magnitude: u128,
    negative: bool,
    rounding: CastValueRoundingMode,
) -> f64 {
    let signed = |value: f64| if negative { -value } else { value };
    // magnitudes below 2^53 convert exactly; skip the correction logic and its
    // expensive `f64`/`u128` round-trip conversions
    if magnitude < (1_u128 << 53) {
        #[expect(clippy::cast_precision_loss)]
        return signed(magnitude as f64);
    }
    #[expect(clippy::cast_precision_loss)]
    let approx = magnitude as f64;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let approx_int = (approx < TWO_POW_128).then_some(approx as u128);
    if approx_int == Some(magnitude) {
        return signed(approx);
    }
    let rounded_up = approx_int.is_none_or(|approx_int| approx_int > magnitude);
    // Whether the magnitude must round away from zero for the requested mode
    let round_up_magnitude = match rounding {
        // `as` rounds ties-to-even and is symmetric in sign
        CastValueRoundingMode::NearestEven => return signed(approx),
        CastValueRoundingMode::TowardsZero => false,
        CastValueRoundingMode::TowardsPositive => !negative,
        CastValueRoundingMode::TowardsNegative => negative,
        CastValueRoundingMode::NearestAway => {
            if rounded_up {
                // the nearest value, and also away from zero on a tie
                return signed(approx);
            }
            // on an exact tie, ties-to-even may round down where away-from-zero must round up
            let Some(approx_int) = approx_int else {
                unreachable!("`approx_int` is `None` only when rounded up");
            };
            let upper = approx.next_up();
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ulp = (upper - approx) as u128; // adjacent float difference is exact
            let below = magnitude - approx_int;
            return if 2 * below == ulp {
                signed(upper)
            } else {
                signed(approx)
            };
        }
    };
    match (round_up_magnitude, rounded_up) {
        (true, true) | (false, false) => signed(approx),
        (true, false) => signed(approx.next_up()),
        (false, true) => signed(approx.next_down()),
    }
}

fn next_up_f16(value: f16) -> f16 {
    if value.is_nan() || value == f16::INFINITY {
        return value;
    }
    if value == f16::NEG_INFINITY {
        return f16::MIN;
    }
    let bits = value.to_bits();
    if bits == 0x8000 {
        f16::from_bits(0x0001)
    } else if value.is_sign_negative() {
        f16::from_bits(bits - 1)
    } else {
        f16::from_bits(bits + 1)
    }
}

fn next_down_f16(value: f16) -> f16 {
    if value.is_nan() || value == f16::NEG_INFINITY {
        return value;
    }
    if value == f16::INFINITY {
        return f16::MAX;
    }
    let bits = value.to_bits();
    if bits == 0x0000 {
        f16::from_bits(0x8001)
    } else if value.is_sign_negative() {
        f16::from_bits(bits + 1)
    } else {
        f16::from_bits(bits - 1)
    }
}

fn next_up_bf16(value: bf16) -> bf16 {
    if value.is_nan() || value == bf16::INFINITY {
        return value;
    }
    if value == bf16::NEG_INFINITY {
        return bf16::MIN;
    }
    let bits = value.to_bits();
    if bits == 0x8000 {
        bf16::from_bits(0x0001)
    } else if value.is_sign_negative() {
        bf16::from_bits(bits - 1)
    } else {
        bf16::from_bits(bits + 1)
    }
}

fn next_down_bf16(value: bf16) -> bf16 {
    if value.is_nan() || value == bf16::NEG_INFINITY {
        return value;
    }
    if value == bf16::INFINITY {
        return bf16::MAX;
    }
    let bits = value.to_bits();
    if bits == 0x0000 {
        bf16::from_bits(0x8001)
    } else if value.is_sign_negative() {
        bf16::from_bits(bits + 1)
    } else {
        bf16::from_bits(bits - 1)
    }
}

// Exact float comparisons are intentional: `rounded` either equals `value`
// exactly or differs from it by at least one ULP of the target type.
#[expect(clippy::float_cmp)]
fn adjust_float_rounding<T>(
    value: f64,
    rounded: T,
    rounding: CastValueRoundingMode,
    to_f64: impl Fn(T) -> f64,
    next_down: impl Fn(T) -> T,
    next_up: impl Fn(T) -> T,
) -> T
where
    T: Copy,
{
    if !value.is_finite() {
        return rounded;
    }
    let rounded_value = to_f64(rounded);
    if rounded_value == value {
        return rounded;
    }
    match rounding {
        CastValueRoundingMode::NearestEven => rounded,
        CastValueRoundingMode::TowardsPositive => {
            if rounded_value < value {
                next_up(rounded)
            } else {
                rounded
            }
        }
        CastValueRoundingMode::TowardsNegative => {
            if rounded_value > value {
                next_down(rounded)
            } else {
                rounded
            }
        }
        CastValueRoundingMode::TowardsZero => {
            if value.is_sign_negative() {
                if rounded_value < value {
                    next_up(rounded)
                } else {
                    rounded
                }
            } else if rounded_value > value {
                next_down(rounded)
            } else {
                rounded
            }
        }
        CastValueRoundingMode::NearestAway => {
            let lower = if rounded_value < value {
                rounded
            } else {
                next_down(rounded)
            };
            let upper = if rounded_value > value {
                rounded
            } else {
                next_up(rounded)
            };
            let lower_value = to_f64(lower);
            let upper_value = to_f64(upper);
            if lower_value.is_finite()
                && upper_value.is_finite()
                && (value - lower_value).abs() == (upper_value - value).abs()
            {
                if value.is_sign_negative() {
                    lower
                } else {
                    upper
                }
            } else {
                rounded
            }
        }
    }
}

/// Round a scalar value to a custom floating point target.
#[doc(hidden)]
#[must_use]
pub fn round_float_with_next_after<T>(
    value: f64,
    rounded: T,
    rounding: CastValueRoundingMode,
    to_f64: impl Fn(T) -> f64,
    next_down: impl Fn(T) -> T,
    next_up: impl Fn(T) -> T,
) -> T
where
    T: Copy,
{
    adjust_float_rounding(value, rounded, rounding, to_f64, next_down, next_up)
}

/// Round a scalar value to `f16`.
#[doc(hidden)]
#[must_use]
pub fn round_f64_to_f16(value: f64, rounding: CastValueRoundingMode) -> f16 {
    adjust_float_rounding(
        value,
        f16::from_f64(value),
        rounding,
        f16::to_f64,
        next_down_f16,
        next_up_f16,
    )
}

/// Round a scalar value to `bf16`.
#[doc(hidden)]
#[must_use]
pub fn round_f64_to_bf16(value: f64, rounding: CastValueRoundingMode) -> bf16 {
    adjust_float_rounding(
        value,
        bf16::from_f64(value),
        rounding,
        bf16::to_f64,
        next_down_bf16,
        next_up_bf16,
    )
}

/// Round a scalar value to `f32`.
#[doc(hidden)]
#[expect(clippy::cast_possible_truncation)]
#[must_use]
pub fn round_f64_to_f32(value: f64, rounding: CastValueRoundingMode) -> f32 {
    adjust_float_rounding(
        value,
        value as f32,
        rounding,
        f64::from,
        f32::next_down,
        f32::next_up,
    )
}

/// Round a scalar value to `f64`.
#[doc(hidden)]
#[must_use]
pub fn round_f64_to_f64(value: f64, rounding: CastValueRoundingMode) -> f64 {
    adjust_float_rounding(
        value,
        value,
        rounding,
        |value| value,
        f64::next_down,
        f64::next_up,
    )
}

/// Cast a scalar to a signed integer.
///
/// # Errors
/// Returns an error if the scalar is not representable.
pub fn cast_scalar_to_signed_integer(
    value: CastValueScalar,
    bits: u32,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<i128, CastValueError> {
    let min = signed_min(bits);
    let max = signed_max(bits);
    if let CastValueScalar::Unsigned(value) = value {
        #[expect(clippy::cast_sign_loss)]
        let max_unsigned = max as u128; // `max` is non-negative
        if value <= max_unsigned {
            #[expect(clippy::cast_possible_wrap)]
            return Ok(value as i128); // `value <= max` cannot wrap
        }
        return match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => Ok(max),
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_signed_unsigned(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        };
    }
    let value = match scalar_to_rounded_integer(value, rounding)? {
        RoundedScalar::Int(value) => value,
        RoundedScalar::Big(value) => {
            return match out_of_range {
                Some(CastValueOutOfRangeMode::Clamp) => Ok(if value < 0.0 { min } else { max }),
                Some(CastValueOutOfRangeMode::Wrap) => {
                    Ok(wrap_signed_unsigned(wrap_float(value, bits), bits))
                }
                None => Err(CastValueError::NotRepresentable),
            };
        }
    };
    if value < min || value > max {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => Ok(value.clamp(min, max)),
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_signed(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        }
    } else {
        Ok(value)
    }
}

/// Cast a scalar to an unsigned integer.
///
/// # Errors
/// Returns an error if the scalar is not representable.
pub fn cast_scalar_to_unsigned_integer(
    value: CastValueScalar,
    bits: u32,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<u128, CastValueError> {
    let max = unsigned_max(bits);
    if let CastValueScalar::Unsigned(value) = value {
        if value <= max {
            return Ok(value);
        }
        return match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => Ok(max),
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_unsigned_unsigned(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        };
    }
    let value = match scalar_to_rounded_integer(value, rounding)? {
        RoundedScalar::Int(value) => value,
        RoundedScalar::Big(value) => {
            return match out_of_range {
                Some(CastValueOutOfRangeMode::Clamp) => Ok(if value < 0.0 { 0 } else { max }),
                Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_float(value, bits)),
                None => Err(CastValueError::NotRepresentable),
            };
        }
    };
    if value < 0 || u128::try_from(value).map_or(true, |value| value > max) {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => {
                Ok(u128::try_from(value).map_or(0, |value| value.min(max)))
            }
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_unsigned(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        }
    } else {
        #[expect(clippy::cast_sign_loss)]
        let value = value as u128; // non-negative per the check above
        Ok(value)
    }
}

/// Cast a scalar to an `f64` quantity with target-range handling.
///
/// # Errors
/// Returns an error if the scalar is not representable.
pub fn cast_scalar_to_float_quantity(
    value: CastValueScalar,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
    min: f64,
    max: f64,
    has_nan: bool,
    has_infinity: bool,
) -> Result<f64, CastValueError> {
    float_quantity_from_f64(
        scalar_to_f64(value, rounding),
        rounding,
        out_of_range,
        min,
        max,
        has_nan,
        has_infinity,
    )
}

/// The target-range handling of [`cast_scalar_to_float_quantity`] for a value
/// already exactly rounded to `f64`.
fn float_quantity_from_f64(
    value: f64,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
    min: f64,
    max: f64,
    has_nan: bool,
    has_infinity: bool,
) -> Result<f64, CastValueError> {
    if value.is_nan() {
        return has_nan
            .then_some(f64::NAN)
            .ok_or(CastValueError::NotRepresentable);
    }
    if value.is_infinite() {
        if has_infinity {
            return Ok(value);
        }
        return match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) if value.is_sign_negative() => Ok(min),
            Some(CastValueOutOfRangeMode::Clamp) => Ok(max),
            Some(CastValueOutOfRangeMode::Wrap) => Err(CastValueError::InvalidWrapTarget),
            None => Err(CastValueError::NotRepresentable),
        };
    }
    if value < min {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => {
                Ok(if has_infinity { f64::NEG_INFINITY } else { min })
            }
            // The cast_value spec is currently underspecified for inverse decoding
            // when the configured encoded data type is integral with
            // out_of_range=wrap, but the decoded array data type is floating
            // point. Match zarr-python 3.2.1 by applying overflow rounding
            // to either the finite bound or infinity when the floating target
            // supports infinities.
            Some(CastValueOutOfRangeMode::Wrap) if has_infinity => match rounding {
                CastValueRoundingMode::TowardsZero | CastValueRoundingMode::TowardsPositive => {
                    Ok(min)
                }
                CastValueRoundingMode::NearestEven
                | CastValueRoundingMode::TowardsNegative
                | CastValueRoundingMode::NearestAway => Ok(f64::NEG_INFINITY),
            },
            Some(CastValueOutOfRangeMode::Wrap) => Err(CastValueError::InvalidWrapTarget),
            None => Err(CastValueError::NotRepresentable),
        }
    } else if value > max {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => {
                Ok(if has_infinity { f64::INFINITY } else { max })
            }
            Some(CastValueOutOfRangeMode::Wrap) if has_infinity => match rounding {
                CastValueRoundingMode::TowardsZero | CastValueRoundingMode::TowardsNegative => {
                    Ok(max)
                }
                CastValueRoundingMode::NearestEven
                | CastValueRoundingMode::TowardsPositive
                | CastValueRoundingMode::NearestAway => Ok(f64::INFINITY),
            },
            Some(CastValueOutOfRangeMode::Wrap) => Err(CastValueError::InvalidWrapTarget),
            None => Err(CastValueError::NotRepresentable),
        }
    } else {
        Ok(value)
    }
}

/// Macro to implement `cast_value` for signed integer data types.
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_cast_value_data_type_traits_signed_integer {
    ($marker:ty, $stored:ty, $bits:expr) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_is_integral(&self) -> bool {
                true
            }

            fn cast_value_repr(
                &self,
            ) -> Option<$crate::codec_traits::cast_value::CastValueRepr> {
                Some($crate::codec_traits::cast_value::CastValueRepr::Int {
                    bits: $bits,
                    stored: <$stored as
                        $crate::codec_traits::cast_value::CastValueIntStoredPrimitive>::STORED,
                })
            }

            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; core::mem::size_of::<$stored>()] =
                    bytes.try_into().map_err(|_| {
                        $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                    })?;
                let value = <$stored>::from_ne_bytes(bytes) as i128;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Signed(
                    value,
                ))
            }

            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
                output: &mut Vec<u8>,
            ) -> Result<(), $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_signed_integer(
                    value,
                    $bits,
                    rounding,
                    out_of_range,
                )?;
                let value = <$stored>::try_from(value).map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::NotRepresentable
                })?;
                output.extend_from_slice(&value.to_ne_bytes());
                Ok(())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
}

/// Macro to implement `cast_value` for unsigned integer data types.
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_cast_value_data_type_traits_unsigned_integer {
    ($marker:ty, $stored:ty, $bits:expr) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_is_integral(&self) -> bool {
                true
            }

            fn cast_value_repr(
                &self,
            ) -> Option<$crate::codec_traits::cast_value::CastValueRepr> {
                Some($crate::codec_traits::cast_value::CastValueRepr::Int {
                    bits: $bits,
                    stored: <$stored as
                        $crate::codec_traits::cast_value::CastValueIntStoredPrimitive>::STORED,
                })
            }

            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; core::mem::size_of::<$stored>()] =
                    bytes.try_into().map_err(|_| {
                        $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                    })?;
                let value = <$stored>::from_ne_bytes(bytes) as u128;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Unsigned(
                    value,
                ))
            }

            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
                output: &mut Vec<u8>,
            ) -> Result<(), $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_unsigned_integer(
                    value,
                    $bits,
                    rounding,
                    out_of_range,
                )?;
                let value = <$stored>::try_from(value).map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::NotRepresentable
                })?;
                output.extend_from_slice(&value.to_ne_bytes());
                Ok(())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
}

/// Macro to implement `cast_value` for floating point data types.
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_cast_value_data_type_traits_float {
    ($marker:ty, f16) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 2] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F16(
                        half::f16::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_repr(&self) -> Option<$crate::codec_traits::cast_value::CastValueRepr> {
                Some($crate::codec_traits::cast_value::CastValueRepr::F16)
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
                output: &mut Vec<u8>,
            ) -> Result<(), $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    half::f16::MIN.to_f64(),
                    half::f16::MAX.to_f64(),
                    true,
                    true,
                )?;
                output.extend_from_slice(
                    &$crate::codec_traits::cast_value::round_f64_to_f16(value, rounding)
                        .to_ne_bytes(),
                );
                Ok(())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, bf16) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 2] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::BF16(
                        half::bf16::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_repr(&self) -> Option<$crate::codec_traits::cast_value::CastValueRepr> {
                Some($crate::codec_traits::cast_value::CastValueRepr::BF16)
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
                output: &mut Vec<u8>,
            ) -> Result<(), $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    half::bf16::MIN.to_f64(),
                    half::bf16::MAX.to_f64(),
                    true,
                    true,
                )?;
                output.extend_from_slice(
                    &$crate::codec_traits::cast_value::round_f64_to_bf16(value, rounding)
                        .to_ne_bytes(),
                );
                Ok(())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, f32) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 4] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F32(
                        f32::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_repr(&self) -> Option<$crate::codec_traits::cast_value::CastValueRepr> {
                Some($crate::codec_traits::cast_value::CastValueRepr::F32)
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
                output: &mut Vec<u8>,
            ) -> Result<(), $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    f64::from(f32::MIN),
                    f64::from(f32::MAX),
                    true,
                    true,
                )?;
                output.extend_from_slice(
                    &$crate::codec_traits::cast_value::round_f64_to_f32(value, rounding)
                        .to_ne_bytes(),
                );
                Ok(())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, f64) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F64(
                        f64::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_repr(&self) -> Option<$crate::codec_traits::cast_value::CastValueRepr> {
                Some($crate::codec_traits::cast_value::CastValueRepr::F64)
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
                output: &mut Vec<u8>,
            ) -> Result<(), $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    f64::MIN,
                    f64::MAX,
                    true,
                    true,
                )?;
                output.extend_from_slice(
                    &$crate::codec_traits::cast_value::round_f64_to_f64(value, rounding)
                        .to_ne_bytes(),
                );
                Ok(())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, microfloat, $float_type:ty) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 1] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F32(
                        <$float_type>::from_bits(bytes[0]).to_f32(),
                    ),
                ))
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
                output: &mut Vec<u8>,
            ) -> Result<(), $crate::codec_traits::cast_value::CastValueError> {
                let has_nan = <$float_type>::NAN.is_nan();
                let has_infinity = <$float_type>::INFINITY.to_f32().is_infinite();
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    f64::from(<$float_type>::MIN.to_f32()),
                    f64::from(<$float_type>::MAX.to_f32()),
                    has_nan,
                    has_infinity,
                )?;
                let quantity = value;
                let value = <$float_type>::from_f64(quantity);
                let value = $crate::codec_traits::cast_value::round_float_with_next_after(
                    quantity,
                    value,
                    rounding,
                    |value| f64::from(value.to_f32()),
                    <$float_type>::next_down,
                    <$float_type>::next_up,
                );
                output.push(value.to_bits());
                Ok(())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
}

pub use crate::{
    _impl_cast_value_data_type_traits_float as impl_cast_value_data_type_traits_float,
    _impl_cast_value_data_type_traits_signed_integer as impl_cast_value_data_type_traits_signed_integer,
    _impl_cast_value_data_type_traits_unsigned_integer as impl_cast_value_data_type_traits_unsigned_integer,
};
